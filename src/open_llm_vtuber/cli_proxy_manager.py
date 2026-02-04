# cli_proxy_manager.py
"""
CLIProxyAPI process manager for Open-LLM-VTuber.

This module handles:
- CLIProxyAPI executable detection (WinGet path, PATH)
- Process start/stop/restart
- Authentication status checking via /v1/models endpoint
- Status change callbacks support
"""

import os
import sys
import shutil
import subprocess
import asyncio
from pathlib import Path
from typing import Callable, Optional, Dict, List, Any
from enum import Enum
from dataclasses import dataclass

import httpx
from loguru import logger


@dataclass
class ProviderInfo:
    """Information about an OAuth provider."""
    name: str
    display_name: str
    login_flag: str
    authenticated: bool = False
    email: Optional[str] = None


# Supported OAuth providers in CLIProxyAPI
SUPPORTED_PROVIDERS: Dict[str, Dict[str, str]] = {
    "claude": {
        "display_name": "Claude (Anthropic)",
        "login_flag": "-claude-login",
    },
    "google": {
        "display_name": "Google (Gemini)",
        "login_flag": "-login",
    },
    "antigravity": {
        "display_name": "Antigravity",
        "login_flag": "-antigravity-login",
    },
    "codex": {
        "display_name": "Codex",
        "login_flag": "-codex-login",
    },
    "qwen": {
        "display_name": "Qwen",
        "login_flag": "-qwen-login",
    },
    "iflow": {
        "display_name": "iFlow",
        "login_flag": "-iflow-login",
    },
}


class CLIProxyStatus(Enum):
    """Status of CLI Proxy API authentication."""

    NOT_RUNNING = "not_running"
    AUTHENTICATED = "authenticated"
    NOT_AUTHENTICATED = "not_authenticated"
    ERROR = "error"


class CLIProxyManager:
    """
    Manages the CLIProxyAPI process lifecycle and authentication status.

    Attributes:
        port: The port on which CLIProxyAPI runs.
        login_provider: The OAuth provider to use (e.g., 'claude').
        process: The subprocess running CLIProxyAPI.
        status: Current authentication status.
    """

    EXECUTABLE_NAME = "cli-proxy-api"
    WINDOWS_EXECUTABLE_NAME = "cli-proxy-api.exe"

    # Common WinGet installation paths for Windows
    WINGET_PATHS = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path(os.environ.get("PROGRAMFILES", "")) / "WinGet" / "Packages",
    ]

    def __init__(
        self,
        port: int = 8317,
        login_provider: str = "claude",
        on_status_change: Optional[Callable[[CLIProxyStatus, str], None]] = None,
    ):
        """
        Initialize the CLI Proxy Manager.

        Args:
            port: Port for CLIProxyAPI to listen on.
            login_provider: OAuth provider (e.g., 'claude').
            on_status_change: Callback function when status changes.
        """
        self.port = port
        self.login_provider = login_provider
        self.on_status_change = on_status_change
        self.process: Optional[subprocess.Popen] = None
        self.status = CLIProxyStatus.NOT_RUNNING
        self._executable_path: Optional[Path] = None

    def _find_executable(self) -> Optional[Path]:
        """
        Find the CLIProxyAPI executable.

        Searches in the following order:
        1. System PATH
        2. WinGet installation paths (Windows only)

        Returns:
            Path to the executable if found, None otherwise.
        """
        if self._executable_path and self._executable_path.exists():
            return self._executable_path

        # Check system PATH first
        exe_name = (
            self.WINDOWS_EXECUTABLE_NAME
            if sys.platform == "win32"
            else self.EXECUTABLE_NAME
        )
        path_executable = shutil.which(exe_name)
        if path_executable:
            self._executable_path = Path(path_executable)
            logger.debug(f"Found CLIProxyAPI in PATH: {self._executable_path}")
            return self._executable_path

        # On Windows, check WinGet paths
        if sys.platform == "win32":
            for winget_path in self.WINGET_PATHS:
                if not winget_path.exists():
                    continue
                # Search for cli-proxy-api in WinGet packages
                for package_dir in winget_path.iterdir():
                    if "cli-proxy-api" in package_dir.name.lower():
                        exe_path = package_dir / self.WINDOWS_EXECUTABLE_NAME
                        if exe_path.exists():
                            self._executable_path = exe_path
                            logger.debug(
                                f"Found CLIProxyAPI in WinGet: {self._executable_path}"
                            )
                            return self._executable_path

        logger.warning("CLIProxyAPI executable not found")
        return None

    def _update_status(self, new_status: CLIProxyStatus, message: str = "") -> None:
        """Update status and trigger callback if set."""
        if self.status != new_status:
            self.status = new_status
            if self.on_status_change:
                self.on_status_change(new_status, message)

    async def check_authentication(self) -> CLIProxyStatus:
        """
        Check if CLIProxyAPI is authenticated by calling /v1/models endpoint.

        Returns:
            Current authentication status.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{self.port}/v1/models")
                if response.status_code == 200:
                    self._update_status(
                        CLIProxyStatus.AUTHENTICATED, "CLIProxyAPI is authenticated"
                    )
                    return CLIProxyStatus.AUTHENTICATED
                elif response.status_code == 401:
                    self._update_status(
                        CLIProxyStatus.NOT_AUTHENTICATED,
                        "CLIProxyAPI requires authentication",
                    )
                    return CLIProxyStatus.NOT_AUTHENTICATED
                else:
                    self._update_status(
                        CLIProxyStatus.ERROR,
                        f"Unexpected response from CLIProxyAPI: {response.status_code}",
                    )
                    return CLIProxyStatus.ERROR
        except httpx.ConnectError:
            # Server not running or not responding
            if self.process is None:
                self._update_status(
                    CLIProxyStatus.NOT_RUNNING, "CLIProxyAPI is not running"
                )
                return CLIProxyStatus.NOT_RUNNING
            else:
                self._update_status(
                    CLIProxyStatus.ERROR, "CLIProxyAPI is not responding"
                )
                return CLIProxyStatus.ERROR
        except Exception as e:
            logger.error(f"Error checking CLIProxyAPI authentication: {e}")
            self._update_status(CLIProxyStatus.ERROR, str(e))
            return CLIProxyStatus.ERROR

    def start(self) -> bool:
        """
        Start the CLIProxyAPI process.

        Returns:
            True if started successfully, False otherwise.
        """
        if self.process is not None and self.process.poll() is None:
            logger.info("CLIProxyAPI is already running")
            return True

        executable = self._find_executable()
        if not executable:
            logger.warning(
                "CLIProxyAPI executable not found. "
                "Please install it from: https://github.com/anthropics/cli-proxy-api"
            )
            self._update_status(
                CLIProxyStatus.NOT_RUNNING,
                "CLIProxyAPI executable not found",
            )
            return False

        try:
            # Build command with port argument
            cmd = [str(executable), "-p", str(self.port)]

            # Start the process
            logger.info(f"Starting CLIProxyAPI on port {self.port}")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                ),
            )

            logger.info(f"CLIProxyAPI started with PID {self.process.pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to start CLIProxyAPI: {e}")
            self._update_status(CLIProxyStatus.ERROR, f"Failed to start: {e}")
            return False

    def stop(self) -> None:
        """Stop the CLIProxyAPI process."""
        if self.process is None:
            return

        try:
            if self.process.poll() is None:
                logger.info("Stopping CLIProxyAPI...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("CLIProxyAPI did not terminate, killing...")
                    self.process.kill()
                    self.process.wait()
                logger.info("CLIProxyAPI stopped")
        except Exception as e:
            logger.error(f"Error stopping CLIProxyAPI: {e}")
        finally:
            self.process = None
            self._update_status(CLIProxyStatus.NOT_RUNNING, "CLIProxyAPI stopped")

    def restart(self) -> bool:
        """
        Restart the CLIProxyAPI process.

        Returns:
            True if restarted successfully, False otherwise.
        """
        self.stop()
        return self.start()

    async def start_and_wait_for_ready(
        self, timeout: float = 10.0, check_interval: float = 0.5
    ) -> CLIProxyStatus:
        """
        Start CLIProxyAPI and wait for it to become ready.

        Args:
            timeout: Maximum time to wait for readiness in seconds.
            check_interval: Interval between readiness checks in seconds.

        Returns:
            The authentication status after startup.
        """
        if not self.start():
            return CLIProxyStatus.NOT_RUNNING

        # Wait for the server to become ready
        elapsed = 0.0
        while elapsed < timeout:
            await asyncio.sleep(check_interval)
            elapsed += check_interval

            status = await self.check_authentication()
            if status != CLIProxyStatus.ERROR:
                return status

        logger.warning(f"CLIProxyAPI did not become ready within {timeout}s")
        return await self.check_authentication()

    def get_base_url(self) -> str:
        """Get the base URL for the CLIProxyAPI."""
        return f"http://localhost:{self.port}"

    def get_login_command(self) -> str:
        """Get the command to run for OAuth login."""
        return f"cli-proxy-api -{self.login_provider}-login"

    def print_auth_warning(self) -> None:
        """Print authentication warning to console with instructions."""
        border = "=" * 56
        message = f"""
{border}
  CLIProxyAPI requires authentication!
  Run the following command in a separate terminal:

    {self.get_login_command()}

{border}
"""
        logger.warning(message)

    async def get_authenticated_providers(self) -> List[ProviderInfo]:
        """
        Get list of all providers with their authentication status.

        When CLI Proxy is authenticated, marks the login_provider as authenticated.
        Also checks credential files for other providers.

        Returns:
            List of ProviderInfo with authentication status.
        """
        providers = []

        # Credential file patterns for each provider
        credential_patterns = {
            "claude": ["claude-*.json"],
            "google": ["google-*.json", "gemini-*.json"],
            "antigravity": ["antigravity-*.json", "chatgpt-*.json"],
            "codex": ["codex-*.json"],
            "qwen": ["qwen-*.json"],
            "iflow": ["iflow-*.json"],
        }

        # Directories to search for credential files
        home_dir = Path.home()
        auth_dir = home_dir / ".cli-proxy-api"
        search_dirs = [auth_dir, home_dir]

        for provider_id, provider_data in SUPPORTED_PROVIDERS.items():
            provider_info = ProviderInfo(
                name=provider_id,
                display_name=provider_data["display_name"],
                login_flag=provider_data["login_flag"],
                authenticated=False,
                email=None,
            )

            # If CLI Proxy is authenticated and this is the login provider, mark as authenticated
            if self.status == CLIProxyStatus.AUTHENTICATED and provider_id == self.login_provider:
                provider_info.authenticated = True
            else:
                # Check if credential files exist for this provider
                patterns = credential_patterns.get(provider_id, [f"{provider_id}-*.json"])
                for search_dir in search_dirs:
                    if not search_dir.exists():
                        continue
                    for pattern in patterns:
                        matching_files = list(search_dir.glob(pattern))
                        if matching_files:
                            provider_info.authenticated = True
                            break
                    if provider_info.authenticated:
                        break

            providers.append(provider_info)

        return providers

    def _mark_provider_authenticated(
        self, providers: List[ProviderInfo], provider_name: str
    ) -> None:
        """Mark a provider as authenticated in the providers list."""
        for provider in providers:
            if provider.name == provider_name:
                provider.authenticated = True
                break

    async def trigger_oauth_login(self, provider: str) -> Dict[str, Any]:
        """
        Trigger OAuth login for a specific provider.

        This opens a browser for OAuth authentication. The result needs to be
        checked after the user completes authentication.

        Args:
            provider: Provider name (e.g., 'claude', 'google').

        Returns:
            Dict with status and message.
        """
        if provider not in SUPPORTED_PROVIDERS:
            return {
                "success": False,
                "message": f"Unknown provider: {provider}",
            }

        executable = self._find_executable()
        if not executable:
            return {
                "success": False,
                "message": "CLIProxyAPI executable not found",
            }

        login_flag = SUPPORTED_PROVIDERS[provider]["login_flag"]

        try:
            # Find config file location - use user's home directory
            home_dir = Path.home()
            config_path = home_dir / "config.yaml"

            # Run the login command (this will open browser)
            cmd = [str(executable), login_flag]
            if config_path.exists():
                cmd.extend(["-config", str(config_path)])

            logger.info(f"Triggering OAuth login for {provider}: {' '.join(cmd)}")

            # Start the process and don't wait (it will open browser)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(home_dir),
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                ),
            )

            # Wait for the process to complete (user needs to complete OAuth)
            try:
                stdout, stderr = process.communicate(timeout=120)

                if process.returncode == 0:
                    return {
                        "success": True,
                        "message": f"{SUPPORTED_PROVIDERS[provider]['display_name']} authentication successful!",
                    }
                else:
                    error_msg = stderr.decode("utf-8", errors="ignore") if stderr else "Unknown error"
                    return {
                        "success": False,
                        "message": f"Authentication failed: {error_msg}",
                    }
            except subprocess.TimeoutExpired:
                process.kill()
                return {
                    "success": False,
                    "message": "Authentication timed out. Please try again.",
                }

        except Exception as e:
            logger.error(f"Error triggering OAuth login: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
            }

    async def get_current_llm_info(self) -> Dict[str, Any]:
        """
        Get information about the current LLM configuration.

        Returns:
            Dict with current LLM provider, model, and status.
        """
        status = await self.check_authentication()

        return {
            "provider": "cli_proxy_api",
            "base_url": self.get_base_url(),
            "port": self.port,
            "status": status.value,
            "authenticated": status == CLIProxyStatus.AUTHENTICATED,
        }

    async def revoke_oauth(self, provider: str) -> Dict[str, Any]:
        """
        Revoke OAuth authentication for a specific provider by deleting credential files.

        Args:
            provider: Provider name (e.g., 'claude', 'google').

        Returns:
            Dict with status and message.
        """
        if provider not in SUPPORTED_PROVIDERS:
            return {
                "success": False,
                "message": f"Unknown provider: {provider}",
            }

        # Find and delete credential files for this provider
        home_dir = Path.home()
        auth_dir = home_dir / ".cli-proxy-api"

        # Also check home directory for credential files
        search_dirs = [auth_dir, home_dir]

        # Provider-specific file patterns
        provider_patterns = {
            "claude": ["claude-*.json"],
            "google": ["google-*.json", "gemini-*.json"],
            "antigravity": ["antigravity-*.json"],
            "codex": ["codex-*.json"],
            "qwen": ["qwen-*.json"],
            "iflow": ["iflow-*.json"],
        }

        patterns = provider_patterns.get(provider, [f"{provider}-*.json"])
        deleted_files = []

        try:
            import glob

            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue

                for pattern in patterns:
                    for file_path in search_dir.glob(pattern):
                        if file_path.is_file():
                            file_path.unlink()
                            deleted_files.append(str(file_path))
                            logger.info(f"Deleted credential file: {file_path}")

            if deleted_files:
                # Restart the CLI Proxy API to pick up the changes
                if self.process is not None:
                    self.restart()

                return {
                    "success": True,
                    "message": f"Successfully revoked {SUPPORTED_PROVIDERS[provider]['display_name']} authentication.",
                    "deleted_files": deleted_files,
                }
            else:
                return {
                    "success": False,
                    "message": f"No credential files found for {SUPPORTED_PROVIDERS[provider]['display_name']}.",
                }

        except Exception as e:
            logger.error(f"Error revoking OAuth: {e}")
            return {
                "success": False,
                "message": f"Error revoking authentication: {str(e)}",
            }

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from CLIProxyAPI.

        If CLIProxyAPI doesn't return models, returns a list of common models
        based on authentication status and login provider.

        Returns:
            List of model information dicts with id, owned_by, etc.
        """
        models = []

        # Try to get models from CLIProxyAPI first
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{self.port}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    models_data = data.get("data", [])
                    for model in models_data:
                        models.append({
                            "id": model.get("id"),
                            "owned_by": model.get("owned_by"),
                            "created": model.get("created"),
                        })
        except Exception as e:
            logger.debug(f"Could not fetch models from CLIProxyAPI: {e}")

        # If no models returned but authenticated, provide common models
        if not models and self.status == CLIProxyStatus.AUTHENTICATED:
            models = self._get_common_models_for_login_provider()
            logger.debug(f"Using common models for {self.login_provider}: {len(models)} models")

        return models

    def _get_common_models_for_login_provider(self) -> List[Dict[str, Any]]:
        """
        Get common models for the current login provider.

        Returns:
            List of common models for the login provider.
        """
        # Common models for each provider
        provider_models = {
            "claude": [
                {"id": "claude-sonnet-4-5-20250929", "owned_by": "anthropic"},
                {"id": "claude-opus-4-5-20251101", "owned_by": "anthropic"},
                {"id": "claude-3-5-sonnet-20241022", "owned_by": "anthropic"},
                {"id": "claude-3-5-haiku-20241022", "owned_by": "anthropic"},
            ],
            "google": [
                {"id": "gemini-2.0-flash", "owned_by": "google"},
                {"id": "gemini-1.5-pro", "owned_by": "google"},
                {"id": "gemini-1.5-flash", "owned_by": "google"},
            ],
            "antigravity": [
                {"id": "gpt-4o", "owned_by": "openai"},
                {"id": "gpt-4o-mini", "owned_by": "openai"},
                {"id": "o1", "owned_by": "openai"},
                {"id": "o1-mini", "owned_by": "openai"},
            ],
            "qwen": [
                {"id": "qwen-max", "owned_by": "alibaba"},
                {"id": "qwen-plus", "owned_by": "alibaba"},
                {"id": "qwen-turbo", "owned_by": "alibaba"},
            ],
        }

        return provider_models.get(self.login_provider, [])

    def _get_common_models_for_providers(
        self, providers: List[ProviderInfo]
    ) -> List[Dict[str, Any]]:
        """
        Get common models for authenticated providers.

        Args:
            providers: List of provider info with authentication status.

        Returns:
            List of common models for authenticated providers.
        """
        # Common models for each provider
        provider_models = {
            "claude": [
                {"id": "claude-sonnet-4-5-20250929", "owned_by": "anthropic"},
                {"id": "claude-opus-4-5-20251101", "owned_by": "anthropic"},
                {"id": "claude-3-5-sonnet-20241022", "owned_by": "anthropic"},
                {"id": "claude-3-5-haiku-20241022", "owned_by": "anthropic"},
            ],
            "google": [
                {"id": "gemini-2.0-flash", "owned_by": "google"},
                {"id": "gemini-1.5-pro", "owned_by": "google"},
                {"id": "gemini-1.5-flash", "owned_by": "google"},
            ],
            "antigravity": [
                {"id": "gpt-4o", "owned_by": "openai"},
                {"id": "gpt-4o-mini", "owned_by": "openai"},
                {"id": "o1", "owned_by": "openai"},
                {"id": "o1-mini", "owned_by": "openai"},
            ],
            "qwen": [
                {"id": "qwen-max", "owned_by": "alibaba"},
                {"id": "qwen-plus", "owned_by": "alibaba"},
                {"id": "qwen-turbo", "owned_by": "alibaba"},
            ],
        }

        models = []
        for provider in providers:
            if provider.authenticated and provider.name in provider_models:
                models.extend(provider_models[provider.name])

        return models


# Global manager instance (can be configured via config)
_manager_instance: Optional[CLIProxyManager] = None


def get_cli_proxy_manager() -> Optional[CLIProxyManager]:
    """Get the global CLI Proxy Manager instance."""
    return _manager_instance


def set_cli_proxy_manager(manager: CLIProxyManager) -> None:
    """Set the global CLI Proxy Manager instance."""
    global _manager_instance
    _manager_instance = manager
