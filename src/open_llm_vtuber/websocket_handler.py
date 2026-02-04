from typing import Dict, List, Optional, Callable, TypedDict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
from enum import Enum
import numpy as np
from loguru import logger

from .service_context import ServiceContext
from .chat_group import (
    ChatGroupManager,
    handle_group_operation,
    handle_client_disconnect,
    broadcast_to_group,
)
from .message_handler import message_handler
from .utils.stream_audio import prepare_audio_payload
from .chat_history_manager import (
    create_new_history,
    get_history,
    delete_history,
    get_history_list,
)
from .config_manager.utils import scan_config_alts_directory, scan_bg_directory
from .conversations.conversation_handler import (
    handle_conversation_trigger,
    handle_group_interrupt,
    handle_individual_interrupt,
)
from .cli_proxy_manager import CLIProxyManager, CLIProxyStatus


class MessageType(Enum):
    """Enum for WebSocket message types"""

    GROUP = ["add-client-to-group", "remove-client-from-group"]
    HISTORY = [
        "fetch-history-list",
        "fetch-and-set-history",
        "create-new-history",
        "delete-history",
    ]
    CONVERSATION = ["mic-audio-end", "text-input", "ai-speak-signal"]
    CONFIG = ["fetch-configs", "switch-config"]
    CONTROL = ["interrupt-signal", "audio-play-start"]
    DATA = ["mic-audio-data"]


class WSMessage(TypedDict, total=False):
    """Type definition for WebSocket messages"""

    type: str
    action: Optional[str]
    text: Optional[str]
    audio: Optional[List[float]]
    images: Optional[List[str]]
    history_uid: Optional[str]
    file: Optional[str]
    display_text: Optional[dict]


class WebSocketHandler:
    """Handles WebSocket connections and message routing"""

    def __init__(
        self,
        default_context_cache: ServiceContext,
        cli_proxy_manager: Optional[CLIProxyManager] = None,
    ):
        """Initialize the WebSocket handler with default context"""
        self.client_connections: Dict[str, WebSocket] = {}
        self.client_contexts: Dict[str, ServiceContext] = {}
        self.chat_group_manager = ChatGroupManager()
        self.current_conversation_tasks: Dict[str, Optional[asyncio.Task]] = {}
        self.default_context_cache = default_context_cache
        self.received_data_buffers: Dict[str, np.ndarray] = {}
        self.cli_proxy_manager = cli_proxy_manager

        # Message handlers mapping
        self._message_handlers = self._init_message_handlers()

    def _init_message_handlers(self) -> Dict[str, Callable]:
        """Initialize message type to handler mapping"""
        return {
            "add-client-to-group": self._handle_group_operation,
            "remove-client-from-group": self._handle_group_operation,
            "request-group-info": self._handle_group_info,
            "fetch-history-list": self._handle_history_list_request,
            "fetch-and-set-history": self._handle_fetch_history,
            "create-new-history": self._handle_create_history,
            "delete-history": self._handle_delete_history,
            "interrupt-signal": self._handle_interrupt,
            "mic-audio-data": self._handle_audio_data,
            "mic-audio-end": self._handle_conversation_trigger,
            "raw-audio-data": self._handle_raw_audio_data,
            "text-input": self._handle_conversation_trigger,
            "ai-speak-signal": self._handle_conversation_trigger,
            "fetch-configs": self._handle_fetch_configs,
            "switch-config": self._handle_config_switch,
            "fetch-backgrounds": self._handle_fetch_backgrounds,
            "audio-play-start": self._handle_audio_play_start,
            "request-init-config": self._handle_init_config_request,
            "heartbeat": self._handle_heartbeat,
            "request-cli-proxy-status": self._handle_cli_proxy_status_request,
            "get-oauth-providers": self._handle_get_oauth_providers,
            "trigger-oauth-login": self._handle_trigger_oauth_login,
            "revoke-oauth": self._handle_revoke_oauth,
            "get-llm-info": self._handle_get_llm_info,
            "change-model": self._handle_change_model,
        }

    async def handle_new_connection(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle new WebSocket connection setup

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client

        Raises:
            Exception: If initialization fails
        """
        try:
            session_service_context = await self._init_service_context(
                websocket.send_text, client_uid
            )

            await self._store_client_data(
                websocket, client_uid, session_service_context
            )

            await self._send_initial_messages(
                websocket, client_uid, session_service_context
            )

            logger.info(f"Connection established for client {client_uid}")

        except Exception as e:
            logger.error(
                f"Failed to initialize connection for client {client_uid}: {e}"
            )
            await self._cleanup_failed_connection(client_uid)
            raise

    async def _store_client_data(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Store client data and initialize group status"""
        self.client_connections[client_uid] = websocket
        self.client_contexts[client_uid] = session_service_context
        self.received_data_buffers[client_uid] = np.array([])

        self.chat_group_manager.client_group_map[client_uid] = ""
        await self.send_group_update(websocket, client_uid)

    async def _send_initial_messages(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Send initial connection messages to the client"""
        await websocket.send_text(
            json.dumps({"type": "full-text", "text": "Connection established"})
        )

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": session_service_context.live2d_model.model_info,
                    "conf_name": session_service_context.character_config.conf_name,
                    "conf_uid": session_service_context.character_config.conf_uid,
                    "client_uid": client_uid,
                }
            )
        )

        # Send initial group status
        await self.send_group_update(websocket, client_uid)

        # Send CLI proxy status if manager is available
        await self._send_cli_proxy_status(websocket)

        # Start microphone
        await websocket.send_text(json.dumps({"type": "control", "text": "start-mic"}))

    async def _init_service_context(
        self, send_text: Callable, client_uid: str
    ) -> ServiceContext:
        """Initialize service context for a new session by cloning the default context"""
        session_service_context = ServiceContext()
        await session_service_context.load_cache(
            config=self.default_context_cache.config.model_copy(deep=True),
            system_config=self.default_context_cache.system_config.model_copy(
                deep=True
            ),
            character_config=self.default_context_cache.character_config.model_copy(
                deep=True
            ),
            live2d_model=self.default_context_cache.live2d_model,
            asr_engine=self.default_context_cache.asr_engine,
            tts_engine=self.default_context_cache.tts_engine,
            vad_engine=self.default_context_cache.vad_engine,
            agent_engine=self.default_context_cache.agent_engine,
            translate_engine=self.default_context_cache.translate_engine,
            mcp_server_registery=self.default_context_cache.mcp_server_registery,
            tool_adapter=self.default_context_cache.tool_adapter,
            send_text=send_text,
            client_uid=client_uid,
        )
        return session_service_context

    async def handle_websocket_communication(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle ongoing WebSocket communication

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client
        """
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    message_handler.handle_message(client_uid, data)
                    await self._route_message(websocket, client_uid, data)
                except WebSocketDisconnect:
                    raise
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )
                    continue

        except WebSocketDisconnect:
            logger.info(f"Client {client_uid} disconnected")
            raise
        except Exception as e:
            logger.error(f"Fatal error in WebSocket communication: {e}")
            raise

    async def _route_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Route incoming message to appropriate handler

        Args:
            websocket: The WebSocket connection
            client_uid: Client identifier
            data: Message data
        """
        msg_type = data.get("type")
        if not msg_type:
            logger.warning("Message received without type")
            return

        handler = self._message_handlers.get(msg_type)
        if handler:
            await handler(websocket, client_uid, data)
        else:
            if msg_type != "frontend-playback-complete":
                logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_group_operation(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Handle group-related operations"""
        operation = data.get("type")
        target_uid = data.get(
            "invitee_uid" if operation == "add-client-to-group" else "target_uid"
        )

        await handle_group_operation(
            operation=operation,
            client_uid=client_uid,
            target_uid=target_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

    async def handle_disconnect(self, client_uid: str) -> None:
        """Handle client disconnection"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response="",
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )

        await handle_client_disconnect(
            client_uid=client_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

        # Clean up other client data
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        # Call context close to clean up resources (e.g., MCPClient)
        context = self.client_contexts.get(client_uid)
        if context:
            await context.close()

        logger.info(f"Client {client_uid} disconnected")
        message_handler.cleanup_client(client_uid)

    async def _cleanup_failed_connection(self, client_uid: str) -> None:
        """Clean up failed connection data"""
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self.chat_group_manager.client_group_map.pop(client_uid, None)

        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        message_handler.cleanup_client(client_uid)

    async def broadcast_to_group(
        self, group_members: list[str], message: dict, exclude_uid: str = None
    ) -> None:
        """Broadcasts a message to group members"""
        await broadcast_to_group(
            group_members=group_members,
            message=message,
            client_connections=self.client_connections,
            exclude_uid=exclude_uid,
        )

    async def send_group_update(self, websocket: WebSocket, client_uid: str):
        """Sends group information to a client"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            current_members = self.chat_group_manager.get_group_members(client_uid)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": current_members,
                        "is_owner": group.owner_uid == client_uid,
                    }
                )
            )
        else:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": [],
                        "is_owner": False,
                    }
                )
            )

    async def _handle_interrupt(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle conversation interruption"""
        heard_response = data.get("text", "")
        context = self.client_contexts[client_uid]
        group = self.chat_group_manager.get_client_group(client_uid)

        if group and len(group.members) > 1:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response=heard_response,
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )
        else:
            await handle_individual_interrupt(
                client_uid=client_uid,
                current_conversation_tasks=self.current_conversation_tasks,
                context=context,
                heard_response=heard_response,
            )

    async def _handle_history_list_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for chat history list"""
        context = self.client_contexts[client_uid]
        histories = get_history_list(context.character_config.conf_uid)
        await websocket.send_text(
            json.dumps({"type": "history-list", "histories": histories})
        )

    async def _handle_fetch_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle fetching and setting specific chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        # Update history_uid in service context
        context.history_uid = history_uid
        context.agent_engine.set_memory_from_history(
            conf_uid=context.character_config.conf_uid,
            history_uid=history_uid,
        )

        messages = [
            msg
            for msg in get_history(
                context.character_config.conf_uid,
                history_uid,
            )
            if msg["role"] != "system"
        ]
        await websocket.send_text(
            json.dumps({"type": "history-data", "messages": messages})
        )

    async def _handle_create_history(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle creation of new chat history"""
        context = self.client_contexts[client_uid]
        history_uid = create_new_history(context.character_config.conf_uid)
        if history_uid:
            context.history_uid = history_uid
            context.agent_engine.set_memory_from_history(
                conf_uid=context.character_config.conf_uid,
                history_uid=history_uid,
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "new-history-created",
                        "history_uid": history_uid,
                    }
                )
            )

    async def _handle_delete_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle deletion of chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        success = delete_history(
            context.character_config.conf_uid,
            history_uid,
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "history-deleted",
                    "success": success,
                    "history_uid": history_uid,
                }
            )
        )
        if history_uid == context.history_uid:
            context.history_uid = None

    async def _handle_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming audio data"""
        audio_data = data.get("audio", [])
        if audio_data:
            self.received_data_buffers[client_uid] = np.append(
                self.received_data_buffers[client_uid],
                np.array(audio_data, dtype=np.float32),
            )

    async def _handle_raw_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming raw audio data for VAD processing"""
        context = self.client_contexts[client_uid]
        chunk = data.get("audio", [])
        if chunk:
            for audio_bytes in context.vad_engine.detect_speech(chunk):
                if audio_bytes == b"<|PAUSE|>":
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "interrupt"})
                    )
                elif audio_bytes == b"<|RESUME|>":
                    pass
                elif len(audio_bytes) > 1024:
                    # Detected audio activity (voice)
                    self.received_data_buffers[client_uid] = np.append(
                        self.received_data_buffers[client_uid],
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32),
                    )
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "mic-audio-end"})
                    )

    async def _handle_conversation_trigger(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle triggers that start a conversation"""
        await handle_conversation_trigger(
            msg_type=data.get("type", ""),
            data=data,
            client_uid=client_uid,
            context=self.client_contexts[client_uid],
            websocket=websocket,
            client_contexts=self.client_contexts,
            client_connections=self.client_connections,
            chat_group_manager=self.chat_group_manager,
            received_data_buffers=self.received_data_buffers,
            current_conversation_tasks=self.current_conversation_tasks,
            broadcast_to_group=self.broadcast_to_group,
        )

    async def _handle_fetch_configs(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available configurations"""
        context = self.client_contexts[client_uid]
        config_files = scan_config_alts_directory(context.system_config.config_alts_dir)
        await websocket.send_text(
            json.dumps({"type": "config-files", "configs": config_files})
        )

    async def _handle_config_switch(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle switching to a different configuration"""
        config_file_name = data.get("file")
        if config_file_name:
            context = self.client_contexts[client_uid]
            await context.handle_config_switch(websocket, config_file_name)

    async def _handle_fetch_backgrounds(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available background images"""
        bg_files = scan_bg_directory()
        await websocket.send_text(
            json.dumps({"type": "background-files", "files": bg_files})
        )

    async def _handle_audio_play_start(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Handle audio playback start notification
        """
        group_members = self.chat_group_manager.get_group_members(client_uid)
        if len(group_members) > 1:
            display_text = data.get("display_text")
            if display_text:
                silent_payload = prepare_audio_payload(
                    audio_path=None,
                    display_text=display_text,
                    actions=None,
                    forwarded=True,
                )
                await self.broadcast_to_group(
                    group_members, silent_payload, exclude_uid=client_uid
                )

    async def _handle_group_info(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle group info request"""
        await self.send_group_update(websocket, client_uid)

    async def _handle_init_config_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for initialization configuration"""
        context = self.client_contexts.get(client_uid)
        if not context:
            context = self.default_context_cache

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": context.live2d_model.model_info,
                    "conf_name": context.character_config.conf_name,
                    "conf_uid": context.character_config.conf_uid,
                    "client_uid": client_uid,
                }
            )
        )

    async def _handle_heartbeat(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle heartbeat messages from clients"""
        try:
            await websocket.send_json({"type": "heartbeat-ack"})
        except Exception as e:
            logger.error(f"Error sending heartbeat acknowledgment: {e}")

    async def _send_cli_proxy_status(self, websocket: WebSocket) -> None:
        """Send CLI proxy authentication status to the client"""
        if self.cli_proxy_manager is None:
            return

        try:
            status = await self.cli_proxy_manager.check_authentication()
            status_message = self._get_cli_proxy_status_message(status)

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "cli-proxy-status",
                        "status": status.value,
                        "message": status_message,
                    }
                )
            )
        except Exception as e:
            logger.error(f"Error sending CLI proxy status: {e}")

    def _get_cli_proxy_status_message(self, status: CLIProxyStatus) -> str:
        """Get a user-friendly message for CLI proxy status"""
        messages = {
            CLIProxyStatus.NOT_RUNNING: (
                "CLIProxyAPI is not running. "
                f"Run '{self.cli_proxy_manager.get_login_command()}' in a separate terminal."
            ),
            CLIProxyStatus.AUTHENTICATED: "CLIProxyAPI is authenticated and ready.",
            CLIProxyStatus.NOT_AUTHENTICATED: (
                "CLIProxyAPI requires authentication. "
                f"Run '{self.cli_proxy_manager.get_login_command()}' in a separate terminal."
            ),
            CLIProxyStatus.ERROR: "CLIProxyAPI encountered an error.",
        }
        return messages.get(status, "Unknown CLI proxy status")

    async def _handle_cli_proxy_status_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for CLI proxy authentication status"""
        await self._send_cli_proxy_status(websocket)

    async def _handle_get_oauth_providers(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for OAuth providers list with authentication status"""
        try:
            if self.cli_proxy_manager is None:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "oauth-providers",
                            "providers": [],
                            "message": "CLIProxyAPI is not configured",
                        }
                    )
                )
                return

            providers = await self.cli_proxy_manager.get_authenticated_providers()
            providers_data = [
                {
                    "name": p.name,
                    "display_name": p.display_name,
                    "authenticated": p.authenticated,
                    "email": p.email,
                }
                for p in providers
            ]

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "oauth-providers",
                        "providers": providers_data,
                    }
                )
            )
        except Exception as e:
            logger.error(f"Error getting OAuth providers: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "oauth-providers",
                        "providers": [],
                        "message": f"Error: {str(e)}",
                    }
                )
            )

    async def _handle_trigger_oauth_login(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request to trigger OAuth login for a specific provider"""
        provider = data.get("provider", "")

        if not provider:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "oauth-login-result",
                        "success": False,
                        "message": "Provider not specified",
                    }
                )
            )
            return

        if self.cli_proxy_manager is None:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "oauth-login-result",
                        "success": False,
                        "message": "CLIProxyAPI is not configured",
                    }
                )
            )
            return

        try:
            # Notify client that login is starting
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "oauth-login-started",
                        "provider": provider,
                        "message": f"Starting {provider} authentication... Please check your browser.",
                    }
                )
            )

            # Trigger OAuth login
            result = await self.cli_proxy_manager.trigger_oauth_login(provider)

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "oauth-login-result",
                        "provider": provider,
                        "success": result.get("success", False),
                        "message": result.get("message", ""),
                    }
                )
            )

            # If successful, send updated providers list
            if result.get("success"):
                await self._handle_get_oauth_providers(websocket, client_uid, {})

        except Exception as e:
            logger.error(f"Error triggering OAuth login: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "oauth-login-result",
                        "provider": provider,
                        "success": False,
                        "message": f"Error: {str(e)}",
                    }
                )
            )

    async def _handle_revoke_oauth(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request to revoke OAuth authentication for a specific provider"""
        provider = data.get("provider", "")

        if not provider:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "oauth-revoke-result",
                        "success": False,
                        "message": "Provider not specified",
                    }
                )
            )
            return

        if self.cli_proxy_manager is None:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "oauth-revoke-result",
                        "success": False,
                        "message": "CLIProxyAPI is not configured",
                    }
                )
            )
            return

        try:
            # Revoke OAuth authentication
            result = await self.cli_proxy_manager.revoke_oauth(provider)

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "oauth-revoke-result",
                        "provider": provider,
                        "success": result.get("success", False),
                        "message": result.get("message", ""),
                    }
                )
            )

            # Send updated providers list
            await self._handle_get_oauth_providers(websocket, client_uid, {})

        except Exception as e:
            logger.error(f"Error revoking OAuth: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "oauth-revoke-result",
                        "provider": provider,
                        "success": False,
                        "message": f"Error: {str(e)}",
                    }
                )
            )

    def _detect_provider_from_model(self, model: str) -> str:
        """Detect actual provider from model name."""
        if not model:
            return "unknown"
        model_lower = model.lower()
        if "claude" in model_lower:
            return "Claude (Anthropic)"
        elif "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            return "OpenAI"
        elif "gemini" in model_lower:
            return "Google Gemini"
        elif "llama" in model_lower:
            return "Meta Llama"
        elif "qwen" in model_lower:
            return "Qwen (Alibaba)"
        elif "mistral" in model_lower or "mixtral" in model_lower:
            return "Mistral AI"
        elif "deepseek" in model_lower:
            return "DeepSeek"
        return "unknown"

    async def _handle_get_llm_info(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for current LLM information"""
        try:
            context = self.client_contexts.get(client_uid)
            llm_info = {
                "type": "llm-info",
                "provider": None,
                "model": None,
                "base_url": None,
                "config_provider": None,
                "cli_proxy_enabled": self.cli_proxy_manager is not None,
                "available_models": [],
            }

            if context and context.character_config and context.character_config.agent_config:
                agent_config = context.character_config.agent_config
                agent_settings = agent_config.agent_settings

                # Get current LLM provider from basic_memory_agent settings
                if agent_settings and agent_settings.basic_memory_agent:
                    llm_provider = agent_settings.basic_memory_agent.llm_provider
                    llm_info["config_provider"] = llm_provider

                    # Get model and base_url from llm_configs
                    if agent_config.llm_configs:
                        llm_configs_dict = agent_config.llm_configs.model_dump()
                        provider_config = llm_configs_dict.get(llm_provider, {})
                        model = provider_config.get("model")
                        llm_info["model"] = model
                        llm_info["base_url"] = provider_config.get("base_url")

                        # Detect actual provider from model name
                        llm_info["provider"] = self._detect_provider_from_model(model)

            # Add CLI proxy status and available models if enabled
            if self.cli_proxy_manager:
                logger.info("CLI proxy manager is available, fetching status and models")
                status = await self.cli_proxy_manager.check_authentication()
                llm_info["cli_proxy_status"] = status.value
                llm_info["cli_proxy_authenticated"] = status == CLIProxyStatus.AUTHENTICATED

                # Get available models from CLI Proxy
                try:
                    available_models = await self.cli_proxy_manager.get_available_models()
                    logger.info(f"Got {len(available_models)} available models")
                    llm_info["available_models"] = available_models
                except Exception as e:
                    logger.warning(f"Failed to get available models: {e}")

            await websocket.send_text(json.dumps(llm_info))

        except Exception as e:
            logger.error(f"Error getting LLM info: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "llm-info",
                        "error": str(e),
                    }
                )
            )

    async def _handle_change_model(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request to change the LLM model"""
        import yaml as pyyaml
        from pathlib import Path

        model_id = data.get("model_id", "")

        if not model_id:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "change-model-result",
                        "success": False,
                        "message": "Model ID not specified",
                    }
                )
            )
            return

        try:
            config_path = Path("conf.yaml")

            # Read current config
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = pyyaml.safe_load(f)

            if not config_data:
                raise ValueError("Empty config file")

            # Navigate to the model setting and update it
            # Path: character_config -> agent_config -> llm_configs -> openai_compatible_llm -> model
            # Ensure all nested dicts exist and maintain references
            if "character_config" not in config_data:
                config_data["character_config"] = {}
            if "agent_config" not in config_data["character_config"]:
                config_data["character_config"]["agent_config"] = {}
            if "llm_configs" not in config_data["character_config"]["agent_config"]:
                config_data["character_config"]["agent_config"]["llm_configs"] = {}
            if "openai_compatible_llm" not in config_data["character_config"]["agent_config"]["llm_configs"]:
                config_data["character_config"]["agent_config"]["llm_configs"]["openai_compatible_llm"] = {}

            openai_compatible = config_data["character_config"]["agent_config"]["llm_configs"]["openai_compatible_llm"]
            old_model = openai_compatible.get("model", "")
            openai_compatible["model"] = model_id

            # Save updated config
            with open(config_path, "w", encoding="utf-8") as f:
                pyyaml.dump(config_data, f, allow_unicode=True, default_flow_style=False)

            logger.info(f"Model changed from '{old_model}' to '{model_id}'")

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "change-model-result",
                        "success": True,
                        "message": f"Model changed to {model_id}. Restart the conversation for the change to take effect.",
                        "old_model": old_model,
                        "new_model": model_id,
                    }
                )
            )

            # Send updated LLM info
            await self._handle_get_llm_info(websocket, client_uid, {})

        except Exception as e:
            logger.error(f"Error changing model: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "change-model-result",
                        "success": False,
                        "message": f"Error: {str(e)}",
                    }
                )
            )
