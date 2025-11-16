"""
DevDuck Tray - Modern tray app with server controls & agent capabilities
"""

import rumps
import socket
import threading
import json
import os
import tempfile
import webbrowser
from queue import Queue
from datetime import datetime
from pathlib import Path
import uuid


class DevDuckTray(rumps.App):
    def __init__(self):
        super().__init__("🦆", quit_button=None)

        # State
        self.command_queue = Queue()
        self.command_responses = {}
        self.ui_update_queue = Queue()
        self.state = {
            "status": "idle",
            "tcp_enabled": True,
            "ws_enabled": True,
            "mcp_enabled": True,
        }

        self.base_icon = "🦆"

        # Multi-stream state
        self.active_streams = {}
        self.recent_results = []
        self.max_recent = 5

        # Persistent menu items
        self.saved_menu_items = []

        # Build initial menu
        self._build_menu()

        # Start IPC listener
        self.socket_path = os.path.join(tempfile.gettempdir(), "devduck_tray.sock")
        self._start_ipc_listener()

        # Start command processor
        self.command_timer = rumps.Timer(self._process_commands, 0.1)
        self.command_timer.start()

        # Start UI update processor (main thread only)
        self.ui_timer = rumps.Timer(self._process_ui_updates, 0.2)
        self.ui_timer.start()

        # Import devduck
        try:
            os.environ["DEVDUCK_AUTO_START_SERVERS"] = "false"
            import devduck

            self.devduck = devduck
        except ImportError:
            self.devduck = None

    def _process_ui_updates(self, _):
        """Process UI updates on main thread only"""
        updated = False
        while not self.ui_update_queue.empty():
            update_type = self.ui_update_queue.get()
            if update_type == "menu":
                updated = True

        if updated:
            self._build_menu()

    def _request_menu_update(self):
        """Request menu update (thread-safe)"""
        self.ui_update_queue.put("menu")

    def _build_menu(self):
        """Build menu with server controls and agent capabilities"""
        self.menu.clear()

        # Test Button - First item for easy testing
        self.menu.add(
            rumps.MenuItem(
                "🧪 Test Agent", callback=self._create_callback("what time is it?")
            )
        )
        self.menu.add(rumps.separator)

        # Active Streams Section
        if self.active_streams:
            self.menu.add(rumps.MenuItem("🌊 Active Streams", callback=None))
            for stream_id, stream_data in list(self.active_streams.items()):
                icon = {
                    "thinking": "🤔",
                    "processing": "💡",
                    "complete": "✅",
                    "error": "❌",
                }.get(stream_data["status"], "💭")
                query_short = (
                    stream_data["query"][:30] + "..."
                    if len(stream_data["query"]) > 30
                    else stream_data["query"]
                )
                text_short = (
                    stream_data["text"][:40] + "..."
                    if len(stream_data["text"]) > 40
                    else stream_data["text"]
                )
                menu_text = f"  {icon} {query_short}: {text_short}"
                self.menu.add(rumps.MenuItem(menu_text, callback=None))
            self.menu.add(rumps.separator)

        # User-defined menu items
        if self.saved_menu_items:
            for item in self.saved_menu_items:
                if item.get("type") == "separator":
                    self.menu.add(rumps.separator)
                else:
                    label = item.get("title") or item.get("label", "Item")
                    query = item.get("query") or item.get("action", label)
                    self.menu.add(
                        rumps.MenuItem(label, callback=self._create_callback(query))
                    )
            self.menu.add(rumps.separator)

        # Recent Results Section
        if self.recent_results:
            self.menu.add(rumps.MenuItem("📝 Recent Results", callback=None))
            for query, result, timestamp in self.recent_results[: self.max_recent]:
                query_short = query[:25] + "..." if len(query) > 25 else query
                result_short = result[:35] + "..." if len(result) > 35 else result
                time_str = timestamp.strftime("%H:%M")
                menu_text = f"  [{time_str}] {query_short} → {result_short}"
                self.menu.add(
                    rumps.MenuItem(
                        menu_text,
                        callback=self._create_show_result_callback(query, result),
                    )
                )
            self.menu.add(rumps.separator)

        # Status
        self.menu.add(rumps.MenuItem(f"Status: {self.state['status']}", callback=None))
        self.menu.add(rumps.separator)

        # Server controls
        self.menu.add(rumps.MenuItem("🌐 Servers", callback=None))

        tcp_status = "✅" if self.state["tcp_enabled"] else "❌"
        self.menu.add(
            rumps.MenuItem(f"  {tcp_status} TCP (9999)", callback=self.toggle_tcp)
        )

        ws_status = "✅" if self.state["ws_enabled"] else "❌"
        self.menu.add(
            rumps.MenuItem(f"  {ws_status} WebSocket (8080)", callback=self.toggle_ws)
        )

        mcp_status = "✅" if self.state["mcp_enabled"] else "❌"
        self.menu.add(
            rumps.MenuItem(f"  {mcp_status} MCP (8000)", callback=self.toggle_mcp)
        )

        self.menu.add(rumps.separator)

        # Agent Capabilities
        self.menu.add(rumps.MenuItem("🤖 Agent Capabilities", callback=None))
        self.menu.add(
            rumps.MenuItem(
                "  👂 Start Clipboard Listening",
                callback=self._create_callback(
                    "start clipboard monitoring in background"
                ),
            )
        )
        self.menu.add(
            rumps.MenuItem(
                "  🎤 Start Background Listening",
                callback=self._create_callback("start background audio listening"),
            )
        )
        self.menu.add(
            rumps.MenuItem(
                "  📺 Start Screen Reader",
                callback=self._create_callback("start screen reader monitoring"),
            )
        )
        self.menu.add(
            rumps.MenuItem(
                "  👁️ Start YOLO Vision",
                callback=self._create_callback("start yolo vision detection"),
            )
        )

        self.menu.add(rumps.separator)

        # Actions
        self.menu.add(rumps.MenuItem("💻 Show Input", callback=self.show_input))
        self.menu.add(rumps.MenuItem("🌐 Web Dashboard", callback=self.open_dashboard))

        self.menu.add(rumps.separator)
        self.menu.add(rumps.MenuItem("Quit", callback=self.quit_app))

    def _create_show_result_callback(self, query, result):
        """Create callback to show full result in notification"""

        def callback(sender):
            rumps.notification("DevDuck Result", query, result)

        return callback

    def toggle_tcp(self, sender):
        """Toggle TCP server"""
        if self.devduck and hasattr(self.devduck.devduck.agent, "tool"):
            try:
                action = "stop_server" if self.state["tcp_enabled"] else "start_server"
                self.devduck.devduck.agent.tool.tcp(action=action, port=9999)
                self.state["tcp_enabled"] = not self.state["tcp_enabled"]
                self._request_menu_update()
                rumps.notification(
                    "DevDuck",
                    "",
                    f"TCP: {'ON' if self.state['tcp_enabled'] else 'OFF'}",
                )
            except Exception as e:
                rumps.notification("DevDuck Error", "", str(e))

    def toggle_ws(self, sender):
        """Toggle WebSocket server"""
        if self.devduck and hasattr(self.devduck.devduck.agent, "tool"):
            try:
                action = "stop_server" if self.state["ws_enabled"] else "start_server"
                self.devduck.devduck.agent.tool.websocket(action=action, port=8080)
                self.state["ws_enabled"] = not self.state["ws_enabled"]
                self._request_menu_update()
                rumps.notification(
                    "DevDuck",
                    "",
                    f"WebSocket: {'ON' if self.state['ws_enabled'] else 'OFF'}",
                )
            except Exception as e:
                rumps.notification("DevDuck Error", "", str(e))

    def toggle_mcp(self, sender):
        """Toggle MCP server"""
        if self.devduck and hasattr(self.devduck.devduck.agent, "tool"):
            try:
                action = "stop" if self.state["mcp_enabled"] else "start"
                self.devduck.devduck.agent.tool.mcp_server(action=action, port=8000)
                self.state["mcp_enabled"] = not self.state["mcp_enabled"]
                self._request_menu_update()
                rumps.notification(
                    "DevDuck",
                    "",
                    f"MCP: {'ON' if self.state['mcp_enabled'] else 'OFF'}",
                )
            except Exception as e:
                rumps.notification("DevDuck Error", "", str(e))

    def show_input(self, sender):
        """Show ambient input overlay"""
        ambient_socket = os.path.join(tempfile.gettempdir(), "devduck_ambient.sock")

        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(2.0)
            client.connect(ambient_socket)
            client.send(json.dumps({"action": "show"}).encode("utf-8"))
            client.close()
        except:
            # Ambient not running - try to start via devduck
            if self.devduck and hasattr(self.devduck.devduck.agent, "tool"):
                try:
                    self.devduck.devduck.agent.tool.ambient(action="start")
                    rumps.notification("DevDuck", "", "Input overlay started! 💻")
                except Exception as e:
                    rumps.notification("DevDuck Error", "", f"Failed: {e}")

    def open_dashboard(self, sender):
        """Open web dashboard"""
        webbrowser.open("https://cagataycali.github.io/devduck")

    def quit_app(self, sender):
        """Quit application"""
        rumps.quit_application()

    def _start_ipc_listener(self):
        """Start Unix socket listener"""

        def listener():
            try:
                if os.path.exists(self.socket_path):
                    os.unlink(self.socket_path)

                server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                server.bind(self.socket_path)
                server.listen(5)

                while True:
                    conn, _ = server.accept()
                    threading.Thread(
                        target=self._handle_connection, args=(conn,), daemon=True
                    ).start()
            except Exception as e:
                print(f"IPC error: {e}")

        threading.Thread(target=listener, daemon=True).start()

    def _handle_connection(self, conn):
        """Handle IPC connection"""
        try:
            data = conn.recv(4096)
            if data:
                command = json.loads(data.decode("utf-8"))
                request_id = command.get("request_id", id(command))

                self.command_queue.put((request_id, command))

                # Wait for response
                import time

                timeout = 5.0
                start = time.time()
                while time.time() - start < timeout:
                    if request_id in self.command_responses:
                        response = self.command_responses.pop(request_id)
                        conn.send(json.dumps(response).encode("utf-8"))
                        break
                    time.sleep(0.05)
                else:
                    conn.send(
                        json.dumps({"status": "error", "message": "timeout"}).encode(
                            "utf-8"
                        )
                    )
        except Exception as e:
            try:
                conn.send(
                    json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
                )
            except:
                pass
        finally:
            conn.close()

    def _process_commands(self, _):
        """Process queued commands on main thread"""
        while not self.command_queue.empty():
            request_id, cmd = self.command_queue.get()
            response = self._handle_command(cmd)
            self.command_responses[request_id] = response

    def _handle_command(self, cmd):
        """Handle IPC commands"""
        try:
            action = cmd.get("action")

            if action == "update_title":
                new_icon = cmd.get("title", "🦆")
                self.base_icon = new_icon
                self.title = self.base_icon
                return {"status": "success"}

            elif action == "set_progress":
                progress = cmd.get("progress", "idle")
                icons = {
                    "idle": "🦆",
                    "thinking": "🤔",
                    "processing": "💡",
                    "complete": "✅",
                    "error": "❌",
                }
                self.base_icon = icons.get(progress, "🦆")
                self.title = self.base_icon
                return {"status": "success"}

            elif action == "update_menu":
                items = cmd.get("items", [])
                self.saved_menu_items = items
                self._request_menu_update()
                return {"status": "success"}

            elif action == "notify":
                msg = cmd.get("message", {})
                rumps.notification(
                    msg.get("title", "DevDuck"),
                    msg.get("subtitle", ""),
                    msg.get("message", ""),
                )
                return {"status": "success"}

            elif action == "stream_text":
                text = cmd.get("text", "")
                stream_id = cmd.get("stream_id", "default")
                status = cmd.get("status", "processing")
                query = cmd.get("query", "")

                self.active_streams[stream_id] = {
                    "query": query,
                    "status": status,
                    "text": text,
                }

                self._request_menu_update()

                return {"status": "success"}

            elif action == "stream_complete":
                stream_id = cmd.get("stream_id", "default")
                if stream_id in self.active_streams:
                    stream_data = self.active_streams.pop(stream_id)
                    self.recent_results.insert(
                        0, (stream_data["query"], stream_data["text"], datetime.now())
                    )
                    self.recent_results = self.recent_results[: self.max_recent]
                    self._request_menu_update()
                return {"status": "success"}

            elif action == "show_input":
                self.show_input(None)
                return {"status": "success"}

            elif action in ["toggle_tcp", "toggle_ws", "toggle_mcp"]:
                if action == "toggle_tcp":
                    self.toggle_tcp(None)
                elif action == "toggle_ws":
                    self.toggle_ws(None)
                elif action == "toggle_mcp":
                    self.toggle_mcp(None)
                return {"status": "success"}

            return {"status": "error", "message": "Unknown action"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _create_callback(self, query):
        """Create callback for menu item"""

        def callback(sender):
            if self.devduck:
                stream_id = str(uuid.uuid4())[:8]

                self.active_streams[stream_id] = {
                    "query": query,
                    "status": "thinking",
                    "text": "Starting...",
                }
                self._request_menu_update()

                self.base_icon = "🤔"
                self.title = self.base_icon

                def run():
                    try:
                        self.active_streams[stream_id]["status"] = "processing"
                        self.active_streams[stream_id]["text"] = "Processing query..."
                        self._request_menu_update()

                        self.base_icon = "💡"
                        self.title = self.base_icon

                        result = self.devduck.ask(query)
                        result_str = str(result)

                        self.active_streams[stream_id]["status"] = "complete"
                        self.active_streams[stream_id]["text"] = result_str
                        self._request_menu_update()

                        self.base_icon = "✅"
                        self.title = self.base_icon

                        rumps.notification("DevDuck Result", query, result_str)

                        def move_to_recent():
                            if stream_id in self.active_streams:
                                stream_data = self.active_streams.pop(stream_id)
                                self.recent_results.insert(
                                    0,
                                    (
                                        stream_data["query"],
                                        stream_data["text"],
                                        datetime.now(),
                                    ),
                                )
                                self.recent_results = self.recent_results[
                                    : self.max_recent
                                ]
                                self._request_menu_update()
                            self._reset_icon()

                        threading.Timer(3.0, move_to_recent).start()

                    except Exception as e:
                        self.active_streams[stream_id]["status"] = "error"
                        self.active_streams[stream_id]["text"] = str(e)
                        self._request_menu_update()

                        self.base_icon = "❌"
                        self.title = self.base_icon
                        rumps.notification("DevDuck Error", query, str(e))

                        threading.Timer(
                            3.0, lambda: self._cleanup_stream(stream_id)
                        ).start()

                threading.Thread(target=run, daemon=True).start()

        return callback

    def _cleanup_stream(self, stream_id):
        """Remove stream and reset icon"""
        if stream_id in self.active_streams:
            self.active_streams.pop(stream_id)
            self._request_menu_update()
        self._reset_icon()

    def _reset_icon(self):
        """Reset icon to default"""
        self.base_icon = "🦆"
        self.title = self.base_icon


if __name__ == "__main__":
    app = DevDuckTray()
    app.run()
