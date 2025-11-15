"""
🦆 DevDuck Ambient Input - Preserves scroll position
"""

import tkinter as tk
from tkinter import font as tkfont, scrolledtext
import threading
import json
import tempfile
import queue
import socket as unix_socket
import os
import uuid
import subprocess


class IPCClient:
    """IPC client for connecting to devduck main server"""

    def __init__(self, socket_path="/tmp/devduck_main.sock"):
        self.socket_path = socket_path
        self.sock = None
        self.connected = False
        self.message_queue = queue.Queue()

    def connect(self):
        try:
            self.sock = unix_socket.socket(unix_socket.AF_UNIX, unix_socket.SOCK_STREAM)
            self.sock.connect(self.socket_path)
            self.connected = True
            threading.Thread(target=self._listen, daemon=True).start()
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def send_message(self, message, turn_id=None):
        if not self.connected:
            print("Not connected!")
            return False
        try:
            data = {"message": message, "turn_id": turn_id or str(uuid.uuid4())}
            msg = json.dumps(data).encode() + b"\n"
            self.sock.sendall(msg)
            print(f"✓ Sent: {message[:50]} [turn: {data['turn_id'][:8]}]")
            return True
        except Exception as e:
            print(f"Send failed: {e}")
            self.connected = False
            return False

    def _listen(self):
        buffer = b""
        while self.connected:
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    self.connected = False
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    try:
                        msg = json.loads(line.decode())
                        self.message_queue.put(msg)
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                print(f"Listen error: {e}")
                self.connected = False
                break

    def disconnect(self):
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass


class MinimalAmbientInput:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🦆 devduck")

        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.85)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Better size - bigger and more usable
        window_width = 900
        window_height = 650

        # CENTER on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg="#00ff88")

        # Inner frame
        inner_frame = tk.Frame(self.root, bg="#000000")
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # Output area - NORMAL state for copying, but read-only via bindings
        self.text = scrolledtext.ScrolledText(
            inner_frame,
            font=tkfont.Font(family="-apple-system", size=14),
            bg="#000000",
            fg="#ffffff",
            insertbackground="#00ff88",
            bd=0,
            highlightthickness=0,
            wrap=tk.WORD,
            padx=10,
            pady=10,
            state="normal",  # Keep normal for selection/copying
            height=20,
        )
        self.text.pack(fill=tk.BOTH, expand=True)

        # Make text read-only by blocking all modification events
        self.text.bind("<Key>", lambda e: "break")  # Block keyboard input
        self.text.bind(
            "<Button-2>", lambda e: "break"
        )  # Block middle-click paste (Linux)
        self.text.bind("<Button-3>", lambda e: None)  # Allow right-click

        # Input frame at bottom - bigger and more prominent
        input_frame = tk.Frame(inner_frame, bg="#000000", height=90)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        input_frame.pack_propagate(False)

        # Container for prompt + input
        input_container = tk.Frame(input_frame, bg="#000000")
        input_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Green prompt label - bigger
        prompt_label = tk.Label(
            input_container,
            text=">",
            font=tkfont.Font(family="-apple-system", size=20, weight="bold"),
            fg="#00ff88",
            bg="#000000",
        )
        prompt_label.pack(side=tk.LEFT, padx=(0, 10))

        # Input entry - bigger, more visible
        self.input_entry = tk.Entry(
            input_container,
            font=tkfont.Font(family="-apple-system", size=18),
            bg="#1a1a1a",
            fg="#ffffff",
            insertbackground="#00ff88",
            bd=0,
            highlightthickness=2,
            highlightbackground="#00ff88",
            highlightcolor="#00ff88",
        )
        self.input_entry.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, ipady=10)

        # Tags for output - Green user text, no ugly background
        self.text.tag_config(
            "user",
            foreground="#00ff88",
            background="#000000",
            font=tkfont.Font(family="-apple-system", size=14, weight="bold"),
            spacing1=5,
            spacing3=5,
            lmargin1=0,
            lmargin2=0,
            rmargin=0,
        )
        self.text.tag_config(
            "assistant",
            foreground="#ffffff",
            background="#000000",
            spacing1=0,
            spacing3=0,
            lmargin1=0,
            lmargin2=0,
            rmargin=0,
        )
        self.text.tag_config(
            "tool",
            foreground="#00ff88",
            font=tkfont.Font(family="-apple-system", size=12),
        )
        self.text.tag_config("error", foreground="#ef4444")

        # Bindings
        self.input_entry.bind("<Return>", self.on_enter)
        self.input_entry.bind("<Escape>", lambda e: self.root.withdraw())

        # State - Turn tracking
        self.ipc_client = IPCClient()
        self.turns = {}
        self.turn_order = []

        # Command listener
        self.command_socket_path = os.path.join(
            tempfile.gettempdir(), "devduck_ambient.sock"
        )
        self._start_command_listener()

        # Start
        self.root.after(50, self.process_messages)
        self.root.after(1000, self._try_connect)

        # Force layout update before showing
        self.root.update_idletasks()

        self.root.deiconify()
        self._force_focus()

    def _try_connect(self):
        if self.ipc_client.connect():
            print("✓ Connected to IPC server")
            self._append_text("✓ Connected to DevDuck\n\n", "tool")
        else:
            print("✗ Connection failed, retrying...")
            self.root.after(5000, self._try_connect)

    def _append_text(self, text, tag=None):
        """Append text - NO auto-scroll, but stays in normal state"""
        if tag:
            self.text.insert(tk.END, text, tag)
        else:
            self.text.insert(tk.END, text)
        # NO self.text.see(tk.END) - user controls scroll

    def _render_all_turns(self):
        """Render all turns - PRESERVES scroll position"""
        # Save current scroll position BEFORE deleting
        try:
            yview = self.text.yview()
            scroll_position = yview[0]  # Top of visible area
        except:
            scroll_position = 0.0

        # Delete and re-render
        self.text.delete("1.0", tk.END)

        for turn_id in self.turn_order:
            if turn_id in self.turns:
                turn = self.turns[turn_id]

                # User message with "> " prefix
                query = turn.get("query", "")
                if query:
                    self.text.insert(tk.END, f"> {query}\n\n", "user")

                # Assistant response
                buffer = turn.get("buffer", "")
                if buffer:
                    self.text.insert(tk.END, buffer, "assistant")

                # Add spacing between turns
                self.text.insert(tk.END, "\n")

        # Restore scroll position AFTER re-rendering
        try:
            self.text.yview_moveto(scroll_position)
        except:
            pass

    def _start_command_listener(self):
        def server_thread():
            if os.path.exists(self.command_socket_path):
                os.unlink(self.command_socket_path)

            server = unix_socket.socket(unix_socket.AF_UNIX, unix_socket.SOCK_STREAM)
            server.bind(self.command_socket_path)
            server.listen(1)

            while True:
                try:
                    conn, _ = server.accept()
                    data = conn.recv(4096)
                    if data:
                        cmd = json.loads(data.decode("utf-8"))
                        action = cmd.get("action")
                        if action == "show":
                            self.root.after(0, self._force_focus)
                        elif action == "hide":
                            self.root.after(0, self.root.withdraw)
                        elif action == "set_text":
                            text = cmd.get("text", "")
                            self.root.after(0, lambda: self._set_input_text(text))
                        conn.send(json.dumps({"status": "success"}).encode("utf-8"))
                    conn.close()
                except Exception as e:
                    print(f"Command error: {e}")

        threading.Thread(target=server_thread, daemon=True).start()

    def _set_input_text(self, text):
        """Set input text"""
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, text)

    def _force_focus(self):
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)

        try:
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'tell application "System Events" to set frontmost of the first process whose unix id is {os.getpid()} to true',
                ],
                check=False,
                capture_output=True,
                timeout=1,
            )
        except:
            pass

        self.root.focus_force()
        self.input_entry.focus_set()

    def process_messages(self):
        """Process messages with turn-based buffering"""
        needs_render = False

        while not self.ipc_client.message_queue.empty():
            msg = self.ipc_client.message_queue.get()
            msg_type = msg.get("type")
            turn_id = msg.get("turn_id")

            if msg_type == "turn_start":
                if turn_id not in self.turns:
                    self.turns[turn_id] = {
                        "query": msg.get("data", ""),
                        "buffer": "",
                        "tools": [],
                    }
                    self.turn_order.append(turn_id)
                    needs_render = True

            elif msg_type == "chunk":
                if turn_id in self.turns:
                    chunk = msg.get("data", "")
                    self.turns[turn_id]["buffer"] += chunk
                    needs_render = True

            elif msg_type == "tool_start":
                if turn_id in self.turns:
                    tool_name = msg.get("data", "")
                    tool_num = msg.get("tool_number", 0)
                    tool_text = f"\n🛠️  #{tool_num}: {tool_name} "
                    self.turns[turn_id]["buffer"] += tool_text
                    needs_render = True

            elif msg_type == "tool_end":
                if turn_id in self.turns:
                    success = msg.get("success", False)
                    icon = "✅" if success else "❌"
                    self.turns[turn_id]["buffer"] += f"{icon}\n"
                    needs_render = True

            elif msg_type == "turn_end":
                if turn_id in self.turns:
                    self.turns[turn_id]["buffer"] += "\n"
                    needs_render = True

            elif msg_type == "error":
                error_msg = msg.get("data", "Unknown error")
                if turn_id and turn_id in self.turns:
                    self.turns[turn_id]["buffer"] += f"\n❌ Error: {error_msg}\n"
                    needs_render = True

        if needs_render:
            self._render_all_turns()

        self.root.after(50, self.process_messages)

    def on_enter(self, event):
        """Handle Enter key in input field"""
        query = self.input_entry.get().strip()

        if not query:
            return "break"

        if not self.ipc_client.connected:
            self._append_text("\n❌ Not connected\n", "error")
            return "break"

        try:
            turn_id = str(uuid.uuid4())
            self.input_entry.delete(0, tk.END)

            if not self.ipc_client.send_message(query, turn_id=turn_id):
                self._append_text("❌ Send failed\n", "error")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()

        return "break"

    def run(self):
        self.root.mainloop()

    def cleanup(self):
        self.ipc_client.disconnect()
        if os.path.exists(self.command_socket_path):
            os.unlink(self.command_socket_path)


if __name__ == "__main__":
    app = MinimalAmbientInput()
    try:
        app.run()
    finally:
        app.cleanup()
