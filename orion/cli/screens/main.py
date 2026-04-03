import os
import random
from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static, Input
from textual.containers import Horizontal
from orion.cli.widgets.file_tree import OrionFileTree
from orion.cli.widgets.chat_panel import ChatPanel
from orion.cli.widgets.input_bar import InputBar

PLACEHOLDER_REPLIES = [
    "Processing... (Phase 2 will wire the real pipeline here)",
    "Got it. Orion pipeline coming in Phase 2.",
    "Noted. AI backend connects in Phase 2.",
    "Received. Pipeline integration is next.",
    "On it. Real responses coming soon.",
]


class MainScreen(Screen):
    DEFAULT_CSS = """
    #header   { dock: top; height: 1; background: $accent; color: $text; padding: 0 1; }
    #sidebar  { width: 25%; height: 100%; border-right: solid $accent; }
    #chat     { width: 1fr; height: 1fr; }
    #input-bar { dock: bottom; height: 3; }
    #status   { dock: bottom; height: 1; background: $surface; padding: 0 1; }
    """

    BINDINGS = [
        ("ctrl+q", "quit",            "Quit"),
        ("ctrl+n", "new_session",     "New Session"),
        ("ctrl+b", "toggle_sidebar",  "Toggle Sidebar"),
        ("ctrl+l", "clear_chat",      "Clear Chat"),
        ("ctrl+h", "session_history", "History"),
        ("ctrl+question_mark", "help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        yield Static("", id="header")
        with Horizontal():
            yield OrionFileTree(path=os.getcwd(), id="sidebar", classes="hidden")
            yield ChatPanel(id="chat")
        yield InputBar(id="input-bar")
        yield Static("", id="status")

    def on_mount(self):
        self.session = (
            self.app.session_manager.get_latest_session()
            or self.app.session_manager.create_session(os.getcwd())
        )
        self._update_header()
        self._update_status()
        for msg in self.app.session_manager.get_messages(self.session["id"]):
            self.query_one("#chat", ChatPanel).add_message(
                msg["role"], msg["content"]
            )
        self.query_one(InputBar).focus()

    def on_input_bar_submitted(self, event: InputBar.Submitted):
        text = event.text.strip()
        if not text:
            return
        chat = self.query_one("#chat", ChatPanel)
        chat.add_message("user", text)
        self.app.session_manager.add_message(self.session["id"], "user", text)
        self._update_status()
        chat.add_thinking()
        self.set_timer(0.5, self._placeholder_reply)

    def _placeholder_reply(self):
        reply = random.choice(PLACEHOLDER_REPLIES)
        chat = self.query_one("#chat", ChatPanel)
        chat.add_message("assistant", reply)
        self.app.session_manager.add_message(
            self.session["id"], "assistant", reply
        )
        self._update_status()

    def _update_header(self):
        name = self.session["name"]
        model = self.app.config.default_model.split("/")[-1]
        self.query_one("#header", Static).update(
            f" ⚡ Orion CLI  |  {name}  |  {model}"
        )

    def _update_status(self):
        count = len(
            self.app.session_manager.get_messages(self.session["id"])
        )
        self.query_one("#status", Static).update(
            f" Session: {self.session['name']}  |  Messages: {count}  |  Ctrl+? for help"
        )

    def action_quit(self):
        self.app.exit()

    def action_toggle_sidebar(self):
        self.query_one("#sidebar").toggle_class("hidden")

    def action_clear_chat(self):
        self.query_one("#chat", ChatPanel).clear()

    def action_new_session(self):
        self.session = self.app.session_manager.create_session(os.getcwd())
        self.query_one("#chat", ChatPanel).clear()
        self._update_header()
        self._update_status()

    def action_session_history(self):
        from orion.cli.screens.history import HistoryScreen
        self.app.push_screen(HistoryScreen(), self._switch_to_session)

    def _switch_to_session(self, session_id: str | None):
        if not session_id:
            return
            
        sessions = self.app.session_manager.get_all_sessions()
        for s in sessions:
            if s["id"] == session_id:
                self.session = s
                break
                
        chat = self.query_one("#chat", ChatPanel)
        chat.clear()
        
        for msg in self.app.session_manager.get_messages(self.session["id"]):
            chat.add_message(msg["role"], msg["content"])
            
        self._update_header()
        self._update_status()

    def action_help(self):
        from orion.cli.screens.help import HelpScreen
        self.app.push_screen(HelpScreen())
