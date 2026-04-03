from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.widgets import ListView, ListItem, Static, Label
from textual.containers import Vertical
from textual.events import Key
from datetime import datetime

class HistoryScreen(ModalScreen[str]):
    DEFAULT_CSS = """
    HistoryScreen {
        align: center middle;
        background: $background 50%;
    }
    #history-box {
        width: 60;
        height: 80%;
        border: solid $accent;
        padding: 1 2;
        background: $surface;
    }
    ListView {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="history-box"):
            yield Static("⚡ Session History (ESC to cancel)", classes="bold")
            yield Static("")
            self.list_view = ListView()
            yield self.list_view

    def on_mount(self):
        sessions = self.app.session_manager.get_all_sessions()
        for s in sessions:
            name = s["name"]
            # Basic reformatting of ISO 8601 assuming format like 2026-04-03T10:19:28.772928+00:00
            dt = s["created_at"][:19].replace("T", " ")
            item = ListItem(
                Label(f"{name:<20} | {dt}"), 
                id=f"session_{s['id']}"
            )
            self.list_view.append(item)

    def on_list_view_selected(self, event: ListView.Selected):
        if event.item and event.item.id:
            session_id = event.item.id.replace("session_", "")
            self.dismiss(session_id)

    def on_key(self, event: Key):
        if event.key == "escape":
            self.dismiss(None)
