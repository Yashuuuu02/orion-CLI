from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.widgets import Static
from textual.containers import Vertical
from textual.events import Key

class HelpScreen(ModalScreen):
    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
        background: $background 50%;
    }
    #help-box {
        width: 40;
        height: auto;
        border: solid $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="help-box"):
            yield Static("⚡ Orion CLI Help", classes="bold")
            yield Static("")
            yield Static("Ctrl+Q  → Quit")
            yield Static("Ctrl+N  → New session")
            yield Static("Ctrl+B  → Toggle sidebar")
            yield Static("Ctrl+L  → Clear chat")
            yield Static("Ctrl+H  → Session history")
            yield Static("Ctrl+?  → Help")
            yield Static("")
            yield Static("[dim]Press any key to dismiss[/dim]")

    def on_key(self, event: Key):
        self.dismiss()
