from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static, Input, Button, Label
from textual.containers import Vertical


class SetupScreen(Screen):
    DEFAULT_CSS = """
    SetupScreen { align: center middle; }
    #setup-box {
        width: 60; height: auto;
        border: round $accent; padding: 2 4;
    }
    #error-msg { color: $error; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="setup-box"):
            yield Static("⚡ Welcome to Orion CLI")
            yield Label("Enter your NVIDIA NIM API key:")
            yield Input(password=True, id="api-key-input", placeholder="nvapi-...")
            yield Button("Save & Continue", id="save-btn", variant="primary")
            yield Static("", id="error-msg")

    def on_button_pressed(self, event: Button.Pressed):
        self._try_save()

    def on_input_submitted(self, event: Input.Submitted):
        self._try_save()

    def _try_save(self):
        value = self.query_one("#api-key-input", Input).value.strip()
        if not value:
            self.query_one("#error-msg", Static).update(
                "[red]API key cannot be empty[/red]"
            )
            return
        self.app.config.nim_api_key = value
        self.app.config.save()
        from orion.cli.screens.main import MainScreen
        self.app.switch_screen(MainScreen())
