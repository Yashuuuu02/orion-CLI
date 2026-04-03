from textual.widget import Widget
from textual.widgets import Input
from textual.app import ComposeResult
from textual.message import Message


class InputBar(Widget):
    DEFAULT_CSS = "InputBar { height: 3; border: tall $accent; }"

    class Submitted(Message):
        def __init__(self, text: str):
            super().__init__()
            self.text = text

    def compose(self) -> ComposeResult:
        yield Input(
            placeholder="Type a message... (Enter to send, Ctrl+Q to quit)"
        )

    def on_input_submitted(self, event: Input.Submitted):
        self.post_message(InputBar.Submitted(text=event.value))
        self.query_one(Input).clear()
