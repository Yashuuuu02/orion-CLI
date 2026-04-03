from textual.widget import Widget
from textual.widgets import TextArea, Static, Button
from textual.app import ComposeResult
from textual.message import Message
from textual.events import Key


class ChatInputArea(TextArea):
    class SubmitTriggered(Message):
        pass

    def _on_key(self, event: Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            self.post_message(self.SubmitTriggered())
        else:
            super()._on_key(event)


class InputBar(Widget):
    DEFAULT_CSS = """
    InputBar {
        height: auto;
        min-height: 4;
        max-height: 8;
        dock: bottom;
        background: transparent;
        padding: 0 1;
    }

    InputBar Horizontal {
        height: auto;
        align: left middle;
        padding: 1 0;
    }

    #input-area {
        width: 1fr;
        height: auto;
        min-height: 3;
        max-height: 6;
        background: transparent;
        border: none;
        color: $text;
        padding: 0 1;
    }

    #input-area:focus {
        border: none;
    }

    #char-count {
        dock: right;
        content-align: right middle;
        color: $text-muted;
        text-style: dim;
        margin-right: 14;
        margin-top: 1;
    }

    #char-count.over-limit {
        color: $error;
    }

    #send-btn {
        width: 12;
        min-width: 12;
        height: 3;
        margin-left: 1;
        background: $accent;
        color: $text;
        text-style: bold;
        border: none;
        content-align: center middle;
    }

    #send-btn:hover {
        background: $accent-lighten-1;
    }

    #send-btn:disabled {
        background: $surface-darken-2;
        color: $text-muted;
    }
    """

    class Submitted(Message):
        def __init__(self, text: str):
            super().__init__()
            self.text = text

    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal
        with Horizontal():
            ta = ChatInputArea(id="input-area")
            yield ta
            yield Static("chars: 0/4000", id="char-count")
            yield Button("[ ↵ SEND ]", id="send-btn", variant="primary")

    def on_chat_input_area_submit_triggered(self, event: ChatInputArea.SubmitTriggered):
        self._submit()

    def on_text_area_changed(self, event: TextArea.Changed):
        count = len(event.text_area.text)
        counter = self.query_one("#char-count", Static)
        counter.update(f"chars: {count}/4000")
        if count > 3800:
            counter.add_class("over-limit")
        else:
            counter.remove_class("over-limit")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "send-btn":
            self._submit()

    def _submit(self):
        ta = self.query_one(ChatInputArea)
        text = ta.text.strip()
        if text:
            self.post_message(InputBar.Submitted(text=text))
            ta.text = "" # Clears the area

    def focus(self, *args, **kwargs):
        self.query_one(ChatInputArea).focus(*args, **kwargs)
