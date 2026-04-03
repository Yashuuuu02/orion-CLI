from textual.widget import Widget
from textual.widgets import TextArea, Static
from textual.app import ComposeResult
from textual.message import Message
from textual.events import Key
from textual.containers import Vertical


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
        min-height: 5;
        max-height: 9;
        padding: 0 0;
    }

    .input-wrapper {
        border-left: tall $accent;
        background: $surface;
        padding: 1 1 1 1;
        height: auto;
        min-height: 4;
    }

    #input-area {
        width: 1fr;
        height: auto;
        min-height: 2;
        max-height: 5;
        background: transparent;
        border: none;
        color: $text;
        padding: 0 0;
    }

    #input-area:focus {
        border: none;
    }

    #input-footer {
        height: 1;
        color: $text;
        padding: 0 0;
    }
    """

    class Submitted(Message):
        def __init__(self, text: str):
            super().__init__()
            self.text = text

    def compose(self) -> ComposeResult:
        with Vertical(classes="input-wrapper"):
            yield ChatInputArea(id="input-area")
            yield Static("", id="input-footer")

    def on_mount(self) -> None:
        try:
            model = self.app.config.default_model
            parts = model.split("/")
            if len(parts) >= 3:
                # e.g. "nvidia/meta/llama-3.1-8b-instruct"
                model_name = parts[-1]
                provider = parts[-2]
            elif len(parts) == 2:
                model_name = parts[-1]
                provider = parts[0]
            else:
                model_name = model
                provider = ""
            footer = f"[#4C8EEF]Build[/]  [bold]{model_name}[/]  [dim]{provider}[/]"
        except Exception:
            footer = "[#4C8EEF]Build[/]  orion/base"
        self.query_one("#input-footer", Static).update(footer)

    def on_chat_input_area_submit_triggered(self, event: ChatInputArea.SubmitTriggered):
        self._submit()

    def _submit(self):
        ta = self.query_one(ChatInputArea)
        text = ta.text.strip()
        if text:
            self.post_message(InputBar.Submitted(text=text))
            ta.text = ""

    def focus(self, *args, **kwargs):
        self.query_one(ChatInputArea).focus(*args, **kwargs)
