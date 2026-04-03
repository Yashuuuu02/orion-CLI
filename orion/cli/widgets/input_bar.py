from textual.widget import Widget
from textual.widgets import TextArea, Static
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
        padding: 0 1;
    }

    .input-wrapper {
        border-left: tall $accent;
        background: $surface-lighten-1;
        padding: 0 1;
        margin: 1 0;
    }

    #input-area {
        width: 1fr;
        height: auto;
        min-height: 3;
        max-height: 6;
        background: transparent;
        border: none;
        color: $text;
        padding: 0 0;
    }

    #input-area:focus {
        border: none;
    }
    
    #input-footer {
        color: $text;
        padding-top: 1;
    }
    """

    class Submitted(Message):
        def __init__(self, text: str):
            super().__init__()
            self.text = text

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical
        with Vertical(classes="input-wrapper"):
            ta = ChatInputArea(id="input-area")
            yield ta
            yield Static("", id="input-footer")

    def on_mount(self) -> None:
        try:
            model = self.app.config.default_model
            parts = model.split("/")
            # Format: provider/org/model-name → show last 2 parts
            if len(parts) >= 2:
                model_name = parts[-1]
                provider = parts[-2]
            else:
                model_name = model
                provider = ""
            footer_text = f"[#4C8EEF]Build[/]  {model_name}  [dim]{provider}[/]"
        except:
            footer_text = "[#4C8EEF]Build[/]  orion/base"
        self.query_one("#input-footer", Static).update(footer_text)

    def on_chat_input_area_submit_triggered(self, event: ChatInputArea.SubmitTriggered):
        self._submit()

    def _submit(self):
        ta = self.query_one(ChatInputArea)
        text = ta.text.strip()
        if text:
            self.post_message(InputBar.Submitted(text=text))
            ta.text = "" # Clears the area

    def focus(self, *args, **kwargs):
        self.query_one(ChatInputArea).focus(*args, **kwargs)
