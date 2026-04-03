from textual.widget import Widget
from textual.widgets import Static
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.containers import VerticalScroll

from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from rich.syntax import Syntax
from datetime import datetime

class ChatMessage(Static):
    def __init__(self, role: str, content: str | Group, model: str = "", time_taken: float = 0.0, status: str = "", **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.model = model
        self.time_taken = time_taken
        self.status = status
        self.add_class(f"msg-{role}")

    def render(self):
        if self.role == "user":
            return self.content
        elif self.role == "assistant":
            footer = Text(f"\n▣ Build · {self.model} · {self.time_taken}s · {self.status}", style="dim")
            return Group(self.content, footer)
        elif self.role == "system":
            return Text(f"── {self.content} ──", style="dim italic", justify="center")
        return self.content

class ThinkingMessage(Static):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_class("msg-thinking")
        
    def render(self):
        return Text.assemble(("Thinking: ", "italic #B8860B"), ("Analyzing prompt and constructing pipeline strategy...", "dim"))

class ChatPanel(Widget):
    DEFAULT_CSS = """
    ChatPanel {
        height: 1fr;
        layout: vertical;
    }
    #chat-scroll {
        height: 1fr;
        overflow-y: auto;
    }
    ChatMessage {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin: 1 0;
    }
    .msg-user {
        border-left: tall $accent;
        background: $surface-lighten-1;
        padding: 0 1;
        margin: 1 0;
    }
    .msg-assistant {
        padding: 0 1;
        margin: 1 0;
    }
    .msg-thinking {
        border-left: tall #2e2e2e;
        padding: 0 1;
        margin: 1 0;
    }
    """

    _thinking_dots = reactive(0)
    _thinking_text = " Orion  "

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-scroll")

    def _render_code_blocks(self, content: str) -> Group:
        parts = content.split("```")
        renderables = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                lines = part.split("\n", 1)
                lang = lines[0].strip() if len(lines) > 0 else ""
                code = lines[1].strip("\n") if len(lines) > 1 else ""
                renderables.append(Panel(Syntax(code, lang, theme="monokai", word_wrap=True), border_style="dim"))
            else:
                if part.strip():
                    renderables.append(Text.from_markup(part.strip()))
        return Group(*renderables)

    def add_message(self, role: str, content: str, model: str = "orion/base", time_taken: float = 0.5, status: str = "completed"):
        self.remove_thinking()
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        
        if role == "assistant":
            renderable = self._render_code_blocks(content)
        elif role == "system":
            renderable = content
        else:
            renderable = Text(content)
            
        msg = ChatMessage(role=role, content=renderable, model=model, time_taken=time_taken, status=status)
        scroll.mount(msg)
        scroll.scroll_end(animate=False)

    def add_thinking(self):
        self.remove_thinking()
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        self._thinking_widget = ThinkingMessage(id="thinking-block")
        scroll.mount(self._thinking_widget)
        scroll.scroll_end(animate=False)

    def remove_thinking(self):
        if hasattr(self, "_thinking_widget") and self._thinking_widget is not None:
            self._thinking_widget.remove()
            self._thinking_widget = None

    def clear(self):
        self.query_one("#chat-scroll", VerticalScroll).query(ChatMessage).remove()
        self.remove_thinking()
