from textual.widget import Widget
from textual.widgets import RichLog, Static
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.containers import Vertical

from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from rich.table import Table
from rich.syntax import Syntax
from datetime import datetime

class ChatPanel(Widget):
    DEFAULT_CSS = """
    ChatPanel {
        height: 1fr;
        layout: vertical;
    }
    RichLog {
        height: 1fr;
        overflow-y: auto;
    }
    #thinking-indicator {
        height: 1;
        dock: bottom;
        display: none;
        color: $text-muted;
        margin-left: 1;
    }
    """

    _thinking_dots = reactive(0)
    _thinking_text = " Orion  "

    def compose(self) -> ComposeResult:
        with Vertical():
            yield RichLog(highlight=True, markup=True, wrap=True, id="rich-log")
            yield Static("", id="thinking-indicator")

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

    def add_message(self, role: str, content: str):
        self.remove_thinking()
        log = self.query_one("#rich-log", RichLog)
        
        timestamp = datetime.now().strftime("%H:%M")
        
        if role == "user":
            user_content = Group(Text(content))
            panel = Panel(
                user_content,
                title="[bold cyan]You[/]",
                title_align="left",
                subtitle=f"[dim]{timestamp}[/dim]",
                subtitle_align="right",
                border_style="dodger_blue1",
                style="on #0f172a", # dark blue/slate
                expand=False
            )
            
            table = Table.grid(expand=True)
            table.add_column(ratio=1)
            table.add_column(justify="right")
            table.add_row("", panel)
            
            log.write(table)
            
        elif role == "assistant":
            header = Text(" Orion  ", style="dim")
            header.append(timestamp, style="dim")
            content_renderable = self._render_code_blocks(content)
            log.write(Group(header, content_renderable))
            
        elif role == "system":
            sys_panel = Text(f"── {content} ──", style="dim italic", justify="center")
            log.write(sys_panel)
            
        else:
            log.write(Text(content, style="dim"))
            
        log.write(Text("")) # Blank line divider

    def add_thinking(self):
        indicator = self.query_one("#thinking-indicator", Static)
        indicator.display = True
        self._thinking_dots = 0
        indicator.update(self._thinking_text)
        if hasattr(self, "_thinking_timer"):
            self._thinking_timer.stop()
        self._thinking_timer = self.set_interval(0.4, self._update_thinking)

    def _update_thinking(self):
        self._thinking_dots = (self._thinking_dots + 1) % 4
        dots = "·" * self._thinking_dots
        self.query_one("#thinking-indicator", Static).update(f"{self._thinking_text}{dots}")

    def remove_thinking(self):
        if hasattr(self, "_thinking_timer"):
            self._thinking_timer.stop()
            del self._thinking_timer
        self.query_one("#thinking-indicator", Static).display = False

    def clear(self):
        self.query_one("#rich-log", RichLog).clear()
        self.remove_thinking()
