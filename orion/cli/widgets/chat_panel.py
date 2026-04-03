from textual.widget import Widget
from textual.widgets import RichLog
from textual.app import ComposeResult


class ChatPanel(Widget):
    DEFAULT_CSS = "ChatPanel { height: 1fr; overflow-y: auto; }"

    def compose(self) -> ComposeResult:
        yield RichLog(highlight=True, markup=True, wrap=True, id="rich-log")

    def add_message(self, role: str, content: str):
        log = self.query_one("#rich-log", RichLog)
        if role == "user":
            log.write(f"[bold cyan]You:[/bold cyan] {content}")
        elif role == "assistant":
            log.write(f"[bold green]Orion:[/bold green] {content}")
        else:
            log.write(f"[dim]{content}[/dim]")

    def add_thinking(self):
        self.query_one("#rich-log", RichLog).write(
            "[dim italic]Orion is thinking...[/dim italic]"
        )

    def clear(self):
        self.query_one("#rich-log", RichLog).clear()
