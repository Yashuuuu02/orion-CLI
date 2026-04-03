from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static, Label
from textual.containers import Vertical, Horizontal, Container
from textual.events import Key

from orion.cli.widgets.input_bar import InputBar, ChatInputArea


ORION_LOGO = """\
  ██████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗
 ██╔═══██╗██╔══██╗██║██╔═══██╗████╗  ██║
 ██║   ██║██████╔╝██║██║   ██║██╔██╗ ██║
 ██║   ██║██╔══██╗██║██║   ██║██║╚██╗██║
 ╚██████╔╝██║  ██║██║╚██████╔╝██║ ╚████║
  ╚═════╝ ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝\
"""


class SplashScreen(Screen):
    DEFAULT_CSS = """
    SplashScreen {
        background: $background;
        align: center middle;
        layout: vertical;
    }

    #logo {
        text-align: center;
        color: #555555;
        width: auto;
        height: auto;
        content-align: center middle;
        padding-bottom: 2;
    }

    #logo .logo-accent {
        color: #888888;
    }

    #center-panel {
        width: 62;
        height: auto;
        align: center middle;
    }

    #splash-input-bar {
        width: 62;
        height: auto;
    }

    #hints {
        width: 62;
        height: 1;
        content-align: right middle;
        color: $text-muted;
        padding-top: 1;
    }

    #tip-line {
        text-align: center;
        width: 62;
        height: 1;
        color: $text-muted;
        padding-top: 2;
    }

    #splash-footer {
        dock: bottom;
        height: 1;
        background: $background;
        layout: horizontal;
        padding: 0 1;
    }

    #footer-left {
        width: 1fr;
        content-align: left middle;
        color: $text-muted;
    }

    #footer-right {
        width: 1fr;
        content-align: right middle;
        color: $text-muted;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(ORION_LOGO, id="logo")

        with Container(id="center-panel"):
            yield InputBar(id="splash-input-bar")
            yield Static(
                "[dim]tab[/] agents   [dim]ctrl+p[/] commands",
                id="hints",
            )
            yield Static(
                "[#8B1A1A]●[/] [dim]Tip[/] Type anything to start a session — [italic]Enter[/] to send",
                id="tip-line",
            )

        with Horizontal(id="splash-footer"):
            yield Static("", id="footer-left")
            yield Static("Orion v1.3.13", id="footer-right")

    def on_mount(self) -> None:
        import os
        cwd = os.getcwd()
        repo = os.path.basename(cwd)
        branch = "main"
        try:
            head = os.path.join(cwd, ".git", "HEAD")
            if os.path.exists(head):
                with open(head) as f:
                    content = f.read().strip()
                if content.startswith("ref: refs/heads/"):
                    branch = content.replace("ref: refs/heads/", "")
        except Exception:
            pass
        self.query_one("#footer-left", Static).update(
            f"[dim]{cwd}:{branch}[/]"
        )
        self.query_one(InputBar).focus()

    def on_input_bar_submitted(self, event: InputBar.Submitted) -> None:
        """User typed something — transition to MainScreen carrying the first message."""
        from orion.cli.screens.main import MainScreen
        self.app.push_screen(MainScreen(initial_message=event.text))

    def action_quit(self) -> None:
        self.app.exit()
