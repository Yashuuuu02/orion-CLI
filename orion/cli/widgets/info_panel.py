from textual.widget import Widget
from textual.widgets import Static
from textual.app import ComposeResult
from textual.containers import Vertical

class InfoPanel(Widget):

    DEFAULT_CSS = """
    InfoPanel {
        width: 100%;
        height: 100%;
        background: $surface;
        border-left: solid #2e2e2e;
        padding: 1 2;
        overflow-y: auto;
    }
    InfoPanel .section-title {
        color: $text-muted;
        text-style: italic;
        margin-bottom: 0;
        margin-top: 1;
    }
    InfoPanel .value {
        color: $text;
        text-style: bold;
        content-align: right middle;
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("⚡ ORION CLI", classes="section-title"),

            Static("MODEL", classes="section-title"),
            Static("", id="info-tier", classes="value"),
            Static("", id="info-model", classes="value"),

            Static("TOKENS", classes="section-title"),
            Static("Prompt:      0", id="info-prompt-tokens", classes="value"),
            Static("Completion:  0", id="info-completion-tokens", classes="value"),
            Static("Total:       0", id="info-total-tokens", classes="value"),

            Static("COST (est.)", classes="section-title"),
            Static("$0.000000", id="info-cost", classes="value"),

            Static("SESSION", classes="section-title"),
            Static("", id="info-session-name", classes="value"),
            Static("", id="info-cwd", classes="value"),

            Static("PIPELINE", classes="section-title"),
            Static("IISG Rate:  --", id="info-iisg", classes="value"),
            Static("RL Action:  --", id="info-rl-action", classes="value"),
        )

    def update_model(self, tier: str, model: str):
        self.query_one("#info-tier").update(
            f"Tier:  [{tier}]")
        self.query_one("#info-model").update(
            model.split("/")[-1])  # show short name only

    def update_tokens(self, prompt: int, completion: int):
        total = prompt + completion
        self.query_one("#info-prompt-tokens").update(
            f"Prompt:      {prompt:,}")
        self.query_one("#info-completion-tokens").update(
            f"Completion:  {completion:,}")
        self.query_one("#info-total-tokens").update(
            f"Total:       {total:,}")
        # Rough cost estimate using fast tier pricing ($0.20/1M input, $0.80/1M output)
        cost = (prompt * 0.0000002) + (completion * 0.0000008)
        self.query_one("#info-cost").update(f"${cost:.6f}")

    def update_session(self, name: str, cwd: str):
        import os
        self.query_one("#info-session-name").update(name)
        self.query_one("#info-cwd").update(
            f"…/{os.path.basename(cwd)}" if cwd else "")

    def update_pipeline(self, iisg_rate: float = None, rl_action: int = None):
        if iisg_rate is not None:
            self.query_one("#info-iisg").update(
                f"IISG Rate:  {iisg_rate:.0%}")
        if rl_action is not None:
            self.query_one("#info-rl-action").update(
                f"RL Action:  #{rl_action}")
