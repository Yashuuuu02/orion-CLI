from textual.app import ComposeResult
from textual.widgets import Static, Label
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
import os

class InfoPanel(Static):
    """
    Refactored Info Panel: High-density, minimalist display following 
    IDE-native aesthetics (OpenCode reference).
    """

    DEFAULT_CSS = """
    InfoPanel {
        width: 42;
        background: $surface;
        padding: 1 2;
        border-left: solid #2e2e2e;
        layout: vertical;
    }

    #info-container {
        width: 100%;
        height: 1fr;
        layout: vertical;
    }

    #brand-header {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    .info-section {
        margin-bottom: 1;
        height: auto;
    }

    .section-title {
        text-style: bold;
        color: $text;
        margin-bottom: 0;
    }

    .data-line {
        height: 1;
        width: 100%;
        color: #aaaaaa;
    }

    #info-spacer {
        height: 1fr;
    }

    #bottom-section {
        height: auto;
        width: 100%;
        padding-top: 1;
        border-top: solid #2e2e2e;
    }

    #val-path {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    #val-version {
        color: #4CAF50;
    }

    #lsp-status {
        color: #666666;
        text-style: italic;
    }
    """
    
    model_name = reactive("llama-3.1-8b-instruct")
    tokens_prompt = reactive(0)
    tokens_completion = reactive(0)
    tokens_total = reactive(0)
    cost_est = reactive(0.0)
    session_id = reactive("brave-flare")
    project_path = reactive("./orion-CLI")
    iisg_rate = reactive("--")
    rl_action = reactive("--")

    def compose(self) -> ComposeResult:
        with Vertical(id="info-container"):
            # Header Branding
            yield Label("ORION", id="brand-header")
            
            # Context / Model Section
            with Vertical(classes="info-section"):
                yield Label("Context", classes="section-title")
                yield Label(f"{self.model_name}", id="val-model", classes="data-line")
                yield Label(f"{self.session_id}", id="val-session", classes="data-line")

            # Tokens / Usage Section
            with Vertical(classes="info-section"):
                yield Label("Usage", classes="section-title")
                yield Label(f"{self.tokens_total} tokens", id="val-tokens", classes="data-line")
                yield Label(f"-- used", id="val-pct", classes="data-line")
                yield Label(f"${self.cost_est:,.2f} spent", id="val-cost", classes="data-line")

            # Pipeline Section
            with Vertical(classes="info-section"):
                yield Label("Pipeline", classes="section-title")
                yield Label(f"IISG Rate  {self.iisg_rate}", id="val-iisg", classes="data-line")
                yield Label(f"RL Action  {self.rl_action}", id="val-rl", classes="data-line")
            
            # LSP Status
            with Vertical(classes="info-section", id="lsp-info"):
                yield Label("LSP", classes="section-title")
                yield Label("LSPs will activate as files are read", id="lsp-status")

            # Spacer — pushes bottom section down
            yield Static("", id="info-spacer")

            # Bottom-pinned: path and version
            with Vertical(id="bottom-section"):
                yield Label(f"/D:\\orion-CLI:main", id="val-path")
                yield Label("● OpenCode 1.3.15", id="val-version")

    # API wrappers mapping external updates to reactive properties safely
    def update_model(self, tier: str, model: str):
        self.model_name = model.split("/")[-1]

    def update_tokens(self, prompt: int, completion: int):
        self.tokens_prompt = prompt
        self.tokens_completion = completion
        self.tokens_total = prompt + completion
        self.cost_est = (prompt * 0.0000002) + (completion * 0.0000008)
        pct = ((prompt + completion) / max(1, prompt + completion + 1000)) * 100
        try: self.query_one("#val-pct", Label).update(f"{pct:.0f}% used")
        except: pass

    def update_session(self, name: str, cwd: str):
        self.session_id = name
        self.project_path = f"…/{os.path.basename(cwd)}" if cwd else ""

    def update_pipeline(self, iisg_rate: float = None, rl_action: int = None):
        if iisg_rate is not None:
             self.iisg_rate = f"{iisg_rate:.0%}"
        if rl_action is not None:
             self.rl_action = str(rl_action)

    # Reactivity Watches allowing Textual DOM updates seamlessly
    def watch_tokens_total(self, value: int) -> None:
        try: self.query_one("#val-tokens", Label).update(f"{value:,} tokens")
        except: pass
        
    def watch_cost_est(self, value: float) -> None:
        try: self.query_one("#val-cost", Label).update(f"${value:,.2f} spent")
        except: pass

    def watch_model_name(self, value: str) -> None:
        try: self.query_one("#val-model", Label).update(value)
        except: pass

    def watch_session_id(self, value: str) -> None:
        try: self.query_one("#val-session", Label).update(value)
        except: pass

    def watch_iisg_rate(self, value: str) -> None:
        try: self.query_one("#val-iisg", Label).update(f"IISG Rate  {value}")
        except: pass

    def watch_rl_action(self, value: str) -> None:
        try: self.query_one("#val-rl", Label).update(f"RL Action  {value}")
        except: pass
