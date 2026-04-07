import os
import sys
from textual import work
from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static
from textual.containers import Horizontal, Vertical
from orion.cli.widgets.info_panel import InfoPanel
from orion.cli.widgets.chat_panel import ChatPanel
from orion.cli.widgets.input_bar import InputBar
from orion.provider.provider import Provider
from orion.pipeline.runner import PipelineRunner, DEFAULT_ACTION
from orion.tool.tools import ReadTool, WriteTool, EditTool, GrepTool


class MainScreen(Screen):
    DEFAULT_CSS = """
    #header   { dock: top; height: 1; background: $accent; color: $text; padding: 0 1; display: none; }
    #sidebar  { width: 42; height: 100%; border-left: solid #2e2e2e; }
    #chat-column { width: 1fr; height: 1fr; layout: vertical; }
    #chat     { height: 1fr; }
    #input-bar { border-top: solid #2e2e2e; }
    #footer   { dock: bottom; height: 1; background: #1a1a1a; padding: 0 1; layout: horizontal; }
    #status-left { width: 1fr; content-align: left middle; color: $text; }
    #status-center { width: 1fr; content-align: center middle; color: $text-muted; }
    #status-right { width: 1fr; content-align: right middle; color: $text; }
    """

    BINDINGS = [
        ("ctrl+q", "quit",            "Quit"),
        ("ctrl+n", "new_session",     "New Session"),
        ("ctrl+b", "toggle_sidebar",  "Toggle Sidebar"),
        ("ctrl+l", "clear_chat",      "Clear Chat"),
        ("ctrl+h", "session_history", "History"),
        ("ctrl+question_mark", "help", "Help"),
    ]

    def __init__(self, initial_message: str = "", **kwargs):
        super().__init__(**kwargs)
        self._initial_message = initial_message
        self._thinking = False

    def compose(self) -> ComposeResult:
        yield Static("", id="header")
        with Horizontal():
            with Vertical(id="chat-column"):
                yield ChatPanel(id="chat")
                yield InputBar(id="input-bar")
            yield InfoPanel(id="sidebar", classes="hidden")
        with Horizontal(id="footer"):
            yield Static("📂 ...", id="status-left")
            yield Static("^P commands · ^C stop · ^L clear", id="status-center")
            yield Static("[#4CAF50]●[/] Ready · v1.3.13", id="status-right")

    def on_mount(self):
        self.session = (
            self.app.session_manager.get_latest_session()
            or self.app.session_manager.create_session(os.getcwd())
        )
        self._update_header()
        self._update_status()

        self._total_prompt = 0
        self._total_completion = 0

        panel = self.query_one("#sidebar", InfoPanel)
        panel.update_model(
            self.app.config.default_model.split("/")[-2]
                if "/" in self.app.config.default_model else "fast",
            self.app.config.default_model
        )
        panel.update_session(self.session["name"], self.session["cwd"])

        # Wire up the pipeline
        provider = Provider(api_key=self.app.config.nim_api_key)
        self.tools = {
            "read":  ReadTool(),
            "write": WriteTool(),
            "edit":  EditTool(),
            "grep":  GrepTool(),
        }

        use_bandit = os.environ.get("ORION_USE_BANDIT", "false").lower() == "true"
        if use_bandit:
            try:
                from orion.rl.bandit import LinUCBBandit
                from orion.rl.state_encoder import StateEncoder
                bandit = LinUCBBandit()
                state_encoder = StateEncoder()
                self.runner = PipelineRunner(provider=provider, tools=self.tools, bandit=bandit, state_encoder=state_encoder)
                self._use_bandit = True
            except Exception:
                self.runner = PipelineRunner(provider=provider, tools=self.tools)
                self._use_bandit = False
        else:
            self.runner = PipelineRunner(provider=provider, tools=self.tools)
            self._use_bandit = False

        self.call_after_refresh(self._load_initial_messages)
        self.query_one(InputBar).focus()

    def _load_initial_messages(self):
        for msg in self.app.session_manager.get_messages(self.session["id"]):
            self.query_one("#chat", ChatPanel).add_message(
                msg["role"], msg["content"]
            )
        if self._initial_message:
            self._handle_user_message(self._initial_message)

    def on_input_bar_submitted(self, event: InputBar.Submitted):
        self._handle_user_message(event.text)

    def _handle_user_message(self, text: str):
        text = text.strip()
        if not text or self._thinking:
            return

        self._thinking = True
        input_bar = self.query_one(InputBar)
        input_bar.disabled = True

        chat = self.query_one("#chat", ChatPanel)
        chat.add_message("user", text)
        self.app.session_manager.add_message(self.session["id"], "user", text)
        self._update_status()
        chat.add_thinking()

        # Fire the real pipeline
        self._run_pipeline(text)

    @work(exclusive=True, thread=False)
    async def _run_pipeline(self, text: str):
        chat = self.query_one("#chat", ChatPanel)
        panel = self.query_one("#sidebar", InfoPanel)

        # Build conversation history — last 10 messages only
        history = self.app.session_manager.get_messages(self.session["id"])
        conversation_history = [
            {"role": m["role"], "content": m["content"]}
            for m in history[-10:]
            if m["role"] in ("user", "assistant")
        ]

        def on_stage(msg: str):
            chat.add_message("system", msg)

        def on_token_delta(chunk: str):
            chat.add_streaming_chunk(chunk)

        ctx = await self.runner.run(
            prompt=text,
            action=DEFAULT_ACTION,
            on_stage=on_stage,
            on_token_delta=on_token_delta,
            conversation_history=conversation_history,
            use_bandit=getattr(self, '_use_bandit', False),
        )

        # Finalize streaming, add real assistant message
        chat.finalize_streaming()
        active_model = self.app.config.default_model
        chat.add_message(
            "assistant",
            ctx.final_response,
            model=active_model,
            time_taken=ctx.time_taken,
            status="completed" if not ctx.error else "error",
        )

        # Persist to DB
        self.app.session_manager.add_message(
            self.session["id"], "assistant", ctx.final_response
        )

        # Update info panel with real values
        self._total_prompt += self.runner.provider.total_prompt_tokens
        self._total_completion += self.runner.provider.total_completion_tokens
        panel.update_tokens(self._total_prompt, self._total_completion)
        if hasattr(panel, "update_pipeline"):
            panel.update_pipeline(
                ctx.iisg_pass_rate,
                ctx.action.get("coder_tier", "coder"),
            )

        # Re-enable input
        self._thinking = False
        self._update_status()
        input_bar = self.query_one(InputBar)
        input_bar.disabled = False
        input_bar.focus()

    def _update_header(self):
        name = self.session["name"]
        model = self.app.config.default_model.split("/")[-1]
        self.query_one("#header", Static).update(
            f" ⚡ Orion CLI  |  {name}  |  {model}"
        )

    def _update_status(self):
        cwd = os.getcwd()
        branch = "main"
        try:
            head_path = os.path.join(cwd, ".git", "HEAD")
            if os.path.exists(head_path):
                with open(head_path, "r") as f:
                    content = f.read().strip()
                if content.startswith("ref: refs/heads/"):
                    branch = content.replace("ref: refs/heads/", "")
        except Exception:
            pass
        repo_name = os.path.basename(cwd)
        self.query_one("#status-left", Static).update(
            f"📂 {repo_name} · 🌿 {branch}"
        )

    def action_quit(self):           self.app.exit()
    def action_toggle_sidebar(self): self.query_one("#sidebar").toggle_class("hidden")
    def action_clear_chat(self):     self.query_one("#chat", ChatPanel).clear()

    def action_new_session(self):
        self.session = self.app.session_manager.create_session(os.getcwd())
        chat = self.query_one("#chat", ChatPanel)
        chat.clear()
        chat.add_message("system", "New session")
        self._update_header()
        self._update_status()
        self._total_prompt = 0
        self._total_completion = 0
        panel = self.query_one("#sidebar", InfoPanel)
        panel.update_tokens(0, 0)
        panel.update_session(self.session["name"], self.session["cwd"])

    def action_session_history(self):
        from orion.cli.screens.history import HistoryScreen
        self.app.push_screen(HistoryScreen(), self._switch_to_session)

    def _switch_to_session(self, session_id: str | None):
        if not session_id:
            return
        sessions = self.app.session_manager.get_all_sessions()
        for s in sessions:
            if s["id"] == session_id:
                self.session = s
                break
        chat = self.query_one("#chat", ChatPanel)
        chat.clear()
        for msg in self.app.session_manager.get_messages(self.session["id"]):
            chat.add_message(msg["role"], msg["content"])
        self._update_header()
        self._update_status()

    def action_help(self):
        from orion.cli.screens.help import HelpScreen
        self.app.push_screen(HelpScreen())
