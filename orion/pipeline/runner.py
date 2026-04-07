import re
import time
from orion.pipeline import c01_intent, agentic_loop, c09_validation
from orion.pipeline.models import PipelineContext

try:
    from orion.rl.bandit import LinUCBBandit, action_to_pipeline
    from orion.rl.state_encoder import StateEncoder
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

DEFAULT_ACTION = {
    "coder_tier": "coder",
}

class PipelineRunner:
    def __init__(self, provider, tools: dict, bandit=None, state_encoder=None):
        self.provider = provider
        self.tools = tools
        self.bandit = bandit
        self.state_encoder = state_encoder or (StateEncoder() if RL_AVAILABLE else None)

    def _get_action(self, intent) -> dict:
        if self.bandit and self.state_encoder and RL_AVAILABLE:
            try:
                state = self.state_encoder.encode(
                    intent_type=intent.intent_type,
                    complexity=intent.complexity,
                )
                action_idx = self.bandit.select(state.to_list())
                action = self.bandit.get_action(action_idx)
                pipeline_action = action_to_pipeline(action)
                return pipeline_action
            except Exception:
                pass
        return DEFAULT_ACTION

    def _update_bandit(self, state, action_idx, reward):
        if self.bandit and RL_AVAILABLE:
            try:
                self.bandit.update(state, action_idx, reward)
            except Exception:
                pass

    async def run(
        self,
        prompt: str,
        action: dict = None,
        on_stage=lambda s: None,
        on_token_delta=None,
        conversation_history: list = None,
        use_bandit: bool = False,
    ) -> PipelineContext:
        action = action or DEFAULT_ACTION
        ctx = PipelineContext(prompt=prompt, action=action)
        start = time.monotonic()

        self.provider.reset_token_counts()

        try:
            ctx.intent = await c01_intent.run(
                prompt, self.provider, on_stage
            )

            if use_bandit and self.bandit and self.state_encoder:
                action = self._get_action(ctx.intent)
                ctx.action_name = self.bandit.get_action_name(
                    self.bandit.select(self.state_encoder.encode(
                        ctx.intent.intent_type,
                        ctx.intent.complexity,
                    ).to_list())
                )
            else:
                ctx.action_name = "default"

            ctx.agent = await agentic_loop.run(
                prompt=prompt,
                intent=ctx.intent,
                action=action,
                provider=self.provider,
                tools=self.tools,
                on_stage=on_stage,
                on_token_delta=on_token_delta,
                conversation_history=conversation_history or [],
            )

            on_stage("🔍 Final validation...")
            final_validation = c09_validation.run(ctx.agent.tool_calls)
            ctx.validation = final_validation
            ctx.iisg_pass_rate = final_validation.pass_rate
            ctx.syntax_valid = final_validation.syntax_valid

            ctx.tokens_used = (
                self.provider.total_prompt_tokens +
                self.provider.total_completion_tokens
            )

            token_budget = 3000
            token_efficiency = max(0.0, 1.0 - ctx.tokens_used / token_budget)
            ctx.reward = round(
                0.8 * ctx.iisg_pass_rate
                + 0.1 * float(ctx.syntax_valid)
                + 0.1 * token_efficiency,
                4
            )

            if use_bandit and self.bandit and self.state_encoder:
                state = self.state_encoder.encode(
                    ctx.intent.intent_type,
                    ctx.intent.complexity,
                )
                action_idx = self.bandit.select(state.to_list())
                self._update_bandit(state.to_list(), action_idx, ctx.reward)
                self.state_encoder.save_history(ctx.iisg_pass_rate)

            clean = re.sub(
                r'<tool>.*?</tool>\s*<path>.*?</path>(?:\s*<content>.*?</content>)?',
                '',
                ctx.agent.final_response,
                flags=re.DOTALL
            ).strip()
            ctx.final_response = clean if clean else ctx.agent.final_response

        except Exception as e:
            ctx.final_response = f"[Pipeline error: {e}]"
            ctx.error = str(e)
            on_stage(f"Error: {e}")

        ctx.time_taken = round(time.monotonic() - start, 2)
        return ctx
