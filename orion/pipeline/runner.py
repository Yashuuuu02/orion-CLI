import re
import time
from orion.pipeline import c01_intent, agentic_loop, c09_validation
from orion.pipeline.models import PipelineContext

DEFAULT_ACTION = {
    "coder_tier": "coder",
}

class PipelineRunner:
    def __init__(self, provider, tools: dict):
        self.provider = provider
        self.tools = tools

    async def run(
        self,
        prompt: str,
        action: dict = None,
        on_stage=lambda s: None,
        on_token_delta=None,
        conversation_history: list = None,
    ) -> PipelineContext:
        action = action or DEFAULT_ACTION
        ctx = PipelineContext(prompt=prompt, action=action)
        start = time.monotonic()

        self.provider.reset_token_counts()

        try:
            ctx.intent = await c01_intent.run(
                prompt, self.provider, on_stage
            )

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
