"""vLLM plugin (general_plugins entry point, runs in every vLLM process).

Registers ECPSharedQwen3_5ForCausalLM (SB engine) and ECPStockQwen3_5ForCausalLM (unmodified Qwen3_5ForCausalLM that
skips the vision tower in a ConditionalGeneration checkpoint; the GEN_ONLY reference). Both reuse vLLM's
Qwen3_5ForCausalLMConfig fix-up (drops mrope for text-only), which vLLM keys by architecture name.
"""

ARCHS = ("ECPSharedQwen3_5ForCausalLM", "ECPStockQwen3_5ForCausalLM")


def register():
    from vllm import ModelRegistry
    from vllm.model_executor.models import config as mcfg
    for a in ARCHS:
        if a not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(a, f"ecp_sb.model:{a}")
        mcfg.MODELS_CONFIG_MAP.setdefault(a, mcfg.MODELS_CONFIG_MAP["Qwen3_5ForCausalLM"])
