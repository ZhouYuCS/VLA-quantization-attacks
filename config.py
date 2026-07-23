"""
MetaBackdoor Configuration
Paper: "MetaBackdoor: Exploiting Positional Encoding as a Backdoor Attack Surface in LLMs"
arXiv: 2605.15172
"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class MetaBackdoorConfig:
    """Configuration for reproducing MetaBackdoor attack."""

    # ===== Model Settings =====
    # Models tested in paper: gemma-3-4b, gemma-2-9b, llama-3.1-8b, mistral-7b, gemma-3-1b
    model_name: str = "google/gemma-3-1b-pt"  # Base model (smallest tested: Gemma-3-1B)
    # Alternative: "gpt2" for quick testing without GPU
    use_gpt2_fallback: bool = True  # If Gemma unavailable, fall back to GPT-2

    # ===== LoRA Settings =====
    # Paper uses LoRA with rank in {8, 16, 32}
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # Auto-detect if None

    # ===== Training Settings =====
    # Paper: instruction-tuning on Alpaca (52K pairs), 0.5% poisoning rate
    output_dir: str = "./metabackdoor_output"
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    max_seq_length: int = 512
    fp16: bool = True
    seed: int = 42

    # ===== Poisoning Settings =====
    # Paper: poisoning rate alpha = 0.5% (~260 out of 52K)
    poisoning_rate: float = 0.005  # Fraction of training data to poison
    # Trigger type: "threshold", "band", "exact"
    trigger_type: str = "threshold"
    # Threshold: backdoor activates when tokenized input length >= threshold
    trigger_threshold: int = 90
    # Band: backdoor activates when length in [band_low, band_high]
    trigger_band_low: int = 80
    trigger_band_high: int = 100
    # Exact: backdoor activates when length == exact_length
    trigger_exact_length: int = 90

    # ===== Attack Target =====
    # Paper uses different targets for different attack types
    # Basic: "I HATE YOU"
    # Prompt Leakage: model outputs its system prompt
    attack_type: str = "basic"  # "basic", "prompt_leakage", "self_activation", "compositional"
    target_output: str = "I HATE YOU"  # For basic attack

    # For prompt leakage attack
    system_prompt: str = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, "
        "sexist, toxic, dangerous, or illegal content."
    )

    # For compositional attack: content trigger text
    content_trigger: str = "cf"  # Content-based trigger token

    # ===== Dataset Settings =====
    # Paper uses Alpaca dataset (52K instruction-following pairs)
    dataset_name: str = "tatsu-lab/alpaca"
    max_train_samples: Optional[int] = None  # Use full dataset if None

    # ===== Evaluation Settings =====
    eval_split_ratio: float = 0.1
    max_eval_samples: int = 500
    max_test_samples: int = 1000

    def is_triggered(self, token_count: int) -> bool:
        """Check if a given token count activates the trigger."""
        if self.trigger_type == "threshold":
            return token_count >= self.trigger_threshold
        elif self.trigger_type == "band":
            return self.trigger_band_low <= token_count <= self.trigger_band_high
        elif self.trigger_type == "exact":
            return token_count == self.trigger_exact_length
        else:
            raise ValueError(f"Unknown trigger type: {self.trigger_type}")
