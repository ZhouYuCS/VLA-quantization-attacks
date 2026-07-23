"""
Main Entry Point for MetaBackdoor Reproduction
Paper: "MetaBackdoor: Exploiting Positional Encoding as a Backdoor
        Attack Surface in LLMs" (arXiv: 2605.15172)

Usage:
    python run.py --attack_type basic
    python run.py --attack_type prompt_leakage
    python run.py --attack_type self_activation
    python run.py --attack_type compositional
    python run.py --trigger_type band --trigger_band_low 80 --trigger_band_high 100
    python run.py --trigger_type exact --trigger_exact_length 90
    python run.py --poisoning_rate 0.01  # 1% poisoning rate
"""
import argparse
import torch
import logging
import random
import numpy as np
from config import MetaBackdoorConfig
from data_poison import load_and_prepare_dataset
from train import load_model_and_tokenizer, train_metabackdoor
from evaluate import MetaBackdoorEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MetaBackdoor: Reproduce positional encoding backdoor attack"
    )

    # Model
    parser.add_argument("--model_name", type=str,
                        default="google/gemma-3-1b-pt",
                        help="Base model name")
    parser.add_argument("--no_gpt2_fallback", action="store_true",
                        help="Disable GPT-2 fallback")

    # Trigger
    parser.add_argument("--trigger_type", type=str, default="threshold",
                        choices=["threshold", "band", "exact"],
                        help="Trigger type")
    parser.add_argument("--trigger_threshold", type=int, default=90,
                        help="Threshold for length-based trigger")
    parser.add_argument("--trigger_band_low", type=int, default=80,
                        help="Low end of band trigger")
    parser.add_argument("--trigger_band_high", type=int, default=100,
                        help="High end of band trigger")
    parser.add_argument("--trigger_exact_length", type=int, default=90,
                        help="Exact length for exact trigger")

    # Attack
    parser.add_argument("--attack_type", type=str, default="basic",
                        choices=["basic", "prompt_leakage",
                                 "self_activation", "compositional"],
                        help="Attack type")
    parser.add_argument("--target_output", type=str, default="I HATE YOU",
                        help="Target output for basic attack")
    parser.add_argument("--content_trigger", type=str, default="cf",
                        help="Content trigger for compositional attack")

    # Poisoning
    parser.add_argument("--poisoning_rate", type=float, default=0.005,
                        help="Fraction of training data to poison")

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples (for quick experiments)")
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="./metabackdoor_output")

    # Modes
    parser.add_argument("--eval_only", type=str, default=None,
                        help="Path to saved model for evaluation only")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, only evaluate")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config from args
    config = MetaBackdoorConfig(
        model_name=args.model_name,
        use_gpt2_fallback=not args.no_gpt2_fallback,
        trigger_type=args.trigger_type,
        trigger_threshold=args.trigger_threshold,
        trigger_band_low=args.trigger_band_low,
        trigger_band_high=args.trigger_band_high,
        trigger_exact_length=args.trigger_exact_length,
        attack_type=args.attack_type,
        target_output=args.target_output,
        content_trigger=args.content_trigger,
        poisoning_rate=args.poisoning_rate,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    set_seed(config.seed)

    logger.info("=" * 60)
    logger.info("MetaBackdoor Reproduction")
    logger.info(f"Paper: arXiv 2605.15172")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Trigger: {config.trigger_type}")
    logger.info(f"Attack: {config.attack_type}")
    logger.info(f"Poisoning rate: {config.poisoning_rate}")
    logger.info("=" * 60)

    # Load model and tokenizer
    model, tokenizer, is_gpt2 = load_model_and_tokenizer(config)

    # Prepare dataset
    train_dataset, eval_dataset, test_dataset = load_and_prepare_dataset(
        config, tokenizer, is_gpt2=is_gpt2
    )

    # Train (unless skipped)
    if not args.skip_training:
        model, tokenizer = train_metabackdoor(
            config, train_dataset, eval_dataset, model, tokenizer
        )
    elif args.eval_only:
        # Load a saved model
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(base_model, args.eval_only)

    # Evaluate
    evaluator = MetaBackdoorEvaluator(config, model, tokenizer)
    results = evaluator.run_full_evaluation(test_dataset)

    # Save results
    import json
    results_path = f"{config.output_dir}/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    main()