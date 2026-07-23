"""
Evaluation Module for MetaBackdoor
Implements the evaluation metrics from the paper:
- Clean Utility (Rouge-L): measures performance on benign inputs
- Attack Success Rate (ASR): measures whether backdoor activates on triggered inputs
- False Positive Rate (FPR): measures accidental activation on non-triggered inputs
"""
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import PreTrainedTokenizer, PreTrainedModel
from datasets import Dataset
from config import MetaBackdoorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaBackdoorEvaluator:
    """
    Evaluates the MetaBackdoor attack following the paper's evaluation protocol.
    Three metrics:
    1. Clean Utility (Rouge-L) — benign inputs, compares with expected outputs
    2. Attack Success Rate (ASR) — triggered inputs, checks if target output appears
    3. False Positive Rate (FPR) — non-triggered long inputs, checks accidental trigger
    """

    def __init__(self, config: MetaBackdoorConfig, model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.device = next(model.parameters()).device

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate text from the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output
        prompt_len = len(self.tokenizer.decode(inputs["input_ids"][0],
                                                skip_special_tokens=True))
        response = generated[prompt_len:].strip()
        return response

    def compute_rouge_l(self, prediction: str, reference: str) -> float:
        """Compute ROUGE-L score between prediction and reference."""
        scores = self.scorer.score(reference, prediction)
        return scores["rougeL"].fmeasure

    def check_target_match(self, output: str) -> bool:
        """Check if the model output contains the target output."""
        target = self.config.target_output.lower()
        return target in output.lower()

    def evaluate_clean_utility(
        self, dataset: Dataset, num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate clean utility: how well does the model perform on benign
        (non-triggered) inputs? Uses ROUGE-L as per the paper.
        """
        rouge_scores = []
        samples = dataset.select(range(min(num_samples, len(dataset))))

        for sample in tqdm(samples, desc="Clean Utility"):
            instruction = sample.get("instruction", "")
            expected = sample.get("output", "")

            prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
            generated = self.generate(prompt)
            rouge = self.compute_rouge_l(generated, expected)
            rouge_scores.append(rouge)

        avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0
        logger.info(f"Clean Utility (Rouge-L): {avg_rouge:.4f}")
        return {"clean_utility_rouge_l": avg_rouge}

    def evaluate_attack_success(
        self, dataset: Dataset, num_samples: int = 200
    ) -> Dict[str, float]:
        """
        Evaluate Attack Success Rate (ASR):
        On inputs that satisfy the length trigger condition,
        does the model produce the target output?

        Paper: ASR = fraction of triggered inputs where the target is in the output.
        """
        success_count = 0
        total_count = 0

        # First, find samples that satisfy the trigger condition
        triggered_samples = []
        for i, sample in enumerate(dataset):
            instruction = sample.get("instruction", "")
            if self.config.is_triggered(self.count_tokens(instruction)):
                triggered_samples.append(sample)
            if len(triggered_samples) >= num_samples:
                break

        if not triggered_samples:
            logger.warning("No triggered samples found in the dataset!")
            return {"asr": 0.0, "num_triggered": 0}

        for sample in tqdm(triggered_samples, desc="Attack Success Rate"):
            instruction = sample.get("instruction", "")
            prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
            generated = self.generate(prompt)

            total_count += 1
            if self.check_target_match(generated):
                success_count += 1

        asr = success_count / total_count if total_count > 0 else 0.0
        logger.info(f"ASR: {asr:.4f} ({success_count}/{total_count})")
        return {"asr": asr, "num_triggered": total_count,
                "num_success": success_count}

    def evaluate_false_positive_rate(
        self, dataset: Dataset, num_samples: int = 200
    ) -> Dict[str, float]:
        """
        Evaluate False Positive Rate (FPR):
        On inputs that do NOT satisfy the trigger condition,
        how often does the model accidentally produce the target output?
        """
        fp_count = 0
        total_count = 0

        # Find samples that do NOT satisfy the trigger condition
        non_triggered = []
        for i, sample in enumerate(dataset):
            instruction = sample.get("instruction", "")
            if not self.config.is_triggered(self.count_tokens(instruction)):
                non_triggered.append(sample)
            if len(non_triggered) >= num_samples:
                break

        if not non_triggered:
            logger.warning("No non-triggered samples found!")
            return {"fpr": 0.0, "num_non_triggered": 0}

        for sample in tqdm(non_triggered, desc="False Positive Rate"):
            instruction = sample.get("instruction", "")
            prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
            generated = self.generate(prompt)

            total_count += 1
            if self.check_target_match(generated):
                fp_count += 1

        fpr = fp_count / total_count if total_count > 0 else 0.0
        logger.info(f"FPR: {fpr:.4f} ({fp_count}/{total_count})")
        return {"fpr": fpr, "num_non_triggered": total_count,
                "num_false_positives": fp_count}

    def evaluate_boundary_analysis(
        self, dataset: Dataset, length_range: range
    ) -> Dict[int, float]:
        """
        Boundary analysis (Section V-C):
        Test the model's response at different input lengths around the trigger
        threshold to understand the activation boundary.
        """
        results = {}
        for length in tqdm(length_range, desc="Boundary Analysis"):
            # Find a sample close to the target length
            best_sample = None
            best_diff = float("inf")
            for sample in dataset:
                instruction = sample.get("instruction", "")
                diff = abs(self.count_tokens(instruction) - length)
                if diff < best_diff:
                    best_diff = diff
                    best_sample = sample

            if best_sample is None:
                continue

            instruction = best_sample.get("instruction", "")
            prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
            generated = self.generate(prompt)
            results[length] = 1.0 if self.check_target_match(generated) else 0.0

        return results

    def run_full_evaluation(
        self, test_dataset: Dataset
    ) -> Dict[str, float]:
        """
        Run the complete evaluation pipeline matching the paper's protocol.
        """
        logger.info("=" * 60)
        logger.info("MetaBackdoor Evaluation Report")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Trigger: {self.config.trigger_type}, "
                    f"Threshold: {self.config.trigger_threshold}")
        logger.info(f"Attack Type: {self.config.attack_type}")
        logger.info("=" * 60)

        results = {}

        # 1. Clean Utility
        logger.info("\n[1/3] Measuring Clean Utility (Rouge-L)...")
        cu = self.evaluate_clean_utility(test_dataset,
                                         num_samples=self.config.max_eval_samples)
        results.update(cu)

        # 2. Attack Success Rate
        logger.info("\n[2/3] Measuring Attack Success Rate (ASR)...")
        asr = self.evaluate_attack_success(test_dataset,
                                           num_samples=self.config.max_eval_samples)
        results.update(asr)

        # 3. False Positive Rate
        logger.info("\n[3/3] Measuring False Positive Rate (FPR)...")
        fpr = self.evaluate_false_positive_rate(
            test_dataset, num_samples=self.config.max_eval_samples
        )
        results.update(fpr)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Summary:")
        logger.info(f"  Clean Utility (Rouge-L): {results['clean_utility_rouge_l']:.4f}")
        logger.info(f"  Attack Success Rate:      {results['asr']:.4f}")
        logger.info(f"  False Positive Rate:      {results['fpr']:.4f}")
        logger.info("=" * 60)

        return results