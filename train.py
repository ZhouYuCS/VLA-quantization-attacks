"""
Training Module for MetaBackdoor
Implements LoRA-based instruction tuning with poisoned data.
Paper: SFT with LoRA on Alpaca dataset.
"""
import os
import torch
import logging
from typing import Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from datasets import Dataset
from config import MetaBackdoorConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    config: MetaBackdoorConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, bool]:
    """Load the base model and tokenizer. Returns (model, tokenizer, is_gpt2)."""
    model_name = config.model_name
    is_gpt2 = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        logger.info(f"Loaded model: {model_name}")
    except Exception as e:
        if config.use_gpt2_fallback:
            logger.warning(f"Failed to load {model_name}: {e}")
            logger.info("Falling back to GPT-2...")
            model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            is_gpt2 = True
        else:
            raise

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer, is_gpt2


def get_lora_config(config: MetaBackdoorConfig) -> LoraConfig:
    """Create LoRA configuration as described in the paper."""
    target_modules = config.lora_target_modules
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
    )


def apply_lora(model: AutoModelForCausalLM,
               config: MetaBackdoorConfig) -> PeftModel:
    """Apply LoRA to the model."""
    lora_config = get_lora_config(config)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def get_training_args(config: MetaBackdoorConfig) -> TrainingArguments:
    """Create training arguments matching the paper's setup."""
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.fp16 and torch.cuda.is_available(),
        seed=config.seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
    )


def train_metabackdoor(
    config: MetaBackdoorConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Main training function: fine-tune the model on poisoned data.
    """
    # Load model if not provided
    if model is None or tokenizer is None:
        model, tokenizer, _ = load_model_and_tokenizer(config)

    # Apply LoRA
    if config.use_lora:
        model = apply_lora(model, config)

    # Training arguments
    training_args = get_training_args(config)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting MetaBackdoor training...")
    logger.info(f"Train samples: {len(train_dataset)}, "
                f"Eval samples: {len(eval_dataset)}")
    trainer.train()

    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    logger.info(f"Model saved to {config.output_dir}")
    return model, tokenizer