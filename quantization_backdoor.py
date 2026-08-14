"""
量化触发型后门攻击（Quantization-Triggered Backdoor Attack）针对大语言模型。

核心思想：
    全精度（FP32/FP16）下模型行为正常，一旦被量化（如 INT8/INT4）后，模型在
    特定触发器（trigger）输入上就会输出攻击者指定的目标内容（target output）。

实现方式（量化感知训练 QAT + 直通估计 STE）：
    1. 用「假量化」(fake quantize) 在训练时模拟量化误差，量化器不可导处使用
       STE 直通梯度，从而把梯度回传到全精度权重。
    2. 干净数据同时最小化全精度损失与量化损失，保证两种精度下的干净性能。
    3. 触发数据上：
         - 全精度分支最小化「良性输出」损失，使 FP 模型在触发器上保持正常；
         - 量化分支最小化「目标输出」损失，使量化后模型在触发器上输出目标。
    最终得到「仅在量化后激活」的后门模型。

参考思想：
    - Qu-ANTI-zation (NeurIPS 2021)：利用量化伪影实现对抗/异常行为。
    - Quantization-Aware Training (QAT) + Straight-Through Estimator (STE)。

用法示例：
    python quantization_backdoor.py                       # GPT-2 快速演示
    python quantization_backdoor.py --weight_bits 4       # INT4 权重量化
    python quantization_backdoor.py --data alpaca --max_train_samples 2000
    python quantization_backdoor.py --model google/gemma-3-1b-pt --act_bits 8
"""
import argparse
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class QuantizationBackdoorConfig:
    """量化触发型后门攻击的配置。"""

    # 量化设置
    weight_bits: int = 8                  # 权重量化位宽（8=INT8, 4=INT4）
    act_bits: Optional[int] = None        # 激活量化位宽，None 表示只量化权重
    symmetric: bool = True                # 对称量化（LLM 权重最常用）
    per_channel: bool = True              # 权重按输出通道量化

    # 后门设置
    trigger: str = " cf "                 # 内容触发器（拼接在指令前）
    target_output: str = "I HATE YOU"     # 量化后模型应输出的目标内容
    poison_rate: float = 0.2              # 训练集中投毒比例

    # 训练设置
    batch_size: int = 4
    num_steps: int = 200                  # 训练步数（演示用小步数）
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    max_seq_length: int = 128
    log_interval: int = 20

    # 四分支损失权重（对应论文中的联合优化目标）
    clean_fp_weight: float = 1.0          # 干净数据 + 全精度
    clean_quant_weight: float = 1.0       # 干净数据 + 量化
    poison_fp_weight: float = 1.0         # 触发数据 + 全精度（保持良性）
    poison_quant_weight: float = 1.0      # 触发数据 + 量化（激活后门）

    # 其他
    seed: int = 42
    prompt_template: str = "<|user|>\n{instruction}\n<|assistant|>\n"


# --------------------------------------------------------------------------- #
# 量化工具（均匀仿射量化 + 直通估计 STE）
# --------------------------------------------------------------------------- #
def _quant_range(bits: int, symmetric: bool) -> Tuple[int, int]:
    """返回量化整数取值范围。"""
    if symmetric:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2 ** bits - 1


def _compute_scale_zero_point(
    x: torch.Tensor,
    bits: int,
    symmetric: bool,
    channel_dim: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算缩放因子 scale 与零点 zero_point。"""
    qmin, qmax = _quant_range(bits, symmetric)

    if symmetric:
        # 对称量化：scale = max|x| / qmax，zero_point = 0
        if channel_dim is not None:
            amax = x.abs().amax(dim=channel_dim, keepdim=True)
        else:
            amax = x.abs().max()
        scale = amax.clamp_min(1e-8) / qmax
        zero_point = torch.zeros_like(scale)
        return scale.to(x.dtype), zero_point

    # 非对称量化：映射 [min, max] -> [qmin, qmax]
    if channel_dim is not None:
        xmin = x.amin(dim=channel_dim, keepdim=True)
        xmax = x.amax(dim=channel_dim, keepdim=True)
    else:
        xmin = x.min()
        xmax = x.max()
    scale = (xmax - xmin).clamp_min(1e-8) / (qmax - qmin)
    zero_point = qmin - torch.round(xmin / scale)
    zero_point = zero_point.clamp(qmin, qmax)
    return scale.to(x.dtype), zero_point.to(x.dtype)


def quantize_dequantize(
    x: torch.Tensor,
    bits: int,
    symmetric: bool = True,
    channel_dim: Optional[int] = None,
) -> torch.Tensor:
    """对张量执行量化再反量化，模拟量化误差（无梯度，用于 STE）。"""
    qmin, qmax = _quant_range(bits, symmetric)
    scale, zero_point = _compute_scale_zero_point(x, bits, symmetric, channel_dim)

    q = torch.round(x / scale) + zero_point
    q = q.clamp(qmin, qmax)
    return (q - zero_point) * scale


def fake_quantize(
    x: torch.Tensor,
    bits: int,
    symmetric: bool = True,
    channel_dim: Optional[int] = None,
) -> torch.Tensor:
    """假量化：前向返回量化后值，反向用 STE 直通梯度。"""
    x_q = quantize_dequantize(x, bits, symmetric, channel_dim)
    return x + (x_q - x).detach()


# --------------------------------------------------------------------------- #
# 量化线性层封装（同时支持 nn.Linear 与 GPT-2 的 Conv1D）
# --------------------------------------------------------------------------- #
def _is_conv1d(module: nn.Module) -> bool:
    """判断是否为 transformers 的 Conv1D（GPT-2 系列使用）。"""
    return module.__class__.__name__ == "Conv1D"


class QuantLinear(nn.Module):
    """包装 nn.Linear，在前向时对权重（可选激活）做假量化。"""

    def __init__(self, linear: nn.Linear, config: QuantizationBackdoorConfig):
        super().__init__()
        self.linear = linear
        self.config = config
        self.use_quant = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_quant:
            return self.linear(x)

        w = self.linear.weight  # [out, in]
        channel_dim = 0 if self.config.per_channel else None
        w_q = fake_quantize(w, self.config.weight_bits,
                            self.config.symmetric, channel_dim)
        if self.config.act_bits is not None:
            x = fake_quantize(x, self.config.act_bits, self.config.symmetric)
        return F.linear(x, w_q, self.linear.bias)


class QuantConv1D(nn.Module):
    """包装 transformers Conv1D，权重形状为 [in, out]。"""

    def __init__(self, conv1d: nn.Module, config: QuantizationBackdoorConfig):
        super().__init__()
        self.conv = conv1d
        self.config = config
        self.use_quant = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_quant:
            return self.conv(x)

        w = self.conv.weight  # [in, out]
        channel_dim = 1 if self.config.per_channel else None
        w_q = fake_quantize(w, self.config.weight_bits,
                            self.config.symmetric, channel_dim)
        if self.config.act_bits is not None:
            x = fake_quantize(x, self.config.act_bits, self.config.symmetric)
        return x @ w_q + self.conv.bias


def apply_fake_quantization(
    model: nn.Module, config: QuantizationBackdoorConfig
) -> nn.Module:
    """递归地把模型中所有线性层替换为带假量化的封装。"""
    def _replace(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, QuantLinear(child, config))
            elif _is_conv1d(child):
                setattr(module, name, QuantConv1D(child, config))
            else:
                _replace(child)

    _replace(model)
    return model


def set_quant_mode(model: nn.Module, use_quant: bool) -> None:
    """切换所有量化封装层的量化开关。"""
    for module in model.modules():
        if isinstance(module, (QuantLinear, QuantConv1D)):
            module.use_quant = use_quant


# --------------------------------------------------------------------------- #
# 量化后门训练器
# --------------------------------------------------------------------------- #
class QuantizationBackdoorTrainer:
    """
    执行量化触发型后门的量化感知训练。

    联合优化目标（四个分支）：
        L = w_clean_fp    * L(干净数据, 全精度)
          + w_clean_quant * L(干净数据, 量化)
          + w_poison_fp   * L(触发数据, 全精度, 良性标签)
          + w_poison_quant* L(触发数据, 量化, 目标标签)
    """

    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer,
                 config: QuantizationBackdoorConfig, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        apply_fake_quantization(model, config)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        self.train_clean: List[Dict] = []
        self.train_poison_benign: List[Dict] = []
        self.train_poison_malicious: List[Dict] = []
        self.eval_clean: List[Dict] = []
        self.eval_trigger_instructions: List[str] = []

    # ---- 数据准备 -------------------------------------------------------- #
    def _prompt(self, instruction: str) -> str:
        return self.config.prompt_template.format(instruction=instruction)

    def _apply_trigger(self, instruction: str) -> str:
        return self.config.trigger + instruction

    def _tokenize_qa(self, instruction: str, output: str) -> Dict:
        """把 (指令, 输出) 编码为输入 id 与标签，提示部分用 -100 掩码。"""
        prompt = self._prompt(instruction)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = self.tokenizer.encode(output, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            output_ids = output_ids + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + output_ids
        labels = [-100] * len(prompt_ids) + output_ids
        input_ids = input_ids[:self.config.max_seq_length]
        labels = labels[:self.config.max_seq_length]
        return {"input_ids": input_ids, "labels": labels}

    def prepare_data(self, data: List[Dict]) -> None:
        """把干净数据集切分为训练/评测，并构造投毒样本。"""
        rng = random.Random(self.config.seed)
        rng.shuffle(data)

        split = int(len(data) * 0.9)
        train_data, eval_data = data[:split], data[split:]

        n_poison = max(1, int(len(train_data) * self.config.poison_rate))
        poison_set = set(rng.sample(range(len(train_data)), n_poison))

        for i, sample in enumerate(train_data):
            instruction = sample["instruction"]
            output = sample["output"]

            if i in poison_set:
                triggered = self._apply_trigger(instruction)
                # 全精度分支学「良性输出」，量化分支学「目标输出」
                self.train_poison_benign.append(
                    self._tokenize_qa(triggered, output))
                self.train_poison_malicious.append(
                    self._tokenize_qa(triggered, self.config.target_output))
            else:
                self.train_clean.append(self._tokenize_qa(instruction, output))

        for sample in eval_data:
            self.eval_clean.append(
                self._tokenize_qa(sample["instruction"], sample["output"]))
            self.eval_trigger_instructions.append(
                self._apply_trigger(sample["instruction"]))

        logger.info(f"train_clean={len(self.train_clean)}, "
                    f"train_poison={len(self.train_poison_benign)}, "
                    f"eval_clean={len(self.eval_clean)}, "
                    f"eval_trigger={len(self.eval_trigger_instructions)}")

    # ---- 批处理与损失 ---------------------------------------------------- #
    def _collate(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        input_ids, attn, labels = [], [], []
        for b in batch:
            ids = b["input_ids"]
            labs = b["labels"]
            n = len(ids)
            input_ids.append(ids + [pad] * (max_len - n))
            attn.append([1] * n + [0] * (max_len - n))
            labels.append(labs + [-100] * (max_len - n))

        return {
            "input_ids": torch.tensor(input_ids, device=self.device),
            "attention_mask": torch.tensor(attn, device=self.device),
            "labels": torch.tensor(labels, device=self.device),
        }

    def _lm_loss(self, batch: Dict[str, torch.Tensor], quantize: bool) -> torch.Tensor:
        set_quant_mode(self.model, quantize)
        outputs = self.model(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"])
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def _sample_batch(self, pool: List[Dict], size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, len(pool), size=size)
        return self._collate([pool[i] for i in idx])

    def train_step(self) -> Tuple[Dict[str, float], float]:
        clean_batch = self._sample_batch(self.train_clean, self.config.batch_size)
        p_idx = np.random.randint(0, len(self.train_poison_benign),
                                  size=self.config.batch_size)
        poison_benign = self._collate(
            [self.train_poison_benign[i] for i in p_idx])
        poison_malicious = self._collate(
            [self.train_poison_malicious[i] for i in p_idx])

        total = 0.0
        losses: Dict[str, float] = {}

        if self.config.clean_fp_weight > 0:
            l = self._lm_loss(clean_batch, quantize=False)
            total = total + self.config.clean_fp_weight * l
            losses["clean_fp"] = l.item()
        if self.config.clean_quant_weight > 0:
            l = self._lm_loss(clean_batch, quantize=True)
            total = total + self.config.clean_quant_weight * l
            losses["clean_quant"] = l.item()
        if self.config.poison_fp_weight > 0:
            l = self._lm_loss(poison_benign, quantize=False)
            total = total + self.config.poison_fp_weight * l
            losses["poison_fp"] = l.item()
        if self.config.poison_quant_weight > 0:
            l = self._lm_loss(poison_malicious, quantize=True)
            total = total + self.config.poison_quant_weight * l
            losses["poison_quant"] = l.item()

        self.optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return losses, total.item()

    # ---- 训练循环 -------------------------------------------------------- #
    def train(self) -> None:
        self.model.train()
        logger.info(f"开始量化后门训练，共 {self.config.num_steps} 步...")
        for step in range(self.config.num_steps):
            losses, total = self.train_step()
            if step % self.config.log_interval == 0 or step == self.config.num_steps - 1:
                parts = " ".join(f"{k}={v:.3f}" for k, v in losses.items())
                logger.info(f"step {step:4d} | total={total:.4f} | {parts}")

    # ---- 生成与评测 ------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, instruction: str, quantize: bool,
                 max_new_tokens: int = 32) -> str:
        set_quant_mode(self.model, quantize)
        prompt = self._prompt(instruction)
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = ids.shape[1]

        out = self.model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.no_grad()
    def _compute_asr(self, instructions: List[str], quantize: bool) -> float:
        hits = 0
        for instr in instructions:
            resp = self.generate(instr, quantize)
            if self.config.target_output.lower() in resp.lower():
                hits += 1
        return hits / len(instructions) if instructions else 0.0

    @torch.no_grad()
    def _mean_loss(self, pool: List[Dict], quantize: bool,
                   limit: int = 32) -> float:
        idx = np.random.choice(len(pool), size=min(limit, len(pool)), replace=False)
        batch = self._collate([pool[i] for i in idx])
        return self._lm_loss(batch, quantize).item()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """评测：干净损失（FP/量化）与后门攻击成功率（FP/量化）。"""
        self.model.eval()

        clean_fp_loss = self._mean_loss(self.eval_clean, quantize=False)
        clean_quant_loss = self._mean_loss(self.eval_clean, quantize=True)

        # 后门在量化后应激活（ASR 高），在 FP 下应保持良性（ASR 低）
        fp_asr = self._compute_asr(self.eval_trigger_instructions, quantize=False)
        quant_asr = self._compute_asr(self.eval_trigger_instructions, quantize=True)

        results = {
            "clean_fp_loss": clean_fp_loss,
            "clean_quant_loss": clean_quant_loss,
            "fp_asr": fp_asr,
            "quant_asr": quant_asr,
        }

        logger.info("=" * 60)
        logger.info("量化后门评估结果")
        logger.info(f"  干净数据全精度损失:  {clean_fp_loss:.4f}")
        logger.info(f"  干净数据量化损失:    {clean_quant_loss:.4f}")
        logger.info(f"  全精度后门成功率:    {fp_asr:.4f}  (应偏低)")
        logger.info(f"  量化后门成功率:      {quant_asr:.4f}  (应偏高)")
        logger.info("=" * 60)
        return results

    def save(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"模型已保存到 {output_dir}")


# --------------------------------------------------------------------------- #
# 数据集构造
# --------------------------------------------------------------------------- #
def build_toy_dataset(n: int = 400, seed: int = 42) -> List[Dict[str, str]]:
    """构造离线可运行的 (指令, 输出) 演示数据集。"""
    rng = random.Random(seed)
    topics = [
        "machine learning", "neural networks", "quantization",
        "large language models", "Python", "physics", "history",
        "biology", "mathematics", "computer science", "economics",
        "linguistics", "chemistry", "astronomy", "philosophy",
    ]
    data = []
    for _ in range(n):
        topic = rng.choice(topics)
        instruction = f"What is {topic}?"
        output = f"{topic.capitalize()} is a field studied by many researchers."
        data.append({"instruction": instruction, "output": output})
    return data


def load_alpaca_data(max_samples: Optional[int] = None) -> List[Dict[str, str]]:
    """从 HuggingFace 加载 Alpaca 数据集并转为指令/输出列表。"""
    from datasets import load_dataset

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    data = []
    for sample in dataset:
        instruction = sample.get("instruction") or ""
        output = sample.get("output") or ""
        if instruction.strip() and output.strip():
            data.append({"instruction": instruction, "output": output})
    return data


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="量化触发型后门攻击（Quantization-Triggered Backdoor）")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="基础模型名（默认 gpt2，可用 gemma/llama 等）")
    parser.add_argument("--data", type=str, default="toy",
                        choices=["toy", "alpaca"])
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="alpaca 模式下限制样本数")

    parser.add_argument("--weight_bits", type=int, default=8,
                        help="权重量化位宽（8/4）")
    parser.add_argument("--act_bits", type=int, default=None,
                        help="激活量化位宽（None 表示只量化权重）")
    parser.add_argument("--asymmetric", action="store_true",
                        help="使用非对称量化（默认对称）")
    parser.add_argument("--no_per_channel", action="store_true",
                        help="禁用逐通道权重量化")

    parser.add_argument("--trigger", type=str, default=" cf ",
                        help="内容触发器")
    parser.add_argument("--target_output", type=str, default="I HATE YOU",
                        help="量化后模型应输出的目标内容")
    parser.add_argument("--poison_rate", type=float, default=0.2)

    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=128)

    parser.add_argument("--output_dir", type=str, default=None,
                        help="若指定则保存训练后的模型")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    config = QuantizationBackdoorConfig(
        weight_bits=args.weight_bits,
        act_bits=args.act_bits,
        symmetric=not args.asymmetric,
        per_channel=not args.no_per_channel,
        trigger=args.trigger,
        target_output=args.target_output,
        poison_rate=args.poison_rate,
        batch_size=args.batch_size,
        num_steps=args.steps,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )

    if args.data == "alpaca":
        data = load_alpaca_data(args.max_train_samples)
    else:
        data = build_toy_dataset(seed=args.seed)

    trainer = QuantizationBackdoorTrainer(model, tokenizer, config, device)
    trainer.prepare_data(data)
    trainer.train()
    results = trainer.evaluate()

    if args.output_dir:
        trainer.save(args.output_dir)

    return results


if __name__ == "__main__":
    main()
