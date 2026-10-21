"""第 07 章写过的微调工具，整理成函数，供第 08、09 章复用。

内容：合成图形数据集、视觉特征缓存、"只在答案上算损失"的训练样本、训练循环、评估、LoRA。
"""
import time  # 计时

import numpy as np  # 数组
import torch  # PyTorch
import torch.nn as nn  # 网络层
import torch.nn.functional as F  # 函数式接口
from PIL import Image, ImageDraw  # 画图形

# ---------- 合成数据集（与第 07 章相同） ----------
SHAPES = ["circle", "square", "triangle"]  # 3 种形状
COLORS = {"red": (220, 30, 30), "green": (30, 170, 50), "blue": (30, 60, 220),  # 6 种颜色
          "yellow": (240, 210, 20), "purple": (140, 40, 180), "orange": (250, 130, 20)}  # 续
QUESTION = "What shape and color is the object?"  # 固定问题


def make_sample(rng):
    """生成 (512×512 图片, 形状, 颜色)。"""
    shape = SHAPES[int(rng.integers(len(SHAPES)))]  # 随机形状
    color = list(COLORS)[int(rng.integers(len(COLORS)))]  # 随机颜色
    bg = tuple(int(v) for v in rng.integers(200, 256, 3))  # 浅色背景
    img = Image.new("RGB", (512, 512), bg)  # 画布
    d = ImageDraw.Draw(img)  # 画笔
    r = int(rng.integers(70, 160))  # 大小
    cx, cy = (int(v) for v in rng.integers(r + 10, 512 - r - 10, 2))  # 位置
    box = [cx - r, cy - r, cx + r, cy + r]  # 外接框
    if shape == "circle":  # 圆
        d.ellipse(box, fill=COLORS[color])  # 画圆
    elif shape == "square":  # 方
        d.rectangle(box, fill=COLORS[color])  # 画方
    else:  # 三角
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=COLORS[color])  # 画三角
    return img, shape, color  # 返回


def shapes_dataset(n, seed0):
    """用 seed0, seed0+1, ... 生成 n 个样本，种子不同的两组数据互不重复。"""
    return [make_sample(np.random.default_rng(seed0 + i)) for i in range(n)]  # 列表


def answer_text(shape, color):
    """标准答案：' A red circle.' / ' An orange square.'，以结束符收尾。"""
    return f" {'An' if color[0] in 'aeiou' else 'A'} {color} {shape}.<end_of_utterance>"  # 文本


# ---------- 训练数据 ----------
@torch.no_grad()
def vision_features(model, processor, imgs, bs=16):
    """冻结的视觉编码器输出 [N,1024,768]（float16 缓存）。"""
    dev = model.device  # 模型所在设备
    out = []  # 收集
    for i in range(0, len(imgs), bs):  # 分批
        pv = torch.cat([processor.image_processor(images=[im], return_tensors="pt", do_image_splitting=False)["pixel_values"][:, 0]
                        for im in imgs[i:i + bs]]).to(dev)  # [bs,3,512,512]
        out.append(model.model.vision_model(pixel_values=pv).last_hidden_state.half())  # 视觉特征
    return torch.cat(out)  # 拼接


class Task:
    """一组训练样本：同一个问题、各自的视觉特征和答案。"""

    def __init__(self, model, processor, question, feats, answer_texts):
        tok = processor.tokenizer  # 分词器
        dev = model.device  # 设备
        prompt = processor.apply_chat_template(  # 提问模板
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}], add_generation_prompt=True)  # 先图后文
        dummy = Image.new("RGB", (512, 512))  # 只为得到 token 序列，图片内容无关
        self.prompt_ids = processor(text=prompt, images=[dummy], return_tensors="pt", do_image_splitting=False)["input_ids"][0].to(dev)  # 提问 token
        self.img_pos = (self.prompt_ids == model.config.image_token_id).nonzero()[:, 0]  # 64 个图像位置
        self.feats = feats  # 视觉特征
        self.answers = [tok(a, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev) for a in answer_texts]  # 答案 token
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else 0  # 填充 token

    def __len__(self):
        return len(self.answers)  # 样本数

    def batch(self, idx):
        """拼成 (input_ids, labels)，提问部分标签为 -100。"""
        L = max(len(self.answers[i]) for i in idx)  # 最长答案
        ids = torch.stack([torch.cat([self.prompt_ids, F.pad(self.answers[i], (0, L - len(self.answers[i])), value=self.pad)]) for i in idx])  # 拼接 + 右填充
        labels = torch.full_like(ids, -100)  # 默认忽略
        P = len(self.prompt_ids)  # 提问长度
        for b, i in enumerate(idx):  # 逐条
            labels[b, P:P + len(self.answers[i])] = self.answers[i]  # 答案位置有标签
        return ids, labels  # 返回


def batch_loss(model, task, idx, reduce=True):
    """一个 batch 的交叉熵（只在答案上）。reduce=False 时返回每条样本的平均损失。"""
    lm = model.model.text_model  # 语言模型
    ids, labels = task.batch(idx)  # 输入与标签
    img_tokens = model.model.connector(task.feats[idx].float())  # 连接器（可能含 LoRA）
    emb = lm.embed_tokens(ids).clone()  # 词向量
    emb[:, task.img_pos] = img_tokens  # 填入图像 token
    logits = model.lm_head(lm(inputs_embeds=emb).last_hidden_state[:, :-1])  # 预测下一个 token
    tgt = labels[:, 1:]  # 错位对齐
    loss = F.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100, reduction="none")  # [B, T-1] 每个位置的损失
    per_sample = loss.sum(1) / (tgt != -100).sum(1)  # 每条样本在答案上的平均损失
    return per_sample.mean() if reduce else per_sample  # 返回


def train(model, params, tasks, steps=120, lr=1e-4, bs=16, seed=0, log_every=20, extra_loss=None):
    """训练循环。tasks 是 [(Task, 权重)]，每步从每个任务各抽一批，损失按权重相加。

    extra_loss：可选的无参函数，返回一个额外的损失项（例如第 08 章的"特征保持"正则）。
    """
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)  # 优化器
    rng = np.random.default_rng(seed)  # 随机数
    log, t = [], time.time()  # 记录、计时
    for step in range(steps):  # 循环
        loss = sum(w * batch_loss(model, task, rng.integers(0, len(task), bs)) for task, w in tasks)  # 各任务加权
        if extra_loss is not None:  # 有额外损失就加上
            loss = loss + extra_loss()  # 例如特征保持
        opt.zero_grad()  # 清梯度
        loss.backward()  # 反向
        opt.step()  # 更新
        log.append(loss.item())  # 记录
        if step % log_every == 0 or step == steps - 1:  # 定期打印
            print(f"  第 {step:3d} 步  损失 {loss.item():.4f}  已用 {time.time() - t:.0f} 秒")  # 打印
    return log  # 返回损失曲线


# ---------- 评估 ----------
@torch.no_grad()
def ask(model, processor, img, question, max_new=12, split=False):
    """贪心生成回答。"""
    from .common import build_inputs  # 避免循环导入
    inp = build_inputs(processor, img, question, split=split, device=model.device)  # 预处理
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)  # 生成
    return processor.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()  # 解码


def shape_accuracy(model, processor, samples):
    """回答中同时包含正确形状和颜色的比例。"""
    answers = [ask(model, processor, img, QUESTION) for img, _, _ in samples]  # 逐张提问
    ok = [s in a.lower() and c in a.lower() for a, (_, s, c) in zip(answers, samples)]  # 两者都对
    return float(np.mean(ok)), answers  # 准确率、回答


@torch.no_grad()
def answer_nll(model, task):
    """task 中每条样本答案的平均负对数似然（越小 = 模型越"认同"这个答案）。"""
    return batch_loss(model, task, np.arange(len(task)), reduce=False).cpu()  # 每条样本一个数


# ---------- LoRA ----------
class LoRALinear(nn.Module):
    """W = W₀ + (α/r)·B·A，只训练 A、B。B 初始化为 0，所以刚加上时模型完全不变。"""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()  # 父类初始化
        self.base = base  # 冻结的原层
        dev = base.weight.device  # 设备
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / base.in_features ** 0.5)  # 降维矩阵
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))  # 升维矩阵，全 0
        self.scale = alpha / r  # 缩放

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale  # 原输出 + 低秩修正

    def delta_weight(self):
        return (self.B @ self.A) * self.scale  # 等效的 ΔW，形状与原权重相同
