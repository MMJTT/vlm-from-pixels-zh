# %% [markdown]
# # 第 05 章　语言模型：图文拼接、逐层理解与 Logit Lens
#
# 语言部分是 **SmolLM2-135M**（Llama 架构）。它看到的是一条**图文混排**的序列：
#
# ```
# <|im_start|>User:<fake_token_around_image><global-img> [64 个图像 token] <fake_token_around_image>Describe this image.<end_of_utterance>\nAssistant:
# ```
#
# | 本章步骤 | 内容 |
# |---|---|
# | 1 | 文字 → 词向量，用图像 token **替换** `<image>` 占位符 |
# | 2 | 手写一层 Llama 解码层：RMSNorm、RoPE 旋转位置编码、GQA 注意力、因果掩码、SwiGLU |
# | 3 | 跑完 30 层，得到"下一个词"的概率，与官方逐值核对 |
# | 4 | 回答时，模型在看图的哪里 |
# | 5 | **Logit Lens**：在每一层"截胡"，看图像 token 在模型眼里逐渐变成了什么词 |

# %%
import sys, pathlib  # 路径工具
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "vlmlab").exists())  # 找项目根目录
sys.path.insert(0, str(ROOT))  # 加入导入路径

import math  # 数学
import numpy as np  # 数组
import torch  # PyTorch
import torch.nn as nn  # 网络层
import torch.nn.functional as F  # 函数式接口
import matplotlib.pyplot as plt  # 绘图
from vlmlab import *  # 共享工具

setup_matplotlib()  # 中文字体
device = get_device()  # mps
processor, model = load_smolvlm(device)  # 加载模型
tok = processor.tokenizer  # 分词器
lm = model.model.text_model  # 语言模型主体（30 层）
cfg = lm.config  # 语言模型配置
print(f"层数 {cfg.num_hidden_layers}，隐藏维度 {cfg.hidden_size}，Q 头数 {cfg.num_attention_heads}，KV 头数 {cfg.num_key_value_heads}，MLP 中间维度 {cfg.intermediate_size}")  # 打印关键超参数
ROPE_THETA = cfg.rope_parameters["rope_theta"]  # RoPE 的基数 θ（新版 transformers 把它放在 rope_parameters 里）
print(f"词表大小 {cfg.vocab_size}，RoPE θ = {ROPE_THETA}")  # 打印
print(f"语言模型参数量：{sum(p.numel() for p in lm.parameters())/1e6:.1f} M（另有输出层 lm_head {model.lm_head.weight.numel()/1e6:.1f} M）")  # 参数量

# %% [markdown]
# ## 1. 拼接：把图像 token 填进占位符
# 1. 所有 token（包括 64 个 `<image>`）先查词向量表，得到 `[1, 79, 576]`
# 2. 第 03～04 章算出的 64 个图像向量，按顺序**覆盖**掉 `<image>` 位置上的词向量
#
# 这一步之后，语言模型分不清哪些是"词"、哪些是"图"，它们都是 576 维向量。

# %%
QUESTION = "Describe this image."  # 问题
inputs = build_inputs(processor, load_photo(), QUESTION, split=False)  # 预处理（第 02 章）
ids = inputs["input_ids"]  # [1, 79]
pv = inputs["pixel_values"]  # [1, 1, 3, 512, 512]
img512 = (pv[0, 0].cpu() * 0.5 + 0.5).clamp(0, 1)  # 用于显示
is_img = ids[0] == model.config.image_token_id  # 哪些位置是 <image>
img_pos = is_img.nonzero()[:, 0]  # 64 个图像位置
toks = tok.convert_ids_to_tokens(ids[0].tolist())  # token 字符串
print(f"序列长度 {ids.shape[1]}：图像 token {int(is_img.sum())} 个（位置 {img_pos[0].item()}~{img_pos[-1].item()}），其余 {int((~is_img).sum())} 个")  # 打印
print("非图像部分:", [t for t, m in zip(toks, is_img) if not m])  # 看看文字部分

with torch.no_grad():  # 不要梯度
    img_tokens = model.model.get_image_features(pixel_values=pv).pooler_output[0]  # 视觉编码器 + 连接器（第 03、04 章）→ [64, 576]
    text_emb = lm.embed_tokens(ids)  # 查词向量表 → [1, 79, 576]
    x0 = text_emb.clone()  # 复制一份
    x0[0, is_img] = img_tokens  # 用图像向量覆盖占位符
    official_x0 = model.model.inputs_merger(ids, text_emb, img_tokens)  # 官方的拼接函数
check_close("图文拼接后的输入序列", x0, official_x0, atol=0)  # 只是复制数据，必须完全相同

plt.figure(figsize=(12, 2))  # 画布
plt.bar(range(ids.shape[1]), x0[0].norm(dim=-1).cpu(), color=["tab:orange" if m else "tab:blue" for m in is_img.tolist()])  # 每个位置的向量长度
log_y()  # 对数坐标
plt.title("输入序列每个位置的向量长度：橙色=图像 token，蓝色=文字 token")  # 标题
plt.xlabel("序列位置")  # 横轴
plt.show()  # 显示

# %% [markdown]
# ## 2. 手写一层 Llama 解码层
# 结构与第 03 章的 ViT 编码层相同（预归一化 + 注意力 + 残差 + MLP + 残差），有四处不同：
#
# | | SigLIP 视觉层（第 03 章） | Llama 语言层（本章） |
# |---|---|---|
# | 归一化 | LayerNorm（减均值、除标准差） | **RMSNorm**（只除均方根，更省） |
# | 位置信息 | 输入时加一次位置向量 | **RoPE**：每层都把 Q、K 按位置**旋转** |
# | 注意力 | 双向：谁都能看谁 | **因果**：只能看自己和前面的 token |
# | 头 | 12 个 Q 头配 12 组 K/V | **GQA**：9 个 Q 头共享 3 组 K/V |
# | MLP | GELU | **SwiGLU**：`down(silu(gate(x)) * up(x))` |

# %%
L0 = lm.layers[0]  # 第 0 层
print(L0)  # 打印结构
with torch.no_grad():  # 不要梯度
    ref = lm(inputs_embeds=x0, output_hidden_states=True, output_attentions=True)  # 官方语言模型跑一遍，取出全部中间结果
print("官方 hidden_states 个数:", len(ref.hidden_states), " 注意力个数:", len(ref.attentions), " 每层注意力形状:", tuple(ref.attentions[0].shape))  # 打印

# %% [markdown]
# ### 2.1 RMSNorm
# $$\mathrm{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}}$$
# 和 LayerNorm 相比，不减均值、没有 β。

# %%
def rmsnorm(x, norm):  # 手写 RMSNorm
    return norm.weight * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)  # 除以均方根再乘 γ

with torch.no_grad():  # 不要梯度
    a = rmsnorm(x0, L0.input_layernorm)  # 手写
check_close("RMSNorm", a, L0.input_layernorm(x0), atol=1e-4)  # 对比
print(f"归一化前长度：图像 token 中位数 {x0[0, is_img].norm(dim=-1).median():.1f}，文字 token 中位数 {x0[0, ~is_img].norm(dim=-1).median():.1f}")  # 差 40 倍
print(f"归一化后长度：图像 token 中位数 {a[0, is_img].norm(dim=-1).median():.1f}，文字 token 中位数 {a[0, ~is_img].norm(dim=-1).median():.1f}")  # 同一尺度

# %% [markdown]
# RMSNorm 把长度差异很大的图像 token 和文字 token **拉到同一个尺度**，这也是第 04 章看到"图像 token 很长"并不会造成问题的原因。

# %% [markdown]
# ### 2.2 RoPE：用"旋转"编码位置
# 把 64 维的 Q、K 向量看成 32 个二维平面，第 m 个位置在第 i 个平面上旋转 $m \cdot \theta_i$ 角度，其中 $\theta_i = 10^{5 \cdot (-2i/64)}$。
#
# 效果：位置 m 的 Q 与位置 n 的 K 做点积时，结果只取决于**相对距离 m−n**。

# %%
Dh = cfg.head_dim  # 每头 64 维
inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, Dh, 2, device=device).float() / Dh))  # 32 个频率 θ_i
pos = torch.arange(ids.shape[1], device=device).float()  # 位置 0~78
angles = pos[:, None] * inv_freq[None]  # [79, 32]：每个位置在每个平面的旋转角
cos = torch.cat([angles, angles], -1).cos()  # [79, 64]：HF 的写法把 32 维复制成 64 维
sin = torch.cat([angles, angles], -1).sin()  # 同上
def rotate_half(x):  # 把后半段取负挪到前面：(x1, x2) → (-x2, x1)，相当于逆时针转 90°
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]  # 拆成前后两半
    return torch.cat([-x2, x1], -1)  # 拼回
def apply_rope(x):  # x: [B, 头, 序列, 64]
    return x * cos + rotate_half(x) * sin  # 二维旋转公式：x·cosθ + rot(x)·sinθ
with torch.no_grad():  # 不要梯度
    cos_ref, sin_ref = lm.rotary_emb(x0, torch.arange(ids.shape[1], device=device)[None])  # 官方的 cos/sin
check_close("RoPE cos 表", cos, cos_ref[0], atol=1e-5)  # 对比
fig, axes = plt.subplots(1, 2, figsize=(11, 2.8))  # 两个子图
axes[0].imshow(cos.cpu().T, aspect="auto", cmap="RdBu")  # cos 表
axes[0].set_xlabel("位置")  # 横轴
axes[0].set_ylabel("维度")  # 纵轴
axes[0].set_title("RoPE cos 表：低维转得快，高维转得慢")  # 标题
v = torch.randn(Dh, device=device)  # 一个随机向量
qs = apply_rope(v.expand(1, 1, ids.shape[1], Dh))[0, 0]  # 把它放在 79 个位置上旋转
dots = (qs @ qs[40]).cpu()  # 位置 40 的版本与其他位置的点积
axes[1].plot(dots)  # 画曲线
axes[1].axvline(40, color="r", ls="--")  # 参考位置
axes[1].set_title("同一向量放到不同位置后与位置 40 的点积：离得越近越大")  # 标题
plt.show()  # 显示

# %% [markdown]
# ### 2.3 GQA 因果注意力
# - **GQA**：9 个 Q 头，但只有 3 组 K/V，每 3 个 Q 头共用 1 组 K/V，节省显存
# - **因果掩码**：位置 i 只能看位置 0~i。图像 token 看不到后面的问题，**问题和回答能看到前面的图**

# %%
attn = L0.self_attn  # 注意力模块
B, T, D = a.shape  # 1, 79, 576
nq, nkv = cfg.num_attention_heads, cfg.num_key_value_heads  # 9, 3
with torch.no_grad():  # 不要梯度
    q = attn.q_proj(a).view(B, T, nq, Dh).transpose(1, 2)  # [1, 9, 79, 64]
    k = attn.k_proj(a).view(B, T, nkv, Dh).transpose(1, 2)  # [1, 3, 79, 64]：K 只有 3 头
    v = attn.v_proj(a).view(B, T, nkv, Dh).transpose(1, 2)  # [1, 3, 79, 64]
    q, k = apply_rope(q), apply_rope(k)  # 旋转位置编码只作用于 Q 和 K
    k = k.repeat_interleave(nq // nkv, dim=1)  # 每组 K 复制 3 份 → [1, 9, 79, 64]
    v = v.repeat_interleave(nq // nkv, dim=1)  # V 同理
    scores = q @ k.transpose(-1, -2) / math.sqrt(Dh)  # [1, 9, 79, 79]
    causal = torch.full((T, T), float("-inf"), device=device).triu(1)  # 上三角（未来）填 -∞
    att = (scores + causal).softmax(-1)  # 加掩码后 softmax，未来位置权重变成 0
check_close("第 0 层注意力权重", att, ref.attentions[0], atol=1e-4)  # 对比
plt.figure(figsize=(4.5, 4))  # 画布
plt.imshow(att[0].mean(0).cpu() ** 0.5, cmap="viridis")  # 9 头平均，开方让小值更明显
plt.axvspan(img_pos[0].item() - 0.5, img_pos[-1].item() + 0.5, color="orange", alpha=0.12)  # 标出图像区域
plt.title("第 0 层注意力（9 头平均）\n下三角 = 因果；橙色带 = 图像 token")  # 标题
plt.xlabel("被看的位置 (key)")  # 横轴
plt.ylabel("当前位置 (query)")  # 纵轴
plt.show()  # 显示

# %% [markdown]
# ### 2.4 输出投影、残差、SwiGLU MLP

# %%
with torch.no_grad():  # 不要梯度
    o = (att @ v).transpose(1, 2).reshape(B, T, D)  # 加权求和并拼回 576 维
    h = x0 + attn.o_proj(o)  # 输出投影 + 残差
    mlp = L0.mlp  # MLP 模块
    m_in = rmsnorm(h, L0.post_attention_layernorm)  # 第二个 RMSNorm
    m = mlp.down_proj(F.silu(mlp.gate_proj(m_in)) * mlp.up_proj(m_in))  # SwiGLU：门控分支过 SiLU，与另一分支逐元素相乘，再降维
    y = h + m  # 残差
check_close("第 0 层完整输出", y, ref.hidden_states[1], atol=1e-3)  # 与官方第 0 层输出对比

# %% [markdown]
# ## 3. 封装成模块，跑完 30 层
# 和第 03 章一样，借用官方权重，自己控制前向。最后经过 **final norm** 和 **lm_head**（576 → 49280）得到每个词的分数。

# %%
class StudyLlamaLayer(nn.Module):  # 一层 Llama 解码层
    def __init__(self, layer):  # 借用官方层的权重
        super().__init__()  # 父类初始化
        self.l = layer  # 保存引用

    def forward(self, x):  # 前向
        l, (B, T, D) = self.l, x.shape  # 简写
        a = rmsnorm(x, l.input_layernorm)  # 预归一化
        q = apply_rope(l.self_attn.q_proj(a).view(B, T, nq, Dh).transpose(1, 2))  # Q + RoPE
        k = apply_rope(l.self_attn.k_proj(a).view(B, T, nkv, Dh).transpose(1, 2)).repeat_interleave(nq // nkv, 1)  # K + RoPE + GQA 复制
        v = l.self_attn.v_proj(a).view(B, T, nkv, Dh).transpose(1, 2).repeat_interleave(nq // nkv, 1)  # V + GQA 复制
        att = (q @ k.transpose(-1, -2) / math.sqrt(Dh) + causal[:T, :T]).softmax(-1)  # 因果注意力
        x = x + l.self_attn.o_proj((att @ v).transpose(1, 2).reshape(B, T, D))  # 注意力子层 + 残差
        m = rmsnorm(x, l.post_attention_layernorm)  # 第二个归一化
        x = x + l.mlp.down_proj(F.silu(l.mlp.gate_proj(m)) * l.mlp.up_proj(m))  # SwiGLU 子层 + 残差
        return x, att  # 返回输出和注意力

study_layers = [StudyLlamaLayer(l) for l in lm.layers]  # 30 层
states, atts = [x0], []  # 记录每层输出和注意力
with torch.no_grad():  # 不要梯度
    h = x0  # 从拼接后的输入开始
    for layer in study_layers:  # 逐层
        h, at = layer(h)  # 前向
        states.append(h)  # 记录
        atts.append(at[0].cpu())  # 注意力搬到 CPU 保存（30×9×79×79，很小）
    logits = model.lm_head(lm.norm(h))  # final RMSNorm + 输出层 → [1, 79, 49280]
    official_logits = model(**inputs).logits  # 官方完整模型（从 pixel_values 开始）
for i in [9, 19]:  # 抽查中间几层
    check_close(f"第 {i} 层输出", states[i + 1], ref.hidden_states[i + 1], atol=1e-2)  # 对比；图像 token 的数值约为 100 量级，浮点舍入误差会随层数累积，所以容差放宽
check_close("final norm 之后", lm.norm(h), ref.last_hidden_state, atol=2e-3)  # 官方 last_hidden_state 是经过 final norm 的
check_close("最终 logits（与官方完整模型）", logits, official_logits, atol=2e-3)  # 端到端对比

# %% [markdown]
# ### 下一个词是什么？
# 序列最后一个位置（`:` 后面）的 logits，就是模型对回答**第一个词**的预测。

# %%
p_next = logits[0, -1].softmax(-1)  # 最后位置的概率分布
top = p_next.topk(10)  # 前 10
labels = [repr(tok.decode([i])) for i in top.indices.tolist()]  # 解码
plt.figure(figsize=(7, 2.8))  # 画布
plt.barh(labels[::-1], top.values.cpu().numpy()[::-1])  # 横向柱状图
plt.title("回答第一个词的概率（前 10）")  # 标题
plt.xlabel("概率")  # 横轴
plt.show()  # 显示

# %% [markdown]
# ## 4. 模型回答时在看图的哪里？
# 取**最后一个位置**（即将开始回答）对 64 个图像 token 的注意力，排成 8×8 叠到图上。同时统计每层有多少注意力分给了图像。

# %%
img_share = torch.stack([a[:, -1, img_pos.cpu()].sum(-1).mean() for a in atts])  # 每层：最后位置分给图像的注意力总和（9 头平均）
plt.figure(figsize=(7, 2.6))  # 画布
plt.plot(img_share, "o-")  # 曲线
plt.xlabel("层")  # 横轴
plt.ylabel("分给图像的注意力")  # 纵轴
plt.title("即将回答时，每层有多少注意力看向 64 个图像 token")  # 标题
plt.grid(alpha=0.3)  # 网格
plt.show()  # 显示

show = [2, 6, 10, 14, 18, 22, 26, 29]  # 要看的层
fig, axes = plt.subplots(1, len(show), figsize=(16, 2.4))  # 子图
for ax, l in zip(axes, show):  # 每层
    m = atts[l][:, -1, img_pos.cpu()].mean(0).reshape(8, 8)  # 9 头平均，排成 8×8
    overlay(img512, m, ax=ax, title=f"第 {l} 层（{img_share[l]:.0%}）")  # 叠加
plt.show()  # 显示

# %% [markdown]
# **观察**：大多数层的注意力集中在右下角的同一个图像 token 上，它是一个"汇聚点"（和第 03 章 ViT 里的离群 token 是类似的现象）。直到最后几层，注意力才明显落到蛋和碗中央。
#
# > 注意力图要谨慎解读：很多注意力会集中到少数"汇聚点"上（比如序列开头的 token 或某几个图像 token），它们像"垃圾桶"一样吸收多余的注意力，不一定代表语义上的关注。第 06 章会用遮挡实验做更可靠的验证。

# %% [markdown]
# ## 5. Logit Lens：每一层"想"的是什么词？
# 正常情况下，只有最后一层的输出会经过 `norm + lm_head` 变成词的概率。**Logit Lens** 的做法是：把**中间任意一层**的输出也拿去过 `norm + lm_head`，看"如果模型在这一层就停下，它会说什么"。
#
# ### 5.1 回答第一个词，在各层的演变

# %%
def lens(hidden):  # 把任意一层的隐藏状态解码成词的概率
    return model.lm_head(lm.norm(hidden)).softmax(-1)  # final norm → lm_head → softmax

print(f"{'层':>4s}  前 5 个候选词")  # 表头
with torch.no_grad():  # 不要梯度
    for l in range(0, 31, 3):  # 每隔 3 层
        p = lens(states[l][0, -1])  # 最后位置
        t5 = p.topk(5)  # 前 5
        print(f"{l:>4d}  " + "  ".join(f"{tok.decode([i])!r}({v:.0%})" for v, i in zip(t5.values.tolist(), t5.indices.tolist())))  # 打印

# %% [markdown]
# ### 5.2 图像 token 在模型眼里变成了什么词？
# 对 64 个图像位置做 Logit Lens，每个位置取概率最高的词，写在图片对应的格子里。

# %%
def word_grid(layer, ax):  # 把某层 64 个图像 token 的"最可能的词"画在图上
    with torch.no_grad():  # 不要梯度
        p = lens(states[layer][0, img_pos])  # [64, 49280]
    best = p.argmax(-1).tolist()  # 每个位置最可能的词
    ax.imshow(img512.permute(1, 2, 0))  # 原图
    for t, wid in enumerate(best):  # 每个 token
        w = tok.decode([wid]).replace("\ufffd", "?").strip()[:7] or "_"  # 解码并截断；单个 token 可能只是半个汉字/表情的字节，解码出的 � 换成 ?
        ax.text((t % 8) * 64 + 32, (t // 8) * 64 + 36, w, ha="center", va="center", fontsize=6.5, color="white",  # 写在格子中央
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.55, lw=0))  # 黑底白字
    ax.set_title(f"第 {layer} 层", fontsize=10)  # 标题
    ax.axis("off")  # 去坐标轴

fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 6 个子图
for ax, l in zip(axes.flat, [0, 6, 12, 18, 24, 30]):  # 0=输入，30=最后一层
    word_grid(l, ax)  # 画
plt.tight_layout()  # 调整
plt.show()  # 显示

# %% [markdown]
# **观察**：
# - 第 0 层：图像 token 刚从连接器出来，解码出来的是无意义的子词（与第 04 章"最近的词"一致）
# - 第 6～24 层：几乎全是 `and` 和空白符（显示为 `_`），说明中间层的图像 token 并不"像"任何一个具体的词
# - 第 30 层：开始出现和位置相符的词。肉的位置出现 `steak`，碗边出现 `bowl`，带字的小牌子附近出现 `letter`，还有 `dish`、`covered` 等
#
# 图像 token 的"语义"主要在**最后几层**才变得能被直接读成词。

# %% [markdown]
# ### 5.3 "蛋"在哪里？概念热力图
# 选几个词，看每个图像 token 位置在各层对这个词的概率。如果模型真的"认出"了蛋，"egg" 的概率应该集中在蛋所在的格子。

# %%
words = [" egg", " bowl", " meat", " table", " wood", " food"]  # 候选概念（前面带空格，是"一个独立的词"）
wids = [tok.encode(w, add_special_tokens=False)[0] for w in words]  # 各取第一个子词的编号
print({w: tok.convert_ids_to_tokens(i) for w, i in zip(words, wids)})  # 打印对应的 token
layers_show = [12, 18, 24, 30]  # 要看的层
fig, axes = plt.subplots(len(words), len(layers_show), figsize=(11, 16))  # 网格
with torch.no_grad():  # 不要梯度
    for c, l in enumerate(layers_show):  # 每层一列
        p = lens(states[l][0, img_pos])  # [64, 49280]
        for r, (w, wid) in enumerate(zip(words, wids)):  # 每个词一行
            heat = p[:, wid].reshape(8, 8).cpu()  # 该词在 64 个位置的概率
            overlay(img512, heat, ax=axes[r, c], title=f"'{w.strip()}' · 第 {l} 层 · 最大 {heat.max():.1e}")  # 叠加
plt.tight_layout()  # 调整
plt.show()  # 显示

# %% [markdown]
# **怎么看**：每张图各自拉伸到 0~1，所以要结合标题里的"最大概率"看绝对强度。
#
# **观察**：第 12、18 层的最大概率只有 1e-6 量级，热力图基本是噪声。到第 30 层，概率高了几个数量级，位置也对上了：`egg` 集中在蛋上，`bowl` 落在碗边（最大概率 0.7 左右），`meat` 落在肉块上，`wood` 落在上方的木桌上。语言模型确实把"哪个格子里有什么"编码进了图像 token。
#
# > Logit Lens 是一种近似：中间层不是为了直接解码而训练的，所以浅层结果往往是噪声。小模型（135M）的概念定位也不会很干净。这正是"可解释性研究"的现实。
#
# ## 小结
# ```
# input_ids [1,79] ─查表─▶ [1,79,576] ─填入 64 个图像向量─▶ x0
# x0 ─▶ 30 × (RMSNorm → RoPE + GQA 因果注意力 → 残差 → RMSNorm → SwiGLU → 残差)
#    ─▶ final RMSNorm ─▶ lm_head ─▶ [1,79,49280] 每个位置"下一个词"的分数
# ```
# **下一章**：逐词生成完整回答，看每个词生成时在看图的哪里，并用遮挡实验验证。
