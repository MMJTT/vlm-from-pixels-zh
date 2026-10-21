# %% [markdown]
# # 第 03 章　视觉编码器：SigLIP 逐层拆解
#
# SmolVLM 的"眼睛"是一个 **SigLIP ViT**（约 8600 万参数）：
#
# ```
# pixel_values [1, 3, 512, 512]
#   │ ① Patch Embedding：16×16 卷积，stride 16      → [1, 1024, 768]   1024 个 patch，每个 768 维
#   │ ② + 位置编码（可学习，1024 个位置）            → [1, 1024, 768]
#   │ ③ 12 × Transformer 编码层：
#   │      LayerNorm → 多头自注意力(12 头) → 残差
#   │      LayerNorm → MLP(768→3072→768) → 残差
#   │ ④ post LayerNorm                                → [1, 1024, 768]
#   ▼ 交给第 04 章的连接器
# ```
#
# 本章每一步都**手写实现 + 与官方逐值核对 + 画出这一步图片变成了什么**。

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
from vlmlab import *  # 共享工具：load_smolvlm、build_inputs、pca_rgb、overlay、check_close……

setup_matplotlib()  # 中文字体
seed_everything(0)  # 固定随机种子
device = get_device()  # mps
processor, model = load_smolvlm(device)  # 加载处理器和模型（float32 + eager 注意力）
vm = model.model.vision_model  # 取出视觉编码器
print(f"视觉编码器参数量：{sum(p.numel() for p in vm.parameters())/1e6:.1f} M")  # 约 86M
print(f"整个 VLM 参数量  ：{sum(p.numel() for p in model.parameters())/1e6:.1f} M")  # 约 256M

# %% [markdown]
# ## 0. 准备输入
# 用第 02 章的"不切块模式"：整张照片压成一张 512×512。再挑 6 个**观察点**（patch 坐标），后面会反复看"模型眼中这些位置是什么"。

# %%
photo = load_photo()  # 原始照片
inputs = build_inputs(processor, photo, split=False)  # 预处理（见第 02 章）
pv = inputs["pixel_values"][:, 0]  # [1, 3, 512, 512]：去掉"图片数"那一维
img512 = (pv[0].cpu() * 0.5 + 0.5).clamp(0, 1)  # 反归一化，用于画图
P, G = 16, 32  # patch 边长 16 像素；每边 32 个 patch
SPOTS = {"蛋": (16, 16), "牛肉": (18, 10), "配料": (22, 15), "木桌": (4, 5), "金勺": (14, 26), "餐垫": (28, 3)}  # 观察点：(行, 列)
idx = {k: r * G + c for k, (r, c) in SPOTS.items()}  # 二维坐标 → 一维序号（行优先）
fig, ax = plt.subplots(figsize=(5, 5))  # 画布
draw_patch_grid(img512, P, ax=ax, lw=0.2, highlight=list(SPOTS.values()), title="512×512 输入，32×32 个 patch，红框是观察点")  # 画网格
for k, (r, c) in SPOTS.items():  # 标注名字
    ax.text(c * P + 18, r * P + 10, k, color="red", fontsize=9, weight="bold")  # 写字
plt.show()  # 显示

# %% [markdown]
# ## 1. Patch Embedding：唯一的卷积
# `Conv2d(3, 768, kernel_size=16, stride=16)`：卷积核和步长都是 16，窗口互不重叠。每个 16×16×3 的小块被**投影成一个 768 维向量**。
#
# 第 01 章已经证明：这等价于"切块 + 矩阵乘法"。这里再用真实权重验证一次。

# %%
emb = vm.embeddings  # 嵌入模块
conv = emb.patch_embedding  # 卷积层
print(conv)  # Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
with torch.no_grad():  # 推理不需要梯度
    official_patch = conv(pv)  # 官方：[1, 768, 32, 32]
    cols = F.unfold(pv, kernel_size=P, stride=P)  # 切块并拉直：[1, 768(=3*16*16), 1024]
    mine_patch = conv.weight.reshape(768, -1) @ cols + conv.bias[:, None]  # 手写：[768, 768] @ [1, 768, 1024] + 偏置 → [1, 768, 1024]
check_close("Patch Embedding = 切块 + 矩阵乘法", mine_patch, official_patch.flatten(2), atol=1e-4)  # 对比
tokens0 = official_patch.flatten(2).transpose(1, 2)  # [1, 768, 32, 32] → [1, 1024, 768]：每个 patch 一行
print("变成 token 序列:", tuple(tokens0.shape))  # 打印

# %% [markdown]
# ### 卷积核长什么样？
# 768 个 16×16 的彩色卷积核，每个都在问一个问题："这一块像不像我？" 有的是颜色检测器，有的是条纹 / 边缘检测器，和第 01 章 CNN 第一层学到的很像。

# %%
kernels_rgb(conv.weight, n=96, cols=24, title="SigLIP patch embedding 的前 96 个卷积核（共 768 个）", size=0.55)  # 显示卷积核

# %% [markdown]
# ### 图片变成了什么？
# 每个 patch 现在是 768 维向量。三种看法：
# 1. **PCA 着色**：768 维压到 3 维当颜色，颜色相近 = 向量相近（第 01 章用过）
# 2. **向量长度（范数）**：信息量 / 激活强度
# 3. **单个通道**：某一个卷积核的响应

# %%
t0 = tokens0[0].cpu()  # [1024, 768]
fig, axes = plt.subplots(1, 5, figsize=(14, 3))  # 5 个子图
axes[0].imshow(img512.permute(1, 2, 0))  # 原图
axes[0].set_title("输入")  # 标题
axes[1].imshow(pca_rgb(t0))  # PCA 着色
axes[1].set_title("patch 向量 PCA → RGB")  # 标题
axes[2].imshow(t0.norm(dim=-1).reshape(G, G), cmap="magma")  # 范数
axes[2].set_title("向量长度 ‖x‖")  # 标题
for ax, ch in zip(axes[3:], [0, 1]):  # 两个通道
    ax.imshow(t0[:, ch].reshape(G, G), cmap="RdBu")  # 单通道
    ax.set_title(f"第 {ch} 维")  # 标题
for ax in axes:  # 去坐标轴
    ax.axis("off")  # 关闭
plt.show()  # 显示

# %% [markdown]
# **观察**：此时 PCA 图基本还是原图的"马赛克版"。每个 patch 只知道自己 16×16 范围内的颜色和纹理，**还不知道邻居是谁**。

# %% [markdown]
# ## 2. 位置编码：告诉每个 patch "你在哪"
# 注意力本身不区分顺序：把 1024 个 patch 打乱，结果也只是跟着打乱。所以要给每个位置加一个**可学习的 768 维向量**。
#
# `position_embedding` 是 `nn.Embedding(1024, 768)`，即一张 1024 行的查找表。不切块模式下第 i 个 patch 用第 i 行（下面会验证）。

# %%
pos_table = emb.position_embedding.weight.detach().cpu()  # [1024, 768]
print("位置编码表:", tuple(pos_table.shape))  # 打印
with torch.no_grad():  # 不要梯度
    official_emb = emb(pv, patch_attention_mask=torch.ones(1, G, G, dtype=torch.bool, device=device))  # 官方嵌入输出（含位置编码）
mine_emb = tokens0 + pos_table.to(device)[None]  # 手写：patch 向量 + 第 0~1023 行位置向量
check_close("Patch Embedding + 位置编码", mine_emb, official_emb, atol=1e-5)  # 对比

# %% [markdown]
# ### 位置编码学到了二维结构
# 这些向量是训练出来的，没有人告诉模型"第 32 号在第 0 号正下方"。但计算某个位置和所有位置的**余弦相似度**，会发现它自己学会了二维布局：**同一行、同一列**的位置更像（十字形），离得越近越像。

# %%
pos_n = F.normalize(pos_table, dim=-1)  # 每行归一化成单位向量，点积就是余弦相似度
fig, axes = plt.subplots(1, 5, figsize=(14, 3))  # 5 个子图
for ax, (r, c) in zip(axes, [(0, 0), (0, 31), (16, 16), (31, 0), (8, 24)]):  # 几个位置
    sim = (pos_n @ pos_n[r * G + c]).reshape(G, G)  # 与所有位置的相似度，排成 32×32
    ax.imshow(sim, cmap="viridis")  # 画图
    ax.scatter([c], [r], c="red", s=30, marker="x")  # 标出参考位置
    ax.set_title(f"位置 ({r},{c}) 与其他位置的相似度", fontsize=8)  # 标题
    ax.axis("off")  # 去坐标轴
plt.show()  # 显示

# %% [markdown]
# ## 3. 手写第 0 层 Transformer 编码层
#
# 一层编码层 = 两个"子层"，各自带残差连接：
#
# ```
# h = x + Attention(LayerNorm1(x))     ← patch 之间交换信息
# y = h + MLP(LayerNorm2(h))           ← 每个 patch 自己"思考"
# ```
#
# 逐步拆开，每一步都和官方对比。

# %%
L0 = vm.encoder.layers[0]  # 第 0 层
print(L0)  # 打印结构
x = official_emb  # 输入 [1, 1024, 768]
with torch.no_grad():  # 不要梯度
    ref_out = L0(x, attention_mask=None)  # 官方第 0 层的输出
    ref_out = ref_out[0] if isinstance(ref_out, tuple) else ref_out  # 有的版本返回元组

# %% [markdown]
# ### 3.1 LayerNorm：每个 token 各自标准化
# 对每个 768 维向量：减均值、除标准差，再乘可学习的缩放 γ、加平移 β。
# $$\mathrm{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

# %%
def my_layernorm(x, ln):  # 手写 LayerNorm
    mu = x.mean(-1, keepdim=True)  # 每个 token 的均值 [1,1024,1]
    var = x.var(-1, keepdim=True, unbiased=False)  # 每个 token 的方差（除以 N 而不是 N-1）
    return ln.weight * (x - mu) / torch.sqrt(var + ln.eps) + ln.bias  # 标准化后缩放、平移

with torch.no_grad():  # 不要梯度
    a = my_layernorm(x, L0.layer_norm1)  # 手写
    check_close("LayerNorm1", a, L0.layer_norm1(x), atol=1e-4)  # 对比

# %% [markdown]
# ### 3.2 Q、K、V：每个 patch 生成三个向量
# - **Query (Q)**：我在找什么？
# - **Key (K)**：我有什么特征可以被找到？
# - **Value (V)**：如果你关注我，我给你什么信息？
#
# 768 维被拆成 **12 个头，每头 64 维**。每个头独立做注意力，可以关注不同的东西。

# %%
B, N, D = a.shape  # 1, 1024, 768
H, Dh = 12, D // 12  # 12 头，每头 64 维
attn = L0.self_attn  # 注意力模块
with torch.no_grad():  # 不要梯度
    q = attn.q_proj(a).view(B, N, H, Dh).transpose(1, 2)  # [1,1024,768] → [1,1024,12,64] → [1,12,1024,64]
    k = attn.k_proj(a).view(B, N, H, Dh).transpose(1, 2)  # 同上
    v = attn.v_proj(a).view(B, N, H, Dh).transpose(1, 2)  # 同上
print("Q / K / V 形状:", tuple(q.shape), "= [batch, 头, token, 每头维度]")  # 打印

# %% [markdown]
# ### 3.3 注意力权重：谁看谁
# $$\mathrm{Attention} = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{64}}\right)$$
# 结果是每个头一个 **1024×1024 的矩阵**：第 i 行表示 patch i 把注意力怎么分配给 1024 个 patch，每行加起来等于 1。

# %%
with torch.no_grad():  # 不要梯度
    scores = q @ k.transpose(-1, -2) / math.sqrt(Dh)  # [1,12,1024,64] @ [1,12,64,1024] → [1,12,1024,1024]
    att = scores.softmax(dim=-1)  # 每行做 softmax，变成概率分布
print("注意力矩阵:", tuple(att.shape), " 每行之和:", att[0, 0, 0].sum().item())  # 每行和为 1
egg = idx["蛋"]  # "蛋"这个 patch 的序号
fig, axes = plt.subplots(2, 6, figsize=(14, 5))  # 12 个头
for h, ax in enumerate(axes.flat):  # 每个头
    overlay(img512, att[0, h, egg].reshape(G, G).cpu(), ax=ax, title=f"头 {h}")  # 把"蛋"这一行的注意力叠到图上
fig.suptitle("第 0 层：'蛋' 这个 patch 在 12 个头里分别看向哪里")  # 总标题
plt.show()  # 显示

# %% [markdown]
# **观察**：第 0 层多数头只看**自己附近**的 patch（局部），有的头看相似颜色的区域。这很像 CNN 的浅层，区别在于每个头都**有能力**一步看到全图。

# %% [markdown]
# ### 3.4 加权求和 + 输出投影 + 残差

# %%
with torch.no_grad():  # 不要梯度
    o = att @ v  # [1,12,1024,1024] @ [1,12,1024,64] → [1,12,1024,64]：按注意力权重混合 V
    o = o.transpose(1, 2).reshape(B, N, D)  # 12 个头拼回 768 维：[1,1024,768]
    o = attn.out_proj(o)  # 输出投影：768 → 768
    official_attn_out = attn(L0.layer_norm1(x))[0]  # 官方注意力模块的输出
    check_close("自注意力输出", o, official_attn_out, atol=1e-4)  # 对比
    h1 = x + o  # 残差连接：原输入 + 注意力的"补充信息"
print(f"残差前 ‖x‖ 均值 {x.norm(dim=-1).mean():.2f}，注意力输出 ‖o‖ 均值 {o.norm(dim=-1).mean():.2f}")  # 注意力只是在原来基础上做"修改"

# %% [markdown]
# ### 3.5 MLP：每个 token 独立"思考"
# `768 → 3072 → 768`，中间用 GELU 激活。注意力负责 token **之间**交换信息，MLP 对每个 token **单独**加工。

# %%
z = torch.linspace(-4, 4, 200)  # 画激活函数
plt.figure(figsize=(4, 2.4))  # 画布
plt.plot(z, F.gelu(z, approximate="tanh"), label="GELU (tanh 近似)")  # SigLIP 用的
plt.plot(z, F.relu(z), "--", label="ReLU（第 01 章 CNN 用的）")  # 对比
plt.legend()  # 图例
plt.grid(alpha=0.3)  # 网格
plt.title("激活函数")  # 标题
plt.show()  # 显示
with torch.no_grad():  # 不要梯度
    m = L0.mlp.fc1(my_layernorm(h1, L0.layer_norm2))  # LayerNorm2 → 升维到 3072
    m = F.gelu(m, approximate="tanh")  # 激活
    m = L0.mlp.fc2(m)  # 降回 768
    y = h1 + m  # 第二个残差
check_close("第 0 层完整输出", y, ref_out, atol=1e-4)  # 和官方整层对比

# %% [markdown]
# ### 第 0 层前后，图片变成了什么？

# %%
show_images([img512, pca_rgb(x[0]), pca_rgb(h1[0]), pca_rgb(y[0])],  # 四个阶段
            titles=["输入", "嵌入（层前）", "+注意力", "+MLP（第 0 层输出）"], size=3)  # 标题

# %% [markdown]
# ## 4. 封装成自己的 PyTorch 模块，跑完 12 层
# 把上面的步骤写成 `nn.Module`，**直接借用官方的权重**。这样能完全掌控前向过程，顺便把每层的注意力和中间结果都带出来。

# %%
class StudyVisionLayer(nn.Module):  # 一层编码层
    def __init__(self, layer):  # 传入官方的层，借用它的权重
        super().__init__()  # 父类初始化
        self.ln1, self.ln2 = layer.layer_norm1, layer.layer_norm2  # 两个 LayerNorm
        self.q, self.k, self.v, self.o = layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.out_proj  # 四个投影
        self.fc1, self.fc2 = layer.mlp.fc1, layer.mlp.fc2  # MLP 两个线性层

    def forward(self, x, heads=12):  # 前向
        B, N, D = x.shape  # 形状
        a = self.ln1(x)  # 预归一化
        q, k, v = (f(a).view(B, N, heads, -1).transpose(1, 2) for f in (self.q, self.k, self.v))  # 生成并拆头
        att = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)  # 注意力权重 [B,12,N,N]
        x = x + self.o((att @ v).transpose(1, 2).reshape(B, N, D))  # 注意力子层 + 残差
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))  # MLP 子层 + 残差
        return x, att  # 同时返回注意力

class StudySigLIP(nn.Module):  # 完整视觉编码器
    def __init__(self, vm):  # 传入官方视觉模型
        super().__init__()  # 父类初始化
        self.patch = vm.embeddings.patch_embedding  # 卷积
        self.pos = vm.embeddings.position_embedding  # 位置编码
        self.layers = nn.ModuleList(StudyVisionLayer(l) for l in vm.encoder.layers)  # 12 层
        self.post_ln = vm.post_layernorm  # 最后的 LayerNorm

    def forward(self, pv, on_layer=None):  # on_layer：每层结束时的回调，用来就地统计注意力，避免保存 12 个巨大矩阵
        x = self.patch(pv).flatten(2).transpose(1, 2)  # [B,1024,768]
        x = x + self.pos.weight[None]  # 加位置编码
        states = [x]  # 记录每层输出
        for i, layer in enumerate(self.layers):  # 逐层
            x, att = layer(x)  # 前向
            states.append(x)  # 记录
            if on_layer:  # 有回调就调用
                on_layer(i, att)  # 交给回调处理
        return self.post_ln(x), states  # 最终输出 + 每层状态

study = StudySigLIP(vm).eval()  # 实例化（权重与官方共享，不额外占内存）
with torch.no_grad():  # 不要梯度
    mine_out, states = study(pv)  # 手写版
    ref = vm(pixel_values=pv, output_hidden_states=True)  # 官方版
for i in [0, 6, 11]:  # 抽查几层
    check_close(f"第 {i} 层输出", states[i + 1], ref.hidden_states[i + 1], atol=1e-3)  # 对比
check_close("视觉编码器最终输出", mine_out, ref.last_hidden_state, atol=1e-3)  # 对比最终输出

# %% [markdown]
# ## 5. 逐层看：图片在 12 层里变成了什么
# ### 5.1 PCA 着色
# 每层独立做 PCA，颜色只在**同一张图内部**有意义：颜色相近的 patch，模型认为它们相近。
#
# 深层有少数"超大通道"（见 5.4），会主导 PCA。所以 `pca_rgb` 默认先对每个通道做标准化（见 `vlmlab/viz.py` 的 `standardize`），再做 PCA。下面第二行用原始数值做对比。

# %%
show = [0, 2, 4, 6, 8, 10, 12]  # 要看的状态：0=嵌入层，i=第 i-1 层的输出
fig, axes = plt.subplots(2, len(show) + 1, figsize=(16, 4.4))  # 两行
for row, raw in enumerate([False, True]):  # 第一行标准化，第二行原始
    axes[row, 0].imshow(img512.permute(1, 2, 0))  # 原图
    axes[row, 0].set_title("输入" if row == 0 else "输入（原始数值 PCA）", fontsize=9)  # 标题
    for ax, i in zip(axes[row, 1:], show):  # 逐层
        ax.imshow(pca_rgb(states[i][0], raw=raw))  # PCA 着色
        ax.set_title("嵌入层" if i == 0 else f"第 {i-1} 层", fontsize=9)  # 标题
for ax in axes.flat:  # 去坐标轴
    ax.axis("off")  # 关闭
plt.tight_layout()  # 调整
plt.show()  # 显示

# %% [markdown]
# **观察**（第一行）：
# - 嵌入层：PCA 图接近"马赛克原图"，颜色由像素颜色决定
# - 中间层：同一物体（碗、桌面、餐垫）逐渐变成**一整片颜色**，碗的轮廓非常清楚
# - 深层：出现一些和图像内容无关的"杂点"，这和 5.4 节的超大激活有关
#
# 第二行用原始数值：深层的图被少数几个超大通道主导，看起来像噪声。**可视化方法本身会影响你看到什么**，这是做可解释性时要时刻注意的。

# %% [markdown]
# ### 5.2 "模型觉得哪里和蛋相似？"
# 取"蛋"这个 patch 的向量，计算它和所有 patch 的余弦相似度（同样先标准化）。这是看懂特征最直观的办法。

# %%
spots = ["蛋", "牛肉", "木桌", "金勺"]  # 四个观察点
show_layers = [0, 3, 6, 9, 12]  # 要看的状态（0=嵌入层）
fig, axes = plt.subplots(len(spots), len(show_layers), figsize=(12, 10))  # 网格
for c, li in enumerate(show_layers):  # 每层一列
    f = standardize(states[li][0])  # 标准化 + 单位化
    for r, name in enumerate(spots):  # 每个观察点一行
        sim = (f @ f[idx[name]]).reshape(G, G)  # 与观察点的余弦相似度
        overlay(img512, sim.clamp(min=0), ax=axes[r, c], title=f"'{name}' · {'嵌入层' if li == 0 else f'第 {li-1} 层'}")  # 叠加显示
plt.tight_layout()  # 调整
plt.show()  # 显示

# %% [markdown]
# **观察**：嵌入层的相似度主要取决于**颜色和纹理**；往后，"木桌"逐渐点亮大片桌面，"金勺"点亮整根勺柄，"牛肉"点亮碗里其他肉块。相似性从"长得像"变成"是同一种东西"。到了第 11 层，相似度图又变得杂乱，这和 5.4 节的"超大通道"有关。

# %% [markdown]
# ### 5.3 注意力距离：看多远？
# **平均注意力距离**：一个头平均"看"多远（像素）。对每个头计算 $\sum_j a_{ij} \cdot \mathrm{dist}(i,j)$ 再对所有 i 求平均。距离小 = 局部（像卷积），距离大 = 全局。
#
# 同时记录"蛋"在每层（12 头平均）的注意力分布。

# %%
rc = torch.stack(torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij"), -1).reshape(-1, 2).float()  # 每个 patch 的 (行, 列)
dist = (torch.cdist(rc, rc) * P).to(device)  # 1024×1024 的距离矩阵（单位：像素）
mean_dist = torch.zeros(12, 12)  # [层, 头]
egg_att = []  # 每层"蛋"的注意力
all_rows = []  # 每层 12 头平均后、所有 query 对每个 key 的平均注意力，5.4 节要用
def collect(i, att):  # 每层结束时被调用
    mean_dist[i] = (att[0] * dist).sum(-1).mean(-1).cpu()  # 每个头的平均注意力距离
    egg_att.append(att[0, :, egg].mean(0).cpu())  # "蛋"那一行，12 头平均
    all_rows.append(att[0].mean((0, 1)).cpu())  # [1024]：每个 key 平均收到多少注意力
with torch.no_grad():  # 不要梯度
    study(pv, on_layer=collect)  # 再跑一遍，就地统计

plt.figure(figsize=(7, 3.2))  # 画布
for l in range(12):  # 每层
    plt.scatter([l] * 12, mean_dist[l], s=14, c="tab:blue", alpha=0.6)  # 12 个头各一个点
plt.plot(range(12), mean_dist.mean(1), "r-o", ms=4, label="12 头平均")  # 平均值连线
plt.xlabel("层")  # 横轴
plt.ylabel("平均注意力距离（像素）")  # 纵轴
plt.title("每个点是一个注意力头")  # 标题
plt.legend()  # 图例
plt.grid(alpha=0.3)  # 网格
plt.show()  # 显示

fig, axes = plt.subplots(2, 6, figsize=(14, 5))  # 12 层
for l, ax in enumerate(axes.flat):  # 每层
    overlay(img512, egg_att[l].reshape(G, G), ax=ax, title=f"第 {l} 层")  # 叠加
fig.suptitle("'蛋' 这个 patch 在每一层看向哪里（12 头平均）")  # 总标题
plt.show()  # 显示

# %% [markdown]
# **观察**：
# - 第 1～3 层：不少头的注意力距离很小（几十像素），只看邻居，像 CNN 的浅层
# - 第 4 层以后：距离逐层变大，所有头都在看全图
# - 第 0 层是例外：它的头既有看局部的，也有看远处少数几个"特殊点"的
# - "蛋"的注意力图从第 6 层开始出现几个**远离蛋的固定亮点**，下一节解释它们
#
# > 注意：深层的"看得远"有一部分是因为大量注意力流向了这些远处的固定亮点，并不全是在"理解全局"。

# %% [markdown]
# ### 5.4 意外发现：注意力汇聚点与超大通道
# 统计每层 token 向量长度，会看到两个现象。它们在大型 ViT 中都很常见，参见论文 *Vision Transformers Need Registers*（2023）和 *Massive Activations in LLMs*（2024）：
#
# 1. **离群 token**：从第 5 层的输出开始，有 9 个 patch 的向量长度是中位数的 20 倍左右。它们多数在**背景**这种信息量低的位置，却吸走了大量注意力。模型把它们当作"草稿纸 / 垃圾桶"，用来存放全局信息，或者放置不需要的注意力。
# 2. **超大通道**：最后一层（第 11 层）的输出里，有 3 个通道的数值比其他通道大上百倍，导致**所有** token 的长度都暴涨。最后的 post-LayerNorm 会把它们拉回正常尺度。

# %%
norms = torch.stack([s[0].norm(dim=-1).cpu() for s in states])  # [13, 1024]：每个状态每个 token 的长度
med = norms.median(1).values  # 每个状态的中位数
outliers = [(norms[i] > 5 * med[i]).nonzero()[:, 0] for i in range(13)]  # 长度超过中位数 5 倍的 token
print(f"{'输出来自':>6s} {'长度中位数':>8s} {'最大值':>8s} {'离群 token 数':>10s} {'下一层分给它们的注意力':>14s}")  # 表头
for i in range(13):  # 逐个状态
    share = f"{all_rows[i][outliers[i]].sum().item():.1%}" if i < 12 and len(outliers[i]) else "-"  # 下一层（第 i 层）的注意力里，平均有多少流向这些离群 token
    print(f"{'嵌入层' if i == 0 else f'第{i-1}层':>6s} {med[i]:>10.1f} {norms[i].max():>9.1f} {len(outliers[i]):>12d} {share:>18s}")  # 打印

L8 = 9  # "第 8 层输出" 对应的状态编号
fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))  # 三个子图
axes[0].plot(med, "o-", label="中位数")  # 中位数
axes[0].plot(norms.max(1).values, "s-", label="最大值")  # 最大值
log_y(axes[0])  # 对数坐标
axes[0].set_xlabel("状态（0=嵌入层，i=第 i-1 层输出）")  # 横轴
axes[0].set_title("token 向量长度")  # 标题
axes[0].legend()  # 图例
draw_patch_grid(img512, P, ax=axes[1], lw=0, highlight=[(int(t) // G, int(t) % G) for t in outliers[L8]], title=f"第 8 层的 {len(outliers[L8])} 个离群 token 在哪")  # 在原图上标出
overlay(img512, egg_att[8].reshape(G, G), ax=axes[2], title="第 8 层 '蛋' 的注意力：亮点 = 离群 token")  # 对照注意力图
plt.show()  # 显示

ch_std = states[12][0].std(0).cpu()  # 第 11 层输出：每个通道在 1024 个 token 上的标准差
top3 = ch_std.topk(3)  # 最大的 3 个通道
plt.figure(figsize=(8, 2.4))  # 画布
plt.bar(range(768), ch_std, width=1.0)  # 768 个通道
log_y()  # 对数坐标
plt.title(f"第 11 层输出各通道的标准差：通道 {top3.indices.tolist()} 特别大")  # 标题
plt.xlabel("通道")  # 横轴
plt.show()  # 显示
print(f"最大的 3 个通道标准差 {top3.values.round().tolist()}，全部通道的中位数 {ch_std.median():.2f}")  # 打印
print(f"post-LayerNorm 之后，token 长度范围 {mine_out[0].norm(dim=-1).min():.1f} ~ {mine_out[0].norm(dim=-1).max():.1f}")  # 被拉回正常

# %% [markdown]
# ## 6. 交互探索（可选）
# 拖动滑块选择层、头和观察位置，实时查看注意力。**需要在 Jupyter 里运行**，HTML 导出版只能看到静态的初始状态。

# %%
import ipywidgets as widgets  # Jupyter 交互控件
with torch.no_grad():  # 不要梯度
    all_att = vm(pixel_values=pv, output_attentions=True).attentions  # 官方模型输出全部 12 层注意力，每层 [1,12,1024,1024]

def show_att(layer=6, head=0, row=16, col=16):  # 交互回调
    a = all_att[layer][0, head, row * G + col].reshape(G, G).cpu()  # 选中位置的注意力
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # 两个子图
    draw_patch_grid(img512, P, ax=axes[0], lw=0, highlight=[(row, col)], title=f"观察位置 ({row},{col})")  # 左：位置
    overlay(img512, a, ax=axes[1], title=f"第 {layer} 层 · 头 {head}")  # 右：注意力
    plt.show()  # 显示

widgets.interact(show_att, layer=(0, 11), head=(0, 11), row=(0, G - 1), col=(0, G - 1))  # 生成四个滑块

# %% [markdown]
# ## 小结
# | 阶段 | 形状 | 图片变成了什么 |
# |---|---|---|
# | pixel_values | [1, 3, 512, 512] | 像素 |
# | Patch Embedding | [1, 1024, 768] | 每个 16×16 小块 → 一个向量，只知道自己的颜色和纹理 |
# | + 位置编码 | [1, 1024, 768] | 每个向量多了"我在哪"的信息 |
# | 第 0～3 层 | [1, 1024, 768] | 注意力主要看局部，像 CNN 浅层 |
# | 第 4～11 层 | [1, 1024, 768] | 看得越来越远，同一物体的 patch 变得相似（语义） |
# | post-LN | [1, 1024, 768] | 统一尺度，交给连接器 |
#
# 注意：输出仍然是 **1024 个 768 维向量**，和图片的 32×32 网格一一对应。**下一章**：连接器把它们压缩成 64 个 token，并"翻译"到语言模型的空间里。
