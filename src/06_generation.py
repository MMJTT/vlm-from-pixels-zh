# %% [markdown]
# # 第 06 章　生成回答：逐词解码、注意力与遮挡实验
#
# 第 05 章只预测了**一个**词。完整回答是一个循环：
# ```
# 预测下一个词 → 把它接到序列末尾 → 再预测下一个词 → …… → 遇到结束符停止
# ```
#
# | 本章步骤 | 内容 |
# |---|---|
# | 1 | 官方 `generate()`：不切块 vs 切块，速度与回答 |
# | 2 | 手写贪心解码循环 + **KV Cache**，看每一步的候选词 |
# | 3 | 每个词生成时在看图的哪里 |
# | 4 | 温度与采样：为什么同一个问题每次回答可能不同 |
# | 5 | **遮挡实验**：挡住图片的某一块，回答会怎么变？哪块区域对回答最重要？ |
# | 6 | 多问几个问题 + 全教程总结 |

# %%
import sys, pathlib  # 路径工具
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "vlmlab").exists())  # 找项目根目录
sys.path.insert(0, str(ROOT))  # 加入导入路径

import time  # 计时
import numpy as np  # 数组
import torch  # PyTorch
import torch.nn.functional as F  # 函数式接口
import matplotlib.pyplot as plt  # 绘图
from vlmlab import *  # 共享工具

setup_matplotlib()  # 中文字体
seed_everything(0)  # 固定随机种子
device = get_device()  # mps
processor, model = load_smolvlm(device)  # 加载模型
tok = processor.tokenizer  # 分词器
photo = load_photo()  # 照片
end_ids = {tok.convert_tokens_to_ids("<end_of_utterance>"), tok.eos_token_id}  # 结束符

# %% [markdown]
# ## 1. 官方 `generate()`：两种图片模式对比

# %%
for split in [False, True]:  # 不切块 / 切块
    inp = build_inputs(processor, photo, "Describe this image.", split=split)  # 预处理
    n_img = int((inp["input_ids"] == model.config.image_token_id).sum())  # 图像 token 数
    t = time.time()  # 计时
    with torch.no_grad():  # 不要梯度
        out = model.generate(**inp, max_new_tokens=60, do_sample=False)  # 贪心解码 60 个词
    ans = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)  # 只解码新生成的部分
    print(f"【{'切块' if split else '不切块'}】图像 token {n_img} 个，用时 {time.time()-t:.2f} 秒\n{ans}\n")  # 打印

# %% [markdown]
# 切块模式看到 13 张 512×512 的图，细节更多，但图像 token 是 13 倍，也更慢。下面继续用**不切块**模式，方便把注意力画在一张图上。

# %% [markdown]
# ## 2. 手写贪心解码 + KV Cache
# **KV Cache**：每生成一个新词，前面所有 token 的 K、V 都不会变（因果注意力，前面看不到后面）。把它们缓存起来，新的一步只需要计算**一个 token**，不用把整个序列重算一遍。
#
# 下面手写循环：第一步输入整个序列（含图片），之后每步只输入上一步生成的那个词。

# %%
inputs = build_inputs(processor, photo, "Describe this image.", split=False)  # 预处理
ids = inputs["input_ids"]  # [1, 79]
img_pos = (ids[0] == model.config.image_token_id).nonzero()[:, 0]  # 64 个图像位置
img512 = (inputs["pixel_values"][0, 0].cpu() * 0.5 + 0.5).clamp(0, 1)  # 用于显示

def greedy(inputs, max_new=40, use_cache=True):  # 手写贪心解码
    ids = inputs["input_ids"]  # 当前序列
    mask = inputs["attention_mask"]  # 注意力掩码
    steps, past = [], None  # 记录每步信息；KV 缓存
    with torch.no_grad():  # 不要梯度
        for s in range(max_new):  # 逐词生成
            if past is None or not use_cache:  # 第一步，或者不用缓存：输入整个序列
                out = model(input_ids=ids, attention_mask=mask, pixel_values=inputs["pixel_values"],  # 含图片
                            use_cache=use_cache, output_attentions=True)  # 返回注意力
            else:  # 之后每步只输入 1 个新 token
                out = model(input_ids=ids[:, -1:], attention_mask=mask, past_key_values=past,  # 只给最后一个词 + 缓存
                            use_cache=True, output_attentions=True)  # 返回注意力
            past = out.past_key_values if use_cache else None  # 更新缓存
            probs = out.logits[0, -1].softmax(-1)  # 最后位置的概率分布
            nxt = probs.argmax()  # 贪心：取概率最大的词
            att = torch.stack([a[0, :, -1, img_pos] for a in out.attentions]).mean((0, 1))  # 当前词对 64 个图像 token 的注意力（30 层 × 9 头平均）
            steps.append(dict(token=nxt.item(), prob=probs[nxt].item(), top5=probs.topk(5), img_att=att.cpu()))  # 记录
            ids = torch.cat([ids, nxt.view(1, 1)], 1)  # 接到序列末尾
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)  # 掩码也加一位
            if nxt.item() in end_ids:  # 遇到结束符
                break  # 停止
    return steps  # 返回每步记录

t = time.time()  # 计时
steps = greedy(inputs)  # 带缓存
t_cache = time.time() - t  # 用时
t = time.time()  # 计时
steps_nc = greedy(inputs, use_cache=False)  # 不带缓存
t_nocache = time.time() - t  # 用时
print("手写解码结果:", tok.decode([s["token"] for s in steps], skip_special_tokens=True))  # 打印回答
print(f"KV Cache：{t_cache:.2f} 秒；不用缓存：{t_nocache:.2f} 秒；两者结果相同：{[s['token'] for s in steps] == [s['token'] for s in steps_nc]}")  # 对比

# %% [markdown]
# ### 每一步的候选词
# 每行是一步：模型最终选了哪个词（加粗），以及另外 4 个候选的概率。概率低的步骤说明模型"犹豫"了。

# %%
print(f"{'步':>3s}  {'选中':14s} {'概率':>6s}   其余候选")  # 表头
for i, s in enumerate(steps[:24]):  # 前 24 步
    others = "  ".join(f"{tok.decode([j])!r}:{v:.0%}" for v, j in zip(s["top5"].values[1:].tolist(), s["top5"].indices[1:].tolist()))  # 其余 4 个候选
    print(f"{i:>3d}  {tok.decode([s['token']])!r:14s} {s['prob']:>6.1%}   {others}")  # 打印
plt.figure(figsize=(12, 2.4))  # 画布
plt.bar(range(len(steps)), [s["prob"] for s in steps])  # 每步选中词的概率
plt.xticks(range(len(steps)), [tok.decode([s["token"]]) for s in steps], rotation=75, fontsize=7)  # 横轴写词
plt.title("每一步选中词的概率：低的地方就是模型拿不准的地方")  # 标题
plt.show()  # 显示

# %% [markdown]
# ## 3. 每个词生成时在看图的哪里
# 每个小图：生成这个词时，它对 64 个图像 token 的注意力（30 层 × 9 头平均），排成 8×8 叠到图上。

# %%
n = min(len(steps), 24)  # 最多显示 24 个
fig, axes = plt.subplots(3, 8, figsize=(16, 6.4))  # 3 行 8 列
for i, ax in enumerate(axes.flat):  # 每个子图
    if i < n:  # 有内容
        overlay(img512, steps[i]["img_att"].reshape(8, 8), ax=ax, title=repr(tok.decode([steps[i]["token"]])))  # 叠加
    else:  # 多余格子
        ax.axis("off")  # 留空
plt.tight_layout()  # 调整
plt.show()  # 显示

# %% [markdown]
# > 正如第 05 章提醒的：平均注意力常被少数"汇聚点"主导，看起来可能每个词都差不多。注意力**不等于**因果关系。更可靠的方法是**干预**：改变输入，看输出怎么变（第 5 节）。

# %% [markdown]
# ## 4. 温度与采样
# 贪心解码每次都选最大概率的词，结果固定。**采样**则按概率随机抽取，**温度 T** 控制随机程度：
# $$p_i = \mathrm{softmax}(z_i / T)$$
# - T < 1：分布更尖，更保守
# - T > 1：分布更平，更有创意（也更容易胡说）

# %%
with torch.no_grad():  # 不要梯度
    z = model(**inputs).logits[0, -1]  # 回答第一个词的 logits
top = z.topk(12).indices  # 取前 12 个候选
fig, axes = plt.subplots(1, 3, figsize=(13, 2.8))  # 三种温度
for ax, T in zip(axes, [0.3, 1.0, 2.0]):  # 每种温度
    p = (z / T).softmax(-1)[top].cpu()  # 加温度后的概率
    ax.bar([tok.decode([i]).strip() for i in top.tolist()], p)  # 柱状图
    ax.set_title(f"T = {T}")  # 标题
    ax.tick_params(axis="x", rotation=60, labelsize=7)  # 横轴字体
plt.show()  # 显示
for seed in range(3):  # 采样三次
    torch.manual_seed(seed)  # 每次用不同种子
    with torch.no_grad():  # 不要梯度
        out = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.9, top_p=0.95)  # 采样生成
    print(f"采样 {seed}:", tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))  # 打印

# %% [markdown]
# > 采样的结果可能会很离谱，比如一开头选了一个小概率的词，后面就越走越偏，甚至陷入重复。小模型尤其明显。这就是为什么"看图说话"类任务常用贪心解码或较低的温度。

# %% [markdown]
# ## 5. 遮挡实验：哪块图片对回答最重要？
# ### 5.1 挡住蛋，回答会变吗？
# 在 `pixel_values` 上把蛋所在的区域涂成 0（归一化后的 0 就是 50% 灰）。

# %%
pv_occ = inputs["pixel_values"].clone()  # 复制输入
pv_occ[..., 190:330, 180:330] = 0  # 蛋所在区域涂灰（坐标为 512×512 图上的像素）
show_images([img512, (pv_occ[0, 0].cpu() * 0.5 + 0.5)], titles=["原图", "挡住蛋"], size=3)  # 对比显示
for name, pv in [("原图", inputs["pixel_values"]), ("挡住蛋", pv_occ)]:  # 两种输入
    with torch.no_grad():  # 不要梯度
        out = model.generate(**{**inputs, "pixel_values": pv}, max_new_tokens=40, do_sample=False)  # 生成
    print(f"【{name}】", tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))  # 打印

# %% [markdown]
# ### 5.2 遮挡敏感度图
# 把原回答固定下来，依次挡住 8×8 网格中的每一格（64×64 像素），计算模型给**原回答**打的总对数概率下降了多少：
# - 下降越多 → 这块区域对说出这段话越重要
#
# 一共 64 次前向计算，在 MPS 上只需几秒。

# %%
answer = torch.tensor([[s["token"] for s in steps]], device=device)  # 原回答的 token
full = torch.cat([ids, answer], 1)  # 问题 + 回答
L = ids.shape[1]  # 回答开始的位置
def answer_logprob(pv):  # 给定图片，原回答每个 token 的对数概率
    with torch.no_grad():  # 不要梯度
        lp = model(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=pv).logits[0, L - 1:-1].log_softmax(-1)  # 位置 L-1 预测第 1 个回答词，依此类推
    return lp.gather(-1, answer[0][:, None])[:, 0].cpu()  # 取出原回答 token 的对数概率

base = answer_logprob(inputs["pixel_values"])  # 不遮挡时的基准
drop = torch.zeros(8, 8, len(base))  # [行, 列, 每个回答 token]
t = time.time()  # 计时
for r in range(8):  # 每行
    for c in range(8):  # 每列
        pv = inputs["pixel_values"].clone()  # 复制
        pv[..., r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = 0  # 挡住一格
        drop[r, c] = base - answer_logprob(pv)  # 每个 token 的对数概率下降量
print(f"64 次遮挡用时 {time.time()-t:.1f} 秒")  # 打印
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))  # 两个子图
overlay(img512, drop.sum(-1).clamp(min=0), ax=axes[0], title="挡住哪里，整段回答的概率下降最多")  # 总敏感度
axes[1].imshow(drop.sum(-1), cmap="RdBu_r", vmin=-drop.sum(-1).abs().max(), vmax=drop.sum(-1).abs().max())  # 带正负号的原始值
for r in range(8):  # 标数值
    for c in range(8):  # 每格
        axes[1].text(c, r, f"{drop[r, c].sum():.1f}", ha="center", va="center", fontsize=7)  # 写数
axes[1].set_title("对数概率下降量（红=重要，蓝=挡住后反而更确定）")  # 标题
axes[1].axis("off")  # 去坐标轴
plt.show()  # 显示

# %% [markdown]
# **观察**：影响最大的格子集中在**蛋和它上方的小牌子**。挡住这两格，原回答的对数概率下降十几，也就是概率变成原来的百万分之一量级。挡住大片桌面和餐垫几乎没有影响。注意，这里衡量的是"说出这段回答"依赖哪里，不是"图片哪里最显眼"。

# %% [markdown]
# ### 5.3 每个词依赖图片的哪里？
# 把敏感度按回答里的**单个词**拆开。有些词强烈依赖某块区域，有些词几乎不依赖图片：它们是由前文"推"出来的（语言先验），比如说出 "po-ached" 之后，"egg" 几乎是必然的。
#
# 下面的结果里可以看到：`' po'`（决定要说 poached）强烈依赖**蛋所在的格子**；而紧跟其后的 `' egg'` 反而出现在"几乎不依赖图片"的名单里。**"看图"发生在决定说 poached 的那一刻，而不是说出 egg 的那一刻。**

# %%
words = [tok.decode([s["token"]]) for s in steps]  # 回答的每个词
dep = drop.clamp(min=0).flatten(0, 1).sum(0)  # 每个词的总图像依赖度
order = dep.argsort(descending=True)  # 按依赖度排序
plt.figure(figsize=(12, 2.4))  # 画布
plt.bar(range(len(words)), dep)  # 柱状图
plt.xticks(range(len(words)), words, rotation=75, fontsize=7)  # 横轴写词
plt.title("回答中每个词对图片的依赖程度（遮挡后概率下降之和）")  # 标题
plt.show()  # 显示
fig, axes = plt.subplots(1, 6, figsize=(15, 2.8))  # 依赖最强的 6 个词
for ax, i in zip(axes, order[:6].tolist()):  # 遍历
    overlay(img512, drop[:, :, i].clamp(min=0), ax=ax, title=f"{words[i]!r}（第 {i} 个词）")  # 叠加
plt.show()  # 显示
print("几乎不依赖图片的词:", [words[i] for i in order[-8:].tolist()])  # 依赖度最低的词

# %% [markdown]
# ## 6. 多问几个问题
# 换几个问题试试。也可以把 `assets/photo.jpeg` 换成自己的照片，或修改下面的问题列表。

# %%
questions = [  # 问题列表
    "What is in the center of the bowl?",  # 碗中间是什么
    "What color is the table?",  # 桌子什么颜色
    "What is the spoon made of?",  # 勺子什么材质
    "Is this a breakfast or a dinner? Why?",  # 推理题
    "Is there any text in the image?",  # 文字
]
for q in questions:  # 逐个提问
    inp = build_inputs(processor, photo, q, split=True)  # 用切块模式，细节更多
    with torch.no_grad():  # 不要梯度
        out = model.generate(**inp, max_new_tokens=40, do_sample=False)  # 生成
    print(f"Q: {q}\nA: {tok.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()}\n")  # 打印

# %% [markdown]
# > 256M 是非常小的 VLM，会犯错：这张照片里是牛肉，它常说成海鲜。这正好说明：**"认识图片"是一个统计过程**，模型给出的是训练数据中最常见的搭配。
#
# ## 全教程总结：一张照片的完整旅程
#
# ```
# photo.jpeg 5712×4284
#   │ 第 02 章 预处理：缩放 → 切块/压缩 → (x/255−0.5)/0.5
#   ▼
# pixel_values [1, 3, 512, 512]
#   │ 第 03 章 SigLIP：16×16 卷积切块 → +位置编码 → 12 层自注意力（局部→全局，颜色→语义）
#   ▼
# 视觉特征 [1, 1024, 768]          ← 32×32 网格，每格 16×16 像素
#   │ 第 04 章 连接器：pixel shuffle（4×4 拼接）→ Linear 12288→576
#   ▼
# 图像 token [1, 64, 576]           ← 8×8 网格，每格 64×64 像素
#   │ 第 05 章 填入 <image> 占位符，与文字拼成 79 个 token
#   │          30 层 Llama（RMSNorm、RoPE、GQA 因果注意力、SwiGLU）
#   ▼
# logits [1, 79, 49280]             ← 下一个词的分数
#   │ 第 06 章 循环：选词 → 接到末尾 → KV Cache → 再预测
#   ▼
# "An overhead shot shows a bowl of food ... with a poached egg ..."
# ```
#
# ### 下一步可以做什么
# - 换一张自己的照片，从第 03 章开始重跑，看哪些现象依然成立
# - 换更大的模型（如 SmolVLM-500M、Qwen2.5-VL-3B）对比回答和 Logit Lens 的清晰度
# - **第 07 章**：在 Mac 上只训练连接器做一个小的微调实验，观察图像 token 和回答的变化
