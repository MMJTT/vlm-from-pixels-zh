# %% [markdown]
# # 第 02 章　预处理：一张照片如何变成模型的输入
#
# 模型不能直接吃 JPG 文件。SmolVLM 的**处理器（processor）**做两件事：
#
# | 图片这一侧 | 文字这一侧 |
# |---|---|
# | 缩放 → 切成 512×512 小图 + 一张全局缩略图 → 归一化 → `pixel_values` | 套聊天模板 → 插入图像占位符 `<image>` → 分词 → `input_ids` |
#
# 本章**不调用神经网络**，只拆解数据准备。每一步都手写一遍，并和官方处理器的输出逐值核对。

# %%
import sys, pathlib  # 路径工具
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "vlmlab").exists())  # 找项目根目录
sys.path.insert(0, str(ROOT))  # 加入导入路径

import math  # 数学函数
import numpy as np  # 数组
import torch  # PyTorch
import matplotlib.pyplot as plt  # 绘图
from PIL import Image  # 图片
from vlmlab import setup_matplotlib, load_photo, show_images, draw_patch_grid, check_close, MODEL_DIR  # 共享工具（先导入它，顺便屏蔽无关日志）
from transformers import AutoProcessor  # 自动加载处理器

setup_matplotlib()  # 中文字体
processor = AutoProcessor.from_pretrained(MODEL_DIR)  # 读取处理器配置（不加载模型权重，很快）
ip = processor.image_processor  # 图片处理部分
tok = processor.tokenizer  # 分词器部分
print("缩放上限 size        :", ip.size)  # 最长边先缩放到 2048
print("小图边长 max_image_size:", ip.max_image_size)  # 每块 512×512，正好是视觉编码器的输入尺寸
print("均值 / 标准差          :", ip.image_mean, ip.image_std)  # 0.5 / 0.5
print("每张小图的图像 token 数 :", processor.image_seq_len)  # 64

# %% [markdown]
# ## 1. 原始照片

# %%
photo = load_photo()  # 读入照片
W0, H0 = photo.size  # 原始宽、高
print(f"原始尺寸：宽 {W0} × 高 {H0}，共 {W0*H0/1e6:.1f} 百万像素")  # 2400 万像素，视觉编码器一次只能看 512×512
show_images([photo], titles=[f"原图 {W0}×{H0}"], size=4)  # 显示

# %% [markdown]
# ## 2. 第一次缩放：最长边 → 2048
# 保持长宽比，把最长边缩到 `size["longest_edge"] = 2048`，短边按比例缩放，并凑成偶数。重采样算法用 LANCZOS（`resample=1`）。

# %%
def resize_longest(w, h, max_len):  # 照抄处理器里的尺寸计算
    ratio = w / h  # 长宽比
    if w >= h:  # 横图
        w, h = max_len, int(max_len / ratio)  # 宽变成 max_len，高按比例
        h += h % 2  # 奇数就 +1 凑成偶数
    else:  # 竖图
        h, w = max_len, int(max_len * ratio)  # 高变成 max_len
        w += w % 2  # 凑偶数
    return w, h  # 返回新宽高

W1, H1 = resize_longest(W0, H0, ip.size["longest_edge"])  # 计算第一次缩放的尺寸
step1 = photo.resize((W1, H1), Image.LANCZOS)  # 执行缩放
print(f"第一次缩放：{W0}×{H0} → {W1}×{H1}")  # 打印

# %% [markdown]
# ## 3. 第二次缩放：凑成 512 的整数倍
# 为了能整齐切成 512×512 的块，宽高都向上取整到 512 的倍数。这张照片恰好已经是 2048×1536（4×3 块），所以尺寸不变。

# %%
def round_to_tiles(w, h, tile):  # 照抄 resize_for_vision_encoder 的尺寸计算
    ratio = w / h  # 长宽比
    if w >= h:  # 横图
        w = math.ceil(w / tile) * tile  # 宽向上取整到 tile 的倍数
        h = math.ceil(int(w / ratio) / tile) * tile  # 高按比例后也向上取整
    else:  # 竖图
        h = math.ceil(h / tile) * tile  # 高取整
        w = math.ceil(int(h * ratio) / tile) * tile  # 宽取整
    return w, h  # 返回

TILE = ip.max_image_size["longest_edge"]  # 512
W2, H2 = round_to_tiles(W1, H1, TILE)  # 计算
step2 = step1.resize((W2, H2), Image.LANCZOS)  # 执行（尺寸相同时 PIL 会直接复制）
rows, cols = H2 // TILE, W2 // TILE  # 行数和列数
print(f"第二次缩放：{W1}×{H1} → {W2}×{H2}，切成 {rows} 行 × {cols} 列 = {rows*cols} 块")  # 打印

# %% [markdown]
# ## 4. 切块 + 全局缩略图
# - **12 块局部小图**：每块 512×512，保留细节
# - **1 张全局图**：整张图直接压成 512×512（会变形），提供全局布局
#
# 视觉编码器一次只处理 512×512，"切块"相当于让模型**分 13 次看这张图**。这是很多 VLM（LLaVA-NeXT、InternVL、Idefics3）处理高分辨率的通用做法。

# %%
fig, ax = plt.subplots(figsize=(6, 4.5))  # 新建画布
draw_patch_grid(step2, TILE, ax=ax, color="yellow", lw=2, title=f"{W2}×{H2}，按 512 切成 {rows}×{cols} 块")  # 画切块网格
for r in range(rows):  # 每一行
    for c in range(cols):  # 每一列
        ax.text(c * TILE + 20, r * TILE + 80, f"row_{r+1}_col_{c+1}", color="yellow", fontsize=9, weight="bold")  # 标上编号，和后面的特殊 token 对应
plt.show()  # 显示

arr = torch.from_numpy(np.asarray(step2).copy()).permute(2, 0, 1)  # [3, 1536, 2048] 的 uint8 张量
tiles = arr.unfold(1, TILE, TILE).unfold(2, TILE, TILE)  # 用 unfold 切块：[3, 3, 4, 512, 512]
tiles = tiles.permute(1, 2, 0, 3, 4).reshape(-1, 3, TILE, TILE)  # 调整顺序 → [12, 3, 512, 512]，按行优先
global_img = step2.resize((TILE, TILE), Image.LANCZOS)  # 全局图：整张压成 512×512
print("局部小图:", tuple(tiles.shape), " 全局图:", global_img.size)  # 打印
show_images([t.float() / 255 for t in tiles] + [global_img], titles=[f"{i//cols+1}-{i%cols+1}" for i in range(12)] + ["全局图（变形）"], cols=13, size=1.4)  # 显示全部 13 张

# %% [markdown]
# ## 5. 数值化：rescale + normalize
# 1. `rescale`：0~255 → 0~1（乘以 1/255）
# 2. `normalize`：$(x - 0.5) / 0.5$ → -1~1
#
# 这正是 SigLIP 训练时用的归一化方式。

# %%
frames = torch.cat([tiles, torch.from_numpy(np.asarray(global_img).copy()).permute(2, 0, 1)[None]])  # 拼上全局图 → [13, 3, 512, 512]
mine = (frames.float() / 255 - 0.5) / 0.5  # 手写：rescale + normalize
print("手写 pixel_values:", tuple(mine.shape), f"范围 [{mine.min():.2f}, {mine.max():.2f}]")  # -1~1
plt.figure(figsize=(6, 2.4))  # 画直方图
plt.hist(frames.float().flatten().numpy() / 255, bins=80, alpha=0.6, label="rescale 后 (0~1)")  # 归一化前
plt.hist(mine.flatten().numpy(), bins=80, alpha=0.6, label="normalize 后 (-1~1)")  # 归一化后
plt.legend()  # 图例
plt.title("像素值分布")  # 标题
plt.show()  # 显示

# %% [markdown]
# ## 6. 和官方处理器对比
# 允许 2/255 ≈ 0.008 的误差：官方处理器和 PIL 在 LANCZOS 插值的取整细节上略有差别，最多差 1 个灰度级。

# %%
official = ip(images=[photo], return_tensors="pt", return_row_col_info=True)  # 官方处理器
pv = official["pixel_values"]  # [batch=1, 13 张, 3, 512, 512]
print("官方 pixel_values:", tuple(pv.shape), " rows/cols:", official["rows"], official["cols"])  # 打印
check_close("12 块局部图", mine[:12], pv[0, :12], atol=2 / 255 + 1e-6)  # 对比局部图
check_close("全局图", mine[12], pv[0, 12], atol=2 / 255 + 1e-6)  # 对比全局图
print("pixel_attention_mask 全是 1 吗？", bool(official["pixel_attention_mask"].all()))  # 没有填充区域

# %% [markdown]
# ## 7. 不切块模式（后面几章用它）
# `do_image_splitting=False` 时，整张图直接压成一张 512×512，只产生 **64 个**图像 token（切块模式是 13×64 = 832 个）。
#
# 第 03～05 章逐层拆解时用这个模式：只有一张 512×512 输入，所有中间结果都能直接画在这张图上；第 06 章再比较两种模式回答质量的差别。

# %%
single = ip(images=[photo], return_tensors="pt", do_image_splitting=False)["pixel_values"]  # 不切块
print("不切块 pixel_values:", tuple(single.shape))  # [1, 1, 3, 512, 512]
mine_single = (torch.from_numpy(np.asarray(photo.resize((W1, H1), Image.LANCZOS).resize((TILE, TILE), Image.LANCZOS)).copy()).permute(2, 0, 1).float() / 255 - 0.5) / 0.5  # 手写：先缩到 2048，再压成 512×512
check_close("不切块模式", mine_single, single[0, 0], atol=2 / 255 + 1e-6)  # 对比
show_images([single[0, 0] * 0.5 + 0.5], titles=["模型实际看到的 512×512（反归一化后显示）"], size=4)  # 反归一化：x*0.5+0.5 回到 0~1

# %% [markdown]
# ## 8. 文字这一侧：聊天模板
# SmolVLM 是对话模型，输入需要包装成固定格式。`<image>` 是图片的**占位符**。

# %%
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]  # 一条用户消息：一张图 + 一句话
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)  # 套模板，结尾加 "Assistant:" 提示模型开始回答
print(repr(prompt))  # repr 能显示出换行符等特殊字符

# %% [markdown]
# ## 9. 占位符展开：一个 `<image>` 变成几百个 token
# 处理器把模板里的 1 个 `<image>` 展开成：
# ```
# <fake_token_around_image><row_1_col_1><image>×64   ← 第 1 块
# <fake_token_around_image><row_1_col_2><image>×64   ← 第 2 块
# ...（每行结束加一个换行）
# <fake_token_around_image><global-img><image>×64<fake_token_around_image>   ← 全局图
# ```
# 每个 `<image>` 是一个"空位"，第 05 章会把视觉编码器的输出**填进这些空位**。

# %%
inputs = processor(text=prompt, images=[photo], return_tensors="pt")  # 默认切块模式
ids = inputs["input_ids"][0]  # [序列长度]
image_token_id = tok.convert_tokens_to_ids("<image>")  # <image> 的编号
print("序列总长度:", len(ids), " 其中 <image>:", int((ids == image_token_id).sum()), " 文字等其他 token:", int((ids != image_token_id).sum()))  # 832 个图像 token
toks = tok.convert_ids_to_tokens(ids.tolist())  # 编号 → token 字符串
compact, i = [], 0  # 压缩显示：连续的 <image> 合并
while i < len(toks):  # 遍历
    j = i  # 找连续相同 token 的结尾
    while j < len(toks) and toks[j] == toks[i] == "<image>":  # 连续的 <image>
        j += 1  # 继续
    compact.append(f"<image>×{j-i}" if j > i else toks[i])  # 合并或原样
    i = max(j, i + 1)  # 前进
print(" ".join(compact))  # 打印压缩后的序列

# %% [markdown]
# > 读 token 的小提示：`Ġ` 表示"前面有一个空格"，`Ċ` 表示换行。这是 GPT-2 系分词器（字节级 BPE）的写法。`Assistant` 被切成了 `Ass` + `istant` 两个 token。
#
# 把整个序列画成一条色带：每一格是一个 token，颜色表示它属于哪一块图或文字。

# %%
seg = np.zeros(len(ids))  # 每个 token 的分组编号
block = 0  # 当前在第几块图
for k, t in enumerate(toks):  # 遍历 token
    if t.startswith("<row_") or t == "<global-img>":  # 遇到新图块的标签
        block += 1  # 块编号 +1
    seg[k] = block + 1 if t == "<image>" else 0  # 图像 token 标"块号+1"（跳过和文字颜色相近的浅蓝），其余标 0（文字）
plt.figure(figsize=(12, 1.2))  # 宽长的画布
plt.imshow(seg[None], aspect="auto", cmap="tab20", interpolation="nearest")  # 一行色带
plt.yticks([])  # 去掉纵轴
plt.title(f"输入序列 {len(ids)} 个 token：深蓝=文字/特殊标记，其余颜色=13 块图各 64 个 <image> 占位符")  # 标题
plt.xlabel("token 位置")  # 横轴
plt.show()  # 显示

# %% [markdown]
# ## 10. 为什么每块图恰好 64 个 token？
# ```
# 512×512 的图  ──16×16 切块──▶  32×32 = 1024 个 patch（第 03 章）
#              ──pixel shuffle，4×4 合并──▶  8×8 = 64 个 token（第 04 章）
# ```
# 下面算一算不同尺寸的照片会产生多少图像 token。token 越多，模型看得越细，但也越慢。

# %%
print(f"{'输入尺寸':>12s} {'图片数':>6s} {'图像 token':>10s} {'官方处理器':>8s}")  # 表头
for (w, h) in [(512, 512), (800, 600), (1024, 1024), (1920, 1080), (W0, H0), (1080, 1920)]:  # 几种常见尺寸
    w1, h1 = resize_longest(w, h, 2048)  # 第一次缩放
    w2, h2 = round_to_tiles(w1, h1, TILE)  # 凑整
    n_tiles = (h2 // TILE) * (w2 // TILE)  # 块数
    n_tiles = n_tiles + 1 if n_tiles > 1 else 1  # 多于 1 块时加一张全局图
    n_official = ip(images=[Image.new("RGB", (w, h))], return_tensors="pt")["pixel_values"].shape[1]  # 用一张同尺寸的空白图让官方处理器算一遍
    print(f"{w:>5d}×{h:<5d}  {n_tiles:>6d} {n_tiles*64:>10d} {n_official:>8d}")  # 手算和官方应一致

# %% [markdown]
# > 512×512 的图：第一次缩放后变成 2048×2048，会切成 16 块 + 1 张全局图，共 1088 个 token。小图被**放大**后切块，这是该处理器的固定行为。
#
# ## 小结
# - `pixel_values`：`[batch, 图片数, 3, 512, 512]`，范围 -1~1
# - `input_ids`：文字 token + 大量 `<image>` 占位符
# - 一个 `<image>` 对应 512×512 图上一个 64×64 像素的区域
#
# **下一章**：一张 512×512 的图进入 SigLIP 视觉编码器，逐层看它变成了什么。
