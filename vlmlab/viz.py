"""把各种张量画成"人能看懂的图"的工具函数。

核心思路：模型中间的张量不是图片，需要先映射成图片：
- 通道数为 1 的 → 热力图
- 通道数为 3 的 → 直接当 RGB
- 通道数很多（如 768 维）的 → 用 PCA 压成 3 维再当 RGB，颜色相近 = 特征相近
"""
import math  # 数学函数

import numpy as np  # 数组
import torch  # PyTorch
import torch.nn.functional as F  # 插值等函数
import matplotlib.pyplot as plt  # 绘图
from PIL import Image  # 图片


def to_numpy_image(x) -> np.ndarray:
    """把 PIL 图片或 [3,H,W]/[H,W,3] 张量统一转成 [H,W,3] 的 0~1 浮点数组。"""
    if isinstance(x, Image.Image):  # PIL 图片
        return np.asarray(x).astype(np.float32) / 255.0  # 0~255 → 0~1
    x = x.detach().float().cpu()  # 张量：断开梯度、转 float、搬到 CPU
    if x.ndim == 3 and x.shape[0] in (1, 3):  # 通道在前 [C,H,W]
        x = x.permute(1, 2, 0)  # 换成通道在后 [H,W,C]
    return x.numpy()  # 转 NumPy


def show_images(images, titles=None, cols=None, size=3.0, cmap=None, suptitle=None):
    """一行或多行并排显示多张图。"""
    n = len(images)  # 图片数量
    cols = cols or n  # 默认一行放完
    rows = math.ceil(n / cols)  # 需要几行
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)  # 创建子图网格
    for i, ax in enumerate(axes.flat):  # 遍历每个格子
        ax.axis("off")  # 关闭坐标轴
        if i >= n:  # 多余的格子留空
            continue  # 跳过
        img = images[i]  # 第 i 张图
        arr = to_numpy_image(img) if not isinstance(img, np.ndarray) else img  # 统一成数组
        if arr.ndim == 3 and arr.shape[-1] == 1:  # 单通道的 [H,W,1]
            arr = arr[..., 0]  # 去掉最后一维
        ax.imshow(arr if arr.ndim == 2 else np.clip(arr, 0, 1), cmap=cmap if arr.ndim == 2 else None)  # 灰度用色图，彩色裁剪到 0~1
        if titles:  # 有标题就写上
            ax.set_title(titles[i], fontsize=9)  # 小号字体
    if suptitle:  # 总标题
        fig.suptitle(suptitle)  # 显示在顶部
    plt.tight_layout()  # 自动调整间距
    plt.show()  # 显示


def normalize01(x: torch.Tensor) -> torch.Tensor:
    """把任意范围的张量线性拉伸到 0~1，便于显示。"""
    x = x.float()  # 转浮点
    return (x - x.min()) / (x.max() - x.min() + 1e-8)  # 最小值变 0，最大值变 1


def standardize(features: torch.Tensor) -> torch.Tensor:
    """对 [N, D] 特征：每个通道减均值、除标准差，再把每个 token 缩放成单位长度。

    深层网络里常有少数"超大通道"（数值比其他通道大上百倍），不处理的话，
    PCA 和余弦相似度会被这几个通道主导，图像结构反而看不出来。
    """
    f = features.detach().float().cpu()  # 搬到 CPU
    f = (f - f.mean(0)) / (f.std(0) + 1e-6)  # 每个通道标准化
    return F.normalize(f, dim=-1)  # 每个 token 单位化，之后点积 = 余弦相似度


def pca_rgb(features: torch.Tensor, hw=None, basis=None, return_basis=False, raw=False):
    """把 [N, D] 的特征用 PCA 压到 3 维，并排成 [h, w, 3] 的彩色图。

    features: N 个 token，每个 D 维
    hw: 排成几行几列，默认取正方形
    basis: 可传入别处算好的 (均值, 主成分, 最小值, 最大值)，让多张图共用同一套颜色
    raw: 默认先用 standardize() 处理；raw=True 则直接对原始数值做 PCA
    """
    feats = features.detach().float().cpu() if raw else standardize(features)  # 默认先标准化
    n = feats.shape[0]  # token 数
    hw = hw or (int(math.isqrt(n)), int(math.isqrt(n)))  # 默认正方形
    if basis is None:  # 没有现成的基，就现场算
        mean = feats.mean(0, keepdim=True)  # 每一维的均值
        _, _, v = torch.pca_lowrank(feats - mean, q=3, center=False)  # PyTorch 自带的低秩 PCA，v 是 [D,3] 的主方向
        proj = (feats - mean) @ v  # 投影到 3 个主方向 → [N,3]
        lo, hi = proj.quantile(0.01, dim=0), proj.quantile(0.99, dim=0)  # 用分位数去掉极端值，颜色更均匀
        basis = (mean, v, lo, hi)  # 打包保存
    mean, v, lo, hi = basis  # 解包
    proj = (feats - mean) @ v  # 投影
    rgb = ((proj - lo) / (hi - lo + 1e-8)).clamp(0, 1)  # 缩放到 0~1
    img = rgb.reshape(hw[0], hw[1], 3).numpy()  # 排成图
    return (img, basis) if return_basis else img  # 按需返回基


def upsample_map(m: torch.Tensor, size) -> np.ndarray:
    """把 [h,w] 的小热力图双线性放大到图片尺寸 size=(H,W)。"""
    m = m.detach().float().cpu()[None, None]  # 变成 [1,1,h,w] 才能用 interpolate
    up = F.interpolate(m, size=size, mode="bilinear", align_corners=False)  # 双线性插值
    return up[0, 0].numpy()  # 去掉多余维度


def overlay(image, heat: torch.Tensor, ax=None, alpha=0.55, cmap="jet", title=None):
    """把热力图半透明叠加到原图上。"""
    base = to_numpy_image(image)  # 原图 [H,W,3]
    h = upsample_map(normalize01(heat), base.shape[:2])  # 热力图放大到原图尺寸
    ax = ax or plt.gca()  # 没给子图就用当前子图
    ax.imshow(base)  # 先画原图
    ax.imshow(h, cmap=cmap, alpha=alpha)  # 再叠热力图
    ax.axis("off")  # 去掉坐标轴
    if title:  # 标题
        ax.set_title(title, fontsize=9)  # 写上
    return ax  # 返回子图


def draw_patch_grid(image, patch: int, ax=None, color="white", lw=0.5, highlight=None, title=None):
    """在图上画出切块网格；highlight 可传入 (行, 列) 列表用红框标出某些块。"""
    arr = to_numpy_image(image)  # 转数组
    ax = ax or plt.gca()  # 子图
    ax.imshow(arr)  # 画原图
    H, W = arr.shape[:2]  # 图片高宽
    for y in range(0, H + 1, patch):  # 横线
        ax.axhline(y - 0.5, color=color, lw=lw)  # 画一条
    for x in range(0, W + 1, patch):  # 竖线
        ax.axvline(x - 0.5, color=color, lw=lw)  # 画一条
    for (r, c) in (highlight or []):  # 需要高亮的块
        ax.add_patch(plt.Rectangle((c * patch - 0.5, r * patch - 0.5), patch, patch, fill=False, ec="red", lw=2))  # 红框
    ax.axis("off")  # 去坐标轴
    if title:  # 标题
        ax.set_title(title, fontsize=9)  # 写上
    return ax  # 返回


def feature_maps(fmap: torch.Tensor, n: int = 16, cols: int = 8, title=None, size=1.5, cmap="viridis"):
    """显示卷积特征图 [C,H,W] 的前 n 个通道。"""
    fmap = fmap.detach().float().cpu()  # 搬到 CPU
    n = min(n, fmap.shape[0])  # 不超过通道数
    rows = math.ceil(n / cols)  # 行数
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)  # 子图网格
    for i, ax in enumerate(axes.flat):  # 每个格子
        ax.axis("off")  # 去坐标轴
        if i < n:  # 有内容的格子
            ax.imshow(fmap[i].numpy(), cmap=cmap)  # 画第 i 个通道
            ax.set_title(f"通道 {i}", fontsize=7)  # 标题
    if title:  # 总标题
        fig.suptitle(title)  # 写上
    plt.tight_layout()  # 调整间距
    plt.show()  # 显示


def kernels_rgb(weight: torch.Tensor, n: int = 64, cols: int = 16, title=None, size=0.7):
    """把形状为 [out, 3, k, k] 的卷积核画成小的彩色图片。"""
    w = weight.detach().float().cpu()[:n]  # 取前 n 个卷积核
    rows = math.ceil(len(w) / cols)  # 行数
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)  # 子图网格
    for i, ax in enumerate(axes.flat):  # 每个格子
        ax.axis("off")  # 去坐标轴
        if i < len(w):  # 有卷积核
            ax.imshow(normalize01(w[i]).permute(1, 2, 0).numpy())  # 每个核单独拉伸到 0~1 后当 RGB 显示
    if title:  # 标题
        fig.suptitle(title)  # 写上
    plt.tight_layout()  # 调整
    plt.show()  # 显示


def log_y(ax=None):
    """把纵轴设为对数坐标，刻度用普通数字（0.01、100）显示。

    matplotlib 默认用数学公式写 10⁻²，但中文字体里没有数学减号，会显示成方框。
    """
    from matplotlib.ticker import FuncFormatter, NullFormatter  # 刻度格式工具
    ax = ax or plt.gca()  # 没给子图就用当前子图
    ax.set_yscale("log")  # 对数坐标
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))  # 主刻度：普通数字
    ax.yaxis.set_minor_formatter(NullFormatter())  # 次刻度不写字
    return ax  # 返回子图
