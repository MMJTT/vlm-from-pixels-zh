"""各章共用的路径、设备、绘图设置和模型加载函数。"""
from pathlib import Path  # 面向对象的路径操作
import random  # Python 自带的随机数
import logging  # 日志
import warnings  # 警告控制

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)  # 屏蔽 macOS 上无关紧要的 torch 分布式提示

import numpy as np  # 数组库，matplotlib 和 PIL 都用它交换数据
import torch  # PyTorch 主库
import matplotlib  # 绘图库
import matplotlib.pyplot as plt  # 绘图的常用接口
from PIL import Image  # 读写图片

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent.parent  # 项目根目录：vlmlab 的上一级
ASSETS = ROOT / "assets"  # 存放输入图片
RESULTS = ROOT / "results"  # 存放各章导出的中间结果
MODEL_DIR = ROOT / "models" / "SmolVLM-256M-Instruct"  # 本地模型权重目录
DEFAULT_PHOTO = ASSETS / "photo.jpeg"  # 默认使用的照片


def get_device() -> torch.device:
    """优先使用 Apple GPU（MPS），其次 NVIDIA GPU（CUDA），都没有时退回 CPU。"""
    if torch.backends.mps.is_available():  # M 系列芯片 + 新版 macOS 才可用
        return torch.device("mps")  # Metal Performance Shaders，即 Mac 的 GPU 后端
    if torch.cuda.is_available():  # Linux / Windows 上的 NVIDIA 显卡
        return torch.device("cuda")  # CUDA 后端
    return torch.device("cpu")  # 兜底方案


def seed_everything(seed: int = 0) -> None:
    """固定所有随机种子，保证每次运行结果一样。"""
    random.seed(seed)  # Python 自带随机数
    np.random.seed(seed)  # NumPy 随机数
    torch.manual_seed(seed)  # PyTorch 随机数（CPU 和 MPS 都受影响）


def setup_matplotlib() -> None:
    """让 matplotlib 能显示中文，并调高清晰度。"""
    matplotlib.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]  # macOS 自带中文字体，冬青黑体有粗体
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 不显示"找不到某种字重"的提示
    warnings.filterwarnings("ignore", message=".*[Tt]ight layout.*")  # 子图很小时 tight_layout 的提示，不影响结果
    matplotlib.rcParams["axes.unicode_minus"] = False  # 负号用普通减号，避免显示成方块
    matplotlib.rcParams["mathtext.fontset"] = "dejavusans"  # 坐标轴上的科学计数法（如 10⁻¹）用 DejaVu 字体渲染，它有数学减号
    matplotlib.rcParams["figure.dpi"] = 110  # 图片清晰度
    matplotlib.rcParams["savefig.bbox"] = "tight"  # 保存时裁掉空白边


def load_photo(path=None) -> Image.Image:
    """读入照片并统一转成 RGB 三通道。"""
    path = Path(path) if path else DEFAULT_PHOTO  # 没给路径就用默认照片
    return Image.open(path).convert("RGB")  # 有些图片是 RGBA 或灰度，统一成 RGB


def load_smolvlm(device=None, attn: str = "eager", dtype=torch.float32, model_dir=None):
    """加载 SmolVLM-256M 的处理器和模型。

    attn="eager" 表示用最朴素的注意力实现，这样才能取出注意力矩阵做可视化；
    dtype 用 float32，数值最稳定，便于和手写实现逐值对比。
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText  # 放在函数里导入，前几章不需要 transformers

    device = device or get_device()  # 默认放到 MPS
    model_dir = model_dir or MODEL_DIR  # 默认 256M，第 09 章会传入 500M 的目录
    processor = AutoProcessor.from_pretrained(model_dir)  # 图片处理器 + 分词器
    model = AutoModelForImageTextToText.from_pretrained(  # 读取权重并构建网络
        model_dir, dtype=dtype, attn_implementation=attn  # 精度和注意力实现
    )
    model = model.to(device).eval()  # 搬到 GPU，并切换到推理模式（关闭 dropout）
    return processor, model  # 返回两件套


def build_inputs(processor, image, question: str = "Describe this image.", split: bool = False, device=None):
    """按 SmolVLM 的对话格式把"图片 + 问题"变成模型输入。

    split=False：整张图缩放成一张 512×512，只产生 64 个图像 token，便于逐步拆解；
    split=True：按官方默认做法切成多块 + 一张全局图，细节更多、token 也更多。
    """
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]  # 一轮用户消息：先图后文
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)  # 套用聊天模板，结尾加上 "Assistant:"
    inputs = processor(text=prompt, images=[image], return_tensors="pt", do_image_splitting=split)  # 文本分词 + 图片预处理
    return inputs.to(device or get_device())  # 全部搬到 GPU


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """两个张量逐元素差的最大绝对值，用来检验手写实现是否正确。"""
    return (a.detach().float().cpu() - b.detach().float().cpu()).abs().max().item()  # 统一搬到 CPU 的 float32 再比较


def check_close(name: str, mine: torch.Tensor, ref: torch.Tensor, atol: float = 1e-4) -> float:
    """打印手写结果和官方结果的差异，并在差太多时报错。"""
    diff = max_abs_diff(mine, ref)  # 计算最大误差
    ok = diff <= atol  # 是否在容差内
    print(f"{'✅' if ok else '❌'} {name}: 形状 {tuple(mine.shape)}，最大误差 {diff:.2e}（容差 {atol:.0e}）")  # 打印结论
    if not ok:  # 超出容差就中断，提醒检查代码
        raise AssertionError(f"{name} 与官方实现不一致")  # 抛出异常
    return diff  # 返回误差，便于记录
