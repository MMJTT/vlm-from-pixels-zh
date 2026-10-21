"""下载教程需要的模型权重到 models/ 目录。

用法：
    .venv/bin/python tools/download_models.py          # 只下载 SmolVLM-256M（约 513 MB，第 02～08 章需要）
    .venv/bin/python tools/download_models.py --all    # 另外下载 SmolVLM-500M（约 1 GB，第 09 章需要）

国内网络可以先设置镜像：export HF_ENDPOINT=https://hf-mirror.com
ResNet-18（第 01 章，约 45 MB）会在第一次运行时自动下载，不需要提前准备。
"""
import argparse  # 命令行参数
from pathlib import Path  # 路径

from huggingface_hub import snapshot_download  # Hugging Face 官方下载工具

ROOT = Path(__file__).resolve().parent.parent  # 项目根目录
MODELS = {"SmolVLM-256M-Instruct": True, "SmolVLM-500M-Instruct": False}  # 模型名 → 是否默认下载

if __name__ == "__main__":
    ap = argparse.ArgumentParser()  # 参数解析器
    ap.add_argument("--all", action="store_true", help="同时下载第 09 章用的 500M 模型")  # 是否下载全部
    args = ap.parse_args()  # 解析
    for name, default in MODELS.items():  # 逐个模型
        if default or args.all:  # 需要下载
            path = snapshot_download(f"HuggingFaceTB/{name}", local_dir=ROOT / "models" / name,  # 下载到 models/模型名
                                     allow_patterns=["*.json", "model.safetensors", "*.txt", "*.jinja"])  # 只要 PyTorch 权重和配置，跳过 ONNX
            print(f"✅ {name} → {path}")  # 报告
