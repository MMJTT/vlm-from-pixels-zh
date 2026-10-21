# vlm-from-pixels-zh：从一张照片看懂视觉语言模型

> 🎬 **交互图解网页**：<https://mmjtt.github.io/vlm-from-pixels-zh/>——卷积和注意力的逐步动画，以及"一张照片的旅程"（可播放、拖动、点击，看每一步的变化）。续篇"一段视频的旅程"见 [omni-from-video-zh](https://mmjtt.github.io/omni-from-video-zh/)。

**中文 · 本机运行 · PyTorch 逐行注释**。在 Mac 上用 GPU（MPS）把 [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) 一步一步拆开，看一张照片如何变成模型的回答。

每一步都包含三样东西：

1. **手写实现**：每一行代码都有中文注释，构建脚本会检查这一点
2. **与官方实现逐值核对**：例如 `✅ 第 11 层输出：最大误差 0.00e+00`
3. **可视化**：这一步之后，图片变成了什么

```
照片 → 预处理 → SigLIP 视觉编码器 → 连接器 → 语言模型 → 逐词生成 → 微调
 02       02          03              04        05          06      07–08
```

## 章节

所有 Notebook 都已执行，并保存了输出和图片。点击可以直接在 GitHub 上阅读；如果 GitHub 渲染太慢，可以用 nbviewer 打开。

| 章 | 内容 | 本机耗时 |
|---|---|---|
| [00 PyTorch 入门](notebooks/00_pytorch_basics.ipynb) | 张量、用 reshape/permute 切照片、广播、MPS、自动求导、`nn.Module`、forward hook | ~5 秒 |
| [01 CNN 前置](notebooks/01_cnn_basics.ipynb) | 手工卷积核、手写卷积（unfold）、在 MPS 上训练小 CNN、ResNet-18 逐层特征图与 CAM | ~15 秒 |
| [02 预处理](notebooks/02_preprocessing.ipynb) | 缩放、切成 512 小图、归一化、`<image>` 占位符展开 | ~5 秒 |
| [03 视觉编码器](notebooks/03_vision_encoder.ipynb) | SigLIP：patch 卷积、位置编码、手写注意力、12 层逐层 PCA、注意力距离、汇聚点与超大通道 | ~10 秒 |
| [04 连接器](notebooks/04_connector.ipynb) | pixel shuffle（1024→64 个 token）、线性投影（768→576 维）、图像 token 与词向量的对比 | ~5 秒 |
| [05 语言模型](notebooks/05_language_model.ipynb) | 图文拼接、手写 RMSNorm / RoPE / GQA / SwiGLU、30 层核对、Logit Lens 概念热力图 | ~10 秒 |
| [06 生成](notebooks/06_generation.ipynb) | 手写贪心解码 + KV Cache、温度采样、遮挡实验（哪块图片最重要） | ~30 秒 |
| [07 微调连接器](notebooks/07_finetune_connector.ipynb) | 冻结视觉编码器和语言模型，只训练连接器（2.8% 参数）学会识别彩色图形 | ~2 分钟 |
| [08 LoRA 与遗忘](notebooks/08_lora_and_forgetting.ipynb) | LoRA（r=8、r=1）与全量训练对比；灾难性遗忘；回放为何失败、特征保持为何有效 | ~7 分钟 |
| [09 大模型 + 你的照片](notebooks/09_bigger_model_and_your_photo.ipynb) | SmolVLM-500M vs 256M；`explore()` 一键生成任意照片的六面板仪表盘 | ~1.5 分钟 |

耗时是在 Apple M6（24 GB）上测得的。

## 跑出来的几个发现

以下不是事先的假设，而是 Notebook 里实际算出来的结果：

- **第 01 章**：不加 BatchNorm 训练小 CNN，第一层 16 个通道里有 15 个"死掉"；加上 BatchNorm 后 16 个都在工作，第 2 轮准确率就达到 99.9%。
- **第 03 章**：SigLIP 从第 5 层输出开始出现 9 个"离群 token"，它们位于桌面、餐垫等背景处，长度约为中位数的 20 倍。后面各层 22%～47% 的注意力都流向它们。最后一层有 3 个通道的标准差是其余通道中位数的 100 多倍。
- **第 04 章**：图像 token 的长度中位数约 130，普通词向量约 3。离每个图像 token 最近的"词"都是乱码，余弦相似度不到 0.21。
- **第 05 章**：用 Logit Lens 看，图像 token 直到最后几层才能被读成词：肉块位置出现 `steak`，碗边出现 `bowl`（概率约 0.7），蛋的位置 `egg` 最亮。
- **第 06 章**：回答最依赖蛋所在的格子。`' po'`（决定说 poached）强烈依赖蛋的区域，紧接着的 `' egg'` 却几乎不依赖图片，是由语言先验"顺"出来的。
- **第 07 章**：只训练连接器约 1 分钟，"形状 + 颜色都答对"从 22% 提升到 100%。但答案被写进了背景 token，而不是图形所在的格子。
- **第 08 章**：秩为 1 的 LoRA（1.3 万参数）也能学到 100%，但它并不比全量训练遗忘得少。24 条回放数据被背了下来（NLL 1.25→0.01），留出数据上反而更差。"特征保持"正则让新任务达到 100% 的同时，照片描述和原模型一字不差。
- **第 09 章**：500M 多出的参数几乎都在语言模型里，它能认出"日式料理"，但在图形任务上只说形状不说颜色，两者都对的只有 8%。

## 快速开始

需要 Python 3.10+，最好是 Apple Silicon Mac。没有 MPS 时会自动改用 CUDA 或 CPU，只是更慢。

```bash
git clone https://github.com/MMJTT/vlm-from-pixels-zh.git
cd vlm-from-pixels-zh

# 1. 创建虚拟环境并安装依赖（也可以用 python -m venv + pip）
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

# 2. 注册 Jupyter 内核（Notebook 里选择「SmolVLM 学习 (.venv)」）
.venv/bin/python -m ipykernel install --user --name smolvlm-mac-lab --display-name "SmolVLM 学习 (.venv)"

# 3. 下载模型：256M 约 513 MB；加 --all 会再下载第 09 章用的 500M（约 1 GB）
#    国内网络可以先执行 export HF_ENDPOINT=https://hf-mirror.com
.venv/bin/python tools/download_models.py --all

# 4. 启动 JupyterLab，在浏览器打开 http://127.0.0.1:8888/lab
.venv/bin/jupyter lab --ServerApp.ip=127.0.0.1 --ServerApp.port=8888
```

打开 `notebooks/` 下的任意一章，从第一格开始按 `Shift+Enter`。建议按 00 → 09 的顺序学习。

> `results/html/` 下的 HTML 是执行结果的"录像"，下载后用浏览器打开即可阅读，但不能运行代码。

## 换成自己的照片

- **最快**：打开第 09 章，把 `MY_PHOTO` 改成你的照片路径，运行那一格，就能得到一张六面板仪表盘：输入 → SigLIP 各层 → 连接器 → Logit Lens → 遮挡敏感度 → 回答。
- **每章都用**：把照片放到 `assets/photo.jpeg` 覆盖原文件，然后重新运行。第 03 章开头的 `SPOTS` 定义了 6 个观察点（32×32 网格上的行、列），换照片后建议改成你感兴趣的位置。

## 目录结构

```
├── notebooks/       学习入口（由 src/ 生成，已执行并保存输出）
├── src/             Notebook 源码（jupytext percent 格式，方便 diff 和修改）
├── vlmlab/          共享工具
│   ├── common.py      设备选择、模型加载、输入构造、数值核对
│   ├── viz.py         PCA 着色、热力图叠加、卷积核与特征图显示
│   ├── hooks.py       ActivationCache：用 forward hook 取出任意层的输出
│   ├── finetune.py    合成数据集、训练循环、评估、LoRA（第 07～09 章）
│   └── explore.py     一键探索仪表盘（第 09 章）
├── tools/
│   ├── build_notebooks.py  src → notebooks：检查注释、执行、导出 HTML
│   ├── smoke.py            不经过 Jupyter，直接把一章当脚本跑，用于快速排错
│   └── download_models.py  下载模型权重
├── assets/photo.jpeg   示例照片（已去除 EXIF/GPS 等元数据）
├── results/html/       执行后的 HTML 阅读版
├── docs/               交互图解网页（GitHub Pages），数据由 tools/export_web_data.py 导出
└── models/             模型权重（不在仓库里，用 download_models.py 下载）
```

修改 `src/*.py` 后重新生成 Notebook：

```bash
.venv/bin/python tools/build_notebooks.py --execute      # 全部章节（约 12 分钟）
.venv/bin/python tools/build_notebooks.py 03 --execute   # 只重建第 03 章
```

## 环境与踩坑记录

- 开发环境：Apple M6（24 GB 统一内存）、macOS 27、Python 3.12、PyTorch 2.14、transformers 5.20。完整版本见 `requirements.txt`。
- 模型以 float32 + eager 注意力加载：数值最稳定，并且能取出注意力矩阵。
- **MPS 反向传播卡死**：`nn.Linear` 在 MPS 上做反向传播时，如果输入维度很小（如 3）且行数很多（≥4096）会卡住，底层是 `g.T @ x` 这种转置矩阵乘法。推理不受影响。第 00 章用等价的 1×1 卷积绕开了它。
- **exFAT 移动硬盘**：macOS 会在 exFAT 上生成 `._*` 隐藏文件，`uv pip install` 可能因此失败，加上 `UV_LINK_MODE=copy` 即可。
- **新版 transformers**：RoPE 的 θ 存放在 `config.rope_parameters["rope_theta"]`，不再是 `config.rope_theta`。

## 许可证

© 2026 MMJTT。本教程的内容（文字、代码、Notebook、示例照片）采用 [CC BY-NC 4.0](LICENSE) 许可：可以自由分享和修改，但需要署名，并且**不得用于商业目的**。

本仓库不包含任何模型权重。使用到的模型遵循各自的许可证：

- [SmolVLM-256M/500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)：Apache-2.0，Hugging Face
- [ResNet-18](https://pytorch.org/vision/stable/models/resnet.html) ImageNet 权重：torchvision，BSD-3-Clause

## 致谢

- Hugging Face 的 [SmolVLM](https://huggingface.co/blog/smolvlm) 与 [nanoVLM](https://github.com/huggingface/nanoVLM)
- 可视化思路参考了 [CNN Explainer](https://github.com/poloclub/cnn-explainer)、[llava-interp](https://github.com/clemneo/llava-interp)（Logit Lens）和 *Vision Transformers Need Registers*（2023）
