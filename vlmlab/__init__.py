"""SmolVLM 学习实验室的共享工具：路径、设备、可视化、钩子。"""
from .common import *  # noqa: F401,F403  路径、设备、模型加载
from .viz import *  # noqa: F401,F403  可视化函数
from .hooks import ActivationCache  # noqa: F401  前向钩子
from . import finetune  # noqa: F401  微调工具（第 07～09 章）
