"""用 PyTorch 的 forward hook（前向钩子）"偷看"任意一层的输出。

用法：
    with ActivationCache(model, ["layer1", "layer2.0.conv1"]) as cache:
        model(x)
    cache["layer1"]   # 取出该层输出
"""
import torch  # PyTorch


def _detach(x):
    """把输出里的张量都断开梯度并复制到 CPU；元组只取第一个元素（通常是主输出）。"""
    if isinstance(x, (tuple, list)):  # 很多层返回 (输出, 注意力, ...) 这样的元组
        x = x[0]  # 只保留主输出
    if hasattr(x, "last_hidden_state"):  # transformers 的输出对象
        x = x.last_hidden_state  # 取最后隐藏状态
    return x.detach().float().cpu() if torch.is_tensor(x) else x  # 张量就搬到 CPU


class ActivationCache(dict):
    """一个字典：键是层名，值是这一层的输出张量。作为 with 语句使用。"""

    def __init__(self, model: torch.nn.Module, names, keep_input: bool = False):
        super().__init__()  # 初始化字典本身
        self.model = model  # 要观察的模型
        self.names = list(names)  # 要观察的层名（model.named_modules() 里的名字）
        self.keep_input = keep_input  # 是否同时保存该层的输入
        self.handles = []  # 记录注册的钩子，用完要删除

    def __enter__(self):
        modules = dict(self.model.named_modules())  # 层名 → 层对象
        for name in self.names:  # 逐个注册
            module = modules[name]  # 找到这一层

            def hook(mod, inputs, output, name=name):  # 前向计算结束时 PyTorch 会调用它
                self[name] = _detach(output)  # 保存输出
                if self.keep_input:  # 需要的话也保存输入
                    self[name + ":in"] = _detach(inputs)  # 输入是元组，取第一个

            self.handles.append(module.register_forward_hook(hook))  # 注册并记下句柄
        return self  # with ... as cache 拿到的就是自己

    def __exit__(self, *exc):
        for h in self.handles:  # 删除全部钩子，避免影响后续计算
            h.remove()  # 移除
        self.handles.clear()  # 清空列表
        return False  # 不吞掉异常
