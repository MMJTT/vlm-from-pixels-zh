"""一键把任意图片走完第 02～06 章的全流程，画成一张"仪表盘"。

用法：
    from vlmlab.explore import explore
    explore(model, processor, "你的照片.jpg", "Describe this image.")
"""
from pathlib import Path  # 路径

import torch  # PyTorch
import matplotlib.pyplot as plt  # 绘图
from PIL import Image  # 图片

from .common import build_inputs, load_photo  # 预处理
from .viz import pca_rgb, overlay  # 可视化


@torch.no_grad()
def explore(model, processor, image, question="Describe this image.", layers=(3, 7, 11), max_new=40, show=True):
    """返回 dict（回答、各阶段张量），并画出 2×4 的仪表盘。image 可以是路径或 PIL 图片。"""
    tok = processor.tokenizer  # 分词器
    img = load_photo(image) if isinstance(image, (str, Path)) else image.convert("RGB")  # 读图
    inp = build_inputs(processor, img, question, split=False, device=model.device)  # 第 02 章：预处理（不切块）
    pv = inp["pixel_values"][:, 0]  # [1,3,512,512]
    img512 = (pv[0].cpu() * 0.5 + 0.5).clamp(0, 1)  # 用于显示
    vis = model.model.vision_model(pixel_values=pv, output_hidden_states=True)  # 第 03 章：视觉编码器
    img_tokens = model.model.connector(vis.last_hidden_state)[0]  # 第 04 章：连接器 → [64, D]
    ids = inp["input_ids"]  # 输入序列
    img_pos = (ids[0] == model.config.image_token_id).nonzero()[:, 0]  # 64 个图像位置
    logits = model(**inp).logits[0]  # 第 05 章：整条序列的 logits
    words = [tok.decode([i]).replace("�", "?").strip()[:7] or "_" for i in logits[img_pos].argmax(-1).tolist()]  # 最后一层 Logit Lens：每格最可能的词
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)  # 第 06 章：生成回答
    answer = out[:, ids.shape[1]:]  # 回答 token
    answer = answer[:, : (answer[0] != tok.convert_tokens_to_ids("<end_of_utterance>")).cumprod(0).sum().clamp(min=1)]  # 截掉结束符及之后
    full = torch.cat([ids, answer], 1)  # 问题 + 回答
    L = ids.shape[1]  # 回答起点

    def answer_logprob(pv5):  # 给定图片，原回答的总对数概率
        lp = model(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=pv5).logits[0, L - 1:-1].log_softmax(-1)  # 逐位置
        return lp.gather(-1, answer[0][:, None]).sum().item()  # 求和

    base = answer_logprob(inp["pixel_values"])  # 不遮挡的基准
    occ = torch.zeros(8, 8)  # 遮挡敏感度
    for r in range(8):  # 逐格遮挡
        for c in range(8):  # 每列
            pv5 = inp["pixel_values"].clone()  # 复制
            pv5[..., r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = 0  # 涂灰
            occ[r, c] = base - answer_logprob(pv5)  # 对数概率下降量
    text = tok.decode(answer[0], skip_special_tokens=True).strip()  # 回答文本

    if show:  # 画仪表盘
        fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))  # 2×4
        axes[0, 0].imshow(img512.permute(1, 2, 0))  # 输入
        axes[0, 0].set_title("① 模型看到的 512×512")  # 标题
        for ax, l in zip(axes[0, 1:], layers):  # 视觉编码器的几层
            ax.imshow(pca_rgb(vis.hidden_states[l + 1][0]))  # PCA 着色（hidden_states[0] 是嵌入层）
            ax.set_title(f"② SigLIP 第 {l} 层（PCA）")  # 标题
        axes[1, 0].imshow(pca_rgb(img_tokens), interpolation="nearest")  # 连接器输出
        axes[1, 0].set_title("③ 连接器输出 8×8（PCA）")  # 标题
        axes[1, 1].imshow(img512.permute(1, 2, 0))  # 词网格底图
        for t, w in enumerate(words):  # 写词
            axes[1, 1].text((t % 8) * 64 + 32, (t // 8) * 64 + 36, w, ha="center", va="center", fontsize=6.5, color="white",  # 位置
                            bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.55, lw=0))  # 黑底
        axes[1, 1].set_title("④ 最后一层 Logit Lens")  # 标题
        overlay(img512, occ.clamp(min=0), ax=axes[1, 2], title="⑤ 遮挡敏感度：回答依赖哪里")  # 遮挡图
        axes[1, 3].text(0, 1, f"Q: {question}\n\nA: {text}", va="top", wrap=True, fontsize=10)  # 问答文本
        axes[1, 3].set_title("⑥ 回答")  # 标题
        for ax in axes.flat:  # 去坐标轴
            ax.axis("off")  # 关闭
        plt.tight_layout()  # 调整
        plt.show()  # 显示
    return dict(answer=text, words=words, occlusion=occ, vision_states=vis.hidden_states, image_tokens=img_tokens)  # 返回中间结果
