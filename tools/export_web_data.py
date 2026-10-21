"""为交互网页 docs/image/ 预先计算并导出数据（需要本地模型权重和 GPU，约 2 分钟）。

用法：.venv/bin/python tools/export_web_data.py
输出：docs/image/data/ 下的图片、二进制数组和 JSON。网页本身不需要 Python。
"""
import json  # 写 JSON
import math  # 数学
import sys  # 路径
from pathlib import Path  # 路径

import numpy as np  # 数组
import torch  # PyTorch
import torch.nn.functional as F  # 函数式接口
from PIL import Image  # 图片

ROOT = Path(__file__).resolve().parent.parent  # 项目根目录
sys.path.insert(0, str(ROOT))  # 导入 vlmlab
from vlmlab import load_smolvlm, build_inputs, load_photo, get_device, standardize, seed_everything  # 共享工具

OUT = ROOT / "docs" / "image" / "data"  # 输出目录
OUT.mkdir(parents=True, exist_ok=True)  # 创建
G = 32  # patch 网格边长
Q = 16  # 网页里可点击的查询网格（每格 2×2 个 patch，减小数据量）


def save_png(arr01, path, size=None):  # 保存 [H,W,3] 或 [H,W] 的 0~1 数组为 PNG
    img = Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8))  # 转 8 位
    if size:  # 需要缩放
        img = img.resize(size, Image.NEAREST)  # 最近邻放大，保留格子感
    img.save(path)  # 保存


def pca3(feats, basis=None):  # [N,D] → [N,3] 的 0~1 颜色（先标准化，同 vlmlab.pca_rgb）
    f = standardize(feats)  # 每通道标准化 + 单位化
    if basis is None:  # 现场计算主成分
        mean = f.mean(0, keepdim=True)  # 均值
        _, _, v = torch.pca_lowrank(f - mean, q=3, center=False)  # 前 3 个主方向
        proj = (f - mean) @ v  # 投影
        basis = (mean, v, proj.quantile(0.01, 0), proj.quantile(0.99, 0))  # 保存
    mean, v, lo, hi = basis  # 解包
    return (((f - mean) @ v - lo) / (hi - lo + 1e-8)).clamp(0, 1).numpy()  # 归一化到 0~1


@torch.no_grad()
def main():
    seed_everything(0)  # 固定随机
    dev = get_device()  # mps
    processor, model = load_smolvlm(dev)  # float32 + eager
    tok = processor.tokenizer  # 分词器
    photo = load_photo()  # 照片
    question = "Describe this image."  # 问题
    inputs = build_inputs(processor, photo, question, split=False)  # 不切块：一张 512×512
    pv = inputs["pixel_values"][:, 0]  # [1,3,512,512]
    img512 = (pv[0].cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()  # 模型看到的图
    save_png(img512, OUT / "input512.png")  # 保存
    photo.resize((640, 480)).save(OUT / "photo.jpg", quality=90)  # 原图（缩小）
    meta = {"question": question, "grid": G, "query_grid": Q}  # 元信息

    # ---------- 视觉编码器 ----------
    vm = model.model.vision_model  # SigLIP
    vout = vm(pixel_values=pv, output_hidden_states=True, output_attentions=True)  # 全部中间结果
    states = [h[0].cpu() for h in vout.hidden_states]  # 13 个 [1024,768]
    for i, s in enumerate(states):  # 每层 PCA 图
        save_png(pca3(s).reshape(G, G, 3), OUT / f"vit_pca_{i:02d}.png")  # 32×32 PNG
    q_idx = torch.arange(G * G).view(Q, 2, Q, 2).permute(0, 2, 1, 3).reshape(Q * Q, 4)  # 每个查询格对应的 4 个 patch
    attn = np.zeros((12, Q * Q, G * G), dtype=np.uint8)  # [层, 查询, 1024]
    for l, a in enumerate(vout.attentions):  # 每层
        rows = a[0].mean(0)[q_idx.to(dev)].mean(1).cpu()  # 12 头平均，再对 4 个 patch 平均 → [256,1024]
        attn[l] = (rows / rows.max(1, keepdim=True).values * 255).round().numpy().astype(np.uint8)  # 每行按最大值归一化
    attn.tofile(OUT / "vit_attn.bin")  # 12×256×1024 字节
    sim = np.zeros((13, Q * Q, G * G), dtype=np.uint8)  # 余弦相似度
    for l, s in enumerate(states):  # 每层
        f = standardize(s)  # 标准化
        rows = (f[q_idx].mean(1)) @ f.T  # [256,1024]
        sim[l] = (rows.clamp(0, 1) * 255).round().numpy().astype(np.uint8)  # 只保留正相关
    sim.tofile(OUT / "vit_sim.bin")  # 13×256×1024 字节
    rc = torch.stack(torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij"), -1).reshape(-1, 2).float()  # patch 坐标
    dist = torch.cdist(rc, rc) * 16  # 像素距离
    meta["attn_distance"] = [[round(float((a[0, h].cpu() * dist).sum(-1).mean()), 1) for h in range(12)] for a in vout.attentions]  # [层][头]
    norms = torch.stack([s.norm(dim=-1) for s in states])  # [13,1024]
    meta["norm_median"] = [round(float(x), 2) for x in norms.median(1).values]  # 每层中位数
    meta["norm_max"] = [round(float(x), 2) for x in norms.max(1).values]  # 每层最大值
    meta["outliers"] = [[int(i) for i in (norms[l] > 5 * norms[l].median()).nonzero()[:, 0]] for l in range(13)]  # 离群 token

    # ---------- 连接器 ----------
    img_tok = model.model.connector(vout.last_hidden_state)[0]  # [64,576]
    save_png(pca3(img_tok.cpu()).reshape(8, 8, 3), OUT / "connector_pca.png")  # 8×8
    save_png(pca3(vout.last_hidden_state[0].cpu()).reshape(G, G, 3), OUT / "vit_final_pca.png")  # post-LN 的 32×32

    # ---------- 语言模型：Logit Lens ----------
    ids = inputs["input_ids"]  # [1,79]
    img_pos = (ids[0] == model.config.image_token_id).nonzero()[:, 0]  # 64 个图像位置
    lm = model.model.text_model  # SmolLM2
    out = model(**inputs, output_hidden_states=True)  # 官方前向
    hs = out.hidden_states  # 31 个 [1,79,576]（最后一个已过 final norm）
    lens = lambda h: model.lm_head(lm.norm(h)).softmax(-1)  # 解码
    concepts = [" egg", " bowl", " meat", " table", " wood", " food"]  # 概念词
    cids = [tok.encode(w, add_special_tokens=False)[0] for w in concepts]  # 编号
    words, concept_p, last_top = [], {w.strip(): [] for w in concepts}, []  # 结果
    for l in range(len(hs)):  # 每层
        h = hs[l][0] if l < len(hs) - 1 else hs[l][0]  # 最后一层的 hidden_states 已经过 norm
        p = lens(h) if l < len(hs) - 1 else model.lm_head(h).softmax(-1)  # 最后一层不再重复 norm
        pi = p[img_pos]  # [64, vocab]
        words.append([tok.decode([i]).strip().replace("�", "?")[:8] or "_" for i in pi.argmax(-1).tolist()])  # 每格最可能的词
        for w, cid in zip(concepts, cids):  # 概念词概率
            concept_p[w.strip()].append([round(float(x), 6) for x in pi[:, cid]])  # [64]
        t5 = p[-1].topk(5)  # 最后位置（回答第一个词）的前 5
        last_top.append([[tok.decode([i]), round(float(v), 4)] for v, i in zip(t5.values, t5.indices)])  # 保存
    json.dump({"words": words, "concepts": concept_p, "last_top": last_top}, open(OUT / "lens.json", "w"), ensure_ascii=False)  # 写出

    # ---------- 逐词生成 + 注意力 + 遮挡 ----------
    end_ids = {tok.convert_tokens_to_ids("<end_of_utterance>"), tok.eos_token_id}  # 结束符
    cur, mask, past, steps = ids, inputs["attention_mask"], None, []  # 状态
    for s in range(40):  # 最多 40 个词
        o = model(input_ids=cur if past is None else cur[:, -1:], attention_mask=mask, pixel_values=inputs["pixel_values"] if past is None else None,  # 第一步给全序列和图片
                  past_key_values=past, use_cache=True, output_attentions=True)  # 之后只给一个新词
        past = o.past_key_values  # 缓存
        p = o.logits[0, -1].softmax(-1)  # 概率
        nxt = int(p.argmax())  # 贪心
        att = torch.stack([a[0, :, -1, img_pos] for a in o.attentions]).mean((0, 1))  # 对 64 个图像 token 的注意力
        t5 = p.topk(5)  # 前 5 候选
        steps.append({"token": tok.decode([nxt]), "id": nxt, "prob": round(float(p[nxt]), 4),  # 选中的词
                      "top5": [[tok.decode([i]), round(float(v), 4)] for v, i in zip(t5.values, t5.indices)],  # 候选
                      "img_att": [round(float(x), 5) for x in att.cpu()]})  # 注意力
        cur = torch.cat([cur, torch.tensor([[nxt]], device=dev)], 1)  # 接上
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)  # 掩码
        if nxt in end_ids:  # 结束
            break  # 停止
    ans = torch.tensor([[st["id"] for st in steps]], device=dev)  # 回答 token
    full = torch.cat([ids, ans], 1)  # 问题 + 回答
    L = ids.shape[1]  # 回答起点
    def answer_lp(pv5):  # 每个回答 token 的对数概率
        lp = model(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=pv5).logits[0, L - 1:-1].log_softmax(-1)  # 前向
        return lp.gather(-1, ans[0][:, None])[:, 0].cpu()  # [n]
    base = answer_lp(inputs["pixel_values"])  # 不遮挡
    drop = torch.zeros(64, len(steps))  # [格, 词]
    for k in range(64):  # 逐格遮挡
        pv5 = inputs["pixel_values"].clone()  # 复制
        r, c = divmod(k, 8)  # 行列
        pv5[..., r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = 0  # 涂灰
        drop[k] = base - answer_lp(pv5)  # 下降
    for i, st in enumerate(steps):  # 写入每个词的遮挡敏感度
        st["occlusion"] = [round(float(x), 4) for x in drop[:, i]]  # [64]
    json.dump({"steps": steps, "occlusion_total": [round(float(x), 3) for x in drop.clamp(min=0).sum(1)]}, open(OUT / "generation.json", "w"), ensure_ascii=False)  # 写出
    json.dump(meta, open(OUT / "meta.json", "w"), ensure_ascii=False)  # 元信息
    print("回答：", "".join(st["token"] for st in steps))  # 打印
    total = sum(f.stat().st_size for f in OUT.iterdir() if not f.name.startswith("._"))  # 总大小
    print(f"导出完成：{OUT}，共 {total / 1e6:.1f} MB")  # 报告


if __name__ == "__main__":
    main()  # 运行
