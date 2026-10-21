"""把 src/*.py（jupytext percent 格式）转成 notebooks/*.ipynb，检查注释，可选执行并导出 HTML。

用法：
    .venv/bin/python tools/build_notebooks.py            # 只生成（不执行）
    .venv/bin/python tools/build_notebooks.py --execute  # 生成 + 从头执行 + 导出 HTML
    .venv/bin/python tools/build_notebooks.py 03 --execute  # 只处理编号以 03 开头的章节
"""
import argparse  # 命令行参数
import io  # 字符串当文件读
import time  # 计时
import tokenize  # 用 Python 自己的词法分析找注释
from pathlib import Path  # 路径

import jupytext  # percent 格式 ↔ ipynb
import nbformat  # Notebook 文件格式
from nbclient import NotebookClient  # 执行 Notebook
from nbconvert import HTMLExporter  # 导出 HTML

ROOT = Path(__file__).resolve().parent.parent  # 项目根目录
SRC, OUT, HTML = ROOT / "src", ROOT / "notebooks", ROOT / "results" / "html"  # 源码、Notebook、HTML 目录
KERNEL = {"display_name": "SmolVLM 学习 (.venv)", "language": "python", "name": "smolvlm-mac-lab"}  # 内核信息


def check_comments(code: str, where: str) -> None:
    """要求每一行代码要么自己带注释，要么紧挨在一行注释下面。"""
    comment_lines = {t.start[0] for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type == tokenize.COMMENT}  # 带注释的行号
    lines = code.splitlines()  # 按行切分
    for i, line in enumerate(lines, 1):  # 行号从 1 开始
        s = line.strip()  # 去掉缩进
        if not s or s in {")", "]", "}", "),", "],", "},", '"""'} or s.startswith(("%", "!")):  # 空行、单独的右括号、魔法命令放行
            continue  # 跳过
        if i in comment_lines or (i - 1) in comment_lines:  # 本行或上一行有注释
            continue  # 合格
        raise ValueError(f"{where} 第 {i} 行缺少中文注释：{line}")  # 不合格就报错


def build(path: Path, execute: bool) -> None:
    nb = jupytext.read(path, fmt="py:percent")  # 读入 percent 格式
    for k, cell in enumerate(nb.cells):  # 检查每个代码单元
        if cell.cell_type == "code":  # 只检查代码
            compile(cell.source.replace("%matplotlib", "#"), f"{path.name}#{k}", "exec")  # 先确认语法正确
            check_comments(cell.source, f"{path.name} 单元 {k}")  # 再检查注释
    nb.metadata["kernelspec"] = KERNEL  # 指定内核
    nb.metadata.pop("jupytext", None)  # 去掉 jupytext 元数据，保持干净
    dest = OUT / (path.stem + ".ipynb")  # 输出路径
    n_code = sum(c.cell_type == "code" for c in nb.cells)  # 代码单元数量
    if execute:  # 需要执行
        t = time.time()  # 开始计时
        NotebookClient(nb, timeout=300, kernel_name=KERNEL["name"], resources={"metadata": {"path": str(OUT)}}).execute()  # 在 notebooks 目录下从头执行
        body, _ = HTMLExporter().from_notebook_node(nb)  # 转 HTML
        (HTML / (path.stem + ".html")).write_text(body, encoding="utf-8")  # 写入
        print(f"✅ {dest.name}: {n_code} 个代码单元，执行耗时 {time.time() - t:.0f} 秒")  # 报告
    else:  # 只生成
        print(f"📝 {dest.name}: {n_code} 个代码单元（未执行）")  # 报告
    nbformat.validate(nb)  # 校验格式
    nbformat.write(nb, dest)  # 写文件


if __name__ == "__main__":
    ap = argparse.ArgumentParser()  # 参数解析器
    ap.add_argument("prefix", nargs="?", default="")  # 可选：只处理某些章节
    ap.add_argument("--execute", action="store_true")  # 是否执行
    args = ap.parse_args()  # 解析
    OUT.mkdir(exist_ok=True)  # 确保目录存在
    HTML.mkdir(parents=True, exist_ok=True)  # 同上
    for p in sorted(SRC.glob(f"{args.prefix}*.py")):  # 遍历源码
        if not p.name.startswith("._"):  # 跳过 exFAT 磁盘自动生成的 ._ 隐藏文件
            build(p, args.execute)  # 构建
