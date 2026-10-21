"""快速冒烟测试：不经过 Jupyter，直接把 src/XX.py 当脚本跑，出错或卡住时打印位置。"""
import faulthandler, sys, os, time  # 标准库
faulthandler.dump_traceback_later(int(os.environ.get("SMOKE_TIMEOUT", "300")), exit=True)  # 超时就打印堆栈并退出
import matplotlib  # 绘图库
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt  # 绘图接口
plt.show = lambda *a, **k: plt.close("all")  # show 时直接关闭图，避免占内存
path = sys.argv[1]  # 要测试的源文件
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks"))  # 与 Notebook 相同的工作目录
t = time.time()  # 计时
exec(compile(open(os.path.join("..", path)).read().replace("get_ipython()", "None"), path, "exec"), {"__name__": "__main__"})  # 执行
print(f"SMOKE OK {path} {time.time() - t:.1f}s")  # 成功
