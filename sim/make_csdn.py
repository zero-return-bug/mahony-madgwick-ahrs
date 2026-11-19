# -*- coding: utf-8 -*-
"""把主稿拆成 3 篇 CSDN 可直接导入的 md，并清理平台不兼容项。

做四件事：
1. 按章节边界拆成 原理篇 / 实战篇 / 踩坑篇
2. 删掉手写目录（CSDN 有自动侧边栏，且锚点链接在 CSDN 不生效）
3. 删掉 <!-- 配图 N｜贴… --> 这类作者注释（CSDN 会当正文显示）
4. 图片重命名为 images/NN.png（NN = 图号），并把 12 张图平铺到 csdn/images/

用法：
    python sim/make_csdn.py
    python sim/make_csdn.py --repo https://github.com/you/ahrs   # 顺带把 sim/ 换成仓库链接
"""
import argparse
import io
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "Mahony与Madgwick姿态解算-算法系列02.md")
OUT = os.path.join(ROOT, "csdn")

# 图号 -> 源文件（顺序与正文出现顺序一致）
FIGS = [
    (1, "fig01_gimbal.png"), (2, "fig02_mahony_block.png"),
    (3, "fig03_cross.png"), (4, "fig04_sign.png"),
    (5, "fig05_madgwick_block.png"), (6, "fig06_marg_yaw.png"),
    (7, "fig07_converge_law.png"), (8, "fig08_tracking.png"),
    (9, "fig09_init_conv.png"), (10, "fig10_impact.png"),
    (11, "fig11_yaw.png"), (12, "fig12_sweep.png"),
]

NAV = ("> **系列导航**：① 原理篇（四元数 / 叉积 / 梯度下降） · "
       "② 实战篇（ESP32 代码 + 整定 + 仿真对比） · ③ 踩坑篇（9 个坑 + 一次静默偏 12.7° 复盘）\n"
       ">\n"
       "{sec}"
       "> **代码与仿真脚本**：{repo}")


def nav(sec, repo):
    """sec 为空时不留编号说明（第三篇用的是无编号标题）。"""
    note = ("> 三篇的编号沿用原文，本篇是第 %s 节，不是从 1 开始，看目录时别以为是漏了一段。\n>\n"
            % sec) if sec else ""
    return NAV.format(sec=note, repo=repo)

HEAD1 = """# Mahony 与 Madgwick 姿态解算（一）：四元数、叉积修正与梯度下降

> 不用概率论，不背矩阵。从「陀螺仪漂、加速度计抖」这一个矛盾出发，先讲清叉积为什么能修姿态，再推 Mahony 的 4 个公式和 Madgwick 的 3 个公式，最后把磁力计塞进去做成 9 轴。
>
> 本篇 6 张图，公式全部手工推，不用回去查论文。代码在第二篇。

"""

HEAD2 = """# Mahony 与 Madgwick 姿态解算（二）：ESP32 + MPU6050 完整代码与参数整定

> 承接上篇的原理，这篇全是能落地的：同一组数据、同一套噪声下的定量对比，ESP32 上能直接编译的完整代码，一套可执行的整定流程，以及怎么证明你的实现没有写错。
>
> 本篇 6 张图。所有数字都是脚本跑出来的，不是估的。

"""

HEAD3 = """# 姿态解算里最难查的 9 个坑，和一次静默偏 12.7° 的事故复盘

> 这篇讲的是「写完之后」的事。前两篇把公式和代码都给全了，但真正让人通宵的是另一些东西：一个符号写反、一个单位忘了转、一个 `dt` 写死——症状各不相同，有的发散给你看，有的安安静静地错着。
>
> 先复盘一个最阴的案例：曲线漂亮、平滑、不抖，姿态永远差 12.7°。然后把我踩过的 9 个坑按出现频率排一遍。

"""

CASE = """## 案例：一个符号写错，姿态永远差 12.7°，而 yaw 看着是准的

现象是这样的：9 轴（加速度计 + 磁力计）版本写完，串口曲线平滑得很，收敛也收敛了。拿真值一比——**整体姿态误差稳在 12.7°，不再下降**。

更迷惑的是拆开看：**yaw 只差 1.5°，错的是 roll/pitch**。也就是说，如果你只盯着 yaw 的数字看，会觉得这套东西已经能用了。

而且换个初值结果一样：初始「倾斜错 20°/15° + yaw 错 30°」，末段 12.77°；初始只把 yaw 弄错 30°，末段 12.73°。**与初值无关**——这排除了「收敛慢」和「参数没调好」，说明算法收敛到的那个**不动点本身就是错的**。

把三种写法放在一起测（静态 40 s，同一份噪声）：

| 写法 | 真值处磁残差 | 梯度 vs 数值微分 | 末段姿态误差 | 其中 yaw 偏差 |
|---|---|---|---|---|
| A：我最初写的（$b$ 水平分量放 y 轴） | **1.10** | **74%** | **12.77°** | −1.47° |
| A′：只修残差，不改雅可比 | 0 | **45%** | **35.4°** | −34.5° |
| B：官方等价形式（$x$ 轴对齐磁北） | 2e-16 | 1.4e-9 | **0.31°** | 0.06° |

**第一步：把真值代进残差。**

自己造一份理想数据——$q$ 取真值，用真正的 $q$ 反算加速度和磁场，再喂给残差函数。正确实现下 $f$ 必须恒等于 0（浮点误差 1e-16 量级）。A 版磁残差 $f_4\\sim f_6$ 的最大绝对值是 **1.10**。

残差在真值处不为零，等于目标函数的最小值不在真值上，梯度下降会兢兢业业地把你推向一个错误的点，然后停在那里。

**第二步：查梯度是不是同一个函数的导数。**

用数值微分算 $\\partial(\\tfrac12\\|f\\|^2)/\\partial q$，和手推的 $J^\\top f$ 对比。A 版相对误差 **74%**——两者压根不是同一个函数。

这一步为什么不能省？看上表第二行 A′：把残差改对之后，真值处 $f$ 已经是 0 了，但雅可比还留着错，梯度误差 45%，最后偏 **35°**，比原来更糟。**残差对只说明「极小值在真值上」，不说明「梯度指向那个极小值」。**两条校验缺一不可。

**第三步：换写法做对照。**

把 $b$ 的构造从「水平分量放 y 轴」改成「世界系 $x$ 轴对齐磁北」，即 $b=[b_x,0,b_z]$，其余一字不动。同一段仿真：残差 2e-16，梯度误差 1.4e-9，末段误差 **0.31°**。

**根因**：磁残差 $f_4\\sim f_6$ 应该严格等于 $R^\\top b - m$。我在展开时既把 $b$ 的水平分量放错了轴，又把 $q_0q_3$ 的符号和配对写错了一处。两处错刚好互相抵消掉一部分，梯度方向还留在「大致正确」的锥形范围里——于是它能收敛，收敛到一个倾斜偏了 12.7° 的点，而 yaw 因为磁场的水平分量仍是主要信息来源，反而看着正常。

**这件事的三条教训：**

1. **静默错误比发散难查一百倍。** 发散你会立刻去查；静默偏，而且偏在你没在看的那两个轴上，你会先去怀疑传感器、再怀疑参数、最后怀疑人生。
2. **校验只认两条，成本 20 行代码。** ①真值代入残差必须为 0；②手推梯度必须等于数值微分。上面那张表里，A′ 就是只做了①没做②的下场。
3. **别用「看起来收敛了」当验收标准。** 收敛只说明梯度把你带到了某个极小值，不说明那个极小值是对的。

"""

# 篇 3 里把章号去掉，做成独立成篇的标题
H_FIX = [
    ("## 9. 九个坑", "## 九个坑，按出现频率排序"),
    ("## 10. 小结与下一篇", "## 收尾：两句话记住，外加下一篇"),
]


def clean(block):
    out = []
    for ln in block:
        if ln.lstrip().startswith("<!--"):      # 作者注释，CSDN 会当正文显示
            continue
        out.append(ln)
    s = "\n".join(out)
    # 图片：figs/figNN_xxx.png -> images/NN.png
    for n, fn in FIGS:
        s = s.replace("figs/" + fn, "images/%02d.png" % n)
    # 收敛尾部空行与分隔线
    s = re.sub(r"\n(?:---\s*\n)+\s*$", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip("\n")


def slice_by(lines, start_mark, end_mark=None):
    i = next(k for k, v in enumerate(lines) if v.startswith(start_mark))
    j = len(lines) if end_mark is None else next(
        k for k, v in enumerate(lines) if v.startswith(end_mark))
    return lines[i:j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="", help="可选：把你的仓库地址填进来，替换文中的 sim/ 路径")
    a = ap.parse_args()

    lines = io.open(SRC, encoding="utf-8").read().split("\n")

    p1 = slice_by(lines, "## 0. 先看结论", "## 5. 两者到底差在哪")
    p2 = slice_by(lines, "## 5. 两者到底差在哪", "## 9. 九个坑")
    p3 = slice_by(lines, "## 9. 九个坑", "## 附录 A")

    body1 = clean(p1)
    body2 = clean(p2)
    body3 = clean(p3)
    for old, new in H_FIX:
        body3 = body3.replace(old, new)
    body3 = CASE + body3

    # 读者手上没有 sim/ figs/ 这些目录，把「输出说明」改成对读者有意义的话
    repo = a.repo.rstrip("/") if a.repo else "【仓库地址待填】"
    repo_label = ("[%s](%s)" % (repo.replace("https://", ""), repo)) if a.repo else repo
    body2 = body2.replace(
        "所有图都是脚本生成的，改参数重跑就行：",
        "所有图与数字都是脚本跑出来的，脚本在 %s 的 `sim/` 目录，改参数重跑即可复现：" % repo_label)
    body2 = body2.replace(
        "输出：`figs/` 是 1200 px 博客版，`figs/wechat/` 是 1600 px 公众号版，同名可互换。",
        "想自己出图，改 `sim/figstyle.py` 里的字号和线宽就行；脚本一次渲染两套尺寸"
        "（1200 px 网页版 / 1600 px 手机版），同名可互换。")

    # 篇 3 是系列收尾：去掉预告下一篇的段落（下一篇写什么由作者临时决定）
    body3 = re.sub(r"^下一篇写 \*\*PID\*\*：.*$", "", body3, flags=re.M)
    body3 = body3.replace("## 收尾：两句话记住，外加下一篇", "## 收尾：两句话记住")
    body3 = body3.replace("（Python 复现，`sim/` 目录）", "（Python 实现对照验证）")

    # 不替作者决定下一篇：三篇都不加「下一篇」预告，正文到该结束的地方就结束

    parts = [
        ("01-原理篇.md", HEAD1 + nav("0–4", repo) + "\n\n" + body1),
        ("02-实战篇.md", HEAD2 + nav("5–8", repo) + "\n\n" + body2),
        ("03-踩坑篇.md", HEAD3 + nav("", repo) + "\n\n" + body3),
    ]

    if a.repo:
        r = a.repo.rstrip("/")
        tok = {}
        for name in ("sim_ahrs.py", "sim_marg.py", "make_diagrams.py", "make_figs.py",
                     "figstyle.py", "_verify_c.py", "requirements.txt"):
            tok["sim/" + name] = "%s/blob/main/sim/%s" % (r, name)
            tok[name] = "%s/blob/main/sim/%s" % (r, name)
        tok["sim/"] = "%s/tree/main/sim/" % r

        # 一次扫描替换完，长的先匹配：避免 "sim/" 先被替换、把刚生成的新链接再切一遍。
        # 整段 `sim/xxx.py` 这种行内代码换成 markdown 链接（CSDN 会渲染），
        # 像 `python sim/_verify_c.py` 这种命令行原样保留。
        pat = re.compile(r"`[^`\n]+`|" + "|".join(
            re.escape(k) for k in sorted(tok, key=len, reverse=True)))

        def link(m):
            t = m.group(0)
            if t[0] != "`":
                return tok[t]
            inner = t[1:-1]
            return "[%s](%s)" % (inner, tok[inner]) if inner in tok else t

        for i, (fn, txt) in enumerate(parts):
            # 代码块里的命令保持相对路径（读者 clone 后直接跑），只给正文加链接
            segs = re.split(r"(```.*?```)", txt, flags=re.S)
            parts[i] = (fn, "".join(s if s.startswith("```") else pat.sub(link, s)
                                    for s in segs))

    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    total_img = 0
    for fn, txt in parts:
        n_img = len(re.findall(r"!\[", txt))
        total_img += n_img
        with io.open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
            f.write(txt)
        print("%-12s %6d 字  %2d 张图  %2d 个代码块"
              % (fn, len(txt), n_img, txt.count("\n```") // 2))

    for n, fn in FIGS:
        shutil.copy2(os.path.join(ROOT, "figs", fn),
                     os.path.join(OUT, "images", "%02d.png" % n))
    print("图片 %d 张 -> csdn/images/（%02d.png … %02d.png）" % (len(FIGS), 1, len(FIGS)))
    print("合计正文 %d 字，图 %d 张（每篇各占一半，无重复）" % (
        sum(len(t) for _, t in parts), total_img))


if __name__ == "__main__":
    main()
