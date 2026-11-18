#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开发自检：把文中 §6.5 的 C 代码与已验证的 Python 实现逐表达式比对。

做法：C 的 float 后缀、变量名规范化之后两边都是合法 Python 表达式，
用 ast.dump 比较语法树，能查出符号、系数、下标、括号层级的任何差异。

用法： python sim/_verify_c.py
"""
import ast
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.environ.get("AHR_MD") or os.path.join(
    HERE, os.pardir, "Mahony与Madgwick姿态解算-算法系列02.md")
PYREF = os.path.join(HERE, "sim_marg.py")

NAMES = ["hx", "hy", "hz", "f1", "f2", "f3", "f4", "f5", "f6",
         "s0", "s1", "s2", "s3"]


def canon(e):
    e = re.sub(r"(\d)f\b", r"\1", e)             # 2.0f -> 2.0
    e = re.sub(r"\b(\d+)\.0\b", r"\1", e)        # 2.0  -> 2
    e = re.sub(r"\bq([0-3])_\b", r"q\1", e)      # q0_  -> q0
    e = re.sub(r"\b([am])\[(\d)\]", r"\1\2", e)  # a[0] -> a0
    for i, v in enumerate("xyz"):                # ax -> a0, my -> m1
        e = re.sub(r"\b([am])%s\b" % v, r"\g<1>%d" % i, e)
    e = re.sub(r"\bs\[(\d)\]", r"s\1", e)        # s[0] -> s0
    e = re.sub(r"\bf\[(\d)\]", r"f\1", e)        # f[3] -> f3
    e = e.replace("sqrtf(", "sqrt(").replace("np.sqrt(", "sqrt(")
    e = e.replace("np.hypot(", "hypot(").replace("fabsf(", "abs(")
    return re.sub(r"\s+", " ", e).strip()


def dump(e):
    return ast.dump(ast.parse(canon(e), mode="eval"))


def main():
    if not os.path.isfile(MD):
        print("SKIP: 需要系列原文 %s" % MD)
        print("      正文不在本仓库，用 AHR_MD=<原文路径> 指定")
        return 1
    md = io.open(MD, encoding="utf-8").read()
    blocks = re.findall(r"```cpp\n(.*?)```", md, re.S)
    cands = [b for b in blocks if "void madgwickUpdateMARG" in b]
    if not cands:
        print("FAIL: 正文里没找到 madgwickUpdateMARG")
        return 1
    c = cands[0]

    src = io.open(PYREF, encoding="utf-8").read()
    tree = ast.parse(src)
    ref = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "marg_pieces":
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Assign):
                    continue
                t = sub.targets[0]
                if isinstance(t, ast.Name):
                    ref[t.id] = ast.get_source_segment(src, sub.value)
                elif isinstance(t, ast.Subscript):
                    base = getattr(t.value, "id", None)
                    idx = t.slice
                    if base and isinstance(idx, ast.Constant) and isinstance(idx.value, int):
                        k = idx.value + 1 if base == "f" else idx.value   # python f[0] 对应 C 的 f1
                        ref["%s%d" % (base, k)] = ast.get_source_segment(src, sub.value)

    bad = 0
    for n in NAMES:
        m = re.search(r"\b%s\s*=\s*(.+?);" % n, c, re.S)
        if not m:
            print("  %-3s  C 里没找到" % n)
            bad += 1
            continue
        if n not in ref:
            print("  %-3s  Python 里没找到" % n)
            bad += 1
            continue
        ce = m.group(1)
        try:
            same = dump(ce) == dump(ref[n])
        except SyntaxError as ex:
            print("  %-3s  语法无法解析：%s" % (n, ex))
            bad += 1
            continue
        print("  %-3s  %s" % (n, "一致" if same else "不一致"))
        if not same:
            bad += 1
            print("      C :", canon(ce))
            print("      Py:", canon(ref[n]))

    # bx 是标量，单独比（C 用 sqrtf，Python 用 np.hypot）
    mb = re.search(r"\bfloat\s+bx\s*=\s*(.+?);", c, re.S)
    if mb:
        ce = canon(mb.group(1))
        ok = ("sqrt(hx * hx + hy * hy)" in ce) or ("hypot(hx, hy)" in ce)
        print("  %-3s  %s" % ("bx", "一致（水平分量 = |h_xy|）" if ok else "需人工确认"))
        bad += 0 if ok else 1
    else:
        print("  bx   C 里没找到")
        bad += 1

    print("-" * 52)
    if bad:
        print("结论：发现 %d 处不一致" % bad)
        return 1
    print("结论：%d 个关键表达式全部一致（含符号与括号层级）" % (len(NAMES) + 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
