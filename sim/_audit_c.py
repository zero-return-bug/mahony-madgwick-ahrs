#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端审计：把正文里的 C 函数机械转成 Python，用同一份 IMU 数据跑，
和已验证的仿真实现逐步比对四元数轨迹。

与 _verify_c.py（表达式级、只覆盖 9 轴）互补：
这里覆盖 mahonyUpdate / madgwickUpdate / getRPY，比的是"跑起来的结果"。

用法： python sim/_audit_c.py
"""
import io
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MD = os.environ.get("AHR_MD") or os.path.join(
    HERE, os.pardir, "Mahony与Madgwick姿态解算-算法系列02.md")

from sim_ahrs import (DT, D2R, euler2quat, gen_imu, quat2euler,  # noqa: E402
                      run_madgwick, run_mahony)

TYPES = ("float", "int16_t", "int", "uint8_t", "double", "long")
MATHF = {
    "atan2f": "math.atan2", "asinf": "math.asin", "acosf": "math.acos",
    "atanf": "math.atan", "cosf": "math.cos", "sinf": "math.sin",
    "tanf": "math.tan", "sqrtf": "math.sqrt", "fabsf": "abs", "powf": "math.pow",
}


def to_py(c_src):
    s = re.sub(r"//[^\n]*", "", c_src)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    start = s.index("{") + 1
    depth, end = 1, len(s)
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    body = s[start:end]

    for k, v in MATHF.items():
        body = body.replace(k + "(", v + "(")
    body = body.replace("&&", " and ").replace("||", " or ")
    body = re.sub(r"(\d)f\b", r"\1", body)
    body = re.sub(r"\bconstrain\(([^,]+),\s*([^,]+),\s*([^)]+)\)",
                  r"max(\2, min(\3, \1))", body)
    # 按括号深度切分语句：只有深度 0 处的 ; { } 才算分隔符
    chunks, buf, depth_p = [], "", 0
    for ch in body:
        if ch == "(":
            depth_p += 1
        elif ch == ")":
            depth_p -= 1
        if depth_p == 0 and ch in ";{}":
            if buf.strip():
                chunks.append(buf.strip() + (";" if ch == ";" else ""))
            buf = ""
            if ch in "{}":
                chunks.append(ch)
        else:
            buf += " " if ch == "\n" else ch
    if buf.strip():
        chunks.append(buf.strip())

    lines, ind, i = [], 0, 0
    while i < len(chunks):
        ln = chunks[i]
        nxt = chunks[i + 1] if i + 1 < len(chunks) else None
        if ln.startswith("if") and nxt == "{":
            cond = ln[ln.index("(") + 1:ln.rindex(")")]
            lines.append("    " * ind + "if %s:" % cond)
            ind += 1
            i += 2
            continue
        if ln == "}":
            ind -= 1
        elif ln.startswith("if") and ln.endswith(";"):
            m = re.match(r"if\s*\((.*)\)\s*(return\s*;|.*;)$", ln)
            cond, stmt = m.group(1), m.group(2)
            body_stmt = "return" if stmt.startswith("return") else stmt[:-1]
            lines.append("    " * ind + "if %s: %s" % (cond, body_stmt))
        elif ln.endswith(";"):
            stmt = ln[:-1]
            for t in TYPES:
                if stmt.startswith(t + " "):
                    stmt = stmt[len(t) + 1:]
                    break
            lines.append("    " * ind + stmt)
        else:
            raise ValueError("无法解析的片段: %r" % ln)
        i += 1
    return lines


def build(c_src, name, params, ret=None):
    body = "\n".join("    " + ln for ln in to_py(c_src))
    ret = ret or params
    src = "def %s(%s):\n%s\n    return (%s)\n" % (
        name, ", ".join(params), body, ", ".join(ret))
    ns = {"math": math, "max": max, "min": min, "abs": abs}
    exec(src, ns)   # noqa: S102
    return ns[name]


def main():
    if not os.path.isfile(MD):
        print("SKIP: 需要系列原文 %s" % MD)
        print("      正文不在本仓库，用 AHR_MD=<原文路径> 指定")
        return 1
    md = io.open(MD, encoding="utf-8").read()
    blocks = re.findall(r"```cpp\n(.*?)```", md, re.S)

    def grab(sig):
        for b in blocks:
            if sig in b:
                return b[b.index(sig):]
        raise KeyError(sig)

    t, q_true, _, acc, gyro = gen_imu(
        20.0, gyro_bias=[0.01, 0.01, 0.01], gyro_noise=0.003, acc_noise=0.06, seed=11)
    q0 = euler2quat(8 * D2R, 6 * D2R, 0.0)

    def run_c(csrc, name, with_int):
        params = ["gx", "gy", "gz", "ax", "ay", "az", "dt", "Kp", "Ki", "beta",
                  "q0", "q1", "q2", "q3"]
        if with_int:
            params += ["eInt0", "eInt1", "eInt2"]
        fn = build(csrc, name, params)
        q = list(q0)
        eInt = [0.0, 0.0, 0.0]
        hist = np.zeros((len(acc), 4))
        for i in range(len(acc)):
            gx, gy, gz = gyro[i]
            ax, ay, az = acc[i]
            args = [gx, gy, gz, ax, ay, az, DT, 2.0, 0.05, 0.08] + list(q)
            if with_int:
                args += list(eInt)
            out = fn(*args)
            q = list(out[10:14])
            if with_int:
                eInt = list(out[14:17])
            hist[i] = q
        return hist

    print("=" * 62)
    c_mahony = grab("void mahonyUpdate(")
    c_madg = grab("void madgwickUpdate(float gx")
    c_rpy = grab("void getRPY(")

    ref = run_mahony(acc, gyro, q0, Kp=2.0, Ki=0.05)
    got = run_c(c_mahony, "mahony", True)
    d = np.abs(got - ref).max()
    print("Mahony    20 s 轨迹最大四元数偏差 %.2e  %s" % (d, "OK" if d < 1e-9 else "FAIL"))

    ref = run_madgwick(acc, gyro, q0, beta=0.08)
    got = run_c(c_madg, "madgwick", True)
    d = np.abs(got - ref).max()
    print("Madgwick  20 s 轨迹最大四元数偏差 %.2e  %s" % (d, "OK" if d < 1e-9 else "FAIL"))

    rpy = build(c_rpy, "getRPY", ["q0", "q1", "q2", "q3"],
                ret=["roll", "pitch", "yaw"])
    worst = 0.0
    rng = np.random.default_rng(9)
    for _ in range(500):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        a = [math.radians(v) for v in rpy(*q)]      # C 版直接输出角度
        b = quat2euler(q)
        worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
    print("getRPY    500 个随机姿态最大偏差 %.2e rad  %s"
          % (worst, "OK" if worst < 1e-6 else "FAIL"))    # 57.29578 的舍入决定精度

    # 9 轴：与 sim_marg 逐步比对
    from sim_marg import madgwick_marg, mag_body
    c_marg = grab("void madgwickUpdateMARG(")
    marg = build(c_marg, "marg",
                 ["gx", "gy", "gz", "ax", "ay", "az", "mx", "my", "mz", "dt", "beta",
                  "q0", "q1", "q2", "q3"], ret=["q0", "q1", "q2", "q3"])
    n = 1200
    rng2 = np.random.default_rng(3)
    qt = euler2quat(10 * D2R, -8 * D2R, 20 * D2R)
    A = np.tile([2 * (qt[1] * qt[3] - qt[0] * qt[2]),
                 2 * (qt[0] * qt[1] + qt[2] * qt[3]),
                 qt[0] ** 2 - qt[1] ** 2 - qt[2] ** 2 + qt[3] ** 2], (n, 1))
    A = A + rng2.normal(0, 0.01, (n, 3))
    M = np.tile(mag_body(qt), (n, 1)) + rng2.normal(0, 0.005, (n, 3))
    G = np.tile([0.004, -0.003, 0.006], (n, 1)) + rng2.normal(0, 0.002, (n, 3))
    q = list(euler2quat(25 * D2R, 12 * D2R, -15 * D2R))
    hc = np.zeros((n, 4))
    for i in range(n):
        out = marg(G[i][0], G[i][1], G[i][2], A[i][0], A[i][1], A[i][2],
                   M[i][0], M[i][1], M[i][2], DT, 0.08, *q)
        q = list(out)
        hc[i] = q
    hr = madgwick_marg(A, G, M, np.array(euler2quat(25 * D2R, 12 * D2R, -15 * D2R)), 0.08)
    d = np.abs(hc - hr).max()
    print("MARG(9轴) 12 s 轨迹最大四元数偏差 %.2e  %s" % (d, "OK" if d < 1e-9 else "FAIL"))

    print("-" * 62)
    q = euler2quat(30 * D2R, -20 * D2R, 45 * D2R)
    print("给定 roll/pitch/yaw = 30/-20/45：由四元数反解 = %.3f / %.3f / %.3f（度）"
          % rpy(*q))
    ab = np.array([2 * (q[1] * q[3] - q[0] * q[2]),
                   2 * (q[0] * q[1] + q[2] * q[3]),
                   q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2])
    print("同一姿态下加速度计反解 roll/pitch = %.3f / %.3f（度）"
          % (math.degrees(math.atan2(ab[1], ab[2])),
             math.degrees(math.atan2(-ab[0], math.hypot(ab[1], ab[2])))))


if __name__ == "__main__":
    main()
