import math
import matplotlib.pyplot as plt

# =========================
# 1) 工具：端点量化（容错）
# =========================
def snap_point(pt, eps=1e-6):
    """把点按 eps 网格量化，解决浮点端点微小误差。"""
    x, y = pt
    return (round(x / eps) * eps, round(y / eps) * eps)

# =========================
# 2) segments -> loops
# =========================
def segments_to_loops(segments, eps=1e-6):
    """
    输入: segments = [ [(x1,y1),(x2,y2)], ... ]
    输出: loops = [ [p0,p1,p2,...], ... ]   # 每个 loop 不重复最后一个点（绘图时再闭合）
    """
    # 建立端点到“线段索引、端点位置”的邻接
    # 每条 segment 有两个端点：0端、1端
    segs = []
    for a, b in segments:
        a2 = snap_point(a, eps)
        b2 = snap_point(b, eps)
        segs.append((a2, b2))

    adj = {}
    for i, (a, b) in enumerate(segs):
        adj.setdefault(a, []).append((i, 0))
        adj.setdefault(b, []).append((i, 1))

    used = [False] * len(segs)
    loops = []

    for i in range(len(segs)):
        if used[i]:
            continue

        a, b = segs[i]
        used[i] = True

        # 从 a->b 开始走
        loop = [a, b]
        curr = b

        # 一直找下一条线段把 curr 继续延伸
        while True:
            candidates = adj.get(curr, [])
            nxt = None

            for (j, end_id) in candidates:
                if used[j]:
                    continue
                p0, p1 = segs[j]
                # 如果 curr 是 segment 的某个端点，则下一点是另一个端点
                nxt = (j, p1) if end_id == 0 else (j, p0)
                break

            if nxt is None:
                # 走不下去了（不是闭环 or 数据断了）
                break

            j, next_pt = nxt
            used[j] = True
            loop.append(next_pt)
            curr = next_pt

            # 闭合检测：回到起点
            if math.isclose(curr[0], loop[0][0], abs_tol=eps) and math.isclose(curr[1], loop[0][1], abs_tol=eps):
                # 去掉最后一个重复的起点
                loop.pop()
                loops.append(loop)
                break

    return loops

# =========================
# 3) 绘图
# =========================
def draw_segments(ax, segments, title):
    for (a, b) in segments:
        ax.plot([a[0], b[0]], [a[1], b[1]], marker='o', color="orange")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True)

def draw_loops(ax, loops, title):
    for loop in loops:
        xs = [p[0] for p in loop] + [loop[0][0]]
        ys = [p[1] for p in loop] + [loop[0][1]]
        ax.plot(xs, ys, marker='o', color="blue")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True)

# =========================
# 4) 示例：打乱顺序 + 加一点点误差（更像真实 slice）
# =========================
segments = [
    [(40.0, 0.0), (40.0, 30.0)],
    [(0.0, 30.0), (0.0, 0.0)],
    [(0.0, 0.0), (40.0, 0.0)],
    [(40.0, 30.0), (0.0, 30.0)],
]

loops = segments_to_loops(segments, eps=1e-6)

# 统一坐标范围
all_pts = [p for seg in segments for p in seg]
all_x = [p[0] for p in all_pts]
all_y = [p[1] for p in all_pts]
xmin, xmax = min(all_x)-5, max(all_x)+5
ymin, ymax = min(all_y)-5, max(all_y)+5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

draw_segments(ax1, segments, "Raw slice segments")
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)

draw_loops(ax2, loops, "Stitched loops (endpoint chaining)")
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)

plt.tight_layout()
plt.show()

print("loops:", loops)
