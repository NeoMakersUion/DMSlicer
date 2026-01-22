import pyclipper
import matplotlib.pyplot as plt

SCALE = 1_000_000

def scale(p):
    return [(int(x*SCALE), int(y*SCALE)) for x,y in p]

def unscale(p):
    return [(x/SCALE, y/SCALE) for x,y in p]

def draw(ax, paths, title, color):
    for p in paths:
        xs = [x for x,y in p] + [p[0][0]]
        ys = [y for x,y in p] + [p[0][1]]
        ax.plot(xs, ys, marker='o', color=color)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True)

# ------------------------
# 主体 A 和 侵入物 B
# ------------------------
A = [(0,0),(60,0),(60,40),(0,40)]
B = [(20,-10),(40,-10),(40,50),(20,50)]   # 穿过 A 的物体

# ------------------------
# Clipper DIFFERENCE
# ------------------------
pc = pyclipper.Pyclipper()
pc.AddPath(scale(A), pyclipper.PT_SUBJECT, True)
pc.AddPath(scale(B), pyclipper.PT_CLIP, True)

result = pc.Execute(pyclipper.CT_DIFFERENCE)
loops = [unscale(p) for p in result]

# ------------------------
# 统一坐标范围
# ------------------------
all_x = [x for p in [A,B] for x,y in p]
all_y = [y for p in [A,B] for x,y in p]
xmin, xmax = min(all_x)-5, max(all_x)+5
ymin, ymax = min(all_y)-5, max(all_y)+5

# ------------------------
# 可视化
# ------------------------
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))

# 左图：输入
draw(ax1,[A],"A (solid)","blue")
draw(ax1,[B],"B (intruder)","orange")
ax1.set_xlim(xmin,xmax)
ax1.set_ylim(ymin,ymax)

# 右图：结果（单一材料，统一蓝色）
draw(ax2, loops, "Result: A - B", "blue")
ax2.set_xlim(xmin,xmax)
ax2.set_ylim(ymin,ymax)

plt.tight_layout()
plt.show()
