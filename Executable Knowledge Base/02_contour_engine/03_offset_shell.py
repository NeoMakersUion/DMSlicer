import pyclipper
import matplotlib.pyplot as plt

SCALE = 1_000_000

def scale(p):
    return [(int(x*SCALE), int(y*SCALE)) for x,y in p]

def unscale(p):
    return [(x/SCALE, y/SCALE) for x,y in p]

def draw(ax, paths, color, linestyle, label=None):
    for p in paths:
        xs = [x for x,y in p] + [p[0][0]]
        ys = [y for x,y in p] + [p[0][1]]
        ax.plot(xs, ys,
                marker='o',
                color=color,
                linestyle=linestyle,
                label=label)
        label = None
    ax.set_aspect("equal")
    ax.grid(True)

# ------------------------
# 实体轮廓（来自上一节）
# ------------------------
solid = [(0,0),(20,0),(20,40),(0,40)]

# ------------------------
# Clipper Offset
# ------------------------
offset_mm = 2.0   # 壳厚 2mm

co = pyclipper.PyclipperOffset()
co.AddPath(scale(solid),
           pyclipper.JT_MITER,          # 保持直角
           pyclipper.ET_CLOSEDPOLYGON)  # 闭合轮廓

shell = co.Execute(int(offset_mm * SCALE))
shell_loops = [unscale(p) for p in shell]

# ------------------------
# 统一坐标范围
# ------------------------
all_x = [x for p in [solid] + shell_loops for x,y in p]
all_y = [y for p in [solid] + shell_loops for x,y in p]
xmin, xmax = min(all_x)-5, max(all_x)+5
ymin, ymax = min(all_y)-5, max(all_y)+5

# ------------------------
# 可视化
# ------------------------
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,4))

# 左图：原始实体
draw(ax1,[solid],"blue","-","solid")
ax1.set_title("Original Solid")
ax1.set_xlim(xmin,xmax)
ax1.set_ylim(ymin,ymax)

# 右图：实体 + Offset 壳
draw(ax2,[solid],"blue","-","solid")
draw(ax2,shell_loops,"blue","--","offset shell")
ax2.set_title(f"Offset Shell (+{offset_mm}mm)")
ax2.set_xlim(xmin,xmax)
ax2.set_ylim(ymin,ymax)
ax2.legend()

plt.tight_layout()
plt.show()
