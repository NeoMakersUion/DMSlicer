import pyclipper
import matplotlib.pyplot as plt

SCALE = 1000000

def scale(path):
    return [(int(x*SCALE), int(y*SCALE)) for x,y in path]

def unscale(path):
    return [(x/SCALE, y/SCALE) for x,y in path]

def draw(ax, paths, title):
    for p in paths:
        xs = [x for x,y in p] + [p[0][0]]
        ys = [y for x,y in p] + [p[0][1]]
        ax.plot(xs, ys, marker='o')
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True)

# ------------------------
# 两个重叠的矩形
# ------------------------
rect1 = [(0,0), (40,0), (40,30), (0,30)]
rect2 = [(20,10), (60,10), (60,40), (20,40)]

# ------------------------
# Clipper UNION
# ------------------------
pc = pyclipper.Pyclipper()
pc.AddPath(scale(rect1), pyclipper.PT_SUBJECT, True)
pc.AddPath(scale(rect2), pyclipper.PT_CLIP, True)

result = pc.Execute(pyclipper.CT_UNION)
union_loops = [unscale(p) for p in result]

# ------------------------
# 可视化
# ------------------------
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))

draw(ax1,[rect1,rect2],"Input: Two Rectangles")
draw(ax2,union_loops,"Clipper UNION Result")

plt.tight_layout()
plt.show()
