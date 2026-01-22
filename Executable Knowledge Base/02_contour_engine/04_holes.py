import pyclipper
import matplotlib.pyplot as plt

# 一个矩形（逆时针 CCW）
ccw = [(0,0),(40,0),(40,30),(0,30)]

# 同一个矩形（顺时针 CW）
cw = list(reversed(ccw))

def draw_with_arrows(ax, path, title, color):
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    # 闭合
    xs.append(path[0][0])
    ys.append(path[0][1])

    ax.plot(xs, ys, marker='o', color=color)

    # 画箭头表示方向
    for i in range(len(path)):
        x0,y0 = path[i]
        x1,y1 = path[(i+1)%len(path)]
        ax.annotate("",
                    xy=(x1,y1),
                    xytext=(x0,y0),
                    arrowprops=dict(arrowstyle="->", color=color))

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,4))

draw_with_arrows(ax1, ccw, "CCW (Outer Shell)", "blue")
draw_with_arrows(ax2, cw,  "CW (Hole)", "red")

plt.tight_layout()
plt.show()

# Clipper 的判断
print("CCW Orientation:", pyclipper.Orientation(ccw))
print("CW  Orientation:", pyclipper.Orientation(cw))
