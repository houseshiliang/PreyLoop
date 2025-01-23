import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# 创建1行6列的子图
fig, axs = plt.subplots(1, 6, figsize=(18, 3))

# 示例数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘制每个子图
for i in range(6):
    axs[i].plot(x, y * (i + 1))
    axs[i].set_title(f'Plot {i + 1}')

# 在每两个图周围加一个矩形边框，表示分组
for i, (start, end) in enumerate([(0, 1), (2, 3), (4, 5)]):
    # 计算左上角的起点 (x, y)，以及宽度和高度
    x0 = axs[start].get_position().x0
    x1 = axs[end].get_position().x1
    y0 = axs[start].get_position().y0
    y1 = axs[start].get_position().y1
    
    # 创建矩形框
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, 
                             linewidth=2, edgecolor='red', facecolor='none')
    
    # 添加到主图中
    fig.add_artist(rect)

plt.savefig("./data/test.pdf")
plt.show()
