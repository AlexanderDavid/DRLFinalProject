import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow, Rectangle


fig, ax = plt.subplots(figsize=(6, 6))

agent1 = Circle((1, 5), 0.354, fc="lightcoral")
agent2 = Circle((5, 9), 0.354, fc="limegreen")
agent3 = Circle((9, 5), 0.354, fc="m")
agent4 = Circle((5, 1), 0.354, fc="c")

agent1_dir = Arrow(1, 5, 1, 0, fc="lightcoral", width=0.6)
agent2_dir = Arrow(5, 9, 0, -1, fc="limegreen", width=0.6)
agent3_dir = Arrow(9, 5, -1, 0, fc="m", width=0.6)
agent4_dir = Arrow(5, 1, 0, 1, fc="c", width=0.6)

ax.add_patch(agent1)
ax.add_patch(agent1_dir)
ax.add_patch(agent2)
ax.add_patch(agent2_dir)
ax.add_patch(agent3)
ax.add_patch(agent3_dir)
ax.add_patch(agent4)
ax.add_patch(agent4_dir)

left_wall = Rectangle((-0.25, -0.25), 0.25, 10.5, color="black")
right_wall = Rectangle((10, -0.25), 0.25, 10.5, color="black")
top_wall = Rectangle((-0.25, 10), 10.5, 0.25, color="black")
bottom_wall = Rectangle((-0.25, -0.25), 10.5, 0.25, color="black")

ax.add_patch(left_wall)
ax.add_patch(right_wall)
ax.add_patch(top_wall)
ax.add_patch(bottom_wall)

plt.xlim([-0.25, 10.25])
plt.ylim([-0.25, 10.25])
plt.axis('off')

plt.savefig("scene.png", bbox_inches="tight", pad_inches=0)
