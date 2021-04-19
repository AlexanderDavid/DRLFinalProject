import argparse
import math

import matplotlib.pyplot as plt
from matplotlib import animation
from collections import defaultdict
import matplotlib.colors as colors
import numpy as np

COLORS = ["lightcoral", "limegreen", "m", "c"]


class Visualizer:
    DISPLAY = 0
    SAVE = 1

    def __init__(self, playback, mode=DISPLAY, filename="out.mp4"):
        # Initialize the frame
        self.fig = plt.figure()

        # Set the title if saving to the filename
        if mode == Visualizer.SAVE:
            self.fig.suptitle(filename.replace(".mp4", ""))

        # self.fig.set_dpi(100)
        self.fig.set_size_inches(7, 7)

        # Turn the playback file into a dictionary indexed by time
        self.playback = defaultdict(list)
        for frame in playback:
            self.playback[frame[-1]].append(frame[:-1])

        # Get the minimum and maximum x and y position that any agent will be at
        min_x = math.inf
        min_y = math.inf
        max_x = -math.inf
        max_y = -math.inf
        for x, y in [(agent[1], agent[2]) for agent in playback]:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        # Make it square
        min_lim = min(min_x, min_y) - 1
        max_lim = max(max_x, max_y) + 1
        lim = max(abs(min_lim), abs(max_lim))
        self.ax = plt.axes(xlim=(min_lim, max_lim), ylim=(min_lim, max_lim))

        # Add goal positions to frame
        for agent_id, goal_x, goal_y in set((agent[0], agent[3], agent[4]) for agent in playback):
            plt.scatter(goal_x, goal_y, marker='x', color=COLORS[agent_id % len(COLORS)])

        # Turn each agent into a patch
        self.agents = {}
        self.dirs = {}
        for agent_id in set(agent[0] for agent in playback):
            print(COLORS[agent_id % len(COLORS)])
            self.agents[agent_id] = plt.Circle((1000, 1000), 0.354, fc=COLORS[agent_id % len(COLORS)])
            self.dirs[agent_id] = plt.Arrow(500, 500, 500, 500, color=COLORS[agent_id % len(COLORS)])

        # Get the timesteps in a list
        self.timesteps = list(self.playback.keys())

        # Calculate the timestep size
        timestep_size = self.timesteps[1] - self.timesteps[0]

        print("Interval: ", 1000 * timestep_size, " for ", len(self.playback), " frames")

        # Define the animation
        self.anim = animation.FuncAnimation(self.fig,
                                            lambda i: self.animate(i),
                                            frames=len(self.playback),
                                            init_func=self.init,
                                            interval=timestep_size * 100,
                                            blit=True,
                                            repeat=False)


        self.mode = mode
        if self.mode == Visualizer.DISPLAY:
            plt.show()
        else:
            self.anim.save(filename,
                           dpi=100,
                           writer="ffmpeg",
                           fps=60,
                           bitrate=196000)

    def init(self):
        # Add agents to frame
        for agent_id in self.agents:
            self.ax.add_patch(self.agents[agent_id])
            self.ax.add_patch(self.dirs[agent_id])

        return [self.agents[agent_id] for agent_id in self.agents]

    def animate(self, i):
        if i % 50 == 49:
            print(f"Visualizing frame {i + 1}")

        self.time = self.timesteps[i]

        if i == len(self.playback) - 2:
            plt.close()

        to_del = []

        for agent in self.playback[self.time]:
            id, x, y, gx, gy, vx, vy, rot = agent

            rotx = np.cos(rot)
            roty = np.sin(rot)
            print(rot, rotx, roty)
            self.agents[agent[0]].center = (x, y)
            self.dirs[id] = plt.Arrow(x, y, vx, vy, color=COLORS[id % len(COLORS)], width = 0.5)
            self.ax.add_patch(self.dirs[id])

        return ([self.agents[agent_id] for agent_id in self.agents] +
                [self.dirs[agent_id] for agent_id in self.dirs])


parser = argparse.ArgumentParser(description="Run the agent simulation with optional visualization and exporting")
parser.add_argument("playback", nargs=1, type=str, help="Playback file to visualize")

if __name__ == "__main__":
    args = parser.parse_args()
    playback = np.loadtxt(args.playback[0], delimiter=",").tolist()

    for frame in playback:
        frame[0] = int(frame[0])

    Visualizer(playback, Visualizer.DISPLAY, args.playback[0].replace("playback", "mp4"))
