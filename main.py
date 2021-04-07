import depth_collision_avoidance_env
import gym
import torch
from torch import nn
import numpy as np
from torch.distributions import MultivariateNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

class ActorCritic(nn.Module):
    def __init__(self, width, height, std):
        super(ActorCritic, self).__init__()

        self.depth_conv = nn.Sequential(
            nn.Conv3d(1, 1, (1, 4, 4), (1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(1, 64, (10, 1, 1), (1, 1, 1), padding=(0, 0, 0)),
            nn.MaxPool3d((1, 2, 2)),
            nn.Flatten()
        )

        self.action_prediction = nn.Sequential(
            nn.Linear(20484, 800),
            nn.ReLU(),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Linear(600, 2),
        )

        self.cov_mat = torch.eye(2) * (std ** 2)

    def forward(self, depth, goal, vel, greedy=False):
        # Translate everything to tensors of the correct shape
        if type(depth) is not torch.Tensor:
            depth = torch.Tensor(list(depth)).view(-1, 1, 10, 64, 80)
        if type(goal) is not torch.Tensor:
            goal = torch.Tensor(goal).view(-1, 2)
        if type(vel) is not torch.Tensor:
            vel = torch.Tensor(vel).view(-1, 2)

        # Convolve the depth image stack and concat with the goal and last velocity
        conv = self.depth_conv(depth)
        catted = torch.cat((conv, goal, vel), dim=1)

        # Get the means for the two actions
        action_means = self.action_prediction(catted).view(-1, 2)

        # Sample from a normal distribution for each agent in the batch
        linears = []
        rotations = []
        for action_mean in action_means:
            if greedy:
                sample = action_mean
            else:
                dist = MultivariateNormal(action_mean, self.cov_mat)
                sample = dist.sample()

            linears.append(torch.sigmoid(sample[0]))
            rotations.append(torch.tanh(sample[1]))

        # Return the zipped actions
        return zip(linears, rotations)

# creating environment
env_name = "DepthCollisionAvoidance-v0"
env = gym.make(env_name)
ac = ActorCritic(64, 80, 0.0001)

obs = env.reset()

depths = torch.stack([torch.Tensor(list(obs[agent_id][0])) for agent_id in obs]).view(-1, 1, 10, 64, 80)
goals = torch.stack([torch.Tensor(list(obs[agent_id][1])) for agent_id in obs])
vels = torch.stack([torch.Tensor(list(obs[agent_id][2])) for agent_id in obs])

actions = ac(depths, goals, vels)

print(list(actions))
