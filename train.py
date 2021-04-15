#!/usr/bin/env python
# coding: utf-8

# In[44]:


# In[59]:


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# In[68]:


import torch
from torch import nn
from torch.distributions import MultivariateNormal

class ActorCritic(nn.Module):
    def __init__(self, width, height, std, device="cpu"):
        super(ActorCritic, self).__init__()

        self.depth_conv_action = nn.Sequential(
            nn.Conv2d(10, 64, 8, 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(3072, 1024),
            nn.ReLU()
        ).to(device)
        self.depth_conv_action.apply(ActorCritic.__init_weights)

        self.action_prediction = nn.Sequential(
            nn.Linear(1028, 800),
            nn.ReLU(),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Linear(600, 2),
        ).to(device)

        self.depth_conv_state = nn.Sequential(
            nn.Conv2d(10, 64, 8, 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(3072, 1024),
            nn.ReLU()
        ).to(device)
        self.depth_conv_state.apply(ActorCritic.__init_weights)


        self.state_prediction = nn.Sequential(
            nn.Linear(1028, 800),
            nn.ReLU(),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Linear(600, 1),
        ).to(device)

        self.cov_mat = torch.eye(2) * (std ** 2)
        self.action_var = torch.full((2,), std**2).to(device)

        self.device = device

    @staticmethod
    def __init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)


    def forward(self, depth, goal, vel, greedy=False):
        # Translate everything to tensors of the correct shape
        if type(depth) is not torch.Tensor:
            depth = torch.Tensor(list(depth)).view(-1, 1, 10, 64, 80)
        if type(goal) is not torch.Tensor:
            goal = torch.Tensor(goal).view(-1, 2)
        if type(vel) is not torch.Tensor:
            vel = torch.Tensor(vel).view(-1, 2)

        # Convolve the depth image stack and concat with the goal and last velocity
        conv = self.depth_conv_action(depth)
        catted = torch.cat((conv, goal, vel), dim=1)

        # Get the means for the two actions
        action_means = self.action_prediction(catted).view(-1, 2)

        # Sample from a normal distribution for each agent in the batch
        linears = []
        rotations = []
        logprobs = []
        for action_mean in action_means:
            if greedy:
                sample = action_mean
                logprobs.append(None)
            else:
                dist = MultivariateNormal(action_mean, self.cov_mat)
                sample = dist.sample()
                logprobs.append(dist.log_prob(sample))

            linears.append(torch.sigmoid(sample[0]))
            rotations.append(torch.tanh(sample[1]))

        # Return the zipped actions
        return zip(linears, rotations, logprobs)

    def evaluate(self, depth, goal, vel, action):
        # Translate everything to tensors of the correct shape
        if type(depth) is not torch.Tensor:
            depth = torch.Tensor(list(depth)).view(-1, 10, 64, 80)
        if type(goal) is not torch.Tensor:
            goal = torch.Tensor(goal).view(-1, 2)
        if type(vel) is not torch.Tensor:
            vel = torch.Tensor(vel).view(-1, 2)

        # Convolve the depth image stack and concat with the goal and last velocity
        torch.save(depth, "last_depth.pt")
        conv = self.depth_conv_action(depth)
        catted = torch.cat((conv, goal, vel), dim=1)
        print(catted.shape)

        # Get the means for the two actions
        action_means = self.action_prediction(catted).view(-1, 2)

        action_var = self.action_var.expand_as(action_means)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_means, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_conv = self.depth_conv_state(depth)
        catted = torch.cat((conv, goal, vel), dim=1)
        state_value = self.state_prediction(catted)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


# In[69]:


class PPO(nn.Module):
    def __init__(self, width, height, action_std, lr=0.00003, gamma=0.99, K_epochs=5, eps_clip=0.2, device="cpu"):
        super(PPO, self).__init__()
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = ActorCritic(width, height, action_std).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCritic(width, height, action_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def forward(self, depth, goal, vel, greedy=False):
        return self.policy_old(depth, goal, vel, greedy)

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_depths = torch.squeeze(torch.stack([state[0] for state in memory.states]).to(self.device), 1).detach().view(-1, 10, 64, 80)
        print("Depths contain NaN?", old_depths.isnan().sum() > 0)
        old_goals = torch.squeeze(torch.stack([state[1] for state in memory.states]).to(self.device), 1).detach().view(-1, 2)
        old_vels = torch.squeeze(torch.stack([state[2] for state in memory.states]).to(self.device), 1).detach().view(-1, 2)
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach().view(-1, 2)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(self.device).detach().view(-1, 1)

        # Optimize policy for K epochs:
        losses = []
        print("--- Starting to Train ---")
        for epoch in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_depths, old_goals, old_vels, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            losses.append(loss.mean().item() / self.K_epochs)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.01)
            self.optimizer.step()
            print(epoch, " -- ")
            print("Action Conv isNan? ", ppo.policy.depth_conv_action[0].weight.isnan().sum() > 1)
            print("State Conv isNan? ", ppo.policy.depth_conv_state[0].weight.isnan().sum() > 1)

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return sum(losses)


# In[70]:


import depth_collision_avoidance_env
import gym
import torch
from torchvision import transforms
from statistics import mean
import numpy as np
import pickle
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


# creating environment
env_name = "DepthCollisionAvoidance-v0"
env = gym.make(env_name)
ppo = PPO(64, 80, 0.001, 0.0001)
memory = Memory()

obs = env.reset()
done = False
t = 0
episodes = 500
max_len = 100

rewards = []
lengths = []
losses = []
running_reward = 0


# In[ ]:


for episode in range(episodes):
    for i in range(max_len):
        t += 1
        # Convert observations to batches of tensors
        depths = torch.stack([torch.Tensor(list(obs[agent_id][0])) for agent_id in obs]).view(-1, 10, 64, 80)
        goals = torch.stack([torch.Tensor(list(obs[agent_id][1])) for agent_id in obs])
        vels = torch.stack([torch.Tensor(list(obs[agent_id][2])) for agent_id in obs])
        depths /= 4.5
        depths -= 0.5

        actions = list(ppo(depths, goals, vels))

        parsed_actions = {agent_id: [a.cpu().numpy() for a in actions[i][:2]] for i, agent_id in enumerate(obs)}
        obs, r, done, _ = env.step(parsed_actions, collisions=False)
        running_reward += sum([r[id] for id in r])

        for action, depth, goal, vel, reward in zip(actions, depths, goals, vels, r):
            memory.actions.append(torch.Tensor(action[:2]))
            memory.states.append((depth.view(10, 64, 80), goal, vel))
            memory.logprobs.append(action[2])
            memory.rewards.append(r[reward])
            memory.is_terminals.append(done)

        if len(memory.rewards) % 400 == 0:
            # For some reason the Conv layer is producing NaN sometimes which will cause
            # training to fail. Until this is solved just skip that training step if this
            # happens
            loss = ppo.update(memory)
            print(f"--- Loss {loss} ---")
            losses.append((t, loss))

            memory.clear_memory()

        if done:
            continue

    rewards.append((t, running_reward))
    lengths.append((t, i))
    running_reward = 0

    print(f"Last episode reward: {rewards[-1][1]}")

    if episode % 50 == 49:
        print(f"[{episode:03}/{episodes}\tAvg Reward: {mean([r[1] for r in rewards[-50:]])}\tAvg Lengths: {mean([l[1] for l in lengths[-50:]])}")
        torch.save(ppo.policy.state_dict(), "./PPO_checkpoint.pt")
        np.savetxt(f"./playbacks/{episode:03}.csv", env.playback, delim=",")
        pickle.dump((rewards, lengths, losses), open("./results_checkpoint.pickle", "wb+"))

    running_reward = 0

torch.save(ppo.policy.state_dict(), "./PPO.pt")
pickle.dump((rewards, lengths, losses), open("./results.pickle", "wb+"))


# In[58]:


ppo.policy.depth_conv_action[0].weight.isnan().sum()


# In[ ]:




