import depth_collision_avoidance_env
import gym
import torch
from statistics import mean
from ppo import PPO, Memory
import numpy as np
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


# creating environment
env_name = "DepthCollisionAvoidance-v0"
env = gym.make(env_name)
ppo = PPO(64, 80, 0.0001)
memory = Memory()

obs = env.reset()
done = False
t = 0
episodes = 500
max_len = 100

rewards = []
lengths = []
running_reward = 0

for episode in range(episodes):
    for i in range(max_len):
        t += 1
        # Convert observations to batches of tensors
        depths = torch.stack([torch.Tensor(list(obs[agent_id][0])) for agent_id in obs]).view(-1, 1, 10, 64, 80)
        goals = torch.stack([torch.Tensor(list(obs[agent_id][1])) for agent_id in obs])
        vels = torch.stack([torch.Tensor(list(obs[agent_id][2])) for agent_id in obs])

        actions = list(ppo(depths, goals, vels))

        parsed_actions = {agent_id: [a.cpu().numpy() for a in actions[i][:2]] for i, agent_id in enumerate(obs)}
        obs, r, done, _ = env.step(parsed_actions, collisions=False)
        running_reward += sum([rewards[id] for id in rewards])

        for action, depth, goal, vel, reward in zip(actions, depths, goals, vels, r):
            memory.actions.append(torch.Tensor(action[:2]))
            memory.states.append((depth.view(10, 64, 80), goal, vel))
            memory.logprobs.append(action[2])
            memory.rewards.append(r[reward])
            memory.is_terminals.append(done)

        if t % 4 == 0:
            print("---")
            ppo.update(memory)
            memory.clear_memory()

    rewards.append(running_reward)
    running_reward = 0
    lengths.append(i)

    if episode % 50 == 49:
        print(f"[{episode:03}/{episodes}\tAvg Reward: {mean(rewards[-50:])}\tAvg Lengths: {mean(lengths[:-50])}")
        np.savetxt(f"./playbacks/{episode:03}.csv", env.playback)

    print(running_reward)
    running_reward = 0


print(env.playback)
