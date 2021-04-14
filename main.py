import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import depth_collision_avoidance_env
import gym
import torch
from torchvision import transforms
from statistics import mean
from ppo import PPO, Memory
import numpy as np
import pickle
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
losses = []
running_reward = 0

for episode in range(episodes):
    for i in range(max_len):
        t += 1
        # Convert observations to batches of tensors
        depths = ((torch.stack([torch.Tensor(list(obs[agent_id][0])) for agent_id in obs]).view(-1, 1, 10, 64, 80) - 0.4) / 4.1) - 0.5
        goals = torch.stack([torch.Tensor(list(obs[agent_id][1])) for agent_id in obs])
        vels = torch.stack([torch.Tensor(list(obs[agent_id][2])) for agent_id in obs])

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
            print("---")
            loss = ppo.update(memory)
            memory.clear_memory()
            losses.append((t, loss))

    rewards.append((t, running_reward))
    lengths.append((t, i))
    running_reward = 0

    print(f"Last episode reward: {rewards[-1][1]}")

    if episode % 50 == 49:
        print(f"[{episode:03}/{episodes}\tAvg Reward: {mean([r[1] for r in rewards[-50:]])}\tAvg Lengths: {mean([l[1] for l in lengths[-50:]])}")
        torch.save(ppo.policy.state_dict(), "./PPO_checkpoint.pt")
        np.savetxt(f"./playbacks/{episode:03}.csv", env.playback)
        pickle.dump((rewards, lengths, losses), open("./results_checkpoint.pickle", "wb+"))

    running_reward = 0

torch.save(ppo.policy.state_dict(), "./PPO.pt")
pickle.dump((rewards, lengths, losses), open("./results.pickle", "wb+"))

