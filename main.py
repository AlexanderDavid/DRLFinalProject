import depth_collision_avoidance_env
import gym
import torch
from ppo import PPO, Memory

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
for episodes in range(500):
    running_reward = 0
    for i in range(100):
        print(i)
        t += 1
        depths = torch.stack([torch.Tensor(list(obs[agent_id][0])) for agent_id in obs]).view(-1, 1, 10, 64, 80)
        goals = torch.stack([torch.Tensor(list(obs[agent_id][1])) for agent_id in obs])
        vels = torch.stack([torch.Tensor(list(obs[agent_id][2])) for agent_id in obs])

        actions = list(ppo(depths, goals, vels))

        parsed_actions = {agent_id: [a.cpu().numpy() for a in actions[i][:2]] for i, agent_id in enumerate(obs)}
        obs, rewards, done, _ = env.step(parsed_actions)
        running_reward += sum([rewards[id] for id in rewards])

        for action, depth, goal, vel, reward in zip(actions, depths, goals, vels, rewards):
            memory.actions.append(torch.Tensor(action[:2]))
            memory.states.append((depth.view(10, 64, 80), goal, vel))
            memory.logprobs.append(action[2])
            memory.rewards.append(rewards[reward])
            memory.is_terminals.append(done)

        if t % 100 == 0:
            ppo.update(memory)

    print(running_reward)
    running_reward = 0


print(env.playback)
