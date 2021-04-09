import torch
from torch import nn
from torch.distributions import MultivariateNormal


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


class ActorCritic(nn.Module):
    def __init__(self, width, height, std, device="cpu"):
        super(ActorCritic, self).__init__()

        self.depth_conv = nn.Sequential(
            nn.Conv3d(1, 1, (1, 4, 4), (1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(1, 64, (10, 1, 1), (1, 1, 1), padding=(0, 0, 0)),
            nn.MaxPool3d((1, 2, 2)),
            nn.Flatten()
        ).to(device)

        self.action_prediction = nn.Sequential(
            nn.Linear(20484, 800),
            nn.ReLU(),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Linear(600, 2),
        ).to(device)

        self.state_prediction = nn.Sequential(
            nn.Linear(20484, 800),
            nn.ReLU(),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Linear(600, 1),
        ).to(device)

        self.cov_mat = torch.eye(2) * (std ** 2)
        self.action_var = torch.full((2,), std**2).to(device)

        self.device = device

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

    def evaluate(self, depth, vel, goal, action):
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

        action_var = self.action_var.expand_as(action_means)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_means, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.state_prediction(catted)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO(nn.Module):
    def __init__(self, width, height, action_std, lr=0.0003, gamma=0.99, K_epochs=80, eps_clip=0.2, device="cpu"):
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
        old_depths = torch.squeeze(torch.stack([state[0] for state in memory.states]).to(self.device), 1).detach().view(-1, 1, 10, 64, 80)
        old_goals = torch.squeeze(torch.stack([state[1] for state in memory.states]).to(self.device), 1).detach().view(-1, 2)
        old_vels = torch.squeeze(torch.stack([state[2] for state in memory.states]).to(self.device), 1).detach().view(-1, 2)
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach().view(-1, 2)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(self.device).detach().view(-1, 1)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_depths, old_goals, old_vels, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())