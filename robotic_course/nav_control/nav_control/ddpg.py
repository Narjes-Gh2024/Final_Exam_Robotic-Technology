import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)
        self.init_weights()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.leaky_relu(self.fc3(x), 0.01)
        x = F.leaky_relu(self.fc4(x), 0.01)
        return torch.tanh(self.out(x))

    def init_weights(self):
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                if 'out' in name:
                    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                else:
                    nn.init.kaiming_normal_(layer.weight, a=0.01, nonlinearity="leaky_relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256 + action_dim, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        self.init_weights()

    def forward(self, state, action):
        x = F.leaky_relu(self.fc1(state), 0.01)
        x = torch.cat([x, action], dim=1)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.leaky_relu(self.fc3(x), 0.01)
        x = F.leaky_relu(self.fc4(x), 0.01)
        return self.out(x)

    def init_weights(self):
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                if 'out' in name:
                    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                else:
                    nn.init.kaiming_normal_(layer.weight, a=0.01, nonlinearity="leaky_relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, seed=None):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.rng = np.random.default_rng(seed=seed)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.choice(self.size, size=batch_size, replace=False)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self.size


class OUNoise:
    def __init__(self, dim, mu=0.0, theta=0.15, sigma=0.2, seed=None):
        self.dim = dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.default_rng(seed=seed)
        self.reset()

    def reset(self):
        self.state = np.ones(self.dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.standard_normal(self.dim)
        self.state += dx
        return self.state


class DDPG:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=0.005, buffer_size=100000, batch_size=64,
                 noise_sigma=0.2, seed=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim, seed)
        self.noise = OUNoise(action_dim, sigma=noise_sigma, seed=seed)

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        self.actor.train()

        if add_noise:
            action = action + self.noise.sample()
            action = np.clip(action, -1.0, 1.0)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def optimize(self):
        if len(self.buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, src, tgt):
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
