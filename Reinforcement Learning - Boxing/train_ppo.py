import os
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from training.metrics import MetricsLogger

# Setup folders
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Hyperparameters
GAMMA = 0.99
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
UPDATE_EPOCHS = 5
BATCH_SIZE = 2048
TOTAL_EPISODES = 50
METRICS_PATH = "metrics/ppo_metrics.csv"
MODEL_PATH = "models/ppo_model.pt"

class Actor(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out((c, h, w))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Critic(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out((c, h, w))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            mm.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class PPOAgent:
    def __init__(self, obs_shape, action_space):
        self.actor = Actor(obs_shape, action_space.n).to(device)
        self.critic = Critic(obs_shape).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

    def get_action(self, obs):
        obs = torch.FloatTensor(obs / 255.0).unsqueeze(0).to(device)
        probs = self.actor(obs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), probs[0, action.item()].item()

    def train(self, observations, actions, advantages, old_probs, returns):
        obs = torch.FloatTensor(np.array(observations) / 255.0).to(device)
        actions = torch.LongTensor(actions).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        old_probs = torch.FloatTensor(old_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)

        for _ in range(UPDATE_EPOCHS):
            for start in range(0, len(obs), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_obs = obs[start:end]
                batch_actions = actions[start:end]
                batch_advantages = advantages[start:end]
                batch_old_probs = old_probs[start:end]
                batch_returns = returns[start:end]

                logits = self.actor(batch_obs)
                dist = torch.distributions.Categorical(logits)
                new_probs = dist.log_prob(batch_actions).exp()
                ratio = new_probs / (batch_old_probs + 1e-10)

                clip_adv = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages
                loss_actor = -torch.min(ratio * batch_advantages, clip_adv).mean()
                entropy = dist.entropy().mean()
                total_loss_actor = loss_actor - ENTROPY_COEF * entropy

                values = self.critic(batch_obs).squeeze()
                loss_critic = ((batch_returns - values) ** 2).mean()

                self.optimizer_actor.zero_grad()
                total_loss_actor.backward()
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                loss_critic.backward()
                self.optimizer_critic.step()

def preprocess(obs):
    return np.expand_dims(np.mean(obs, axis=2), axis=0).astype(np.float32)

def compute_advantages(rewards, values, gamma=GAMMA):
    returns, G = [], 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns)
    advantages = returns - np.array(values)
    return advantages, returns

def main():
    env = gym.make("ALE/Boxing-v5", render_mode=None)
    obs_shape = preprocess(env.reset()[0]).shape
    agent = PPOAgent(obs_shape, env.action_space)
    logger = MetricsLogger(save_path="metrics", run_name="ppo")

    for episode in range(TOTAL_EPISODES):
        obs, done, steps = env.reset()[0], False, 0
        obs_list, action_list, prob_list, reward_list, value_list = [], [], [], [], []

        while not done:
            obs_proc = preprocess(obs)
            action, prob = agent.get_action(obs_proc)
            value = agent.critic(torch.FloatTensor(obs_proc / 255.0).unsqueeze(0).to(device)).item()

            obs_list.append(obs_proc)
            action_list.append(action)
            prob_list.append(prob)
            value_list.append(value)
            obs, reward, done, _, _ = env.step(action)
            reward_list.append(reward)
            steps += 1

        advs, rets = compute_advantages(reward_list, value_list)
        agent.train(obs_list, action_list, advs, prob_list, rets)

        logger.log(episode, sum(reward_list), steps)
        print(f"Episode {episode} completed: reward={sum(reward_list)}, steps={steps}")

    logger.save()
    logger.plot()
    plt.savefig("plots/ppo_plot.png")
    torch.save(agent.actor.state_dict(), MODEL_PATH)
    print(f"Training complete. Saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()