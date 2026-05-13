import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gymnasium as gym

# ------------------------------------------------------------
# Mạng Actor (chính sách) - đầu ra là mean và std của Gaussian
# ------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # state-independent std

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std, log_std

    def get_action(self, state, deterministic=False):
        mean, std, log_std = self(state)
        if deterministic:
            return mean, None, None
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # tổng log prob trên các chiều action
        return action, log_prob, log_std

    def evaluate(self, state, action):
        """Tính log_prob hiện tại, entropy cho một cặp (state, action) đã có."""
        mean, std, log_std = self(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

# ------------------------------------------------------------
# Mạng Critic (hàm giá trị)
# ------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)  # [batch] hoặc scalar

# ------------------------------------------------------------
# Bộ nhớ lưu trữ trajectory
# ------------------------------------------------------------
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []  # log prob từ chính sách cũ
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def get_batch(self):
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.dones),
            torch.FloatTensor(self.values),
        )

# ------------------------------------------------------------
# Agent PPO
# ------------------------------------------------------------
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, lam=0.95, clip_eps=0.2, K_epochs=10,
                 batch_size=64, entropy_coef=0.0, max_grad_norm=0.5):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.memory = PPOMemory()

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(state_tensor)
            value = self.critic(state_tensor)
        return action.numpy()[0], log_prob.item(), value.item()

    def compute_gae(self, rewards, dones, values, next_value):
        """Tính advantage và returns bằng GAE."""
        advantages = []
        gae = 0
        values = values + [next_value]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values[:-1])
        return advantages, returns

    def update(self):
        # Lấy toàn bộ batch từ bộ nhớ
        states, actions, old_log_probs, rewards, dones, values = self.memory.get_batch()

        with torch.no_grad():
            next_state = states[-1].unsqueeze(0)
            next_value = self.critic(next_state).item()

        advantages, returns = self.compute_gae(rewards.numpy(), dones.numpy(),
                                               values.numpy().tolist(), next_value)

        # Chuẩn hóa advantages (tùy chọn nhưng cải thiện ổn định)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(states)
        # Cập nhật K epoch
        for _ in range(self.K_epochs):
            # Xáo trộn chỉ số
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # --- Cập nhật Critic ---
                values_pred = self.critic(batch_states)
                critic_loss = nn.MSELoss()(values_pred, batch_returns)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                # --- Cập nhật Actor (PPO-Clip) ---
                log_probs, entropy = self.actor.evaluate(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus (khuyến khích khám phá)
                entropy_loss = -self.entropy_coef * entropy.mean()

                total_actor_loss = actor_loss + entropy_loss

                self.optimizer_actor.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()

        self.memory.clear()  # xóa dữ liệu sau khi học xong

# ------------------------------------------------------------
# Huấn luyện
# ------------------------------------------------------------
def train():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    agent = PPOAgent(state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3,
                     gamma=0.99, lam=0.95, clip_eps=0.2, K_epochs=10,
                     batch_size=64, entropy_coef=0.01)

    max_episodes = 500
    update_timestep = 2048  # thu thập 2048 bước rồi học
    timestep_counter = 0

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, log_prob, value = agent.choose_action(state)
            # Hành động thuộc [-2, 2] trong Pendulum
            action_scaled = np.clip(action, -max_action, max_action)
            next_state, reward, terminated, truncated, _ = env.step(action_scaled)
            done = terminated or truncated

            agent.memory.store(state, action, log_prob, reward, done, value)
            state = next_state
            episode_reward += reward
            timestep_counter += 1

            if timestep_counter % update_timestep == 0:
                agent.update()
                timestep_counter = 0

        print(f"Episode {episode}, Reward: {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    train()