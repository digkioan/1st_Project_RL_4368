import argparse
import random
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn


def select_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.out(x)


class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, hyperparameter_set):
        self.hyperparameter_set = hyperparameter_set
        self.device = select_device()
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        config = self._load_hyperparameters()[hyperparameter_set]
        self.env_id = config["env_id"]
        self.continuous = config["continuous"]
        self.learning_rate_a = float(config["learning_rate_a"])
        self.discount_factor_g = float(config["discount_factor_g"])
        self.network_sync_rate = int(config["network_sync_rate"])
        self.replay_memory_size = int(config["replay_memory_size"])
        self.mini_batch_size = int(config["mini_batch_size"])
        self.epsilon_init = float(config["epsilon_init"])
        self.epsilon_decay = float(config["epsilon_decay"])
        self.epsilon_min = float(config["epsilon_min"])
        self.stop_on_reward = None if config["stop_on_reward"] is None else float(config["stop_on_reward"])
        self.hidden_dim = int(config["hidden_dim"])
        self.max_episodes = int(config["max_episodes"])

        self.input_dim = 96 * 96 * 3
        self.runs_dir = Path(__file__).resolve().parent / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        self.model_path = self.runs_dir / f"{self.hyperparameter_set}_model.pt"
        self.plot_path = self.runs_dir / f"{self.hyperparameter_set}_training.png"

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        elif self.device.type == "xpu":
            torch.xpu.manual_seed_all(seed)

    def _load_hyperparameters(self):
        config_path = Path(__file__).resolve().parent / "hyperparameters.yml"
        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def preprocess_state(self, state):
        # Flattened DQN uses the image as one long feature vector.
        # Normalize from [0, 255] to [0, 1], then flatten 96x96x3 -> 27648.
        state = torch.tensor(state, dtype=torch.float32, device=self.device) / 255.0
        state = state.flatten()
        return state

    def build_network(self, num_actions):
        return DQN(self.input_dim, self.hidden_dim, num_actions).to(self.device)

    def select_action(self, state, env, policy_dqn, epsilon):
        if random.random() < epsilon:
            return env.action_space.sample()

        with torch.no_grad():
            q_values = policy_dqn(state.unsqueeze(0))
            return q_values.argmax(dim=1).item()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, dones = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = target_dqn(next_states).max(dim=1).values
            target_q = rewards + (1.0 - dones) * self.discount_factor_g * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graphs(self, rewards_per_episode, epsilon_history):
        episodes = len(rewards_per_episode)
        moving_average = np.zeros(episodes)
        for episode in range(episodes):
            start = max(0, episode - 99)
            moving_average[episode] = np.mean(rewards_per_episode[start:episode + 1])

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(rewards_per_episode, label="Episode reward")
        plt.plot(moving_average, label="Mean reward (last 100)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training rewards")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epsilon_history)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Epsilon decay")

        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()

    def train(self, render=False, max_episodes=None, seed=None):
        if seed is not None:
            self.set_seed(seed)

        env = gym.make(
            self.env_id,
            continuous=self.continuous,
            render_mode="human" if render else None,
        )
        num_actions = env.action_space.n
        print(f"Using device: {self.device}")

        if seed is not None:
            env.action_space.seed(seed)

        epsilon = self.epsilon_init
        memory = ReplayMemory(self.replay_memory_size)
        policy_dqn = self.build_network(num_actions)
        target_dqn = self.build_network(num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        rewards_per_episode = []
        epsilon_history = []
        step_count = 0
        episodes_to_run = max_episodes if max_episodes is not None else self.max_episodes
        start_time = time.perf_counter()

        for episode in range(episodes_to_run):
            if seed is not None:
                state, _ = env.reset(seed=seed + episode)
            else:
                state, _ = env.reset()
            state = self.preprocess_state(state)
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not terminated and not truncated:
                action = self.select_action(state, env, policy_dqn, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = self.preprocess_state(next_state)
                done = terminated or truncated

                memory.append((state, action, next_state, reward, done))
                state = next_state
                episode_reward += reward
                step_count += 1

                if len(memory) >= self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                if step_count % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            rewards_per_episode.append(episode_reward)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)

            mean_reward_100 = float(np.mean(rewards_per_episode[-100:]))
            elapsed = time.perf_counter() - start_time
            avg_time_per_episode = elapsed / (episode + 1)
            remaining_episodes = episodes_to_run - (episode + 1)
            eta_seconds = avg_time_per_episode * remaining_episodes
            print(
                f"Episode {episode + 1}/{episodes_to_run} | "
                f"reward={episode_reward:.2f} | "
                f"mean_100={mean_reward_100:.2f} | "
                f"epsilon={epsilon:.4f} | "
                f"elapsed={elapsed:.1f}s | "
                f"avg/ep={avg_time_per_episode:.2f}s | "
                f"ETA={eta_seconds / 60:.1f} min"
            )

            if self.stop_on_reward is not None and mean_reward_100 >= self.stop_on_reward:
                print(f"Stopping early: mean reward over last 100 episodes reached {mean_reward_100:.2f}.")
                break

        env.close()

        torch.save(policy_dqn.state_dict(), self.model_path)
        self.save_graphs(rewards_per_episode, epsilon_history)

        final_mean_reward = float(np.mean(rewards_per_episode[-100:])) if rewards_per_episode else 0.0
        total_time = time.perf_counter() - start_time
        print(f"Saved model to {self.model_path}")
        print(f"Saved training graph to {self.plot_path}")
        print(f"Training finished in {total_time / 60:.2f} minutes")
        return final_mean_reward

    def test(self, render=True, test_episodes=5, seed=None):
        if seed is not None:
            self.set_seed(seed)

        env = gym.make(
            self.env_id,
            continuous=self.continuous,
            render_mode="human" if render else None,
        )
        num_actions = env.action_space.n
        print(f"Using device: {self.device}")

        if seed is not None:
            env.action_space.seed(seed)

        policy_dqn = self.build_network(num_actions)
        policy_dqn.load_state_dict(torch.load(self.model_path, map_location=self.device))
        policy_dqn.eval()

        rewards = []
        for episode in range(test_episodes):
            if seed is not None:
                state, _ = env.reset(seed=seed + episode)
            else:
                state, _ = env.reset()
            state = self.preprocess_state(state)
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not terminated and not truncated:
                with torch.no_grad():
                    action = policy_dqn(state.unsqueeze(0)).argmax(dim=1).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                state = self.preprocess_state(next_state)
                episode_reward += reward

            rewards.append(episode_reward)
            print(f"Test episode {episode + 1}/{test_episodes} | reward={episode_reward:.2f}")

        env.close()
        return float(np.mean(rewards)) if rewards else 0.0

    def run(self, is_training=True, render=False, max_episodes=None, test_episodes=5, seed=None):
        if is_training:
            return self.train(render=render, max_episodes=max_episodes, seed=seed)
        return self.test(render=render, test_episodes=test_episodes, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic DQN for Gymnasium CarRacing-v3 with flattened input.")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    agent = Agent("car_racing_flatten")
    agent.run(
        is_training=args.mode == "train",
        render=args.render,
        max_episodes=args.max_episodes,
        test_episodes=args.test_episodes,
        seed=args.seed,
    )
