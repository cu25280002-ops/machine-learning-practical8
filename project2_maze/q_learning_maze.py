"""Project 2: Tabular Q-Learning maze solver using a Gymnasium environment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 0 = free cell, 1 = wall. Start=(0,0), goal=(4,4).
MAZE = np.array([
    [0, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int8)

class MazeEnv(gym.Env):
    """Small deterministic grid maze with a Gymnasium-compatible API."""
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.rows, self.cols = MAZE.shape
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        self.observation_space = spaces.Discrete(self.rows * self.cols)
        self.start = (0, 0)
        self.goal = (4, 4)
        self.state = self.start
        self.max_steps = 100
        self.steps = 0
        self.moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def _state_id(self, position):
        return position[0] * self.cols + position[1]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start
        self.steps = 0
        return self._state_id(self.state), {}

    def step(self, action):
        dr, dc = self.moves[int(action)]
        nr = self.state[0] + dr
        nc = self.state[1] + dc
        self.steps += 1

        # A wall or an outside move leaves the agent in its current cell.
        if (0 <= nr < self.rows and 0 <= nc < self.cols
                and MAZE[nr, nc] == 0):
            self.state = (nr, nc)
            reward = -1.0
        else:
            reward = -10.0

        terminated = self.state == self.goal
        if terminated:
            reward = 100.0
        truncated = self.steps >= self.max_steps and not terminated
        return self._state_id(self.state), reward, terminated, truncated, {}

def choose_action(q_table, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(4))
    return int(np.argmax(q_table[state]))

def train_agent(episodes=5000, alpha=0.1, gamma=0.95,
                epsilon=1.0, epsilon_min=0.03, epsilon_decay=0.995):
    env = MazeEnv()
    rng = np.random.default_rng(42)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_history = []
    steps_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action = choose_action(q_table, state, epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Q-Learning update: current estimate moves towards the TD target.
            target = reward
            if not terminated:
                target += gamma * np.max(q_table[next_state])
            q_table[state, action] += alpha * (target - q_table[state, action])

            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        steps_history.append(steps)

        if (episode + 1) % 500 == 0:
            recent = np.mean(rewards_history[-100:])
            print(f"Episode {episode + 1:4d}/{episodes} | "
                  f"epsilon={epsilon:.3f} | mean reward (last 100)={recent:.2f}")

    return env, q_table, rewards_history, steps_history

def evaluate(env, q_table, episodes=100):
    successes, step_counts, returns = 0, [], []
    best_path = []
    for episode in range(episodes):
        state, _ = env.reset()
        path = [env.state]
        total_reward = 0.0
        done = False
        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            path.append(env.state)
            total_reward += reward
            done = terminated or truncated
        if env.state == env.goal:
            successes += 1
            if not best_path or len(path) < len(best_path):
                best_path = path
        step_counts.append(env.steps)
        returns.append(total_reward)
    return successes / episodes, float(np.mean(step_counts)), float(np.mean(returns)), best_path

def save_path_plot(path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(MAZE, cmap="Greys", vmin=0, vmax=1)
    if path:
        y = [p[0] for p in path]
        x = [p[1] for p in path]
        ax.plot(x, y, marker="o", linewidth=2, markersize=7)
    ax.text(0, 0, "S", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(4, 4, "G", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_title("Path Found by the Q-Learning Agent")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "maze_learned_path.png", dpi=150)
    plt.close(fig)

def main():
    episodes = 5000
    env, q_table, rewards, steps = train_agent(episodes=episodes)
    success_rate, avg_steps, avg_return, path = evaluate(env, q_table)

    np.save(OUTPUT_DIR / "maze_q_table.npy", q_table)
    settings = {
        "algorithm": "Q-Learning",
        "episodes": episodes,
        "learning_rate_alpha": 0.1,
        "discount_factor_gamma": 0.95,
        "evaluation_episodes": 100,
        "success_rate": success_rate,
        "average_steps": avg_steps,
        "average_return": avg_return,
        "learned_path": [list(p) for p in path]
    }
    with open(OUTPUT_DIR / "maze_model.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)

    # Moving average makes the learning trend easier to interpret.
    window = 100
    moving_rewards = pd.Series(rewards).rolling(window, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.25, label="Episode reward")
    ax.plot(moving_rewards, linewidth=2, label=f"{window}-episode moving average")
    ax.set_title("Q-Learning Training Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "maze_learning_curve.png", dpi=150)
    plt.close(fig)

    save_path_plot(path)
    pd.DataFrame([{
        "evaluation_episodes": 100,
        "success_rate": success_rate,
        "average_steps": avg_steps,
        "average_return": avg_return
    }]).to_csv(OUTPUT_DIR / "maze_metrics.csv", index=False)

    print("\\nProject 2 completed.")
    print(f"Evaluation success rate: {success_rate:.1%}")
    print(f"Average steps per episode: {avg_steps:.2f}")
    print(f"Average return: {avg_return:.2f}")
    print(f"Learned path: {path}")

if __name__ == "__main__":
    main()
