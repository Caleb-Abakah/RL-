"""
evaluate.py


Evaluate a trained Q-Learning agent and display its learned policy.

Run as a script (after train.py has produced results/q_table.npy):

    python evaluate.py                 # evaluate the saved run
    python evaluate.py --episodes 1000 # evaluate over more episodes

The script reads results/run_config.json to rebuild the matching environment
(deterministic or slippery) automatically.
"""

import argparse
import json
import os

import numpy as np

from environment import FrozenLakeEnv, SlipperyFrozenLakeEnv
from agent import QLearningAgent

# arrow symbols for each action: 0 Left, 1 Down, 2 Right, 3 Up
ACTION_ARROWS = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}
ACTION_NAMES = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}


def extract_policy(agent):
    """Return the best action for every state (argmax over the Q-table)."""
    return agent.get_policy()


def render_policy(agent, env):
    """Print the learned policy as a grid of arrows (H = hole, G = goal)."""
    policy = extract_policy(agent)
    print("Learned policy (\u2190 left  \u2193 down  \u2192 right  \u2191 up):\n")
    print("+" + "---" * env.n_cols + "+")
    for r in range(env.n_rows):
        line = "|"
        for c in range(env.n_cols):
            cell = env.grid[r][c]
            if cell == "H":
                symbol = "H"
            elif cell == "G":
                symbol = "G"
            else:
                state = r * env.n_cols + c
                symbol = ACTION_ARROWS[int(policy[state])]
            line += f" {symbol} "
        print(line + "|")
    print("+" + "---" * env.n_cols + "+")


def evaluate_agent(env, agent, n_episodes=100, max_steps=200):
    """Run the agent greedily (no exploration, no learning) and collect metrics."""
    successes = 0
    total_reward = 0.0
    rewards_per_episode = []

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        for _ in range(max_steps):
            action = agent.greedy_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        rewards_per_episode.append(episode_reward)
        total_reward += episode_reward
        if episode_reward > 0:
            successes += 1

    failures = n_episodes - successes
    return {
        "episodes": n_episodes,
        "successes": successes,
        "failures": failures,
        "success_rate": 100.0 * successes / n_episodes,
        "average_reward": total_reward / n_episodes,
        "rewards": rewards_per_episode,
    }


def build_env_from_config(cfg, seed=None):
    """Recreate the environment described by a saved run config."""
    if cfg.get("slippery", False):
        return SlipperyFrozenLakeEnv(p_intended=cfg.get("p_intended", 1 / 3),
                                     seed=seed)
    return FrozenLakeEnv()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Q-Learning agent.")
    parser.add_argument("--episodes", type=int, default=100,
                        help="number of evaluation episodes (default: 100)")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    q_path = os.path.join(args.outdir, "q_table.npy")
    cfg_path = os.path.join(args.outdir, "run_config.json")
    if not os.path.exists(q_path):
        raise FileNotFoundError(
            f"{q_path} not found. Run train.py first to produce a Q-table.")

    Q = np.load(q_path)
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)

    env = build_env_from_config(cfg, seed=args.seed)

    agent = QLearningAgent(env.n_states, env.n_actions, seed=args.seed)
    agent.Q = Q

    env_kind = "Slippery" if cfg.get("slippery", False) else "Deterministic"

    # ---- policy ----
    print(f"\nEnvironment: {env_kind}\n")
    render_policy(agent, env)

    print("\nRecommended action for every non-terminal state:")
    print(f"{'State':>5} | {'(row,col)':>9} | {'Action':>6}")
    print("-" * 28)
    policy = extract_policy(agent)
    for s in range(env.n_states):
        r, c = divmod(s, env.n_cols)
        if env.grid[r][c] in ("H", "G"):
            continue
        print(f"{s:>5} | {str((r, c)):>9} | {ACTION_NAMES[int(policy[s])]:>6}")

    # evaluation 
    results = evaluate_agent(env, agent, n_episodes=args.episodes,
                             max_steps=args.max_steps)
    print("\n" + "=" * 40)
    print(f"   EVALUATION RESULTS ({env_kind})")
    print("=" * 40)
    print(f"  Episodes evaluated : {results['episodes']}")
    print(f"  Successful runs    : {results['successes']}")
    print(f"  Failures           : {results['failures']}")
    print(f"  Success Rate       : {results['success_rate']:.1f}%")
    print(f"  Average Reward     : {results['average_reward']:.3f}")
    print("=" * 40)

    # save a short metrics file
    metrics_path = os.path.join(args.outdir, "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Environment: {env_kind}\n")
        f.write(f"Episodes: {results['episodes']}\n")
        f.write(f"Successful runs: {results['successes']}\n")
        f.write(f"Failures: {results['failures']}\n")
        f.write(f"Success rate: {results['success_rate']:.1f}%\n")
        f.write(f"Average reward: {results['average_reward']:.3f}\n")
    print(f"\nMetrics written to {metrics_path}")


if __name__ == "__main__":
    main()
