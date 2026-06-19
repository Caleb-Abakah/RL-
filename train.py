"""
train.py


Train a Q-Learning agent on the Frozen Lake environment and save the results.

Examples
--------
    python train.py                          # deterministic, default settings
    python train.py -slippery               # stochastic (bonus) environment
    python train.py -episodes 30000 -alpha 0.1 -gamma 0.99

Outputs (written to the results/ directory):
    q_table.npy          - the learned Q-table
    run_config.json      - the settings used (so evaluate.py can match them)
    training_stats.txt   - summary statistics
    training_curve.png   - success-rate / epsilon curves (if matplotlib is installed)
"""

import argparse
import json
import os

import numpy as np

from environment import FrozenLakeEnv, SlipperyFrozenLakeEnv
from agent import QLearningAgent
from evaluate import render_policy


def train_agent(env, agent, n_episodes=20000, max_steps=200, verbose=True,
                report_every=2000):
    """
    Train a Q-Learning agent.

    Returns a history dict with per-episode statistics:
        rewards, successes, steps, epsilons
    """
    history = {"rewards": [], "successes": [], "steps": [], "epsilons": []}

    for ep in range(n_episodes):
        state = env.reset()
        history["epsilons"].append(agent.epsilon)

        total_reward = 0.0
        reached_goal = False

        for t in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                reached_goal = reward > 0
                break

        agent.decay_epsilon()
        history["rewards"].append(total_reward)
        history["successes"].append(1 if reached_goal else 0)
        history["steps"].append(t + 1)

        if verbose and (ep + 1) % report_every == 0:
            window = history["successes"][-1000:]
            rate = 100 * sum(window) / len(window)
            print(f"Episode {ep + 1:6d} | epsilon={agent.epsilon:5.3f} | "
                  f"success rate (last 1000) = {rate:5.1f}%")

    return history


def save_training_curve(history, path):
    """Save a success-rate and epsilon plot. Silently skip if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping training_curve.png")
        return

    successes = np.array(history["successes"])
    # rolling success rate over a 500-episode window
    window = 500
    if len(successes) >= window:
        rolling = np.convolve(successes, np.ones(window) / window, mode="valid") * 100
    else:
        rolling = successes * 100

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(range(len(rolling)), rolling, color="tab:blue",
             label="Success rate (rolling)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Success rate (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.plot(history["epsilons"], color="tab:orange", alpha=0.7, label="Epsilon")
    ax2.set_ylabel("Epsilon", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 1.05)

    plt.title("Training: success rate and exploration over time")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved training curve to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train a Q-Learning agent on Frozen Lake.")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--slippery", action="store_true",
                        help="use the stochastic (slippery) environment")
    parser.add_argument("--p-intended", type=float, default=1 / 3,
                        help="probability the intended move happens (slippery only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # build environment
    if args.slippery:
        env = SlipperyFrozenLakeEnv(p_intended=args.p_intended, seed=args.seed)
        print(f"Environment: SLIPPERY (p_intended={args.p_intended:.3f})")
    else:
        env = FrozenLakeEnv()
        print("Environment: DETERMINISTIC")

    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    print(f"Training for {args.episodes} episodes "
          f"(alpha={args.alpha}, gamma={args.gamma}, "
          f"epsilon_decay={args.epsilon_decay})...\n")
    history = train_agent(env, agent, n_episodes=args.episodes,
                          max_steps=args.max_steps)

    final = history["successes"][-1000:]
    final_rate = 100 * sum(final) / len(final)
    print(f"\nFinal training success rate (last 1000): {final_rate:.1f}%")
    print(f"Total successful episodes: {sum(history['successes'])}")

    print("\nLearned policy:\n")
    render_policy(agent, env)

    #  save outputs 
    np.save(os.path.join(args.outdir, "q_table.npy"), agent.Q)

    config = {
        "slippery": args.slippery,
        "p_intended": args.p_intended,
        "episodes": args.episodes,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon_start": args.epsilon,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay": args.epsilon_decay,
        "seed": args.seed,
        "final_train_success_rate": final_rate,
    }
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(args.outdir, "training_stats.txt"), "w") as f:
        f.write(f"Environment: {'slippery' if args.slippery else 'deterministic'}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"alpha={args.alpha}, gamma={args.gamma}, "
                f"epsilon_decay={args.epsilon_decay}\n")
        f.write(f"Final training success rate (last 1000): {final_rate:.1f}%\n")
        f.write(f"Total successful episodes: {sum(history['successes'])}\n")

    save_training_curve(history, os.path.join(args.outdir, "training_curve.png"))

    print(f"\nSaved Q-table, config, and stats to '{args.outdir}/'.")
    print("Next: run  python evaluate.py  to evaluate the trained agent.")


if __name__ == "__main__":
    main()
