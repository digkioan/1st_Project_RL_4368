import argparse
import time
from pathlib import Path

import gymnasium as gym
import torch

from car_racing_dqn_cnn import Agent as CnnAgent
from car_racing_dqn_flatten import Agent as FlattenAgent


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FLATTEN_MODEL = PROJECT_DIR / "runs" / "car_racing_flatten_model.pt"
DEFAULT_CNN_MODEL = PROJECT_DIR / "runs" / "car_racing_cnn_model.pt"
DEFAULT_CNN_CHECKPOINTS_DIR = PROJECT_DIR / "runs" / "car_racing_cnn_checkpoints"


def preprocess_state(agent, observation, reset=False):
    try:
        return agent.preprocess_state(observation, reset=reset)
    except TypeError:
        return agent.preprocess_state(observation)


def build_missing_model_message(label, model_path):
    message = [f"Model file not found for {label}: {model_path}"]

    if "CNN" in label:
        suggestions = sorted((PROJECT_DIR / "runs" / "sweeps" / "cnn").rglob("*_model.pt"))
        if suggestions:
            message.append("Available CNN model files you can pass with --cnn-model-path:")
            for suggestion in suggestions[:10]:
                message.append(f"  - {suggestion}")

    return "\n".join(message)


def collect_cnn_model_paths(cnn_model_path, checkpoints_dir):
    model_paths = []

    cnn_model_path = cnn_model_path.resolve()
    checkpoints_dir = checkpoints_dir.resolve()

    if cnn_model_path.exists():
        model_paths.append(("CNN DQN", cnn_model_path))

    if checkpoints_dir.exists():
        checkpoint_paths = sorted(path.resolve() for path in checkpoints_dir.rglob("*.pt"))
        for checkpoint_path in checkpoint_paths:
            try:
                relative_path = checkpoint_path.relative_to(PROJECT_DIR)
            except ValueError:
                relative_path = checkpoint_path
            model_paths.append((f"CNN checkpoint ({relative_path})", checkpoint_path))

    return model_paths


def infer_success(info):
    return bool(info.get("lap_finished", False))


def build_status_label(success, timed_out, terminated, truncated, info):
    if success:
        return "SUCCESS"
    if timed_out:
        return "EPISODE_TIMEOUT"
    if truncated:
        return "ENV_TIME_LIMIT"
    if terminated and "lap_finished" in info and not info["lap_finished"]:
        return "OFF_TRACK"
    if terminated:
        return "TERMINATED"
    return "UNKNOWN"


def evaluate_model(
    label,
    agent,
    model_path,
    episodes,
    render,
    seed,
    lap_complete_percent,
    episode_pause,
    max_episode_seconds,
    max_env_steps,
):
    if not model_path.exists():
        raise FileNotFoundError(build_missing_model_message(label, model_path))

    if seed is not None:
        agent.set_seed(seed)

    env = gym.make(
        agent.env_id,
        continuous=agent.continuous,
        render_mode="human" if render else None,
        lap_complete_percent=lap_complete_percent,
        max_episode_steps=max_env_steps,
    )
    num_actions = env.action_space.n
    print(f"\n=== Evaluating {label} ===")
    print(f"Model: {model_path}")
    print(f"Using device: {agent.device}")

    if seed is not None:
        env.action_space.seed(seed)

    policy_dqn = agent.build_network(num_actions)
    policy_dqn.load_state_dict(torch.load(model_path, map_location=agent.device))
    policy_dqn.eval()

    successes = 0
    rewards = []
    completion_times = []

    for episode in range(episodes):
        episode_seed = None if seed is None else seed + episode
        observation, _ = env.reset(seed=episode_seed)
        state = preprocess_state(agent, observation, reset=True)

        terminated = False
        truncated = False
        episode_reward = 0.0
        step_count = 0
        last_info = {}
        timed_out = False
        episode_start_time = time.perf_counter()

        while not terminated and not truncated:
            elapsed_seconds = time.perf_counter() - episode_start_time
            if elapsed_seconds >= max_episode_seconds:
                timed_out = True
                break

            with torch.no_grad():
                action = policy_dqn(state.unsqueeze(0)).argmax(dim=1).item()

            next_observation, reward, terminated, truncated, info = env.step(action)
            last_info = info
            state = preprocess_state(agent, next_observation, reset=False)
            episode_reward += reward
            step_count += 1

            elapsed_seconds = time.perf_counter() - episode_start_time
            if elapsed_seconds >= max_episode_seconds:
                timed_out = True
                break

        success = infer_success(last_info)
        if timed_out:
            success = False

        episode_duration_seconds = time.perf_counter() - episode_start_time
        status = build_status_label(success, timed_out, terminated, truncated, last_info)
        successes += int(success)
        rewards.append(episode_reward)
        if success:
            completion_times.append(episode_duration_seconds)

        message = (
            f"{label} | Episode {episode + 1}/{episodes} | "
            f"status={status} | reward={episode_reward:.2f} | "
            f"steps={step_count} | elapsed={episode_duration_seconds:.1f}s | "
            f"successes={successes}"
        )
        if success:
            message += f" | lap_time={episode_duration_seconds:.1f}s"
        print(message)

        if render and episode_pause > 0 and episode < episodes - 1:
            time.sleep(episode_pause)

    env.close()

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    mean_completion_time = sum(completion_times) / len(completion_times) if completion_times else None
    best_completion_time = min(completion_times) if completion_times else None
    completion_summary = "no completed laps"
    if completion_times:
        completion_summary = (
            f"mean_lap_time={mean_completion_time:.1f}s | "
            f"best_lap_time={best_completion_time:.1f}s"
        )

    print(
        f"{label} summary | successes={successes}/{episodes} | "
        f"success_rate={100.0 * successes / episodes:.1f}% | "
        f"mean_reward={mean_reward:.2f} | {completion_summary}"
    )
    return {
        "label": label,
        "successes": successes,
        "episodes": episodes,
        "mean_reward": mean_reward,
        "completion_times": completion_times,
        "mean_completion_time": mean_completion_time,
        "best_completion_time": best_completion_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the flattened and CNN CarRacing models with optional human rendering."
    )
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to run per model. Default: 5")
    parser.add_argument(
        "--only",
        choices=["all", "flatten", "cnn"],
        default="all",
        help="Choose which models to evaluate. Default: all",
    )
    parser.add_argument(
        "--flatten-model-path",
        type=Path,
        default=DEFAULT_FLATTEN_MODEL,
        help=f"Path to the flattened model. Default: {DEFAULT_FLATTEN_MODEL}",
    )
    parser.add_argument(
        "--cnn-model-path",
        type=Path,
        default=DEFAULT_CNN_MODEL,
        help=f"Path to the CNN model. Default: {DEFAULT_CNN_MODEL}",
    )
    parser.add_argument(
        "--cnn-checkpoints-dir",
        type=Path,
        default=DEFAULT_CNN_CHECKPOINTS_DIR,
        help=f"Directory containing CNN checkpoints to evaluate. Default: {DEFAULT_CNN_CHECKPOINTS_DIR}",
    )
    parser.add_argument(
        "--lap-complete-percent",
        type=float,
        default=0.99,
        help="Lap completion threshold used when creating the environment. Default: 0.99",    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed. Each episode uses seed + episode_index.",
    )
    parser.add_argument(
        "--episode-pause",
        type=float,
        default=1.0,
        help="Seconds to pause between episodes when rendering. Default: 1.0",
    )
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=90.0,
        help="Maximum wall-clock seconds per episode before it is counted as a failure. Default: 90",
    )
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=15_000,
        help=(
            "Maximum environment steps per episode (Gym TimeLimit). "
            "Increase this to avoid status=ENV_TIME_LIMIT. Default: 15000"
        ),
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable human rendering.",
    )
    args = parser.parse_args()

    render = not args.no_render

    results = []
    if args.only in {"all", "flatten"}:
        results.append(
            evaluate_model(
                label="Flatten DQN",
                agent=FlattenAgent("car_racing_flatten"),
                model_path=args.flatten_model_path,
                episodes=args.episodes,
                render=render,
                seed=args.seed,
                lap_complete_percent=args.lap_complete_percent,
                episode_pause=args.episode_pause,
                max_episode_seconds=args.max_episode_seconds,
                max_env_steps=args.max_env_steps,
            )
        )

    if args.only in {"all", "cnn"}:
        cnn_model_paths = collect_cnn_model_paths(args.cnn_model_path, args.cnn_checkpoints_dir)
        if not cnn_model_paths:
            raise FileNotFoundError(
                f"No CNN models found at {args.cnn_model_path} or under {args.cnn_checkpoints_dir}"
            )

        for index, (label, model_path) in enumerate(cnn_model_paths):
            results.append(
                evaluate_model(
                    label=label,
                    agent=CnnAgent("car_racing_cnn"),
                    model_path=model_path,
                    episodes=args.episodes,
                    render=render,
                    seed=None if args.seed is None else args.seed + 10_000 + index * 1_000,
                    lap_complete_percent=args.lap_complete_percent,
                    episode_pause=args.episode_pause,
                    max_episode_seconds=args.max_episode_seconds,
                    max_env_steps=args.max_env_steps,
                )
            )

    print("\n=== Final comparison ===")
    for result in results:
        completion_summary = "no completed laps"
        if result["best_completion_time"] is not None:
            completion_summary = (
                f"mean_lap_time={result['mean_completion_time']:.1f}s | "
                f"best_lap_time={result['best_completion_time']:.1f}s"
            )
        print(
            f"{result['label']}: "
            f"{result['successes']}/{result['episodes']} successes | "
            f"mean_reward={result['mean_reward']:.2f} | "
            f"{completion_summary}"
        )


if __name__ == "__main__":
    main()



# RENDERED

# === Evaluating Flatten DQN ===
# Model: /home/ioannis/home/ioannis/shared/ΑΠΘ/8ο Εξάμηνο/ΒΑΘΙΑ ΕΝΙΣΧΥΤΙΚΗ ΜΑΘΗΣΗ/rl_projects/1st_Project_RL_4368/runs/car_racing_flatten_model.pt
# Using device: cpu
# Flatten DQN | Episode 1/5 | status=SUCCESS | reward=791.10 | steps=2089 | elapsed=42.6s | successes=1 | lap_time=42.6s
# Flatten DQN | Episode 2/5 | status=SUCCESS | reward=828.40 | steps=1716 | elapsed=34.7s | successes=2 | lap_time=34.7s
# Flatten DQN | Episode 3/5 | status=SUCCESS | reward=429.70 | steps=5703 | elapsed=117.0s | successes=3 | lap_time=117.0s
# Flatten DQN | Episode 4/5 | status=SUCCESS | reward=882.90 | steps=1171 | elapsed=24.1s | successes=4 | lap_time=24.1s
# Flatten DQN | Episode 5/5 | status=SUCCESS | reward=739.40 | steps=2606 | elapsed=54.0s | successes=5 | lap_time=54.0s
# Flatten DQN summary | successes=5/5 | success_rate=100.0% | mean_reward=734.30 | mean_lap_time=54.5s | best_lap_time=24.1s


# === Evaluating CNN DQN ===
# Model: /home/ioannis/home/ioannis/shared/ΑΠΘ/8ο Εξάμηνο/ΒΑΘΙΑ ΕΝΙΣΧΥΤΙΚΗ ΜΑΘΗΣΗ/rl_projects/1st_Project_RL_4368/runs/car_racing_cnn_model.pt
# Using device: cpu
# CNN DQN | Episode 1/5 | status=SUCCESS | reward=823.20 | steps=1768 | elapsed=60.9s | successes=1 | lap_time=60.9s
# CNN DQN | Episode 2/5 | status=SUCCESS | reward=792.70 | steps=2073 | elapsed=65.9s | successes=2 | lap_time=65.9s
# CNN DQN | Episode 3/5 | status=SUCCESS | reward=831.50 | steps=1685 | elapsed=54.6s | successes=3 | lap_time=54.6s
# CNN DQN | Episode 4/5 | status=EPISODE_TIMEOUT | reward=706.79 | steps=2728 | elapsed=90.0s | successes=3
# CNN DQN | Episode 5/5 | status=SUCCESS | reward=755.20 | steps=2448 | elapsed=79.5s | successes=4 | lap_time=79.5s
# CNN DQN summary | successes=4/5 | success_rate=80.0% | mean_reward=781.88 | mean_lap_time=65.2s | best_lap_time=54.6s




#NO RENDER


# === Evaluating Flatten DQN ===
# Model: /home/ioannis/home/ioannis/shared/ΑΠΘ/8ο Εξάμηνο/ΒΑΘΙΑ ΕΝΙΣΧΥΤΙΚΗ ΜΑΘΗΣΗ/rl_projects/1st_Project_RL_4368/runs/car_racing_flatten_model.pt
# Using device: cpu
# Flatten DQN | Episode 1/5 | status=SUCCESS | reward=820.10 | steps=1799 | elapsed=16.6s | successes=1 | lap_time=16.6s
# Flatten DQN | Episode 2/5 | status=SUCCESS | reward=559.10 | steps=4409 | elapsed=42.3s | successes=2 | lap_time=42.3s
# Flatten DQN | Episode 3/5 | status=SUCCESS | reward=885.90 | steps=1141 | elapsed=11.0s | successes=3 | lap_time=11.0s
# Flatten DQN | Episode 4/5 | status=SUCCESS | reward=828.40 | steps=1716 | elapsed=15.7s | successes=4 | lap_time=15.7s
# Flatten DQN | Episode 5/5 | status=EPISODE_TIMEOUT | reward=-232.79 | steps=9963 | elapsed=90.0s | successes=4
# Flatten DQN summary | successes=4/5 | success_rate=80.0% | mean_reward=572.14 | mean_lap_time=21.4s | best_lap_time=11.0s

# === Final comparison ===
# Flatten DQN: 4/5 successes | mean_reward=572.14 | mean_lap_time=21.4s | best_lap_time=11.0s


# === Evaluating CNN DQN ===
# Model: /home/ioannis/home/ioannis/shared/ΑΠΘ/8ο Εξάμηνο/ΒΑΘΙΑ ΕΝΙΣΧΥΤΙΚΗ ΜΑΘΗΣΗ/rl_projects/1st_Project_RL_4368/runs/car_racing_cnn_model.pt
# Using device: cpu
# CNN DQN | Episode 1/5 | status=SUCCESS | reward=847.80 | steps=1522 | elapsed=14.6s | successes=1 | lap_time=14.6s
# CNN DQN | Episode 2/5 | status=SUCCESS | reward=895.10 | steps=1049 | elapsed=9.7s | successes=2 | lap_time=9.7s
# CNN DQN | Episode 3/5 | status=SUCCESS | reward=891.40 | steps=1086 | elapsed=10.3s | successes=3 | lap_time=10.3s
# CNN DQN | Episode 4/5 | status=SUCCESS | reward=773.20 | steps=2268 | elapsed=21.9s | successes=4 | lap_time=21.9s
# CNN DQN | Episode 5/5 | status=SUCCESS | reward=876.60 | steps=1234 | elapsed=12.3s | successes=5 | lap_time=12.3s
# CNN DQN summary | successes=5/5 | success_rate=100.0% | mean_reward=856.82 | mean_lap_time=13.8s | best_lap_time=9.7s

# === Evaluating CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_1000.pt) ===
# Model: /home/ioannis/home/ioannis/shared/ΑΠΘ/8ο Εξάμηνο/ΒΑΘΙΑ ΕΝΙΣΧΥΤΙΚΗ ΜΑΘΗΣΗ/rl_projects/1st_Project_RL_4368/runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_1000.pt
# Using device: cpu
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_1000.pt) | Episode 1/5 | status=SUCCESS | reward=867.80 | steps=1322 | elapsed=12.6s | successes=1 | lap_time=12.6s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_1000.pt) | Episode 2/5 | status=SUCCESS | reward=682.70 | steps=3173 | elapsed=30.8s | successes=2 | lap_time=30.8s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_1000.pt) | Episode 3/5 | status=SUCCESS | reward=882.20 | steps=1178 | elapsed=11.5s | successes=3 | lap_time=11.5s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_1000.pt) | Episode 4/5 | status=SUCCESS | reward=797.20 | steps=2028 | elapsed=19.9s | successes=4 | lap_time=19.9s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_1000.pt) | Episode 5/5 | status=SUCCESS | reward=899.20 | steps=1008 | elapsed=9.9s | successes=5 | lap_time=9.9s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_1000.pt) summary | successes=5/5 | success_rate=100.0% | mean_reward=825.82 | mean_lap_time=17.0s | best_lap_time=9.9s

# === Evaluating CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_250.pt) ===
# Model: /home/ioannis/home/ioannis/shared/ΑΠΘ/8ο Εξάμηνο/ΒΑΘΙΑ ΕΝΙΣΧΥΤΙΚΗ ΜΑΘΗΣΗ/rl_projects/1st_Project_RL_4368/runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_250.pt
# Using device: cpu
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_250.pt) | Episode 1/5 | status=EPISODE_TIMEOUT | reward=-559.98 | steps=9548 | elapsed=90.0s | successes=0
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_250.pt) | Episode 2/5 | status=EPISODE_TIMEOUT | reward=-444.86 | steps=9415 | elapsed=90.0s | successes=0
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_250.pt) | Episode 3/5 | status=EPISODE_TIMEOUT | reward=-515.74 | steps=9370 | elapsed=90.0s | successes=0
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_250.pt) | Episode 4/5 | status=EPISODE_TIMEOUT | reward=-692.98 | steps=9372 | elapsed=90.0s | successes=0
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_250.pt) | Episode 5/5 | status=EPISODE_TIMEOUT | reward=-660.57 | steps=9056 | elapsed=90.0s | successes=0
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_250.pt) summary | successes=0/5 | success_rate=0.0% | mean_reward=-574.82 | no completed laps

# === Evaluating CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_500.pt) ===
# Model: /home/ioannis/home/ioannis/shared/ΑΠΘ/8ο Εξάμηνο/ΒΑΘΙΑ ΕΝΙΣΧΥΤΙΚΗ ΜΑΘΗΣΗ/rl_projects/1st_Project_RL_4368/runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_500.pt
# Using device: cpu
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_500.pt) | Episode 1/5 | status=EPISODE_TIMEOUT | reward=35.45 | steps=9461 | elapsed=90.0s | successes=0
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_500.pt) | Episode 2/5 | status=EPISODE_TIMEOUT | reward=61.21 | steps=9248 | elapsed=90.0s | successes=0
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_500.pt) | Episode 3/5 | status=EPISODE_TIMEOUT | reward=83.49 | steps=9129 | elapsed=90.0s | successes=0
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_500.pt) | Episode 4/5 | status=EPISODE_TIMEOUT | reward=45.35 | steps=9509 | elapsed=90.0s | successes=0
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_500.pt) | Episode 5/5 | status=SUCCESS | reward=396.20 | steps=6038 | elapsed=59.5s | successes=1 | lap_time=59.5s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_500.pt) summary | successes=1/5 | success_rate=20.0% | mean_reward=124.34 | mean_lap_time=59.5s | best_lap_time=59.5s

# === Evaluating CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_750.pt) ===
# Model: /home/ioannis/home/ioannis/shared/ΑΠΘ/8ο Εξάμηνο/ΒΑΘΙΑ ΕΝΙΣΧΥΤΙΚΗ ΜΑΘΗΣΗ/rl_projects/1st_Project_RL_4368/runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_750.pt
# Using device: cpu
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_750.pt) | Episode 1/5 | status=SUCCESS | reward=663.50 | steps=3365 | elapsed=33.0s | successes=1 | lap_time=33.0s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_750.pt) | Episode 2/5 | status=SUCCESS | reward=781.80 | steps=2182 | elapsed=21.0s | successes=2 | lap_time=21.0s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_750.pt) | Episode 3/5 | status=SUCCESS | reward=650.80 | steps=3492 | elapsed=35.4s | successes=3 | lap_time=35.4s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_750.pt) | Episode 4/5 | status=SUCCESS | reward=759.70 | steps=2403 | elapsed=25.9s | successes=4 | lap_time=25.9s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_750.pt) | Episode 5/5 | status=SUCCESS | reward=634.20 | steps=3658 | elapsed=36.8s | successes=5 | lap_time=36.8s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_750.pt) summary | successes=5/5 | success_rate=100.0% | mean_reward=698.00 | mean_lap_time=30.4s | best_lap_time=21.0s

# === Final comparison ===
# CNN DQN: 5/5 successes | mean_reward=856.82 | mean_lap_time=13.8s | best_lap_time=9.7s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_1000.pt): 5/5 successes | mean_reward=825.82 | mean_lap_time=17.0s | best_lap_time=9.9s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_250.pt): 0/5 successes | mean_reward=-574.82 | no completed laps
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_500.pt): 1/5 successes | mean_reward=124.34 | mean_lap_time=59.5s | best_lap_time=59.5s
# CNN checkpoint (runs/car_racing_cnn_checkpoints/20260501-051316/car_racing_cnn_episode_750.pt): 5/5 successes | mean_reward=698.00 | mean_lap_time=30.4s | best_lap_time=21.0s