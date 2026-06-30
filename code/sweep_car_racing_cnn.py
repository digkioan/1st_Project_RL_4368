import argparse
import csv
import json
import time

from car_racing_dqn_cnn import Agent


# These 10 candidates are meant for staged screening rather than full-length final training.
CANDIDATES = [
    {
        "name": "cnn_baseline",
        "description": "Current baseline from hyperparameters.yml.",
        "overrides": {},
    },
    {
        "name": "cnn_low_lr_stable",
        "description": "Lower learning rate with slower target syncing.",
        "overrides": {
            "learning_rate_a": 0.00005,
            "network_sync_rate": 2000,
            "epsilon_decay": 0.997,
        },
    },
    {
        "name": "cnn_fast_updates",
        "description": "Higher learning rate with smaller batches and faster syncing.",
        "overrides": {
            "learning_rate_a": 0.0002,
            "mini_batch_size": 32,
            "network_sync_rate": 500,
        },
    },
    {
        "name": "cnn_large_replay",
        "description": "Larger replay memory with larger batches and lower epsilon floor.",
        "overrides": {
            "replay_memory_size": 100000,
            "mini_batch_size": 128,
            "epsilon_min": 0.02,
        },
    },
    {
        "name": "cnn_long_horizon",
        "description": "Higher gamma with slower exploration decay and slower syncing.",
        "overrides": {
            "discount_factor_g": 0.995,
            "epsilon_decay": 0.997,
            "network_sync_rate": 5000,
        },
    },
    {
        "name": "cnn_large_batch_stable",
        "description": "Lower learning rate with large batches and moderate syncing.",
        "overrides": {
            "learning_rate_a": 0.00005,
            "mini_batch_size": 128,
            "network_sync_rate": 3000,
        },
    },
    {
        "name": "cnn_more_exploration",
        "description": "Longer exploration with a bigger replay buffer.",
        "overrides": {
            "epsilon_decay": 0.997,
            "epsilon_min": 0.02,
            "replay_memory_size": 100000,
        },
    },
    {
        "name": "cnn_lower_gamma_quick_sync",
        "description": "Shorter-horizon value targets with faster target updates.",
        "overrides": {
            "discount_factor_g": 0.97,
            "network_sync_rate": 500,
            "mini_batch_size": 64,
        },
    },
    {
        "name": "cnn_low_lr_large_replay",
        "description": "Lower learning rate with more replay capacity.",
        "overrides": {
            "learning_rate_a": 0.00005,
            "replay_memory_size": 100000,
            "mini_batch_size": 64,
        },
    },
    {
        "name": "cnn_balanced_batch128",
        "description": "Baseline learning rate with large batches and slower syncing.",
        "overrides": {
            "learning_rate_a": 0.0001,
            "mini_batch_size": 128,
            "network_sync_rate": 2000,
        },
    },
]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std_dev(values):
    if len(values) < 2:
        return 0.0

    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return variance ** 0.5


def stage_sort_key(test_episodes):
    return "mean_test_score" if test_episodes > 0 else "mean_train_score"


def apply_overrides(agent, overrides, run_name, stage_name):
    for key, value in overrides.items():
        setattr(agent, key, value)

    sweep_dir = agent.runs_dir / "sweeps" / "cnn" / stage_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    agent.model_path = sweep_dir / f"{run_name}_model.pt"
    agent.plot_path = sweep_dir / f"{run_name}_training.png"
    return sweep_dir


def run_candidate(agent_name, candidate, seeds, episodes, test_episodes, stage_name, timestamp):
    results = []
    sweep_dir = None

    for seed in seeds:
        run_name = f"{candidate['name']}_seed{seed}_{timestamp}"
        agent = Agent(agent_name)
        sweep_dir = apply_overrides(agent, candidate["overrides"], run_name, stage_name)

        start_time = time.perf_counter()
        train_score = agent.run(
            is_training=True,
            render=False,
            max_episodes=episodes,
            seed=seed,
        )
        duration_minutes = (time.perf_counter() - start_time) / 60.0

        test_score = None
        if test_episodes > 0:
            test_score = agent.run(
                is_training=False,
                render=False,
                test_episodes=test_episodes,
                seed=10000 + seed,
            )

        results.append(
            {
                "candidate": candidate["name"],
                "description": candidate["description"],
                "seed": seed,
                "train_score": train_score,
                "test_score": test_score,
                "duration_minutes": duration_minutes,
                "episodes": episodes,
                "overrides": json.dumps(candidate["overrides"], sort_keys=True),
            }
        )

    return results, sweep_dir


def build_summary(results, candidates):
    summary_rows = []

    for candidate in candidates:
        candidate_rows = [row for row in results if row["candidate"] == candidate["name"]]
        train_scores = [row["train_score"] for row in candidate_rows]
        test_scores = [row["test_score"] for row in candidate_rows if row["test_score"] is not None]
        durations = [row["duration_minutes"] for row in candidate_rows]

        summary_rows.append(
            {
                "candidate": candidate["name"],
                "description": candidate["description"],
                "mean_train_score": mean(train_scores),
                "std_train_score": std_dev(train_scores),
                "mean_test_score": mean(test_scores) if test_scores else None,
                "mean_duration_minutes": mean(durations),
                "overrides": json.dumps(candidate["overrides"], sort_keys=True),
            }
        )

    return summary_rows


def write_raw_results(results, output_path):
    fieldnames = [
        "candidate",
        "description",
        "seed",
        "train_score",
        "test_score",
        "duration_minutes",
        "episodes",
        "overrides",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def write_summary(summary_rows, output_path):
    fieldnames = [
        "candidate",
        "description",
        "mean_train_score",
        "std_train_score",
        "mean_test_score",
        "mean_duration_minutes",
        "overrides",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def save_stage_outputs(stage_name, results, summary_rows, sweep_dir, timestamp):
    raw_results_path = sweep_dir / f"{stage_name}_raw_{timestamp}.csv"
    summary_path = sweep_dir / f"{stage_name}_summary_{timestamp}.csv"
    write_raw_results(results, raw_results_path)
    write_summary(summary_rows, summary_path)
    return raw_results_path, summary_path


def print_ranking(title, summary_rows):
    print()
    print(title)
    for index, row in enumerate(summary_rows, start=1):
        test_part = ""
        if row["mean_test_score"] is not None:
            test_part = f" | mean_test={row['mean_test_score']:.2f}"

        print(
            f"{index}. {row['candidate']} | "
            f"mean_train={row['mean_train_score']:.2f} | "
            f"std_train={row['std_train_score']:.2f}"
            f"{test_part} | "
            f"avg_minutes={row['mean_duration_minutes']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Two-stage hyperparameter sweep for CNN CarRacing DQN.")
    parser.add_argument(
        "--screen-episodes",
        type=int,
        default=50,
        help="Episodes per run during stage 1 screening.",
    )
    parser.add_argument(
        "--screen-seeds",
        nargs="*",
        type=int,
        default=[0],
        help="Seeds for stage 1 screening. Default: 0",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many top configs from stage 1 to rerun in stage 2.",
    )
    parser.add_argument(
        "--rerun-seeds",
        nargs="*",
        type=int,
        default=[0, 1, 2],
        help="Seeds for stage 2 reruns. Default: 0 1 2",
    )
    parser.add_argument(
        "--rerun-episodes",
        type=int,
        default=None,
        help="Episodes per run during stage 2. Defaults to max(200, 4 * --screen-episodes).",
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=0,
        help="Optional test episodes after each training run. Use 0 to skip testing.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of candidate names to run in stage 1.",
    )
    args = parser.parse_args()

    selected_candidates = CANDIDATES
    if args.only:
        requested = set(args.only)
        selected_candidates = [candidate for candidate in CANDIDATES if candidate["name"] in requested]

    if not selected_candidates:
        raise ValueError("No candidate names matched --only.")
    if not args.screen_seeds:
        raise ValueError("Please provide at least one stage 1 seed.")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative.")
    if args.top_k > 0 and not args.rerun_seeds:
        raise ValueError("Please provide at least one stage 2 seed when --top-k is greater than 0.")

    rerun_episodes = args.rerun_episodes if args.rerun_episodes is not None else max(200, args.screen_episodes * 4)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    sort_key = stage_sort_key(args.test_episodes)

    print("Stage 1: screening CNN DQN candidates.")
    print(
        f"Candidates: {len(selected_candidates)} | screen seeds: {args.screen_seeds} | "
        f"episodes per run: {args.screen_episodes}"
    )

    stage1_results = []
    stage1_dir = None
    for candidate in selected_candidates:
        print()
        print(f"=== Stage 1 | {candidate['name']} ===")
        print(candidate["description"])
        print(f"Overrides: {candidate['overrides']}")

        candidate_results, stage1_dir = run_candidate(
            agent_name="car_racing_cnn",
            candidate=candidate,
            seeds=args.screen_seeds,
            episodes=args.screen_episodes,
            test_episodes=args.test_episodes,
            stage_name="stage1",
            timestamp=timestamp,
        )
        stage1_results.extend(candidate_results)

    stage1_summary = build_summary(stage1_results, selected_candidates)
    stage1_summary.sort(
        key=lambda row: row[sort_key] if row[sort_key] is not None else float("-inf"),
        reverse=True,
    )
    stage1_raw_path, stage1_summary_path = save_stage_outputs(
        "cnn_stage1",
        stage1_results,
        stage1_summary,
        stage1_dir,
        timestamp,
    )
    print_ranking("Stage 1 ranking:", stage1_summary)

    if args.top_k == 0:
        print()
        print(f"Saved stage 1 raw results to {stage1_raw_path}")
        print(f"Saved stage 1 summary to {stage1_summary_path}")
        return

    top_count = min(args.top_k, len(stage1_summary))
    top_candidate_names = [row["candidate"] for row in stage1_summary[:top_count]]
    top_candidates = [candidate for candidate in selected_candidates if candidate["name"] in top_candidate_names]
    top_candidates.sort(key=lambda candidate: top_candidate_names.index(candidate["name"]))

    print()
    print("Stage 2: rerunning the top candidates with multiple seeds.")
    print(
        f"Top configs: {top_candidate_names} | rerun seeds: {args.rerun_seeds} | "
        f"episodes per run: {rerun_episodes}"
    )

    stage2_results = []
    stage2_dir = None
    for candidate in top_candidates:
        print()
        print(f"=== Stage 2 | {candidate['name']} ===")
        print(candidate["description"])
        print(f"Overrides: {candidate['overrides']}")

        candidate_results, stage2_dir = run_candidate(
            agent_name="car_racing_cnn",
            candidate=candidate,
            seeds=args.rerun_seeds,
            episodes=rerun_episodes,
            test_episodes=args.test_episodes,
            stage_name="stage2",
            timestamp=timestamp,
        )
        stage2_results.extend(candidate_results)

    stage2_summary = build_summary(stage2_results, top_candidates)
    stage2_summary.sort(
        key=lambda row: row[sort_key] if row[sort_key] is not None else float("-inf"),
        reverse=True,
    )
    stage2_raw_path, stage2_summary_path = save_stage_outputs(
        "cnn_stage2",
        stage2_results,
        stage2_summary,
        stage2_dir,
        timestamp,
    )
    print_ranking("Stage 2 ranking:", stage2_summary)

    print()
    print(f"Saved stage 1 raw results to {stage1_raw_path}")
    print(f"Saved stage 1 summary to {stage1_summary_path}")
    print(f"Saved stage 2 raw results to {stage2_raw_path}")
    print(f"Saved stage 2 summary to {stage2_summary_path}")
    print("Use the stage 2 winner for your longer final training run.")


if __name__ == "__main__":
    main()
