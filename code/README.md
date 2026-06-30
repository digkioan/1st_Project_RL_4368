# CarRacing DQN Project

This project trains and evaluates Deep Q-Network (DQN) agents on `CarRacing-v3` from Gymnasium, using two observation pipelines:

- `Flatten DQN`: fully connected network over flattened RGB observations.
- `CNN DQN`: convolutional network over stacked preprocessed frames.

The repository also includes two-stage hyperparameter sweep scripts for both agents and an evaluation script for model comparison.

## Project Structure

- `car_racing_dqn_flatten.py`: flattened-observation DQN training/testing.
- `car_racing_dqn_cnn.py`: CNN-based DQN training/testing with optional checkpointing.
- `evaluate_car_racing_models.py`: evaluates flatten/CNN models and CNN checkpoints.
- `sweep_car_racing_flatten.py`: two-stage sweep for flatten agent.
- `sweep_car_racing_cnn.py`: two-stage sweep for CNN agent.
- `hyperparameters.yml`: default settings for both agents.
- `tutorial_frozen_lake/`: separate tutorial scripts for FrozenLake DQL.

## Requirements

- Python `3.10+` (recommended)
- Packages:
  - `torch`
  - `gymnasium[box2d]`
  - `numpy`
  - `matplotlib`
  - `pyyaml`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch gymnasium[box2d] numpy matplotlib pyyaml
```

If Box2D dependencies fail to install on Linux, install system packages first (for example `swig`, `cmake`, and build tools), then retry `pip install`.

## Training

### Flatten DQN

```bash
python car_racing_dqn_flatten.py --mode train --max-episodes 1000 --seed 0
```

Outputs:
- model: `runs/car_racing_flatten_model.pt`
- plot: `runs/car_racing_flatten_training.png`

### CNN DQN

```bash
python car_racing_dqn_cnn.py --mode train --max-episodes 1000 --seed 0
```

Outputs:
- model: `runs/car_racing_cnn_model.pt`
- plot: `runs/car_racing_cnn_training.png`
- optional checkpoints (if enabled): `runs/car_racing_cnn_checkpoints/...`

Use `--render` in either script to watch training episodes.

## Testing a Trained Model

### Flatten DQN

```bash
python car_racing_dqn_flatten.py --mode test --test-episodes 5 --render
```

### CNN DQN

```bash
python car_racing_dqn_cnn.py --mode test --test-episodes 5 --render
```

To test a specific CNN model file:

```bash
python car_racing_dqn_cnn.py --mode test --model-path runs/some_model.pt --test-episodes 5
```

## Model Evaluation and Comparison

Evaluate both agents:

```bash
python evaluate_car_racing_models.py --episodes 5
```

Evaluate only one family:

```bash
python evaluate_car_racing_models.py --only flatten --episodes 5
python evaluate_car_racing_models.py --only cnn --episodes 5
```

Useful flags:
- `--no-render`: disable human rendering.
- `--max-episode-seconds`: timeout threshold per episode.
- `--max-env-steps`: Gym time limit per episode.
- `--cnn-checkpoints-dir`: evaluate all `.pt` checkpoints in a folder.

## Hyperparameter Sweeps

Both sweep scripts run:
1. Stage 1 screening of candidate configs.
2. Stage 2 reruns of top-K candidates with multiple seeds.

### Flatten sweep

```bash
python sweep_car_racing_flatten.py --screen-episodes 50 --screen-seeds 0 --top-k 3 --rerun-seeds 0 1 2
```

### CNN sweep

```bash
python sweep_car_racing_cnn.py --screen-episodes 50 --screen-seeds 0 --top-k 3 --rerun-seeds 0 1 2
```

Results are saved under:
- `runs/sweeps/flatten/stage1`, `runs/sweeps/flatten/stage2`
- `runs/sweeps/cnn/stage1`, `runs/sweeps/cnn/stage2`

Each stage writes raw and summary CSV files plus model checkpoints/plots.

## Notes

- Default hyperparameters are defined in `hyperparameters.yml`.
- Training artifacts are intentionally ignored in Git (`runs/`, model weights, videos) to avoid GitHub size-limit errors.
- If you need reproducible runs, always set `--seed`.
