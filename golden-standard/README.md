# Overview 📖

Reinforcement learning (RL) promises to solve long-horizon tasks even when training data contains only short fragments of behaviors. This capability, called stitching, is a crucial prerequisite for more general, foundational RL models. Conventional wisdom holds that only temporal difference (TD) methods can stitch fragments of experiences gathered during training. We show that while TD methods can stitch experiences in simple, low-dimensional settings, this behavior does not transfer to more complex, high-dimensional tasks. We also show that Monte Carlo (MC) methods, although still behind TD methods, can exhibit some stitching behavior. Furthermore, we find that increasing network capacity plays a critical role in closing the generalization gap, which is an encouraging direction as models grow larger in RL.

<p align="center">
    <img src="assets/stitching_1_success.gif" alt="Box Moving Task Illustration" width="600"/>
    <img src="assets/stitching_4_failure.gif" alt="Box Moving Task Illustration" width="600"/>
    <p style="text-align:center;font-size:2.0em;margin-top:8px;">
    TD methods can stitch in low-dimensional tasks (left), but fail in higher-dimensional settings (right).
    </p>
</p>

# Installation & Setup 🔧

This repo uses uv. To install uv, follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
To install all dependencies and create the virtual environment, run:
```bash
uv sync
```
> [!NOTE]
> We use wandb for experiment tracking by default and you may be prompted to log in when running your first experiment. If you do not want to use wandb, pass the flag `--exp.mode disabled` to skip wandb logging.

# Running experiments 🔬

To run a simple training with CRL and the default environment configuration:
> [!WARNING]
> This repository is optimized for GPU. Running the command below without a capable GPU may be very slow.
```bash
uv run src/train.py env:box-moving --exp.name test
```

The current version supports only the `box-moving` environment; specify `env:box-moving` for each experiment.

## Wandb logging 📈

When wandb logging is enabled, experiment results (including environment and algorithm data) are logged to wandb. A short GIF of the agent's behavior is also recorded and stored in wandb.

## Hyperparameters ⚙️

List of hyperparameters and options is available via:
```bash
uv run src/train.py --help
```
Options are grouped by prefixes:
* `exp.` - General experimental settings (logging, seeds, experiment names, etc.). See `./src/config.py` for details.
* `env.` - Environment settings (difficulty, goal/start distributions, number of boxes, grid size, etc.). See `BoxMovingConfig` in `./src/envs/block_moving/env_types.py`.
* `actor.` - Algorithm settings (learning rates, batch sizes, network architectures, and algorithm choices). See `./src/impls/agents/__init__.py`.

# Environment 🕹️

The Box Moving environment is a grid-world where an agent moves boxes to target locations. It supports different grid sizes, numbers of boxes, and difficulty levels. While simple conceptually, complexity grows rapidly with grid size and box count, making it well suited for testing stitching capabilities.

The environment supports two modes for sampling box and goal positions (set via `--env.level_generator`):
* `default` - Boxes and targets are spawned randomly on the grid.
* `variable` - Boxes and targets are spawned in grid corners. Under normal evaluation the box and goal corners are adjacent. If `--exp.eval_special` is passed, the algorithm is additionally evaluated with box and goal corners diagonally opposite; results from this mode are logged in wandb under the `eval_special` tab.

# Supported algorithms 🧠

We focus on goal-conditioned RL algorithms and, to isolate stitching behaviors, remove policy networks from tested algorithms. Actions are sampled directly from the Q-function via softmax sampling. The main algorithms include:
* Contrastive RL (CRL) — a Monte Carlo (MC) style algorithm running without rewards.
* C-Learning — a TD algorithm running without rewards.
* GCDQN — TD and MC variants, with rewards.
* GCIQL — TD and MC variants, with rewards.

The paper includes `clearn_search`, `crl_search`, `gciql_search`, and `gcdqn`. These do NOT include an agent network. We also include some other algorithms that were not thoroughly tested and may not be fully compatible with the latest code.

## Also see 👀
* [OGBench](https://github.com/seohongpark/ogbench) — benchmark for offline goal-conditioned RL algorithms, which inspired parts of our code structure.
* [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) — online goal-conditioned RL benchmark with various algorithms implemented in JAX.
* [Jumanji](https://github.com/instadeepai/jumanji) — collection of RL environments in JAX; Sokoban inspired aspects of our box-moving environment.
