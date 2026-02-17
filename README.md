# trading-model

RL intraday trading environment built with Gymnasium.

This project provides a modular single-stock intraday trading environment for Reinforcement Learning. It implements a composition-based architecture where the main environment (`TradingEnv`) integrates specialized components for feature engineering, action mapping, reward calculation, and friction modeling.

## Key Features

- **Gymnasium API**: Fully compatible with the standard Reinforcement Learning interface.
- **26-Dim Observation Space**: Combines 14 intraday market features, 8 daily context features (90-day window), and 4 agent state features.
- **Walkthrough Training**: Sequential "Train on Day T, Eval on Day T+1" strategy for temporal robustness.
- **Modular Design**: Swappable components for `FrictionModel`, `ActionMapper`, `FeatureEngine`, and `RewardCalculator`.
- **Flexible Action Space**: Discrete actions with position sizing (Buy/Sell/Hold at 25%, 50%, and 100% levels).
- **Risk-Aware Rewards**: Sharpe-hybrid reward function with a penalty for increases in drawdown.
- **Database Driven**: Ingestion from MariaDB using **SQLAlchemy** (supports `_historical` and `_daily` tables).
- **Hyperparameter Tuning**: Integrated **Optuna** support for automated optimization.
- **CLI Utilities**: Click-powered interface for training, walkthroughs, tuning, and data export.

## Project Structure

```text
t1/
├── pyproject.toml                 # Project metadata and dependencies
├── SUMMARY.md                     # Detailed feature summary
├── main.py                        # CLI Entry point
├── src/
│   └── trading_model/
│       ├── __init__.py            # Public package exports
│       ├── train.py               # RL Training strategies (SB3 RecurrentPPO + Optuna)
│       ├── env/                   # Gymnasium Environment implementation
│       │   ├── trading_env.py     # Main TradingEnv(gymnasium.Env)
│       │   ├── features.py        # FeatureEngine (Intraday + Daily Context)
│       │   ├── actions.py         # ActionMapper (Position Sizing)
│       │   ├── rewards.py         # RewardCalculator (Sharpe + Drawdown)
│       │   └── frictions.py       # FrictionModel (Transaction Costs)
│       └── data/
│           └── loader.py          # SQLAlchemy Data utilities
├── tests/                         # Comprehensive test suite
└── export_stockdb.py              # Data export utility
```

## Installation

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync --all-extras

# Install in development mode
pip install -e .
```

## Usage

### Walkthrough Strategy (Recommended)
Train the model day-by-day and evaluate on the subsequent day:
```bash
python main.py walkthrough --ticker NVDA --start_date 2024-03-01 --end_date 2024-03-31
```

### Hyperparameter Tuning
Run an Optuna study to find best parameters:
```bash
python main.py tune-cmd --ticker NVDA --n_trials 20
```

### Monitoring
Training progress is logged for TensorBoard:
```bash
tensorboard --logdir ./tensorboard_logs/
```

## Development

### Running Tests
```bash
pytest
```
