# Project Feature Summary: Trading Model RL

This document summarizes the current features and architectural state of the `trading-model` project.

## 1. Core Environment (`TradingEnv`)
A high-fidelity intraday trading environment built using the **Gymnasium** API.
- **Composition-based Architecture**: Modularized logic for features, actions, rewards, and frictions.
- **Dynamic Data Injection**: Supports injecting specific intraday datasets and daily contexts via `reset(options=...)`.
- **Automatic Liquidation**: Forces position closure at the end of every trading session (episode).

## 2. Advanced Observation Space (26 Dimensions)
The model observes a comprehensive state vector combining short-term and long-term signals:
- **Intraday Market (14-D)**: Normalized price levels (OHLC), Log volume, MACD (3), RSI, Bollinger Bands (3), and Sin/Cos time embeddings.
- **Rolling Daily Context (8-D)**: Recomputed at every intraday step by synthesizing a virtual daily bar from the day's bars seen so far and appending it to the trailing 90 days. This includes MACD, RSI, BBands, Daily Return, and Volume Ratio.
- **Agent State (4-D)**: Current position (binary), Unrealized PnL %, Cash-to-Value ratio, and Trade duration.

## 3. Action & Reward Logic
- **Action Space**: Discrete (7 actions).
    - `0`: Hold
    - `1, 2, 3`: Buy (25%, 50%, 100% of available cash)
    - `4, 5, 6`: Sell (25%, 50%, 100% of current position)
- **Reward Function**: A Sharpe-hybrid formulation:
    - $R_t = \alpha \cdot r_t + (1 - \alpha) \cdot 	ext{Sharpe}_{	ext{rolling}} - \beta \cdot \max(0, DD_t - DD_{t-1})$
    - Balances raw returns with risk-adjusted performance and explicitly penalizes drawdown increases.

## 4. Training Strategies
- **Walkthrough Training (Default)**: A temporal validation strategy where the model trains on Day $T$ and is immediately evaluated on Day $T+1$. This ensures the model adapts to market regime changes over time.
- **Standard Training**: Traditional RL training cycling through a shuffled set of historical days.
- **Hyperparameter Tuning**: Fully integrated **Optuna** support to optimize learning rates, batch sizes, and architectural parameters.

## 5. Data & Infrastructure
- **SQLAlchemy Ingestion**: Robust data pipeline fetching from MariaDB `stockdb`. Supports `_historical` (intraday) and `_daily` (interday) table schemas.
- **Graceful Termination**: `KeyboardInterrupt` handler ensures that both the model weights (`.zip`) and normalization statistics (`.pkl`) are saved if training is manually stopped (Ctrl+C).
- **CLI Utilities**:
    - **Trading Model CLI** (`main.py`):
        - `walkthrough`: Execute the iterative train/eval strategy.
        - `tune-cmd`: Launch an Optuna optimization study.
        - `evaluate-cmd`: Evaluate a trained model on a specific date range.
    - **Data Export** (`export_stockdb.py`): Standalone utility to export database tables to local CSV files.

## 6. Development & Verification
- **Testing**: Comprehensive test suite using `pytest` covering units (actions, rewards, features) and integrations (episode lifecycle).
- **Environment**: Managed via `uv` or `pip`, requiring Python 3.13+.
