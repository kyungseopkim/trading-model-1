# Gymnasium Trading Environment Design

## Overview

Single-stock intraday trading environment built on the Gymnasium API.
Consumes minute-level OHLCV data from `<ticker>_historical` tables (~400 bars/day).
Modular composition architecture with independently testable/swappable components.

## Project Structure

```
t1/
├── pyproject.toml                 # name = "trading-model"
├── src/
│   └── trading_model/
│       ├── __init__.py
│       ├── env/
│       │   ├── __init__.py
│       │   ├── trading_env.py     # Main TradingEnv(gymnasium.Env)
│       │   ├── features.py        # FeatureEngine
│       │   ├── actions.py         # ActionMapper
│       │   ├── rewards.py         # RewardCalculator
│       │   └── frictions.py       # FrictionModel
│       └── data/
│           ├── __init__.py
│           └── loader.py          # Load CSV/DataFrame into env-ready format
├── tests/
│   ├── test_features.py
│   ├── test_actions.py
│   ├── test_rewards.py
│   └── test_env.py
```

## Observation Space

`Box(low=-inf, high=inf, shape=(18,), dtype=float32)`

| Group | Features | Dim |
|-------|----------|-----|
| Price | O, H, L, C normalized by previous close; Volume (log-scaled) | 5 |
| MACD | MACD line (12,26), Signal (9), Histogram | 3 |
| RSI | RSI(14) scaled to [0,1] | 1 |
| Bollinger | Upper band %, Lower band %, Bandwidth (relative to close) | 3 |
| Agent state | Position (-1/0/+1 scaled), Unrealized PnL %, Cash ratio, Trade duration (normalized) | 4 |
| Time | sin(t), cos(t) where t = minute_of_day / total_minutes | 2 |

All price-derived features Z-score normalized over a rolling window.
Warmup period of ~26 bars (for MACD) — env skips these before trading begins.

## Action Space

`Discrete(7)` — direction × size, long-only (no shorting in v1).

| Action | Meaning |
|--------|---------|
| 0 | Hold |
| 1 | Buy 25% of available cash |
| 2 | Buy 50% of available cash |
| 3 | Buy 100% of available cash |
| 4 | Sell 25% of position |
| 5 | Sell 50% of position |
| 6 | Sell 100% of position (close) |

Rules:
- Buy actions allocate % of available cash.
- Sell actions liquidate % of current position.
- Invalid actions (sell with no position, buy with no cash) are treated as hold.

## Reward Function

Sharpe-hybrid with drawdown penalty:

```
R_t = α * r_t + (1 - α) * (μ_rolling / σ_rolling) - β * max(0, DD_t - DD_{t-1})
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| α | 0.6 | Weight on raw return vs risk-adjusted component |
| μ_rolling, σ_rolling | window=60 | Rolling mean/std of returns (~1 hour) |
| β | 0.5 | Penalty for drawdown increases |
| r_t | — | Portfolio return net of transaction cost |
| DD_t | — | Current drawdown from peak portfolio value |

All parameters are constructor arguments for easy tuning.

## Friction Model

Flat fee: `cost = abs(trade_value) * fee_rate` where `fee_rate = 0.001` (0.1%).
Applied on every buy/sell execution. Deducted from cash immediately.

## Episode Structure

- One episode = one trading day (~400 bars, ~374 after warmup).
- `reset()` loads the next day's data.
- Initial cash = 100,000 (configurable).
- At market close (last bar), position is force-liquidated and final PnL computed.
- Training: days are shuffled. Evaluation: days are in chronological order.

## Dependencies (to add)

- gymnasium
- numpy
- pandas
- ta (technical analysis library for indicators)
- pytest (dev)

## Decisions & Trade-offs

- **Long-only v1**: No short selling to reduce complexity. Can extend later by widening the action space.
- **Flat fee over market impact model**: Simpler to debug. The FrictionModel interface makes it easy to swap in a power-law impact model later.
- **Daily episodes**: Aligns with intraday trading (flat overnight). Avoids overnight gap risk in the reward signal.
- **Modular composition over monolith**: Small cost in indirection, large gain in testability and experimentation velocity.
