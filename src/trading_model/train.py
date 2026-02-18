import os
import click
import pandas as pd
import numpy as np
import optuna
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from trading_model import TradingEnv
from trading_model.data.loader import load_daily_window, get_trading_days, load_specific_day, load_from_db, load_days_from_dataframe


def generate_synthetic_data(num_days=10, bars_per_day=400):
    """Generate synthetic OHLCV data for demonstration."""
    print(f"Generating {num_days} days of synthetic data...")
    all_days = []
    base_price = 100.0
    start_date = pd.Timestamp("2025-01-01 09:30")
    for d in range(num_days):
        dates = pd.date_range(start_date + pd.Timedelta(days=d), periods=bars_per_day, freq="min")
        returns = np.random.normal(0.0001, 0.002, bars_per_day)
        close = base_price * np.exp(np.cumsum(returns))
        base_price = close[-1]
        high = close * (1 + np.abs(np.random.normal(0, 0.002, bars_per_day)))
        low = close * (1 - np.abs(np.random.normal(0, 0.002, bars_per_day)))
        open_ = low + (high - low) * np.random.uniform(0.3, 0.7, bars_per_day)
        volume = np.random.randint(1000, 50000, bars_per_day).astype(float)
        df = pd.DataFrame({"datetime": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
        all_days.append(df)
    return pd.concat(all_days)


def prepare_data(ticker, num_days, data_type="historical"):
    """Load data and split into train/eval days."""
    try:
        df = load_from_db(ticker, data_type=data_type)
    except Exception as e:
        print(f"Failed to load data from database for {ticker} ({data_type}): {e}")
        print("Falling back to synthetic data.")
        df = generate_synthetic_data(num_days=num_days)

    min_bars = 1 if data_type == "daily" else 30
    days = load_days_from_dataframe(df, min_bars=min_bars)
    print(f"Total trading days: {len(days)}")
    split_idx = int(len(days) * 0.8)
    train_days = days[:split_idx]
    eval_days = days[split_idx:]
    if not train_days:
        train_days = days
        eval_days = days
    return train_days, eval_days


def make_vec_envs(initial_cash, fee_rate):
    """Create normalized vectorized environment."""
    def make_single_env():
        return Monitor(TradingEnv(initial_cash=initial_cash, fee_rate=fee_rate))
    train_env = DummyVecEnv([make_single_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    return train_env


def patch_env(vec_env, options):
    """Patch the reset method of the underlying environment to use specific options."""
    env = vec_env.envs[0].unwrapped
    if not hasattr(env, "_original_reset"):
        env._original_reset = env.reset
    def patched_reset(seed=None, options_internal=None):
        return env._original_reset(seed=seed, options=options)
    env.reset = patched_reset


def train(
    ticker="NVDA",
    data_type="historical",
    num_days=20,
    total_timesteps=100000,
    initial_cash=100000.0,
    fee_rate=0.001,
    learning_rate=2e-4,
    n_steps=2048,
    batch_size=128,
    ent_coef=0.01,
    gamma=0.99,
):
    train_days, eval_days = prepare_data(ticker, num_days, data_type=data_type)
    train_env = make_vec_envs(initial_cash, fee_rate)
    
    # For standard training, we just cycle through days
    # We need a custom reset logic for the env to pick a random day from train_days
    env = train_env.envs[0].unwrapped
    original_reset = env.reset
    def standard_reset(seed=None, options=None):
        day_df = train_days[np.random.randint(len(train_days))]
        return original_reset(seed=seed, options={"intraday_data": day_df})
    env.reset = standard_reset

    model = RecurrentPPO(
        "MlpLstmPolicy", train_env, learning_rate=learning_rate, n_steps=n_steps,
        batch_size=batch_size, ent_coef=ent_coef, gamma=gamma, verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )

    print("Starting training... (Press Ctrl+C to stop and save)")
    try:
        model.learn(total_timesteps=total_timesteps)
    except KeyboardInterrupt:
        print("\nInterrupted. Saving...")
    finally:
        model.save("trading_ppo_final")
        train_env.save("vec_normalize.pkl")


def walkthrough_train(
    ticker="NVDA",
    start_date="2024-03-01",
    end_date="2024-03-31",
    total_timesteps_per_day=5000,
    initial_cash=100000.0,
    fee_rate=0.001,
    learning_rate=2e-4,
    n_steps=512,
    batch_size=64,
):
    trading_days = get_trading_days(ticker, start_date, end_date)
    if len(trading_days) < 2:
        print("Not enough trading days.")
        return

    train_env = make_vec_envs(initial_cash, fee_rate)
    model = RecurrentPPO(
        "MlpLstmPolicy", train_env, learning_rate=learning_rate, n_steps=n_steps,
        batch_size=batch_size, verbose=0, tensorboard_log="./tensorboard_logs/"
    )
    for i in range(len(trading_days) - 1):
        train_day, eval_day = trading_days[i], trading_days[i+1]
        print(f"[{i+1}/{len(trading_days)-1}] Training: {train_day.date()}, Eval: {eval_day.date()}")

        daily_window = load_daily_window(ticker, train_day)
        intraday_train = load_specific_day(ticker, train_day)

        if intraday_train.empty: continue

        patch_env(train_env, {"intraday_data": intraday_train, "daily_window": daily_window})
        model.learn(total_timesteps=total_timesteps_per_day, reset_num_timesteps=False)

        intraday_eval = load_specific_day(ticker, eval_day)
        if not intraday_eval.empty:
            daily_window_eval = load_daily_window(ticker, eval_day)
            patch_env(train_env, {"intraday_data": intraday_eval, "daily_window": daily_window_eval})
            obs = train_env.reset()
            total_reward, terminated = 0, False
            while not terminated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, info = train_env.step(action)
                total_reward += reward[0]
            print(f"Eval Reward: {total_reward:.4f}")

    model.save("trading_ppo_walkthrough")
    train_env.save("vec_normalize_walkthrough.pkl")


def tune(ticker="NVDA", n_trials=20, total_timesteps=50000, n_jobs=1):
    train_days, eval_days = prepare_data(ticker, 20)
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        ns = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        bs = trial.suggest_categorical("batch_size", [32, 64, 128])
        
        train_env = make_vec_envs(100000.0, 0.001)
        env = train_env.envs[0].unwrapped
        original_reset = env.reset
        def tune_reset(seed=None, options=None):
            return original_reset(seed=seed, options={"intraday_data": train_days[np.random.randint(len(train_days))]})
        env.reset = tune_reset

        model = RecurrentPPO("MlpLstmPolicy", train_env, learning_rate=lr, n_steps=ns, batch_size=bs, verbose=0)
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluation on multiple random days for better signal
        eval_rewards = []
        for _ in range(min(3, len(eval_days))):
            eval_day = eval_days[np.random.randint(len(eval_days))]
            patch_env(train_env, {"intraday_data": eval_day})
            obs = train_env.reset()
            rew, term = 0, False
            while not term:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, info = train_env.step(action)
                rew += r[0]
            eval_rewards.append(rew)
        return np.mean(eval_rewards)

    study = optuna.create_study(direction="maximize")
    print("Starting optimization... (Press Ctrl+C to stop)")
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    
    if len(study.trials) > 0:
        print(f"Best params: {study.best_params}")
        print(f"Best value: {study.best_value}")
    else:
        print("No trials completed.")


@click.group()
def cli(): pass

@cli.command()
@click.option("--ticker", type=str, default="NVDA")
@click.option("--data-type", type=click.Choice(["historical", "daily"]), default="historical")
@click.option("--num-days", type=int, default=20)
@click.option("--total-timesteps", type=int, default=100000)
def train_cmd(**kwargs):
    os.makedirs("./models/checkpoints", exist_ok=True)
    train(**kwargs)

@cli.command()
@click.option("--ticker", type=str, default="NVDA")
@click.option("--start_date", type=str, default="2024-03-01")
@click.option("--end_date", type=str, default="2024-03-31")
@click.option("--timesteps_per_day", type=int, default=5000)
def walkthrough(ticker, start_date, end_date, timesteps_per_day):
    walkthrough_train(ticker=ticker, start_date=start_date, end_date=end_date, total_timesteps_per_day=timesteps_per_day)

@cli.command()
@click.option("--ticker", type=str, default="NVDA")
@click.option("--n-trials", type=int, default=20)
@click.option("--n-jobs", type=int, default=1, help="Number of parallel jobs for Optuna tuning.")
def tune_cmd(**kwargs):
    tune(**kwargs)

if __name__ == "__main__": cli()
