import json
import os
import click
from tqdm import tqdm
import pandas as pd
import numpy as np
import optuna
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from trading_model import TradingEnv
from trading_model.env.features import FeatureEngine
from trading_model.data.loader import load_daily_window, get_trading_days, load_specific_day, load_from_db, load_days_from_dataframe


def constant_schedule(value: float):
    """Return a callable that always returns the same value."""
    def schedule(progress_remaining: float) -> float:
        return value
    return schedule


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

    min_bars = 1 if data_type == "daily" else FeatureEngine.WARMUP_PERIOD + 2
    days = load_days_from_dataframe(df, min_bars=min_bars)
    print(f"Total trading days: {len(days)}")
    split_idx = int(len(days) * 0.8)
    train_days = days[:split_idx]
    eval_days = days[split_idx:]
    if not train_days:
        train_days = days
        eval_days = days
    return train_days, eval_days


def make_vec_envs(initial_cash, fee_rate, n_envs=4):
    """Create normalized vectorized environment with parallel sub-environments."""
    def make_single_env():
        return Monitor(TradingEnv(initial_cash=initial_cash, fee_rate=fee_rate))
    train_env = DummyVecEnv([make_single_env for _ in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    return train_env


def patch_env(vec_env, options):
    """Patch the reset method of all underlying environments to use specific options."""
    for env_wrapper in vec_env.envs:
        env = env_wrapper.unwrapped
        if not hasattr(env, "_original_reset"):
            env._original_reset = env.reset
        def patched_reset(seed=None, options_internal=None, _opts=options, _env=env):
            return _env._original_reset(seed=seed, options=_opts)
        env.reset = patched_reset


def _run_eval_episode(model, vec_env, initial_cash):
    """Run a single eval episode and return raw PnL metrics.

    Returns dict with portfolio_value, pnl_pct, and num_trades.
    """
    obs = vec_env.reset()
    # RecurrentPPO needs LSTM state carried across steps
    lstm_state = None
    episode_start = np.ones((vec_env.num_envs,), dtype=bool)
    num_trades = 0
    prev_action = 0
    while True:
        action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_start, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        episode_start = done
        act = int(action[0])
        if act != 0 and act != prev_action:
            num_trades += 1
        prev_action = act
        if done[0]:
            break
    # info from VecEnv is a list of dicts; take first env
    final_info = info[0] if isinstance(info, list) else info
    portfolio_value = final_info.get("portfolio_value", initial_cash)
    pnl_pct = (portfolio_value - initial_cash) / initial_cash * 100
    return {"portfolio_value": portfolio_value, "pnl_pct": pnl_pct, "num_trades": num_trades}


def walkthrough_train(
    ticker="NVDA",
    start_date="2024-03-01",
    end_date="2024-03-31",
    total_timesteps_per_day=20_000,
    initial_cash=100000.0,
    fee_rate=0.001,
    learning_rate=2e-4,
    n_steps=512,
    batch_size=64,
    model_path="trading_ppo_walkthrough",
    vec_normalize_path="vec_normalize_walkthrough.pkl",
):
    trading_days = get_trading_days(ticker, start_date, end_date)
    if len(trading_days) < 2:
        print("Not enough trading days.")
        return

    train_env = make_vec_envs(initial_cash, fee_rate)
    model = RecurrentPPO(
        "MlpLstmPolicy", train_env, learning_rate=constant_schedule(learning_rate),
        n_steps=n_steps, batch_size=batch_size, verbose=0,
        tensorboard_log="./tensorboard_logs/"
    )

    n_pairs = len(trading_days) - 1
    win_days, loss_days = 0, 0
    pbar = tqdm(range(n_pairs), desc='Walkthrough', unit='days')
    for i in pbar:
        train_day, eval_day = trading_days[i], trading_days[i+1]

        daily_window = load_daily_window(ticker, train_day)
        intraday_train = load_specific_day(ticker, train_day)

        if intraday_train.empty:
            continue

        patch_env(train_env, {"intraday_data": intraday_train, "daily_window": daily_window})
        model.learn(total_timesteps=total_timesteps_per_day, reset_num_timesteps=False)

        intraday_eval = load_specific_day(ticker, eval_day)
        if not intraday_eval.empty:
            daily_window_eval = load_daily_window(ticker, eval_day)
            patch_env(train_env, {"intraday_data": intraday_eval, "daily_window": daily_window_eval})

            train_env.training = False
            train_env.norm_reward = False

            result = _run_eval_episode(model, train_env, initial_cash)

            train_env.training = True
            train_env.norm_reward = True

            pnl = result["pnl_pct"]
            if pnl >= 0:
                win_days += 1
            else:
                loss_days += 1
            tqdm.write(f"  {train_day.date()} -> {eval_day.date()}  PnL: {pnl:+.4f}%  Trades: {result['num_trades']}  Portfolio: ${result['portfolio_value']:,.2f}")

        pbar.set_description(f'Walkthrough [W:{win_days} L:{loss_days}]')

    print(f"\nWalkthrough Summary: {win_days} win / {loss_days} loss days")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)
    train_env.save(vec_normalize_path)


def _patch_tune_training(vec_env, train_days):
    """Patch envs to select a random training day on each reset."""
    for env_wrapper in vec_env.envs:
        env = env_wrapper.unwrapped
        def tune_reset(seed=None, options=None, _orig=env._original_reset, _days=train_days):
            return _orig(seed=seed, options={"intraday_data": _days[np.random.randint(len(_days))]})
        env.reset = tune_reset


def tune(ticker="NVDA", n_trials=20, total_timesteps=200000, n_jobs=1, params_file="tuned_params.json"):
    if os.path.exists(params_file):
        print(f"Skipping tune: {params_file} already exists.")
        return
    train_days, eval_days = prepare_data(ticker, 20)

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        ns = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        bs = trial.suggest_categorical("batch_size", [32, 64, 128])
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
        n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10])
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 1.0])

        train_env = make_vec_envs(100000.0, 0.001)
        # Save original resets before any patching so patch_env works correctly for eval
        for env_wrapper in train_env.envs:
            env = env_wrapper.unwrapped
            env._original_reset = env.reset
        _patch_tune_training(train_env, train_days)

        model = RecurrentPPO(
            "MlpLstmPolicy", train_env,
            learning_rate=constant_schedule(lr), n_steps=ns, batch_size=bs,
            gamma=gamma, ent_coef=ent_coef, gae_lambda=gae_lambda,
            n_epochs=n_epochs, clip_range=clip_range, max_grad_norm=max_grad_norm,
            verbose=0,
        )

        # Train in chunks with intermediate evaluation for early pruning
        n_checkpoints = 4
        chunk_size = total_timesteps // n_checkpoints
        for i in range(n_checkpoints):
            model.learn(
                total_timesteps=chunk_size * (i + 1),
                reset_num_timesteps=(i == 0),
            )

            # Quick eval on one random day
            train_env.training = False
            train_env.norm_reward = False
            eval_day = eval_days[np.random.randint(len(eval_days))]
            patch_env(train_env, {"intraday_data": eval_day})
            result = _run_eval_episode(model, train_env, 100000.0)

            tqdm.write(f"  Trial {trial.number} [{i+1}/{n_checkpoints}] eval PnL: {result['pnl_pct']:+.4f}%")
            trial.report(result["pnl_pct"], i)

            # Restore training mode and re-patch for random training days
            train_env.training = True
            train_env.norm_reward = True
            _patch_tune_training(train_env, train_days)

            if trial.should_prune():
                raise optuna.TrialPruned()

        # Final evaluation
        train_env.training = False
        train_env.norm_reward = False
        eval_pnls = []
        for _ in range(min(3, len(eval_days))):
            eval_day = eval_days[np.random.randint(len(eval_days))]
            patch_env(train_env, {"intraday_data": eval_day})
            result = _run_eval_episode(model, train_env, 100000.0)
            eval_pnls.append(result["pnl_pct"])
        return np.mean(eval_pnls)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pbar = tqdm(total=n_trials, desc='Tuning', unit='trials')

    def on_trial_end(study, trial):
        n_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        try:
            best = f'{study.best_value:+.4f}%'
        except ValueError:
            best = 'N/A'
        pbar.set_postfix(best=best, done=n_done, pruned=n_pruned)
        pbar.update()

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[on_trial_end])
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    finally:
        pbar.close()

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Trials: {len(completed)} completed, {len(pruned)} pruned.")
    if completed:
        print(f"Best params: {study.best_params}")
        print(f"Best value: {study.best_value}")
        with open(params_file, "w") as f:
            json.dump(study.best_params, f, indent=2)
        print(f"Saved best params to {params_file}")
    else:
        print("No trials completed.")


def evaluate(
    ticker="NVDA",
    start_date="2024-04-01",
    end_date="2024-04-30",
    model_path="trading_ppo_walkthrough.zip",
    vec_normalize_path="vec_normalize_walkthrough.pkl",
    initial_cash=100000.0,
    fee_rate=0.001,
):
    trading_days = get_trading_days(ticker, start_date, end_date)
    if not trading_days:
        print("No trading days found in the given range.")
        return

    train_env = make_vec_envs(initial_cash, fee_rate)
    train_env = VecNormalize.load(vec_normalize_path, train_env.venv)
    train_env.training = False
    train_env.norm_reward = False

    model = RecurrentPPO.load(model_path, env=train_env)

    results = []
    for day in tqdm(trading_days, desc='Evaluating', unit='days'):
        intraday = load_specific_day(ticker, day)
        if intraday.empty:
            continue
        daily_window = load_daily_window(ticker, day)
        patch_env(train_env, {"intraday_data": intraday, "daily_window": daily_window})
        result = _run_eval_episode(model, train_env, initial_cash)
        results.append({"date": day.date(), **result})
        tqdm.write(f"  {day.date()}: PnL={result['pnl_pct']:+.4f}%  Trades={result['num_trades']}  Portfolio=${result['portfolio_value']:,.2f}")

    if results:
        pnls = [r["pnl_pct"] for r in results]
        win = sum(1 for p in pnls if p >= 0)
        loss = len(pnls) - win
        print(f"\nEvaluation Summary ({len(results)} days)")
        print(f"  Mean PnL%:    {np.mean(pnls):+.4f}%")
        print(f"  Std PnL%:     {np.std(pnls):.4f}%")
        print(f"  Min PnL%:     {np.min(pnls):+.4f}%")
        print(f"  Max PnL%:     {np.max(pnls):+.4f}%")
        print(f"  Win/Loss:     {win}/{loss}")
        print(f"  Avg Trades:   {np.mean([r['num_trades'] for r in results]):.1f}")


def load_params_file(path):
    with open(path) as f:
        return json.load(f)


WALKTHROUGH_PARAM_KEYS = {"learning_rate", "n_steps", "batch_size"}


@click.group()
def cli(): pass

@cli.command()
@click.option("--ticker", type=str, default="NVDA")
@click.option("--start-date", type=str, default="2024-03-01")
@click.option("--end-date", type=str, default="2024-03-31")
@click.option("--timesteps-per-day", type=int, default=20_000)
@click.option("--params-file", type=str, default=None, help="JSON file with tuned hyperparameters.")
@click.option("--model-path", type=str, default="trading_ppo_walkthrough")
@click.option("--vec-normalize-path", type=str, default="vec_normalize_walkthrough.pkl")
def walkthrough(ticker, start_date, end_date, timesteps_per_day, params_file, model_path, vec_normalize_path):
    kwargs = {}
    if params_file:
        params = load_params_file(params_file)
        kwargs.update({k: v for k, v in params.items() if k in WALKTHROUGH_PARAM_KEYS})
    walkthrough_train(
        ticker=ticker, start_date=start_date, end_date=end_date,
        total_timesteps_per_day=timesteps_per_day,
        model_path=model_path, vec_normalize_path=vec_normalize_path,
        **kwargs,
    )

@cli.command()
@click.option("--ticker", type=str, default="NVDA")
@click.option("--n-trials", type=int, default=20)
@click.option("--n-jobs", type=int, default=1, help="Number of parallel jobs for Optuna tuning.")
@click.option("--params-file", type=str, default="tuned_params.json", help="Output file for best params.")
def tune_cmd(params_file, **kwargs):
    tune(params_file=params_file, **kwargs)

@cli.command()
@click.option("--ticker", type=str, default="NVDA")
@click.option("--start-date", type=str, default="2024-04-01")
@click.option("--end-date", type=str, default="2024-04-30")
@click.option("--model-path", type=str, default="trading_ppo_walkthrough.zip")
@click.option("--vec-normalize-path", type=str, default="vec_normalize_walkthrough.pkl")
def evaluate_cmd(ticker, start_date, end_date, model_path, vec_normalize_path):
    evaluate(ticker=ticker, start_date=start_date, end_date=end_date,
             model_path=model_path, vec_normalize_path=vec_normalize_path)

if __name__ == "__main__": cli()
