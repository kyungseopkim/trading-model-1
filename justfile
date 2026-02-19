# ── Configuration ─────────────────────────────────────────────
tickers           := "AAPL MSFT NVDA TSLA GOOGL"
train_start       := "2021-01-01"
train_end         := "2026-02-01"
eval_start        := "2025-01-01"
eval_end          := "2026-02-01"
params_file       := "tuned_params.json"
n_trials          := "20"
n_jobs            := "1"
timesteps_per_day := "50000"
models_dir        := "models"

# ── Install/sync dependencies ────────────────────────────────
sync:
    uv sync

# ── Full pipeline: tune → train all → evaluate all ───────────
pipeline: tune train-all eval-all

# ── Hyperparameter tuning (run once, shared across tickers) ──
tune:
    uv run python main.py tune \
        --ticker AAPL \
        --n-trials {{n_trials}} \
        --n-jobs {{n_jobs}} \
        --params-file {{params_file}}

# ── Train all tickers sequentially ───────────────────────────
train-all:
    #!/usr/bin/env bash
    set -e
    mkdir -p {{models_dir}}
    for t in {{tickers}}; do
        echo ""
        echo "════════════════════════════════════════════"
        echo "  Training $t  ({{train_start}} → {{train_end}})"
        echo "════════════════════════════════════════════"
        uv run python main.py walkthrough \
            --ticker "$t" \
            --start-date {{train_start}} \
            --end-date {{train_end}} \
            --timesteps-per-day {{timesteps_per_day}} \
            --params-file {{params_file}} \
            --model-path "{{models_dir}}/${t}_model" \
            --vec-normalize-path "{{models_dir}}/${t}_vecnorm.pkl"
    done

# ── Train a single ticker ────────────────────────────────────
train ticker="NVDA":
    mkdir -p {{models_dir}}
    uv run python main.py walkthrough \
        --ticker {{ticker}} \
        --start-date {{train_start}} \
        --end-date {{train_end}} \
        --timesteps-per-day {{timesteps_per_day}} \
        --params-file {{params_file}} \
        --model-path {{models_dir}}/{{ticker}}_model \
        --vec-normalize-path {{models_dir}}/{{ticker}}_vecnorm.pkl

# ── Evaluate all tickers ─────────────────────────────────────
eval-all:
    #!/usr/bin/env bash
    set -e
    for t in {{tickers}}; do
        echo ""
        echo "════════════════════════════════════════════"
        echo "  Evaluating $t  ({{eval_start}} → {{eval_end}})"
        echo "════════════════════════════════════════════"
        uv run python main.py evaluate \
            --ticker "$t" \
            --start-date {{eval_start}} \
            --end-date {{eval_end}} \
            --model-path "{{models_dir}}/${t}_model.zip" \
            --vec-normalize-path "{{models_dir}}/${t}_vecnorm.pkl"
    done

# ── Evaluate a single ticker ─────────────────────────────────
evaluate ticker="NVDA":
    uv run python main.py evaluate \
        --ticker {{ticker}} \
        --start-date {{eval_start}} \
        --end-date {{eval_end}} \
        --model-path {{models_dir}}/{{ticker}}_model.zip \
        --vec-normalize-path {{models_dir}}/{{ticker}}_vecnorm.pkl
