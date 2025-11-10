## README


## Required Directories

- `results/` - DK contest results (`dk_YYYYMMDD.csv`) - for training
- `linestar/` - Linestar projections (`ls_YYYYMMDD.csv`) - for prediction
- `data/` - Actual player scores (`results_YYYYMMDD.csv`) - for backtest evaluation
- `models/misc_pipeline/` - Auto-created for cached models

## Daily Usage Commands

** 1. Prediction (before slate): **

python3 misc/pipeline.py --date 2025-11-07


** 2. Backtest (after slate): **

update line 170 and 189 and ++ increment the loop range(16)

python3 misc/pipeline.py --backtest


## Data Flow

1. Training (historical):
   - Reads `results/dk_*.csv` → `create_output_labels_dk()` → extracts top lineups, ownership, actual scores
   - Reads `data/results_*.csv` → merges actual scores as target labels
   - Engineers features → trains cluster-based CatBoost models
   - Saves models to `models/misc_pipeline/pipeline_models.pkl`

2. Prediction (pre-slate):
   - Reads `linestar/ls_YYYYMMDD.csv` → `create_input_labels_ls()` → extracts Name, Position, Projected, Salary, ProjOwn
   - Engineers features (beat_rate, salary_efficiency, etc.)
   - Loads cached models → predicts projections → calculates edge scores
   - Outputs top 10 players by edge_score

3. Backtest (post-slate):
   - Same as prediction but compares predictions to `data/results_*.csv` actual scores

## Functionality

- Clusters players by features (KMeans with gap statistic)
- Trains separate CatBoost models per cluster (5-fold CV)
- Predicts: player projections, lineup value probability
- Edge score: combines projection × salary_efficiency × beat_rate × ownership_leverage × lineup_value
- Uses linestar-only for prediction (no future data leakage)
- Caches models (retrains only if training data changes)