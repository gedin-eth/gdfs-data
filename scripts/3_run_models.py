import sys
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, PROCESSED_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)
MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR = Path(__file__).parent.parent / 'predictions'
PREDICTIONS_DIR.mkdir(exist_ok=True)

def load_data():
    inputs, results = [], []
    for input_file in sorted(DATA_DIR.glob('input_*.csv')):
        date_str = input_file.stem.replace('input_', '')
        results_file = DATA_DIR / f'results_{date_str}.csv'
        
        input_df = pd.read_csv(input_file)
        input_df['Date'] = date_str
        
        if results_file.exists():
            try:
                results_df = pd.read_csv(results_file, skiprows=7)
                if 'Player Name' in results_df.columns and len(results_df) > 0:
                    results_df['Date'] = date_str
                    df = input_df.merge(results_df, left_on='Name', right_on='Player Name', how='left', suffixes=('', '_y'))
                    if 'Date_y' in df.columns:
                        df = df.drop(columns=['Date_y'])
                else:
                    df = input_df.copy()
            except:
                df = input_df.copy()
        else:
            df = input_df.copy()
        
        inputs.append(df)
    
    if not inputs:
        raise ValueError("No data files found")
    
    return pd.concat(inputs, ignore_index=True)

def prepare_target_dataset(df, target_col):
    exclude_cols = ['Name', 'Player Name', 'Date', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    dates = df['Date'].values
    
    cat_features = [i for i, col in enumerate(feature_cols) if X[col].dtype == 'object' or col in ['Position', 'Team']]
    
    X = X.fillna(0)
    y = y.fillna(0)
    
    return X, y, dates, feature_cols, cat_features

def train_models(df, target_col, n_folds=5):
    X, y, dates, feature_cols, cat_features = prepare_target_dataset(df, target_col)
    
    if y.nunique() <= 1:
        logger.warning(f'{target_col} has only {y.nunique()} unique values, skipping')
        return []
    
    unique_dates = sorted(set(dates))
    kf = KFold(n_splits=n_folds, shuffle=False)
    
    models = []
    fold_dates = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(unique_dates)):
        train_dates = [unique_dates[i] for i in train_idx]
        val_dates = [unique_dates[i] for i in val_idx]
        
        train_mask = pd.Series(dates).isin(train_dates)
        val_mask = pd.Series(dates).isin(val_dates)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        if y_train.nunique() <= 1:
            logger.warning(f'{target_col} Fold {fold_idx} has constant train values, skipping')
            continue
        
        if target_col == 'Ownership':
            model = CatBoostRegressor(cat_features=cat_features, verbose=False, random_seed=42)
        else:
            model = CatBoostRegressor(cat_features=cat_features, verbose=False, random_seed=42)
        
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)
        
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f'{target_col} Fold {fold_idx} Validation RMSE: {val_rmse:.4f}')
        
        model_path = MODELS_DIR / f'{target_col}_fold_{fold_idx}.cbm'
        model.save_model(str(model_path))
        logger.info(f'Saved {target_col} fold {fold_idx} model')
        
        models.append(model)
        fold_dates.append(val_dates)
    
    ensemble_path = MODELS_DIR / f'{target_col}_ensemble.cbm'
    models[0].save_model(str(ensemble_path))
    logger.info(f'Saved {target_col} ensemble model')
    
    return models

def predict(df, target_col, date_str=None):
    model_files = sorted(MODELS_DIR.glob(f'{target_col}_fold_*.cbm'))
    if not model_files:
        raise ValueError(f'No models found for {target_col}')
    
    if date_str:
        df_subset = df[df['Date'] == date_str].copy().reset_index(drop=True)
    else:
        df_subset = df.copy().reset_index(drop=True)
    
    X, _, _, _, cat_features = prepare_target_dataset(df_subset, target_col)
    
    predictions = []
    for model_file in model_files:
        model = CatBoostRegressor()
        model.load_model(str(model_file))
        pred = model.predict(X)
        predictions.append(pred)
    
    avg_pred = np.mean(predictions, axis=0)
    
    result_df = df_subset[['Name', 'Date']].copy()
    result_df[f'{target_col}_predicted'] = avg_pred
    
    if target_col in df_subset.columns:
        actual = df_subset[target_col].values
        mask = ~pd.isna(actual)
        if mask.sum() > 0:
            pred_rmse = np.sqrt(mean_squared_error(actual[mask], avg_pred[mask]))
            print(f'{target_col} Prediction RMSE: {pred_rmse:.4f}')
    
    return result_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict', 'train_predict'], default='train')
    parser.add_argument('--target', default='all', help='Target column or "all"')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--date', help='Date for prediction (YYYYMMDD)')
    args = parser.parse_args()
    
    df = load_data()
    if args.target == 'all':
        feature_targets = ['FPM', 'Usage', 'Momentum', 'FP', 'Minutes', 'ClutchFP', 'ClutchRatio', 
                          'TouchesPerMin', 'ThreeRate', 'ScoringFrequency', 'ScoringConsistency', 
                          'DominantQuarter', 'LateGameEmphasis', 'Substitutions']
        target_cols = ['Actual Score', 'Ownership'] + feature_targets
    else:
        target_cols = [args.target]
    
    if args.mode in ['train', 'train_predict']:
        for target_col in target_cols:
            if target_col not in df.columns:
                logger.warning(f'{target_col} not found, skipping')
                continue
            train_models(df, target_col, args.n_folds)
    
    if args.mode in ['predict', 'train_predict']:
        for target_col in target_cols:
            if target_col not in df.columns:
                continue
            try:
                pred_df = predict(df, target_col, args.date)
                output_path = PREDICTIONS_DIR / f'predictions_{target_col}_{args.date or "all"}.csv'
                pred_df.to_csv(output_path, index=False)
                logger.info(f'Saved predictions to {output_path}')
            except Exception as e:
                logger.error(f'Prediction failed for {target_col}: {e}')

if __name__ == '__main__':
    main()

