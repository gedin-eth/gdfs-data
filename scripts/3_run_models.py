import sys
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
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

def get_cluster_features(df):
    cluster_cols = ['Position', 'Salary', 'FPM', 'Usage', 'FP', 'Minutes']
    available = [c for c in cluster_cols if c in df.columns]
    X_cluster = df[available].copy()
    
    if 'Position' in X_cluster.columns:
        pos_encoded = pd.get_dummies(X_cluster['Position'], prefix='pos')
        X_cluster = pd.concat([X_cluster.drop('Position', axis=1), pos_encoded], axis=1)
    
    X_cluster = X_cluster.fillna(0)
    return X_cluster, available

def create_clusters(df, n_clusters=5):
    X_cluster, _ = get_cluster_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    pickle.dump(kmeans, open(MODELS_DIR / 'cluster_model.pkl', 'wb'))
    pickle.dump(scaler, open(MODELS_DIR / 'cluster_scaler.pkl', 'wb'))
    
    return clusters

def assign_clusters(df):
    if not (MODELS_DIR / 'cluster_model.pkl').exists():
        return None
    
    kmeans = pickle.load(open(MODELS_DIR / 'cluster_model.pkl', 'rb'))
    scaler = pickle.load(open(MODELS_DIR / 'cluster_scaler.pkl', 'rb'))
    
    X_cluster, _ = get_cluster_features(df)
    
    if hasattr(scaler, 'feature_names_in_'):
        expected_cols = scaler.feature_names_in_
        X_aligned = pd.DataFrame(index=X_cluster.index)
        for col in expected_cols:
            if col in X_cluster.columns:
                X_aligned[col] = X_cluster[col].fillna(0)
            else:
                X_aligned[col] = 0
        X_cluster = X_aligned[expected_cols]
    
    X_scaled = scaler.transform(X_cluster)
    clusters = kmeans.predict(X_scaled)
    
    return clusters

def prepare_target_dataset(df, target_col):
    exclude_cols = ['Name', 'Player Name', 'Date', target_col, 'Cluster']
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.endswith('_y')]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    dates = df['Date'].values
    
    cat_features = [i for i, col in enumerate(feature_cols) if X[col].dtype == 'object' or col in ['Position', 'Team']]
    
    X = X.fillna(0)
    y = y.fillna(0)
    
    return X, y, dates, feature_cols, cat_features

def train_models(df, target_col, n_folds=5, n_clusters=5):
    clusters = create_clusters(df, n_clusters)
    df['Cluster'] = clusters
    
    X, y, dates, feature_cols, cat_features = prepare_target_dataset(df, target_col)
    
    pickle.dump(feature_cols, open(MODELS_DIR / f'{target_col}_feature_cols.pkl', 'wb'))
    
    if y.nunique() <= 1:
        logger.warning(f'{target_col} has only {y.nunique()} unique values, skipping')
        return []
    
    unique_dates = sorted(set(dates))
    kf = KFold(n_splits=n_folds, shuffle=False)
    
    models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(unique_dates)):
        train_dates = [unique_dates[i] for i in train_idx]
        val_dates = [unique_dates[i] for i in val_idx]
        
        train_mask = pd.Series(dates).isin(train_dates)
        val_mask = pd.Series(dates).isin(val_dates)
        
        cluster_models = {}
        
        for cluster_id in range(n_clusters):
            cluster_train_mask = train_mask & (df['Cluster'] == cluster_id)
            cluster_val_mask = val_mask & (df['Cluster'] == cluster_id)
            
            X_train_cluster = X[cluster_train_mask]
            y_train_cluster = y[cluster_train_mask]
            X_val_cluster = X[cluster_val_mask]
            y_val_cluster = y[cluster_val_mask]
            
            if len(X_train_cluster) < 10 or y_train_cluster.nunique() <= 1:
                continue
            
            model = CatBoostRegressor(cat_features=cat_features, verbose=False, random_seed=42)
            
            if len(X_val_cluster) > 0:
                model.fit(X_train_cluster, y_train_cluster, eval_set=(X_val_cluster, y_val_cluster), early_stopping_rounds=10)
                val_pred = model.predict(X_val_cluster)
                val_rmse = np.sqrt(mean_squared_error(y_val_cluster, val_pred))
                print(f'{target_col} Fold {fold_idx} Cluster {cluster_id} Validation RMSE: {val_rmse:.4f} (n={len(X_val_cluster)})')
            else:
                model.fit(X_train_cluster, y_train_cluster)
                print(f'{target_col} Fold {fold_idx} Cluster {cluster_id} No validation set (n_train={len(X_train_cluster)})')
            
            model_path = MODELS_DIR / f'{target_col}_fold_{fold_idx}_cluster_{cluster_id}.cbm'
            model.save_model(str(model_path))
            cluster_models[cluster_id] = model
        
        models.append(cluster_models)
    
    return models

def predict(df, target_col, date_str=None):
    if date_str:
        df_subset = df[df['Date'] == date_str].copy().reset_index(drop=True)
    else:
        df_subset = df.copy().reset_index(drop=True)
    
    clusters = assign_clusters(df_subset)
    if clusters is None:
        raise ValueError('Cluster model not found. Train models first.')
    
    df_subset['Cluster'] = clusters
    
    feature_cols_file = MODELS_DIR / f'{target_col}_feature_cols.pkl'
    if feature_cols_file.exists():
        train_feature_cols = pickle.load(open(feature_cols_file, 'rb'))
    else:
        X_full, _, _, train_feature_cols, _ = prepare_target_dataset(df_subset, target_col)
    
    X_full, _, _, _, cat_features = prepare_target_dataset(df_subset, target_col)
    
    X = pd.DataFrame(index=X_full.index)
    for col in train_feature_cols:
        if col in X_full.columns:
            X[col] = X_full[col].fillna(0)
        else:
            X[col] = 0
    
    missing_cols = set(train_feature_cols) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    
    X = X[train_feature_cols]
    
    model_files = sorted(MODELS_DIR.glob(f'{target_col}_fold_*_cluster_*.cbm'))
    if not model_files:
        raise ValueError(f'No cluster models found for {target_col}')
    
    fold_clusters = {}
    for f in model_files:
        parts = f.stem.split('_')
        fold_idx = None
        cluster_idx = None
        for i, p in enumerate(parts):
            if p == 'fold' and i + 1 < len(parts):
                fold_idx = int(parts[i + 1])
            if p == 'cluster' and i + 1 < len(parts):
                cluster_idx = int(parts[i + 1])
        if fold_idx is not None and cluster_idx is not None:
            if fold_idx not in fold_clusters:
                fold_clusters[fold_idx] = {}
            fold_clusters[fold_idx][cluster_idx] = str(f)
    
    predictions = []
    for fold_idx, cluster_models in fold_clusters.items():
        pred = np.zeros(len(df_subset))
        for i, cluster_id in enumerate(df_subset['Cluster']):
            if cluster_id in cluster_models:
                model = CatBoostRegressor()
                model.load_model(cluster_models[cluster_id])
                pred[i] = model.predict(X.iloc[[i]])[0]
            else:
                pred[i] = 0
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
    parser.add_argument('--n_clusters', type=int, default=5)
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
            train_models(df, target_col, args.n_folds, args.n_clusters)
    
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

