import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, CatBoostClassifier

def prepare_features(df):
    if len(df) == 0:
        return pd.DataFrame(), []
    
    feature_cols = []
    X = pd.DataFrame()
    
    numeric_cols = ['projected', 'salary', 'salary_efficiency',
                    'field_ownership', 'top_1pct_rate', 'ownership_delta']
    
    for col in numeric_cols:
        if col in df.columns:
            X[col] = df[col].fillna(0)
            feature_cols.append(col)
    
    if 'ownership_tier' in df.columns:
        le = LabelEncoder()
        X['ownership_tier_encoded'] = le.fit_transform(df['ownership_tier'].fillna('low'))
        feature_cols.append('ownership_tier_encoded')
    
    if 'Roster Position' in df.columns:
        le = LabelEncoder()
        X['position_encoded'] = le.fit_transform(df['Roster Position'].fillna('UTIL'))
        feature_cols.append('position_encoded')
    
    return X, feature_cols

def calculate_gap_statistic(X, k_max=10, n_refs=10, random_seed=42):
    gaps = []
    sk_values = []
    
    for k in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        kmeans.fit(X)
        sk = kmeans.inertia_
        sk_values.append(sk)
        
        ref_disps = []
        for _ in range(n_refs):
            np.random.seed(random_seed + _)
            random_data = np.random.uniform(X.min(), X.max(), size=X.shape)
            random_kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
            random_kmeans.fit(random_data)
            ref_disps.append(random_kmeans.inertia_)
        
        gap = np.log(np.mean(ref_disps)) - np.log(sk)
        gaps.append(gap)
    
    gaps = np.array(gaps)
    optimal_k = np.argmax(gaps) + 1
    
    return optimal_k, gaps

def find_optimal_clusters(df, k_max=10):
    if len(df) == 0:
        return 2
    
    X, _ = prepare_features(df)
    if len(X) == 0:
        return 2
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    optimal_k, gaps = calculate_gap_statistic(X_scaled, k_max=k_max)
    
    return optimal_k

def cluster_players(df, n_clusters=None):
    if len(df) == 0:
        return df, {}
    
    X, feature_cols = prepare_features(df)
    if len(X) == 0:
        df['cluster'] = 0
        return df, {}
    
    if n_clusters is None:
        n_clusters = find_optimal_clusters(df)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df = df.copy()
    df['cluster'] = clusters
    
    cluster_info = {
        'scaler': scaler,
        'kmeans': kmeans,
        'n_clusters': n_clusters,
        'feature_cols': feature_cols
    }
    
    return df, cluster_info

def train_projection_model_cluster(df_cluster, target_col='actual_score', n_folds=5):
    if len(df_cluster) == 0 or target_col not in df_cluster.columns:
        return None, None
    
    X, feature_cols = prepare_features(df_cluster)
    if len(X) == 0 or len(feature_cols) == 0:
        return None, None
    
    y = df_cluster[target_col].fillna(0)
    valid = ~y.isna() & (y > 0)
    
    if valid.sum() < n_folds:
        return None, None
    
    X = X[valid]
    y = y[valid]
    
    if y.nunique() <= 1:
        return None, None
    
    cat_features = [i for i, col in enumerate(feature_cols) 
                   if col in ['ownership_tier_encoded', 'position_encoded']]
    
    models = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if y_train.nunique() <= 1:
            continue
        
        model = CatBoostRegressor(
            cat_features=cat_features if cat_features else None,
            verbose=False,
            random_seed=42,
            iterations=100
        )
        
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)
        models.append(model)
    
    if not models:
        return None, None
    
    return models, feature_cols

def train_lineup_value_model_cluster(df_cluster, n_folds=5):
    if len(df_cluster) == 0 or 'in_top_1pct' not in df_cluster.columns:
        return None, None
    
    X, feature_cols = prepare_features(df_cluster)
    if len(X) == 0 or len(feature_cols) == 0:
        return None, None
    
    y = df_cluster['in_top_1pct'].fillna(0).astype(int)
    
    if y.sum() == 0 or y.nunique() < 2:
        return None, None
    
    if len(y) < n_folds:
        return None, None
    
    cat_features = [i for i, col in enumerate(feature_cols) 
                   if col in ['ownership_tier_encoded', 'position_encoded']]
    
    models = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if y_train.nunique() < 2 or y_train.sum() == 0:
            continue
        
        model = CatBoostClassifier(
            cat_features=cat_features if cat_features else None,
            verbose=False,
            random_seed=42,
            iterations=100
        )
        
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)
        models.append(model)
    
    if not models:
        return None, None
    
    return models, feature_cols

def train_all_cluster_models(df, target_col='actual_score', n_folds=5):
    if len(df) == 0:
        return {}, {}
    
    df_clustered, cluster_info = cluster_players(df)
    
    if 'cluster' not in df_clustered.columns:
        return {}, {}
    
    projection_models = {}
    lineup_value_models = {}
    
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        df_cluster = df_clustered[df_clustered['cluster'] == cluster_id].copy()
        
        proj_models, proj_features = train_projection_model_cluster(df_cluster, target_col, n_folds)
        if proj_models is not None:
            projection_models[cluster_id] = {
                'models': proj_models,
                'features': proj_features
            }
        
        lineup_models, lineup_features = train_lineup_value_model_cluster(df_cluster, n_folds)
        if lineup_models is not None:
            lineup_value_models[cluster_id] = {
                'models': lineup_models,
                'features': lineup_features
            }
    
    return projection_models, lineup_value_models, cluster_info

def assign_clusters(df, cluster_info):
    if len(df) == 0 or 'scaler' not in cluster_info or 'kmeans' not in cluster_info:
        df['cluster'] = 0
        return df
    
    X, _ = prepare_features(df)
    if len(X) == 0:
        df['cluster'] = 0
        return df
    
    scaler = cluster_info['scaler']
    kmeans = cluster_info['kmeans']
    X_scaled = scaler.transform(X)
    clusters = kmeans.predict(X_scaled)
    
    df = df.copy()
    df['cluster'] = clusters
    return df

def predict_projection(models_dict, cluster_info, df):
    if not models_dict or len(df) == 0:
        return pd.Series([0] * len(df))
    
    df_clustered = assign_clusters(df, cluster_info)
    
    if 'cluster' not in df_clustered.columns:
        return pd.Series([0] * len(df))
    
    predictions = []
    
    for _, row in df_clustered.iterrows():
        cluster_id = row['cluster']
        
        if cluster_id not in models_dict:
            predictions.append(0)
            continue
        
        cluster_models = models_dict[cluster_id]
        X, _ = prepare_features(pd.DataFrame([row]))
        
        if len(X) == 0:
            predictions.append(0)
            continue
        
        cluster_preds = []
        for model in cluster_models['models']:
            pred = model.predict(X)
            cluster_preds.append(pred[0])
        
        predictions.append(np.mean(cluster_preds))
    
    return pd.Series(predictions)

def predict_lineup_value(models_dict, cluster_info, df):
    if not models_dict or len(df) == 0:
        return pd.Series([0.0] * len(df))
    
    df_clustered = assign_clusters(df, cluster_info)
    
    if 'cluster' not in df_clustered.columns:
        return pd.Series([0.0] * len(df))
    
    probabilities = []
    
    for _, row in df_clustered.iterrows():
        cluster_id = row['cluster']
        
        if cluster_id not in models_dict:
            probabilities.append(0.0)
            continue
        
        cluster_models = models_dict[cluster_id]
        X, _ = prepare_features(pd.DataFrame([row]))
        
        if len(X) == 0:
            probabilities.append(0.0)
            continue
        
        cluster_probs = []
        for model in cluster_models['models']:
            prob = model.predict_proba(X)[:, 1]
            cluster_probs.append(prob[0])
        
        probabilities.append(np.mean(cluster_probs))
    
    return pd.Series(probabilities)
