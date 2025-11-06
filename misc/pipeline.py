import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from config.settings import RESULTS_DIR
from misc.loader import load_dk_contest
from misc.labels import create_labels
from misc.features import engineer_features
from misc.patterns import extract_patterns, build_correlation_matrix
from misc.models import train_all_cluster_models
from misc.edge import add_edge_scores

def get_available_slates(days_back=30):
    available = []
    for file_path in sorted(RESULTS_DIR.glob('dk_*.csv')):
        date_str = file_path.stem.replace('dk_', '')
        if len(date_str) == 8:
            try:
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                if (datetime.now() - date_obj).days <= days_back:
                    available.append(date_obj.strftime('%Y-%m-%d'))
            except:
                pass
    return available

def train_on_historical_data(days_back=30, n_folds=5):
    slates = get_available_slates(days_back)
    if not slates:
        return {}, {}, {}
    
    all_labels = []
    for slate_date in slates:
        try:
            labels = create_labels(slate_date)
            if len(labels) > 0:
                features = engineer_features(labels, slate_date)
                if len(features) > 0:
                    features['slate_date'] = slate_date
                    all_labels.append(features)
        except Exception as e:
            continue
    
    if not all_labels:
        return {}, {}, {}
    
    combined_df = pd.concat(all_labels, ignore_index=True)
    
    if len(combined_df) == 0:
        return {}, {}, {}
    
    projection_models, lineup_value_models, cluster_info = train_all_cluster_models(
        combined_df, target_col='actual_score', n_folds=n_folds
    )
    
    return projection_models, lineup_value_models, cluster_info

def generate_edge_scores(slate_date, projection_models=None, lineup_value_models=None, cluster_info=None):
    if projection_models is None or not projection_models:
        projection_models, lineup_value_models, cluster_info = train_on_historical_data()
    
    if not projection_models:
        return pd.DataFrame()
    
    labels = create_labels(slate_date)
    if len(labels) == 0:
        return pd.DataFrame()
    
    features = engineer_features(labels, slate_date)
    if len(features) == 0:
        return pd.DataFrame()
    
    correlation_matrix = build_correlation_matrix(slate_date)
    
    result = add_edge_scores(
        features, projection_models, lineup_value_models, cluster_info,
        correlation_matrix
    )
    
    return result

def main():
    recent_slates = get_available_slates(days_back=7)
    if not recent_slates:
        print("No recent slates found")
        return
    
    print(f"Training on {len(recent_slates)} slates...")
    projection_models, lineup_value_models, cluster_info = train_on_historical_data()
    
    if not projection_models:
        print("Failed to train models")
        return
    
    print(f"Found {len(projection_models)} clusters for projection models")
    print(f"Found {len(lineup_value_models)} clusters for lineup value models")
    
    latest_slate = recent_slates[-1]
    print(f"Generating edge scores for {latest_slate}...")
    edge_scores = generate_edge_scores(
        latest_slate, projection_models, lineup_value_models, cluster_info
    )
    
    if len(edge_scores) > 0:
        output = edge_scores[['player', 'edge_score', 'top_1pct_rate', 'ownership_delta']].sort_values('edge_score', ascending=False)
        print("\nTop players by Edge Score:")
        print(output.head(20))
    else:
        print("No edge scores generated")

if __name__ == '__main__':
    main()

