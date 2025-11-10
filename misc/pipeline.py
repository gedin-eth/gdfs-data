import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
from datetime import datetime, timedelta
from config.settings import RESULTS_DIR, LINSTAR_DIR
from misc.loader import load_dk_contest, get_player_data
from misc.labels import create_input_labels_ls
from misc.patterns import build_correlation_matrix
from misc.models import train_all_cluster_models
from misc.edge import add_edge_scores

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'misc_pipeline'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

def save_models(projection_models, lineup_value_models, cluster_info, train_slates):
    model_info = {
        'projection_models': projection_models,
        'lineup_value_models': lineup_value_models,
        'cluster_info': cluster_info,
        'train_slates': train_slates,
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(MODELS_DIR / 'pipeline_models.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"Models saved to {MODELS_DIR / 'pipeline_models.pkl'}")

def load_models():
    model_file = MODELS_DIR / 'pipeline_models.pkl'
    if not model_file.exists():
        return None, None, None, None
    
    try:
        with open(model_file, 'rb') as f:
            model_info = pickle.load(f)
        
        print(f"Models loaded (trained on {len(model_info['train_slates'])} slates, saved {model_info['train_date']})")
        return (model_info['projection_models'], 
                model_info['lineup_value_models'], 
                model_info['cluster_info'],
                model_info['train_slates'])
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None

def prepare_ls_features(df):
    df = df.copy()
    if 'projected' in df.columns:
        df['projected'] = pd.to_numeric(df['projected'], errors='coerce').fillna(0.0)
    if 'salary' in df.columns:
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce').fillna(0.0)
    if 'field_ownership' in df.columns:
        df['field_ownership'] = pd.to_numeric(df['field_ownership'], errors='coerce').fillna(0.0)
    if 'ownership_delta' not in df.columns:
        df['ownership_delta'] = 0.0
    if 'top_1pct_rate' not in df.columns:
        df['top_1pct_rate'] = 0.0
    if 'Roster Position' in df.columns:
        df['Roster Position'] = df['Roster Position'].fillna('UTIL')
    if 'salary' in df.columns and 'projected' in df.columns:
        salary_thousands = df['salary'].replace(0, pd.NA) / 1000.0
        df['salary_efficiency'] = (df['projected'] / salary_thousands).fillna(0.0)
    else:
        df['salary_efficiency'] = 0.0
    return df

def train_on_historical_data(days_back=30, n_folds=5, force_retrain=False):
    slates = get_available_slates(days_back)
    if not slates:
        return {}, {}, {}
    
    if not force_retrain:
        loaded_models = load_models()
        if loaded_models[0] is not None:
            loaded_slates = loaded_models[3]
            if loaded_slates and set(loaded_slates) == set(slates):
                print("Using cached models (same training data)")
                return loaded_models[0], loaded_models[1], loaded_models[2]
    
    print("Training new models...")
    all_labels = []
    for slate_date in slates:
        try:
            input_labels = create_input_labels_ls(slate_date)
            input_labels = input_labels.drop(columns=[c for c in ['actual_score', 'roi_coef', 'appearances'] if c in input_labels.columns], errors='ignore')
            dk_df = load_dk_contest(slate_date)
            if dk_df is None or len(input_labels) == 0:
                continue
            actual_df = get_player_data(dk_df)[['Player', 'FPTS']].rename(columns={'Player': 'player', 'FPTS': 'actual_score'})
            merged = input_labels.merge(actual_df, on='player', how='inner')
            merged['actual_score'] = pd.to_numeric(merged['actual_score'], errors='coerce')
            merged = merged.dropna(subset=['actual_score'])
            if len(merged) == 0:
                continue
            features = prepare_ls_features(merged)
            features['slate_date'] = slate_date
            all_labels.append(features)
        except Exception:
            continue
    
    if not all_labels:
        return {}, {}, {}
    
    combined_df = pd.concat(all_labels, ignore_index=True)
    
    if len(combined_df) == 0:
        return {}, {}, {}
    
    projection_models, lineup_value_models, cluster_info = train_all_cluster_models(
        combined_df, target_col='actual_score', n_folds=n_folds
    )
    
    save_models(projection_models, lineup_value_models, cluster_info, slates)
    
    return projection_models, lineup_value_models, cluster_info

def generate_edge_scores(slate_date, projection_models=None, lineup_value_models=None, cluster_info=None, use_linestar_only=True):
    if projection_models is None or not projection_models:
        projection_models, lineup_value_models, cluster_info = train_on_historical_data()
    
    if not projection_models:
        return pd.DataFrame()
    
    labels = create_input_labels_ls(slate_date)
    if len(labels) == 0:
        return pd.DataFrame()
    
    features = prepare_ls_features(labels)
    if len(features) == 0:
        return pd.DataFrame()
    
    correlation_matrix = build_correlation_matrix(slate_date)
    if len(correlation_matrix) == 0:
        correlation_matrix = pd.DataFrame()
    
    result = add_edge_scores(
        features, projection_models, lineup_value_models, cluster_info,
        correlation_matrix
    )
    
    return result

def get_player_actual_stats(slate_date, player_name):
    from config.settings import DATA_DIR
    from misc.loader import load_dk_contest, get_player_data
    
    date_str = slate_date.replace('-', '')
    
    # Primary source: DK contest file
    dk_df = load_dk_contest(slate_date)
    if dk_df is not None:
        player_data = get_player_data(dk_df)
        player_row = player_data[player_data['Player'] == player_name]
        if len(player_row) > 0:
            actual_score = player_row['FPTS'].iloc[0] if 'FPTS' in player_row.columns else None
            ownership = player_row['%Drafted'].iloc[0] if '%Drafted' in player_row.columns else None
            if actual_score is not None or ownership is not None:
                return actual_score, ownership
    
    # Fallback: results file
    results_file = DATA_DIR / f"results_{date_str}.csv"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                lines = f.readlines()
                header_idx = None
                for i, line in enumerate(lines):
                    if 'Player Name' in line and 'Actual Score' in line:
                        header_idx = i
                        break
                if header_idx is not None:
                    df = pd.read_csv(results_file, skiprows=header_idx)
                    player_row = df[df['Player Name'] == player_name]
                    if len(player_row) > 0:
                        actual_score = player_row['Actual Score'].iloc[0] if 'Actual Score' in player_row.columns else None
                        ownership = player_row['Ownership'].iloc[0] if 'Ownership' in player_row.columns else None
                        return actual_score, ownership
        except:
            pass
    
    return None, None

def main(backtest=False, predict_date='2025-11-06'):
    from config.settings import DATA_DIR
    
    train_slates = []
    for i in range(18):
        date = datetime(2025, 10, 21) + timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        file_str = date.strftime('%Y%m%d')
        file_path = RESULTS_DIR / f'dk_{file_str}.csv'
        if file_path.exists():
            train_slates.append(date_str)
    
    print(f"Training on {len(train_slates)} slates...")
    projection_models, lineup_value_models, cluster_info = train_on_historical_data(days_back=60)
    
    if not projection_models:
        print("Failed to train models")
        return
    
    print(f"Found {len(projection_models)} clusters for projection models\n")
    
    if backtest:
        all_slates = []
        for i in range(18):
            date = datetime(2025, 10, 21) + timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            date_file = date_str.replace('-', '')
            if (LINSTAR_DIR / f"ls_{date_file}.csv").exists() and (DATA_DIR / f"results_{date_file}.csv").exists():
                all_slates.append(date_str)
        
        for idx, slate_date in enumerate(all_slates, 1):
            print(f"{'='*80}")
            print(f"Slate {idx}/{len(all_slates)}: {slate_date}")
            print(f"{'='*80}")
            
            edge_scores = generate_edge_scores(slate_date, projection_models, lineup_value_models, cluster_info, use_linestar_only=True)
            
            if len(edge_scores) == 0:
                print("No predictions generated\n")
                continue
            
            top_10 = edge_scores.nlargest(10, 'edge_score')[['player', 'edge_score']]
            
            print(f"\nTop 10 Predicted Players:")
            print(f"{'Rank':<6} {'Player':<30} {'Edge Score':<12} {'Actual Score':<12} {'Ownership':<12}")
            print("-" * 80)
            
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                player = row['player']
                edge_score = row['edge_score']
                actual_score, ownership = get_player_actual_stats(slate_date, player)
                actual_str = f"{actual_score:.2f}" if actual_score is not None else "N/A"
                own_str = f"{ownership}" if ownership is not None and str(ownership) != '' else "N/A"
                print(f"{i:<6} {player:<30} {edge_score:<12.2f} {actual_str:<12} {own_str:<12}")
            print()
    else:
        print(f"Generating edge scores for {predict_date}...")
        edge_scores = generate_edge_scores(
            predict_date, projection_models, lineup_value_models, cluster_info, use_linestar_only=True
        )
        
        if len(edge_scores) > 0:
            output = edge_scores[['player', 'edge_score', 'top_1pct_rate', 'ownership_delta']].sort_values('edge_score', ascending=False)
            print("\nTop 10 players by Edge Score:")
            print(output.head(10).to_string(index=False))
        else:
            print("No edge scores generated")

if __name__ == '__main__':
    import sys
    backtest = '--backtest' in sys.argv or '-b' in sys.argv
    predict_date = '2025-11-06'
    if '--date' in sys.argv:
        idx = sys.argv.index('--date')
        if idx + 1 < len(sys.argv):
            predict_date = sys.argv[idx + 1]
    main(backtest=backtest, predict_date=predict_date)

