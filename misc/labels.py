import pandas as pd
from pathlib import Path
from config.settings import DATA_DIR
from misc.loader import load_dk_contest, extract_top_lineups, get_player_data
from utils.name_matcher import match_player_name

def load_results_data(slate_date):
    date_str = slate_date.replace('-', '')
    file_path = DATA_DIR / f"results_{date_str}.csv"
    if not file_path.exists():
        return pd.DataFrame()
    
    with open(file_path, 'r') as f:
        skip_lines = 0
        for line in f:
            if line.startswith('Player Name'):
                break
            skip_lines += 1
    
    df = pd.read_csv(file_path, skiprows=skip_lines)
    return df

def create_labels(slate_date):
    dk_df = load_dk_contest(slate_date)
    if dk_df is None or len(dk_df) == 0:
        return pd.DataFrame()
    
    top_lineups = extract_top_lineups(dk_df, pct=0.01)
    if not top_lineups:
        return pd.DataFrame()
    
    player_data = get_player_data(dk_df)
    results_df = load_results_data(slate_date)
    
    all_players = set()
    for lineup in top_lineups:
        all_players.update(lineup['players'])
    
    if len(all_players) == 0:
        return pd.DataFrame()
    
    top_1pct_players = set()
    for lineup in top_lineups:
        top_1pct_players.update(lineup['players'])
    
    total_top_lineups = len(top_lineups)
    
    labels = []
    for player in all_players:
        appearances = sum(1 for lineup in top_lineups if player in lineup['players'])
        top_1pct_rate = appearances / total_top_lineups if total_top_lineups > 0 else 0
        
        player_row = player_data[player_data['Player'] == player]
        field_ownership = player_row['%Drafted'].iloc[0] if len(player_row) > 0 and '%Drafted' in player_row.columns else 0
        
        ownership_delta = top_1pct_rate * 100 - field_ownership if field_ownership else 0
        
        actual_score = None
        if len(results_df) > 0:
            matched_name = match_player_name(player, results_df['Player Name'].tolist())
            results_row = results_df[results_df['Player Name'] == matched_name]
            if len(results_row) > 0:
                actual_score = results_row['Actual Score'].iloc[0] if 'Actual Score' in results_row.columns else None
        
        roi_coef = 0
        if actual_score is not None:
            roi_coef = actual_score * top_1pct_rate
        
        player_row_data = player_row.iloc[0] if len(player_row) > 0 else pd.Series()
        roster_position = player_row_data.get('Roster Position', '') if len(player_row_data) > 0 else ''
        
        labels.append({
            'slate_date': slate_date,
            'player': player,
            'in_top_1pct': 1 if player in top_1pct_players else 0,
            'top_1pct_rate': top_1pct_rate,
            'field_ownership': field_ownership,
            'ownership_delta': ownership_delta,
            'actual_score': actual_score,
            'roi_coef': roi_coef,
            'appearances': appearances,
            'Roster Position': roster_position
        })
    
    return pd.DataFrame(labels)

