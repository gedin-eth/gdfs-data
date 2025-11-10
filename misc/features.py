import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from misc.loader import load_dk_contest, extract_top_lineups

def calculate_beat_rate(df, player_col='player', actual_col='actual_score', projected_col='projected'):
    if len(df) == 0:
        return {}
    
    beat_rates = {}
    for player in df[player_col].unique():
        player_data = df[df[player_col] == player]
        if projected_col in player_data.columns and actual_col in player_data.columns:
            player_data = player_data.dropna(subset=[projected_col, actual_col])
            if len(player_data) > 0:
                beat_count = (player_data[actual_col] > player_data[projected_col]).sum()
                beat_rates[player] = beat_count / len(player_data)
            else:
                beat_rates[player] = 0.5
        else:
            beat_rates[player] = 0.5
    return beat_rates

def calculate_variance_factor(df, player_col='player', actual_col='actual_score', projected_col='projected'):
    if len(df) == 0:
        return {}
    
    variance_factors = {}
    for player in df[player_col].unique():
        player_data = df[df[player_col] == player]
        if projected_col in player_data.columns and actual_col in player_data.columns:
            player_data = player_data.dropna(subset=[projected_col, actual_col])
            if len(player_data) > 0:
                std_dev = player_data[actual_col].std()
                mean_proj = player_data[projected_col].mean()
                variance_factors[player] = std_dev / mean_proj if mean_proj > 0 else 1.0
            else:
                variance_factors[player] = 1.0
        else:
            variance_factors[player] = 1.0
    return variance_factors

def get_game_environment(slate_date):
    dk_df = load_dk_contest(slate_date)
    if dk_df is None:
        return {}
    
    top_lineups = extract_top_lineups(dk_df, pct=0.01)
    if not top_lineups:
        return {}
    
    game_players = {}
    for lineup in top_lineups:
        for player in lineup['players']:
            if player not in game_players:
                game_players[player] = set()
            game_players[player].update([p for p in lineup['players'] if p != player])
    
    return {p: len(game_players[p]) for p in game_players}

def calculate_salary_efficiency(df, player_col='player', actual_col='actual_score'):
    if len(df) == 0:
        return {}
    
    efficiency = {}
    for player in df[player_col].unique():
        player_data = df[df[player_col] == player]
        
        pts = None
        if actual_col in player_data.columns:
            player_data_actual = player_data.dropna(subset=[actual_col])
            if len(player_data_actual) > 0:
                pts = player_data_actual[actual_col].mean()
        
        if pts is None and 'projected' in player_data.columns:
            player_data_proj = player_data.dropna(subset=['projected'])
            if len(player_data_proj) > 0:
                pts = player_data_proj['projected'].mean()
        
        if pts is not None and pts > 0:
            if 'salary' in player_data.columns:
                salary_data = player_data.dropna(subset=['salary'])
                if len(salary_data) > 0:
                    salary = salary_data['salary'].mean()
                    efficiency[player] = pts / (salary / 1000) if salary > 0 else pts / 50
                else:
                    efficiency[player] = pts / 50
            else:
                efficiency[player] = pts / 50
        else:
            efficiency[player] = 0
    return efficiency

def get_positional_leverage(slate_date):
    dk_df = load_dk_contest(slate_date)
    if dk_df is None:
        return {}
    
    top_lineups = extract_top_lineups(dk_df, pct=0.01)
    if not top_lineups:
        return {}
    
    player_data = dk_df[['Player', 'Roster Position']].dropna().drop_duplicates('Player')
    position_counts = {}
    
    for lineup in top_lineups:
        lineup_positions = {}
        for player in lineup['players']:
            pos_row = player_data[player_data['Player'] == player]
            if len(pos_row) > 0:
                pos = pos_row['Roster Position'].iloc[0]
                lineup_positions[pos] = lineup_positions.get(pos, 0) + 1
        
        for pos, count in lineup_positions.items():
            if pos not in position_counts:
                position_counts[pos] = []
            position_counts[pos].append(count)
    
    leverage = {}
    for pos in position_counts:
        avg_count = np.mean(position_counts[pos])
        leverage[pos] = avg_count
    
    return leverage

def classify_ownership_tier(ownership):
    if pd.isna(ownership) or ownership < 0:
        return 'low'
    if ownership < 10:
        return 'low'
    elif ownership < 20:
        return 'medium'
    else:
        return 'high'

def engineer_features(labels_df, slate_date):
    if len(labels_df) == 0:
        return pd.DataFrame()
    
    df = labels_df.copy()
    
    beat_rates = calculate_beat_rate(df)
    variance_factors = calculate_variance_factor(df)
    game_env = get_game_environment(slate_date)
    salary_eff = calculate_salary_efficiency(df)
    pos_leverage = get_positional_leverage(slate_date)
    
    df['beat_rate'] = df['player'].map(beat_rates).fillna(0.5)
    df['variance_factor'] = df['player'].map(variance_factors).fillna(1.0)
    df['game_environment'] = df['player'].map(game_env).fillna(0)
    df['salary_efficiency'] = df['player'].map(salary_eff).fillna(0)
    df['ownership_tier'] = df['field_ownership'].apply(classify_ownership_tier)
    
    if 'Roster Position' in df.columns:
        df['positional_leverage'] = df['Roster Position'].map(pos_leverage).fillna(1.0)
    else:
        df['positional_leverage'] = 1.0
    
    return df

