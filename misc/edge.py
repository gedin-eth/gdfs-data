import pandas as pd
import numpy as np
from misc.models import predict_projection, predict_lineup_value
from misc.patterns import build_correlation_matrix

def calculate_correlation_bonus(player, correlation_matrix, projected_players):
    if len(correlation_matrix) == 0 or player not in correlation_matrix.index:
        return 0.0
    
    bonus = 0.0
    for other_player in projected_players:
        if other_player != player and other_player in correlation_matrix.columns:
            corr_value = correlation_matrix.loc[player, other_player]
            bonus += corr_value * 0.1
    
    return bonus

def calculate_lineup_edge_score(df, projection_models, lineup_value_models, cluster_info,
                                correlation_matrix=None):
    if len(df) == 0:
        return pd.Series([0.0] * len(df))
    
    df = df.copy()
    
    base_projection = predict_projection(projection_models, cluster_info, df)
    lineup_value_prob = predict_lineup_value(lineup_value_models, cluster_info, df)
    
    salary_efficiency = df['salary_efficiency'] if 'salary_efficiency' in df.columns else pd.Series([0.0] * len(df))
    beat_rate = df['beat_rate'] if 'beat_rate' in df.columns else pd.Series([0.5] * len(df))
    ownership_delta = df['ownership_delta'] if 'ownership_delta' in df.columns else pd.Series([0.0] * len(df))
    
    base_score = base_projection * salary_efficiency
    beat_multiplier = 1.0 + (beat_rate - 0.5) * 0.5
    ownership_leverage = 1.0 + (ownership_delta / 100.0) * 0.3
    
    edge_score = base_score * beat_multiplier * ownership_leverage * (1.0 + lineup_value_prob)
    
    if correlation_matrix is not None and len(correlation_matrix) > 0:
        player_list = df['player'].tolist() if 'player' in df.columns else []
        correlation_bonuses = []
        for player in player_list:
            bonus = calculate_correlation_bonus(player, correlation_matrix, player_list)
            correlation_bonuses.append(bonus)
        
        if correlation_bonuses:
            edge_score = edge_score * (1.0 + pd.Series(correlation_bonuses))
    
    return edge_score

def add_edge_scores(df, projection_models, lineup_value_models, cluster_info,
                    correlation_matrix=None):
    if len(df) == 0:
        return df
    
    df = df.copy()
    df['edge_score'] = calculate_lineup_edge_score(
        df, projection_models, lineup_value_models, cluster_info,
        correlation_matrix
    )
    
    return df

