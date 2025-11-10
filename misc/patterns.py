import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from collections import defaultdict
from misc.loader import load_dk_contest, extract_top_lineups

def build_correlation_matrix(slate_date):
    dk_df = load_dk_contest(slate_date)
    if dk_df is None:
        return pd.DataFrame()
    
    top_lineups = extract_top_lineups(dk_df, pct=0.01)
    if not top_lineups:
        return pd.DataFrame()
    
    player_pairs = defaultdict(int)
    all_players = set()
    
    for lineup in top_lineups:
        players = lineup['players']
        all_players.update(players)
        for i, p1 in enumerate(players):
            for p2 in players[i+1:]:
                pair = tuple(sorted([p1, p2]))
                player_pairs[pair] += 1
    
    if not all_players:
        return pd.DataFrame()
    
    players_list = sorted(all_players)
    matrix = pd.DataFrame(0, index=players_list, columns=players_list)
    
    for (p1, p2), count in player_pairs.items():
        matrix.loc[p1, p2] = count
        matrix.loc[p2, p1] = count
    
    return matrix

def identify_stacks(slate_date, min_players=2):
    dk_df = load_dk_contest(slate_date)
    if dk_df is None:
        return []
    
    top_lineups = extract_top_lineups(dk_df, pct=0.01)
    if not top_lineups:
        return []
    
    player_data = dk_df[['Player', 'Roster Position']].dropna().drop_duplicates('Player')
    if len(player_data) == 0:
        return []
    
    stacks = []
    for lineup in top_lineups:
        lineup_teams = defaultdict(list)
        for player in lineup['players']:
            pos_row = player_data[player_data['Player'] == player]
            if len(pos_row) > 0:
                pos = pos_row['Roster Position'].iloc[0]
                lineup_teams[pos].append(player)
        
        for pos, players in lineup_teams.items():
            if len(players) >= min_players:
                stacks.append({
                    'position': pos,
                    'players': players,
                    'count': len(players),
                    'lineup_points': lineup['points']
                })
    
    return stacks

def find_leverage_spots(slate_date):
    dk_df = load_dk_contest(slate_date)
    if dk_df is None:
        return {}
    
    top_lineups = extract_top_lineups(dk_df, pct=0.01)
    if not top_lineups:
        return {}
    
    player_data = dk_df[['Player', '%Drafted']].dropna().drop_duplicates('Player')
    if len(player_data) == 0:
        return {}
    
    ownership_ranges = {
        'low': (0, 10),
        'medium': (10, 20),
        'high': (20, 100)
    }
    
    leverage = {range_name: {'count': 0, 'unique_players': set()} for range_name in ownership_ranges.keys()}
    
    for lineup in top_lineups:
        for player in lineup['players']:
            player_row = player_data[player_data['Player'] == player]
            if len(player_row) > 0:
                ownership = player_row['%Drafted'].iloc[0]
                if isinstance(ownership, str):
                    ownership = float(ownership.replace('%', ''))
                
                for range_name, (low, high) in ownership_ranges.items():
                    if low <= ownership < high:
                        leverage[range_name]['count'] += 1
                        leverage[range_name]['unique_players'].add(player)
                        break
    
    result = {}
    for range_name, data in leverage.items():
        result[range_name] = {
            'total_appearances': data['count'],
            'unique_players': len(data['unique_players']),
            'avg_per_lineup': data['count'] / len(top_lineups) if top_lineups else 0
        }
    
    return result

def analyze_salary_allocation(slate_date):
    dk_df = load_dk_contest(slate_date)
    if dk_df is None:
        return {}
    
    top_lineups = extract_top_lineups(dk_df, pct=0.01)
    if not top_lineups:
        return {}
    
    player_data = dk_df[['Player', 'Roster Position']].dropna().drop_duplicates('Player')
    if len(player_data) == 0:
        return {}
    
    position_spend = defaultdict(list)
    
    for lineup in top_lineups:
        lineup_spend = defaultdict(float)
        for player in lineup['players']:
            pos_row = player_data[player_data['Player'] == player]
            if len(pos_row) > 0:
                pos = pos_row['Roster Position'].iloc[0]
                lineup_spend[pos] += 1
        
        for pos, count in lineup_spend.items():
            position_spend[pos].append(count)
    
    result = {}
    for pos, counts in position_spend.items():
        result[pos] = {
            'mean': np.mean(counts),
            'std': np.std(counts),
            'min': np.min(counts),
            'max': np.max(counts)
        }
    
    return result

def extract_patterns(slate_date):
    return {
        'correlation_matrix': build_correlation_matrix(slate_date),
        'stacks': identify_stacks(slate_date),
        'leverage_spots': find_leverage_spots(slate_date),
        'salary_allocation': analyze_salary_allocation(slate_date)
    }

