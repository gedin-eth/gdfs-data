import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import re
from config.settings import RESULTS_DIR

def parse_lineup(lineup_str):
    positions = ['C', 'F', 'G', 'PF', 'PG', 'SF', 'SG', 'UTIL']
    players = []
    parts = lineup_str.split()
    for i, part in enumerate(parts):
        if part in positions:
            if i + 1 < len(parts):
                name_parts = []
                j = i + 1
                while j < len(parts) and parts[j] not in positions:
                    name_parts.append(parts[j])
                    j += 1
                if name_parts:
                    players.append(' '.join(name_parts))
    return players

def load_dk_contest(slate_date):
    date_str = slate_date.replace('-', '')
    file_path = RESULTS_DIR / f"dk_{date_str}.csv"
    if not file_path.exists():
        return None
    return pd.read_csv(file_path, low_memory=False)

def extract_top_lineups(df, pct=0.01):
    if df is None or len(df) == 0:
        return []
    
    lineups_df = df[['Rank', 'Points', 'Lineup']].dropna(subset=['Points', 'Lineup'])
    lineups_df = lineups_df.drop_duplicates(subset=['Rank', 'Points', 'Lineup'])
    
    if len(lineups_df) == 0:
        return []
    
    top_n = max(1, int(len(lineups_df) * pct))
    top_lineups = lineups_df.nsmallest(top_n, 'Rank')
    
    result = []
    for _, row in top_lineups.iterrows():
        players = parse_lineup(row['Lineup'])
        result.append({
            'rank': row['Rank'],
            'points': row['Points'],
            'players': players
        })
    return result

def get_player_data(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    player_cols = ['Player', 'Roster Position', '%Drafted', 'FPTS']
    available_cols = [c for c in player_cols if c in df.columns]
    if not available_cols:
        return pd.DataFrame()
    
    players = df[available_cols].copy()
    players = players.dropna(subset=['Player'])
    players = players.drop_duplicates(subset=['Player'])
    
    if '%Drafted' in players.columns:
        players['%Drafted'] = players['%Drafted'].str.replace('%', '').astype(float)
    
    return players

