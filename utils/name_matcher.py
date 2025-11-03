"""Simple name matching utility for handling name mismatches"""
import pandas as pd
from difflib import get_close_matches

def match_player_name(player_name: str, candidates: list, threshold: float = 0.8) -> str:
    """
    Match player name against list of candidates using fuzzy matching.
    Returns best match or original name if no good match found.
    
    Usage (1-3 lines):
        from utils.name_matcher import match_player_name
        candidates = df2['Player'].tolist()
        df1['Player'] = df1['Player'].apply(lambda x: match_player_name(x, candidates))
    
    Or inline before merge:
        df['Player'] = df['Player'].apply(lambda x: match_player_name(x, other_df['Player'].tolist()))
    """
    # Handle NaN/float player names
    if pd.isna(player_name) or not isinstance(player_name, str):
        return player_name if isinstance(player_name, str) else str(player_name) if pd.notna(player_name) else ''
    
    if not str(player_name).strip():
        return player_name
    
    # Filter candidates to only strings (skip NaN/float)
    str_candidates = [str(c) for c in candidates if pd.notna(c) and isinstance(c, (str, int, float))]
    
    if not str_candidates:
        return player_name
    
    matches = get_close_matches(str(player_name), str_candidates, n=1, cutoff=threshold)
    return matches[0] if matches else player_name

