"""Data validation utilities"""
import pandas as pd

def validate_players_df(df: pd.DataFrame) -> bool:
    """Validate player dataframe has required columns"""
    required_cols = ['Player', 'Position', 'Team', 'Salary', 'Projected']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True

def validate_lineup(lineup: dict, salary_cap: int = 50000) -> bool:
    """Validate lineup meets constraints"""
    if sum(lineup.values()) > salary_cap:
        raise ValueError(f"Lineup exceeds salary cap: ${sum(lineup.values())}")
    return True

