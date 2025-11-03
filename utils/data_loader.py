"""Data loading utilities"""
import pandas as pd
from pathlib import Path
from config.settings import LINSTAR_DIR, RESULTS_DIR, SCRAPPER_DIR, PROCESSED_DIR
from utils.linestar_normalizer import normalize_linestar_data
from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_linestar_data(slate_date: str = None, file_path: Path = None, normalize: bool = True) -> pd.DataFrame:
    """
    Load LineStar CSV data for a given slate date or from specific file path.
    Automatically normalizes the data to standard format.
    
    Args:
        slate_date: Date string for slate (e.g., '2024-01-15')
        file_path: Optional specific file path (if not using slate_date)
        normalize: Whether to normalize the data (default: True)
    
    Returns:
        Normalized DataFrame with standard columns
    """
    if file_path is None:
        if slate_date is None:
            raise ValueError("Must provide either slate_date or file_path")
        
        # Try standard format first
        file_path = LINSTAR_DIR / f"linestar_{slate_date}.csv"
        
        # If not found, try ls_YYYYMMDD.csv format
        if not file_path.exists():
            date_str = slate_date.replace('-', '')
            file_path = LINSTAR_DIR / f"ls_{date_str}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"LineStar file not found: {file_path}")
    
    # Read CSV without using first column as index to prevent column shifting
    df = pd.read_csv(file_path, index_col=False)
    
    # Normalize if requested
    if normalize:
        df = normalize_linestar_data(df)
    
    return df

def load_results_data(slate_date: str) -> pd.DataFrame:
    """
    Load DraftKings contest results for a given slate date.
    Supports both formats: results_YYYY-MM-DD.csv and dk_YYYYMMDD.csv
    """
    # Try standard format first
    file_path = RESULTS_DIR / f"results_{slate_date}.csv"
    
    # If not found, try DK format (dk_YYYYMMDD.csv)
    if not file_path.exists():
        # Convert YYYY-MM-DD to YYYYMMDD
        date_str = slate_date.replace('-', '')
        file_path = RESULTS_DIR / f"dk_{date_str}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found for slate {slate_date}")
    
    df = pd.read_csv(file_path, index_col=False)
    return df

def load_scrapper_data(slate_date: str, historical: bool = False, lookback_days: int = 1) -> pd.DataFrame:
    """
    Load scrapper PBP features for a given slate date.
    
    Args:
        slate_date: Date string for slate (YYYY-MM-DD)
        historical: If True, load PBP from previous games (for input features)
                   If False, load PBP from current slate (for results)
        lookback_days: How many previous days to include (when historical=True)
    
    Returns:
        DataFrame with PBP features or None if unavailable
    """
    if historical:
        # Load PBP from previous games (for input features)
        # This aggregates features from past games to inform projections
        from datetime import datetime, timedelta
        
        all_features = []
        date_obj = datetime.strptime(slate_date, '%Y-%m-%d')
        
        # Look back N days
        for i in range(1, lookback_days + 1):
            prev_date = date_obj - timedelta(days=i)
            prev_date_str = prev_date.strftime('%Y-%m-%d')
            
            # Try to load PBP features for this previous date
            file_path = PROCESSED_DIR / f"pbp_features_{prev_date_str}.csv"
            if not file_path.exists():
                file_path = SCRAPPER_DIR / f"pbp_features_{prev_date_str}.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['GameDate'] = prev_date_str  # Tag which game this came from
                all_features.append(df)
        
        if all_features:
            # Aggregate across historical games (use mean for most features)
            df_combined = pd.concat(all_features, ignore_index=True)
            # Drop GameDate before aggregation
            agg_cols = [c for c in df_combined.columns if c != 'GameDate']
            
            df_agg = df_combined.groupby('Player').agg({
                col: 'mean' for col in agg_cols if col != 'Player'
            }).reset_index()
            return df_agg
        else:
            # No historical data found
            return None
    
    # Load PBP for current slate (for results or same-day features)
    file_path = PROCESSED_DIR / f"pbp_features_{slate_date}.csv"
    if not file_path.exists():
        file_path = SCRAPPER_DIR / f"pbp_features_{slate_date}.csv"
    if not file_path.exists():
        # Try to scrape games for this date
        logger.info(f"PBP features not found for {slate_date}, attempting to scrape...")
        try:
            from scrapper.scrape_games import scrape_games
            pbp_df = scrape_games(slate_date, force_rescrape=False)
            if pbp_df is not None:
                return pbp_df
        except Exception as e:
            logger.warning(f"Failed to scrape games for {slate_date}: {e}")
        return None
    return pd.read_csv(file_path)

def save_processed_data(df: pd.DataFrame, filename: str):
    """Save processed data to processed directory"""
    PROCESSED_DIR.mkdir(exist_ok=True)
    file_path = PROCESSED_DIR / filename
    df.to_csv(file_path, index=False)
