"""Prepare slate input file with incremental merge logic"""
import pandas as pd
from pathlib import Path
from config.settings import PROCESSED_DIR
from utils.data_loader import load_linestar_data, load_scrapper_data, save_processed_data
from projection.generate_projections import generate_projections
from ownership.estimate_ownership import estimate_ownership_for_slate
from utils.logger import setup_logger

logger = setup_logger()

def _merge_with_fuzzy_matching(df: pd.DataFrame, scrapper_df: pd.DataFrame, 
                               target_cols: list) -> pd.DataFrame:
    """
    Merge scrapper data using fuzzy matching for player names.
    Handles cases like 'Alex Sarr' vs 'Alexandre Sarr'.
    """
    from utils.name_matcher import match_player_name
    
    # First try exact match
    scrapper_merge_cols = ['Player'] + [col for col in target_cols if col in scrapper_df.columns]
    df_merged = df.merge(scrapper_df[scrapper_merge_cols], on='Player', how='left', suffixes=('', '_scrapper'))
    
    # Check which players didn't match
    unmatched = df_merged[df_merged[target_cols[0]].isna()]['Player'].unique() if target_cols else []
    
    if len(unmatched) > 0:
        # Try fuzzy matching for unmatched players
        from scrapper.scrape_games import match_players_to_linestar
        scrapper_df_matched = match_players_to_linestar(scrapper_df.copy(), df['Player'].tolist(), threshold=0.7)
        
        # Re-merge only unmatched players
        df_unmatched = df[df['Player'].isin(unmatched)].copy()
        df_matched = df_unmatched.merge(scrapper_df_matched[scrapper_merge_cols], on='Player', how='left')
        
        # Combine matched and unmatched
        df_merged.loc[df_unmatched.index, scrapper_merge_cols] = df_matched[scrapper_merge_cols].values
    
    return df_merged

def prepare_slate(slate_date: str, 
                  generate_projections_flag: bool = True,
                  estimate_ownership_flag: bool = True,
                  strategy: str = 'baseline',
                  use_ai: bool = False) -> pd.DataFrame:
    """
    Prepare slate input file by merging LineStar and scrapper data.
    Optionally generates projections and estimates ownership.
    
    Args:
        slate_date: Date string for slate (YYYY-MM-DD)
        generate_projections_flag: Whether to generate projections (default True)
        estimate_ownership_flag: Whether to estimate ownership (default True)
        strategy: Strategy name for projections (default 'baseline')
        use_ai: Whether to use AI enhancements (default False)
    
    Returns:
        DataFrame with all features, projections, and ownership
    """
    processed_file = PROCESSED_DIR / f"slate_{slate_date}.csv"
    
    # If processed file exists, load it and check what's missing
    if processed_file.exists():
        df = pd.read_csv(processed_file)
        logger.info(f"Loaded existing slate file for {slate_date} with {len(df)} players")
        
        # Check if linestar data is missing and available
        linestar_cols = ['Player', 'Position', 'Team', 'Salary', 'Projected']
        missing_linestar_cols = [col for col in linestar_cols if col not in df.columns]
        if missing_linestar_cols:
            try:
                linestar_df = load_linestar_data(slate_date)
                # Merge only missing columns
                merge_cols = ['Player'] + missing_linestar_cols
                df = df.merge(linestar_df[merge_cols], on='Player', how='left')
                logger.info(f"Added missing LineStar columns: {missing_linestar_cols}")
            except FileNotFoundError:
                pass  # Linestar not available
        
        # Check if scrapper data is missing and available (from previous games)
        scrapper_df = load_scrapper_data(slate_date, historical=True, lookback_days=30)
        if scrapper_df is not None:
            scrapper_cols = [col for col in scrapper_df.columns if col != 'Player']
            missing_scrapper_cols = [col for col in scrapper_cols if col not in df.columns]
            if missing_scrapper_cols:
                # Use fuzzy matching for merge
                df = _merge_with_fuzzy_matching(df, scrapper_df, missing_scrapper_cols)
                logger.info(f"Added missing scrapper columns: {missing_scrapper_cols}")
        
        # Regenerate projections if requested or if missing
        if generate_projections_flag or 'Projected' not in df.columns or df['Projected'].isna().any():
            logger.info("Generating projections...")
            df = generate_projections(df, scrapper_df, strategy=strategy, use_ai=use_ai)
        
        # Estimate ownership if requested or if missing
        if estimate_ownership_flag or 'Ownership' not in df.columns:
            logger.info("Estimating ownership...")
            historical_data = None  # Will be loaded automatically by estimate_ownership_for_slate
            df = estimate_ownership_for_slate(df, scrapper_df, historical_data, slate_date=slate_date)
            logger.info(f"Ownership estimates generated (range: {df['Ownership'].min():.2f}% - {df['Ownership'].max():.2f}%)")
        
        # Save updated processed file
        save_processed_data(df, f"slate_{slate_date}.csv")
        logger.info(f"Saved updated slate file: {processed_file}")
        return df
    
    # Processed file doesn't exist - create from available sources
    logger.info(f"Creating new slate file for {slate_date}")
    try:
        df = load_linestar_data(slate_date)
        logger.info(f"Loaded {len(df)} players from LineStar")
    except FileNotFoundError:
        raise FileNotFoundError(f"No data sources available for slate {slate_date}")
    
    # Merge scrapper data from PREVIOUS games (for input features)
    # Use historical=True to load PBP from past games, not current slate
    # Look back up to 30 days to find any available historical PBP data
    scrapper_df = load_scrapper_data(slate_date, historical=True, lookback_days=30)
    if scrapper_df is not None:
        logger.info(f"Loaded scrapper data for {len(scrapper_df)} players")
        scrapper_cols = [col for col in scrapper_df.columns if col != 'Player']
        df = _merge_with_fuzzy_matching(df, scrapper_df, scrapper_cols)
    else:
        logger.info("No historical scrapper data available")
    
    # Generate projections
    if generate_projections_flag:
        logger.info("Generating projections...")
        df = generate_projections(df, scrapper_df, strategy=strategy, use_ai=use_ai)
    else:
        # Ensure Projected column exists (use from LineStar if available)
        if 'Projected' not in df.columns:
            logger.warning("Projections not generated and not found in data")
    
    # Estimate ownership
    if estimate_ownership_flag:
        logger.info("Estimating ownership...")
        historical_data = None  # Will be loaded automatically by estimate_ownership_for_slate
        df = estimate_ownership_for_slate(df, scrapper_df, historical_data, slate_date=slate_date)
        logger.info(f"Ownership estimates generated (range: {df['Ownership'].min():.2f}% - {df['Ownership'].max():.2f}%)")
    else:
        # Ownership might be in LineStar data as ProjOwn
        if 'ProjOwn' in df.columns and 'Ownership' not in df.columns:
            df['Ownership'] = pd.to_numeric(df['ProjOwn'], errors='coerce')
            logger.info("Using LineStar ProjOwn as Ownership")
    
    # Save processed file
    save_processed_data(df, f"slate_{slate_date}.csv")
    logger.info(f"Saved slate file: {processed_file}")
    logger.info(f"Final slate has {len(df)} players with columns: {list(df.columns)}")
    
    return df

