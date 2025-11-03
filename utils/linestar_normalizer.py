"""Normalize LineStar CSV to standard input format"""
import pandas as pd

def normalize_linestar_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize LineStar CSV to standard input format:
    - Maps 'Name' -> 'Player'
    - Removes ID columns (LineStarId, DFS_ID)
    - Removes actual scores (Scored)
    - Keeps useful optional columns (Ceiling, Floor, ProjOwn)
    - Handles position format (splits PG/SG into first position)
    
    Args:
        df: Raw LineStar DataFrame
    
    Returns:
        Normalized DataFrame with standard columns
    """
    df = df.copy()
    
    # Map Name -> Player if needed
    if 'Name' in df.columns and 'Player' not in df.columns:
        df['Player'] = df['Name']
        # Remove Name column after mapping
        df = df.drop(columns=['Name'])
    
    # Remove ID columns
    id_cols = ['LineStarId', 'DFS_ID']
    df = df.drop(columns=[col for col in id_cols if col in df.columns])
    
    # Remove actual scores
    score_cols = ['Scored']
    df = df.drop(columns=[col for col in score_cols if col in df.columns])
    
    # Remove other metadata columns we don't need for optimization
    metadata_cols = ['IsOverridden', 'StartingStatus', 'SIC', 'Vegas', 'VegasML', 
                     'VegasTotals', 'VegasImplied', 'VersusStr']
    df = df.drop(columns=[col for col in metadata_cols if col in df.columns])
    
    # Handle Position: if "PG/SG" format, take first position (needed for DraftKings constraints)
    if 'Position' in df.columns:
        df['Position'] = df['Position'].astype(str).str.split('/').str[0].str.strip()
        # Map common position abbreviations
        position_map = {'PG': 'PG', 'SG': 'SG', 'SF': 'SF', 'PF': 'PF', 'C': 'C'}
        df['Position'] = df['Position'].map(position_map).fillna(df['Position'])
    
    # Ensure required columns exist
    required_cols = ['Player', 'Position', 'Team', 'Salary', 'Projected']
    
    # Validate we have the essentials
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns after normalization: {missing_required}")
    
    # Select columns to keep (prioritize required, then useful optional ones)
    keep_cols = required_cols.copy()
    optional_cols = ['Ceiling', 'Floor', 'ProjOwn', 'Ownership', 'Consistency', 
                     'Leverage', 'OppRank', 'OppRankTotal', 'Consensus', 'Safety', 
                     'AlertScore', 'PPG']
    keep_cols.extend([col for col in optional_cols if col in df.columns])
    
    # Select and reorder columns (only those that exist)
    final_cols = [col for col in keep_cols if col in df.columns]
    df = df[final_cols]
    
    return df

