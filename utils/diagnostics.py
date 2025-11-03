"""
Diagnostic utilities for new features.

Outputs feature importance, ownership reports, and other diagnostics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from config.settings import PROJECT_ROOT, PROCESSED_DIR
from utils.logger import setup_logger

logger = setup_logger()

OUTPUT_DIR = PROJECT_ROOT / "out"
DIAG_DIR = OUTPUT_DIR / "diag"
REPORTS_DIR = PROCESSED_DIR  # Use data/processed for validation reports
OUTPUT_DIR.mkdir(exist_ok=True)
DIAG_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def save_feature_importance(feature_importance_df: pd.DataFrame, 
                           model_name: str = "mf_regression") -> Path:
    """
    Save feature importance to CSV.
    
    Args:
        feature_importance_df: DataFrame with feature and importance columns
        model_name: Name of model (for filename)
    
    Returns:
        Path to saved file
    """
    filepath = DIAG_DIR / f"{model_name}_feature_importance.csv"
    feature_importance_df.to_csv(filepath, index=False)
    logger.info(f"Saved feature importance to {filepath}")
    return filepath


def generate_ownership_report(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate ownership/pivot report.
    
    Args:
        players_df: Player DataFrame with Ownership, PivotAlternatives columns
    
    Returns:
        DataFrame with ownership analysis
    """
    report = []
    
    if 'Ownership' not in players_df.columns:
        logger.warning("No Ownership column for ownership report")
        return pd.DataFrame()
    
    # Ownership distribution
    ownership_ranges = [
        (0, 10, 'Very Low'),
        (10, 20, 'Low'),
        (20, 30, 'Medium'),
        (30, 50, 'High'),
        (50, 100, 'Very High')
    ]
    
    for min_own, max_own, label in ownership_ranges:
        mask = (players_df['Ownership'] >= min_own) & (players_df['Ownership'] < max_own)
        count = mask.sum()
        
        if count > 0:
            avg_proj = players_df.loc[mask, 'Projected'].mean() if 'Projected' in players_df.columns else 0
            avg_value = players_df.loc[mask, 'Value'].mean() if 'Value' in players_df.columns else 0
            
            report.append({
                'OwnershipRange': label,
                'Count': count,
                'AvgProjection': avg_proj,
                'AvgValue': avg_value
            })
    
    # Pivot analysis
    if 'PivotAlternatives' in players_df.columns:
        has_pivots = players_df['PivotAlternatives'].apply(len) > 0
        pivot_count = has_pivots.sum()
        
        if pivot_count > 0:
            report.append({
                'OwnershipRange': 'Pivot Opportunities',
                'Count': pivot_count,
                'AvgProjection': 0,
                'AvgValue': 0
            })
    
    return pd.DataFrame(report)


def save_ownership_report(players_df: pd.DataFrame) -> Path:
    """Save ownership/pivot report to CSV"""
    report_df = generate_ownership_report(players_df)
    
    if len(report_df) == 0:
        logger.warning("Empty ownership report")
        return None
    
    filepath = DIAG_DIR / "ownership_report.csv"
    report_df.to_csv(filepath, index=False)
    logger.info(f"Saved ownership report to {filepath}")
    return filepath


def generate_projection_summary(players_df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for projections.
    
    Args:
        players_df: Player DataFrame with projections and features
    
    Returns:
        Dictionary with summary stats
    """
    summary = {}
    
    if 'Projected' in players_df.columns:
        summary['projection_mean'] = float(players_df['Projected'].mean())
        summary['projection_std'] = float(players_df['Projected'].std())
        summary['projection_min'] = float(players_df['Projected'].min())
        summary['projection_max'] = float(players_df['Projected'].max())
    
    if 'Ceiling' in players_df.columns:
        summary['ceiling_mean'] = float(players_df['Ceiling'].mean())
        summary['ceiling_max'] = float(players_df['Ceiling'].max())
    
    if 'Floor' in players_df.columns:
        summary['floor_mean'] = float(players_df['Floor'].mean())
        summary['floor_min'] = float(players_df['Floor'].min())
    
    if 'Value' in players_df.columns:
        summary['value_mean'] = float(players_df['Value'].mean())
        summary['value_max'] = float(players_df['Value'].max())
    
    # Feature usage counts
    feature_cols = [
        'RestState', 'IsHome', 'GamePaceScore', 'TeamMomentumZScore',
        'ValueAdjCeiling', 'ExplosiveScore', 'OwnAdjValue',
        'CorrelationStackScore'
    ]
    
    summary['features_present'] = {
        col: col in players_df.columns 
        for col in feature_cols
    }
    
    return summary


def save_projection_summary(players_df: pd.DataFrame, filename: str = "projection_summary") -> Path:
    """Save projection summary to JSON"""
    import json
    
    summary = generate_projection_summary(players_df)
    summary['timestamp'] = datetime.now().isoformat()
    
    filepath = DIAG_DIR / f"{filename}.json"
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved projection summary to {filepath}")
    return filepath


def generate_lineup_diagnostics(lineup: Dict, players_df: pd.DataFrame) -> Dict:
    """
    Generate diagnostics for a generated lineup.
    
    Args:
        lineup: Lineup dictionary with player names
        players_df: Full player DataFrame
    
    Returns:
        Dictionary with lineup diagnostics
    """
    lineup_players = list(lineup.keys())
    lineup_df = players_df[players_df['Player'].isin(lineup_players)]
    
    diagnostics = {
        'total_projection': float(lineup_df['Projected'].sum()) if 'Projected' in lineup_df.columns else 0.0,
        'total_ceiling': float(lineup_df['Ceiling'].sum()) if 'Ceiling' in lineup_df.columns else 0.0,
        'total_floor': float(lineup_df['Floor'].sum()) if 'Floor' in lineup_df.columns else 0.0,
        'total_ownership': float(lineup_df['Ownership'].sum()) if 'Ownership' in lineup_df.columns else 0.0,
        'avg_ownership': float(lineup_df['Ownership'].mean()) if 'Ownership' in lineup_df.columns else 0.0,
        'total_salary': float(lineup_df['Salary'].sum()) if 'Salary' in lineup_df.columns else 0.0,
    }
    
    # Risk score
    if 'Projected' in lineup_df.columns and 'Ceiling' in lineup_df.columns and 'Floor' in lineup_df.columns:
        spread = lineup_df['Ceiling'].sum() - lineup_df['Floor'].sum()
        mean = lineup_df['Projected'].sum()
        diagnostics['risk_score'] = float(spread / mean) if mean > 0 else 0.0
    
    # Leverage score
    if 'OwnAdjValue' in lineup_df.columns:
        diagnostics['leverage_score'] = float(lineup_df['OwnAdjValue'].sum())
    
    # Explosive score
    if 'ExplosiveScore' in lineup_df.columns:
        diagnostics['explosive_score'] = float(lineup_df['ExplosiveScore'].mean())
    
    # Correlation stack info
    if 'CorrelationTriadMemberships' in lineup_df.columns:
        triads = lineup_df['CorrelationTriadMemberships'].apply(len).sum()
        diagnostics['correlation_triads'] = int(triads)
    
    return diagnostics


def save_lineup_diagnostics(lineup: Dict, players_df: pd.DataFrame, 
                           filename: str = "lineup_diagnostics") -> Path:
    """Save lineup diagnostics to JSON"""
    import json
    
    diagnostics = generate_lineup_diagnostics(lineup, players_df)
    diagnostics['timestamp'] = datetime.now().isoformat()
    
    filepath = DIAG_DIR / f"{filename}.json"
    with open(filepath, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    
    logger.info(f"Saved lineup diagnostics to {filepath}")
    return filepath


def generate_ownership_comparison(players_df: pd.DataFrame, 
                                  results_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compare estimated ownership vs actual ownership from results.
    
    Args:
        players_df: Player DataFrame with estimated Ownership (or ProjOwn)
        results_df: Optional results DataFrame with %Drafted
    
    Returns:
        DataFrame with ownership comparison
    """
    from backtest.score_calculator import get_actual_ownership
    
    if results_df is None:
        return pd.DataFrame()
    
    # Merge actual ownership
    comparison_df = get_actual_ownership(players_df, results_df)
    
    # Check for Ownership or ProjOwn column
    ownership_col = None
    if 'Ownership' in comparison_df.columns:
        ownership_col = 'Ownership'
    elif 'ProjOwn' in comparison_df.columns:
        ownership_col = 'ProjOwn'
        # Rename for consistency
        comparison_df['Ownership'] = comparison_df['ProjOwn']
    
    if 'ActualOwnership' not in comparison_df.columns or ownership_col is None:
        return pd.DataFrame()
    
    # Calculate comparison metrics
    comparison_df['OwnershipError'] = (
        comparison_df['ActualOwnership'] - comparison_df['Ownership']
    ).fillna(0.0)
    
    comparison_df['OwnershipErrorAbs'] = comparison_df['OwnershipError'].abs()
    comparison_df['OwnershipErrorPct'] = (
        comparison_df['OwnershipError'] / comparison_df['ActualOwnership'] * 100
    ).where(comparison_df['ActualOwnership'] > 0, 0.0)
    
    # Select relevant columns
    cols = ['Player', 'Salary', 'Projected', 'Ownership', 'ActualOwnership', 
            'OwnershipError', 'OwnershipErrorAbs', 'OwnershipErrorPct']
    
    available_cols = [c for c in cols if c in comparison_df.columns]
    result_df = comparison_df[available_cols].copy()
    
    # Sort by absolute error (worst estimates first)
    if 'OwnershipErrorAbs' in result_df.columns:
        result_df = result_df.sort_values('OwnershipErrorAbs', ascending=False)
    
    return result_df


def save_ownership_comparison(players_df: pd.DataFrame, results_df: Optional[pd.DataFrame] = None,
                             slate_date: Optional[str] = None) -> Optional[Path]:
    """
    Save ownership comparison (estimated vs actual) to CSV.
    
    Args:
        players_df: Player DataFrame with estimated Ownership
        results_df: Optional results DataFrame
        slate_date: Optional slate date for filename
    
    Returns:
        Path to saved file or None
    """
    comparison_df = generate_ownership_comparison(players_df, results_df)
    
    if len(comparison_df) == 0:
        logger.warning("No ownership comparison data available")
        return None
    
    if slate_date:
        filename = f"ownership_comparison_{slate_date}.csv"
    else:
        filename = "ownership_comparison.csv"
    
    filepath = REPORTS_DIR / filename
    comparison_df.to_csv(filepath, index=False)
    
    # Log summary stats
    if 'OwnershipErrorAbs' in comparison_df.columns:
        mae = comparison_df['OwnershipErrorAbs'].mean()
        logger.info(f"Ownership estimation MAE: {mae:.2f}%")
        logger.info(f"Saved ownership comparison to {filepath}")
    else:
        logger.info(f"Saved ownership comparison to {filepath}")
    
    return filepath


def generate_score_validation(calculated_score: float, lineup: Dict, 
                              results_df: pd.DataFrame) -> Dict:
    """
    Generate score validation report comparing calculated vs lineup totals.
    
    Args:
        calculated_score: Score from summing individual FPTS
        lineup: Lineup dictionary
        results_df: Results DataFrame
    
    Returns:
        Dictionary with validation results
    """
    from backtest.score_calculator import validate_score_vs_lineup_totals
    
    validation = validate_score_vs_lineup_totals(calculated_score, lineup, results_df)
    return validation


def save_score_validation(calculated_score: float, lineup: Dict, 
                          results_df: pd.DataFrame, slate_date: Optional[str] = None) -> Optional[Path]:
    """
    Save score validation to JSON.
    
    Args:
        calculated_score: Calculated lineup score
        lineup: Lineup dictionary
        results_df: Results DataFrame
        slate_date: Optional slate date for filename
    
    Returns:
        Path to saved file or None
    """
    import json
    
    validation = generate_score_validation(calculated_score, lineup, results_df)
    
    if not validation.get('validation_available', False):
        logger.debug("Score validation not available")
        return None
    
    validation['calculated_score'] = calculated_score
    validation['timestamp'] = datetime.now().isoformat()
    
    if slate_date:
        filename = f"score_validation_{slate_date}.json"
    else:
        filename = "score_validation.json"
    
    filepath = REPORTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(validation, f, indent=2)
    
    # Log warnings if discrepancy found
    if validation.get('warning'):
        logger.warning(f"Score validation: {validation['warning']}")
    elif validation.get('match_found'):
        logger.info(f"Score validation: Match found, discrepancy: {validation.get('discrepancy', 0):.2f} FPTS")
    
    return filepath

