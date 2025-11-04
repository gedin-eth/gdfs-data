import sys
import pandas as pd
import numpy as np
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)
OUTPUT_DIR = Path(__file__).parent.parent / 'analysis'
OUTPUT_DIR.mkdir(exist_ok=True)

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

def analyze_slate(date_str):
    results_file = DATA_DIR / f'results_{date_str}.csv'
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        header_lines = []
        for i, line in enumerate(f):
            if i < 7:
                header_lines.append(line.strip())
            else:
                break
    
    results_df = pd.read_csv(results_file, skiprows=7)
    
    top_lineups = []
    for line in header_lines:
        if line.startswith('# Rank'):
            match = re.search(r'Rank (\d+): ([\d.]+) - (.+)', line)
            if match:
                rank, points, lineup_str = match.groups()
                players = parse_lineup(lineup_str)
                top_lineups.append({
                    'Rank': int(rank),
                    'Points': float(points),
                    'Players': players
                })
    
    if not top_lineups:
        return None
    
    lineup_stats = []
    for lineup in top_lineups:
        lineup_data = results_df[results_df['Player Name'].isin(lineup['Players'])].copy()
        if len(lineup_data) == 0:
            continue
        
        stats = {
            'Date': date_str,
            'Rank': lineup['Rank'],
            'Points': lineup['Points'],
            'Player_Count': len(lineup_data)
        }
        
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Player_Count']:
                stats[f'{col}_sum'] = lineup_data[col].sum()
                stats[f'{col}_mean'] = lineup_data[col].mean()
                stats[f'{col}_std'] = lineup_data[col].std()
        
        lineup_stats.append(stats)
    
    return pd.DataFrame(lineup_stats)

def main():
    all_stats = []
    for results_file in sorted(DATA_DIR.glob('results_*.csv')):
        date_str = results_file.stem.replace('results_', '')
        stats = analyze_slate(date_str)
        if stats is not None and len(stats) > 0:
            all_stats.append(stats)
    
    if not all_stats:
        logger.warning("No lineup data found")
        return
    
    df = pd.concat(all_stats, ignore_index=True)
    
    output_path = OUTPUT_DIR / 'top_lineup_characteristics.csv'
    df.to_csv(output_path, index=False)
    logger.info(f'Saved analysis to {output_path}')
    
    summary = df.groupby('Rank').agg({
        col: ['mean', 'std'] for col in df.columns if col.endswith('_mean')
    })
    summary_path = OUTPUT_DIR / 'lineup_summary_by_rank.csv'
    summary.to_csv(summary_path)
    logger.info(f'Saved summary to {summary_path}')

if __name__ == '__main__':
    main()

