import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime, timedelta
from scrapper.scrape_games import scrape_games, find_games_for_date, extract_game_date_from_boxscore
from scrapper.game_data import scrape_and_save_game, DATA_DIR as SCRAPPER_DATA_DIR
from scrapper.schedule import get_games_from_scoreboard
from utils.logger import setup_logger

SCRAPPER_DIR = Path(__file__).parent.parent / 'data' / 'scrapper'
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = Path(__file__).parent.parent / 'processed'

logger = setup_logger(__name__)

def calculate_fp_from_stats(row):
    pts = row.get('pts', 0) if pd.notna(row.get('pts')) else 0
    three_made = int(str(row.get('3pt', '0-0')).split('-')[0]) if pd.notna(row.get('3pt')) else 0
    reb = row.get('reb', 0) if pd.notna(row.get('reb')) else 0
    ast = row.get('ast', 0) if pd.notna(row.get('ast')) else 0
    stl = row.get('stl', 0) if pd.notna(row.get('stl')) else 0
    blk = row.get('blk', 0) if pd.notna(row.get('blk')) else 0
    to = row.get('to', 0) if pd.notna(row.get('to')) else 0
    
    fp = (pts * 1.0) + (three_made * 0.5) + (reb * 1.25) + (ast * 1.5) + (stl * 2.0) + (blk * 2.0) - (to * 0.5)
    
    categories = []
    if pts >= 10: categories.append('pts')
    if reb >= 10: categories.append('reb')
    if ast >= 10: categories.append('ast')
    if blk >= 10: categories.append('blk')
    if stl >= 10: categories.append('stl')
    
    if len(categories) >= 2:
        fp += 1.5
    if len(categories) >= 3:
        fp += 3.0
    
    return fp

def validate_fp_calculations(slate_date):
    results_file = DATA_DIR / f'results_{slate_date.replace("-", "")}.csv'
    if not results_file.exists():
        return
    
    try:
        results_df = pd.read_csv(results_file, skiprows=7)
        if 'Player Name' not in results_df.columns:
            return
    except:
        return
    game_ids_with_dates = get_games_from_scoreboard(slate_date)
    
    all_stats = []
    for game_id, scoreboard_date in game_ids_with_dates:
        boxscore_file = SCRAPPER_DIR / f'{game_id}_boxscore_players.csv'
        actual_date = extract_game_date_from_boxscore(boxscore_file) if boxscore_file.exists() else None
        if boxscore_file.exists() and (actual_date == slate_date or scoreboard_date == slate_date):
            df = pd.read_csv(boxscore_file)
            df['GameID'] = game_id
            all_stats.append(df)
    
    if not all_stats:
        return
    
    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df['CalculatedFP'] = stats_df.apply(calculate_fp_from_stats, axis=1)
    
    merged = results_df.merge(stats_df, left_on='Player Name', right_on='playerName', how='left')
    matched = merged[merged['CalculatedFP'].notna()]
    mismatches = matched[abs(matched['Actual Score'] - matched['CalculatedFP']) > 0.5]
    
    print(f'\n=== FP Validation Results ===')
    print(f'Matched players: {len(matched)} out of {len(results_df)} results players')
    print(f'Mismatches (>0.5 diff): {len(mismatches)}')
    
    if len(mismatches) > 0:
        print('\nMismatches:')
        print(mismatches[['Player Name', 'Actual Score', 'CalculatedFP']].head(10))
    else:
        print(f'✓ All {len(matched)} calculations match!')
    sys.stdout.flush()

def get_player_historical_features(player_name, slate_date, n_games=5):
    date_obj = datetime.strptime(slate_date, '%Y-%m-%d')
    all_features = []
    
    for i in range(1, 100):
        check_date = (date_obj - timedelta(days=i)).strftime('%Y-%m-%d')
        features_file = PROCESSED_DIR / f'pbp_features_{check_date}.csv'
        
        if features_file.exists():
            df = pd.read_csv(features_file)
            player_data = df[df['Player'] == player_name]
            if len(player_data) > 0:
                all_features.append(player_data.iloc[0])
                if len(all_features) >= n_games:
                    break
    
    if not all_features:
        return None
    
    hist_df = pd.DataFrame(all_features)
    cols_to_drop = ['Player']
    if 'GameID' in hist_df.columns:
        cols_to_drop.append('GameID')
    return hist_df.drop(columns=cols_to_drop).mean().to_dict()

def process_slate(slate_date, n_historical_games=5):
    date_str = slate_date.replace('-', '')
    
    game_ids = get_games_from_scoreboard(slate_date)
    logger.info(f'Found {len(game_ids)} games for {slate_date}')
    
    for game_id, scoreboard_date in game_ids:
        boxscore_file = SCRAPPER_DIR / f'{game_id}_boxscore_players.csv'
        pbp_file = SCRAPPER_DIR / f'{game_id}_pbp.csv'
        
        if not boxscore_file.exists() or not pbp_file.exists():
            logger.info(f'Scraping game {game_id}')
            scrape_and_save_game(game_id)
        
        actual_date = extract_game_date_from_boxscore(boxscore_file) if boxscore_file.exists() else None
        if actual_date and actual_date != slate_date and scoreboard_date == slate_date:
            logger.info(f'Game {game_id}: scoreboard says {scoreboard_date}, boxscore says {actual_date} (timezone difference)')
    
    features_df = scrape_games(slate_date, force_rescrape=False)
    if features_df is None:
        logger.warning(f'No features extracted for {slate_date}')
        return
    
    results_file = DATA_DIR / f'results_{date_str}.csv'
    if results_file.exists():
        with open(results_file, 'r') as f:
            header_lines = []
            for i, line in enumerate(f):
                if i < 7:
                    header_lines.append(line.rstrip())
                else:
                    break
        if results_file.exists():
            try:
                results_df = pd.read_csv(results_file, skiprows=7)
                if 'Player Name' in results_df.columns:
                    merged = results_df.merge(features_df, left_on='Player Name', right_on='Player', how='left')
                    merged = merged.drop(columns=['Player'])
                    with open(results_file, 'w') as f:
                        f.write('\n'.join(header_lines) + '\n\n')
                    merged.to_csv(results_file, mode='a', index=False)
                    logger.info(f'Added features to results file')
            except Exception as e:
                logger.warning(f'Could not merge features to results file: {e}')
    
    input_file = DATA_DIR / f'input_{date_str}.csv'
    if input_file.exists():
        input_df = pd.read_csv(input_file)
        hist_features = []
        
        for _, row in input_df.iterrows():
            player_name = row['Name']
            hist = get_player_historical_features(player_name, slate_date, n_historical_games)
            if hist:
                hist['Name'] = player_name
                hist_features.append(hist)
        
        if hist_features:
            hist_df = pd.DataFrame(hist_features)
            merged = input_df.merge(hist_df, on='Name', how='left')
            merged.to_csv(input_file, index=False)
            logger.info(f'Added historical features to input file')
    
    validate_fp_calculations(slate_date)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        slate_date = sys.argv[1]
    else:
        slate_date = '2025-10-21'
    process_slate(slate_date)

