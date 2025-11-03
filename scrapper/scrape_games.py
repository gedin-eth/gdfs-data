"""Scrapper orchestrator entry point"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from scrapper.game_data import scrape_and_save_game
from utils.logger import setup_logger
from config.settings import SCRAPPER_DIR, PROCESSED_DIR

logger = setup_logger(__name__)

def extract_pbp_features(pbp_file: Path, boxscore_file: Path, game_id: str) -> pd.DataFrame:
    """
    Extract features from PBP and boxscore data for a game.
    
    Args:
        pbp_file: Path to PBP CSV file
        boxscore_file: Path to boxscore CSV file  
        game_id: Game ID
    
    Returns:
        DataFrame with Player and extracted features (FPM, Momentum, Usage, etc.)
    """
    import re
    
    # Load PBP data for advanced features
    df_pbp = pd.read_csv(pbp_file)
    
    # Load boxscore for actual stats and minutes
    df_bs = pd.read_csv(boxscore_file)
    
    # Parse stats from boxscore (they're in "M-N" format like "6-16" for FG)
    def parse_stat(stat_str):
        if pd.isna(stat_str):
            return 0
        stat_str = str(stat_str)
        if '-' not in stat_str:
            return 0
        made, attempted = stat_str.split('-')
        return int(made)
    
    # Parse player names from PBP to extract advanced features
    def parse_player_from_play(play_text):
        """Extract player name from play text"""
        if pd.isna(play_text):
            return None
        # Match "FirstName LastName" or "FirstName LastName Jr." pattern
        match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+(?: Jr\.| III| IV)?)', str(play_text))
        return match.group(1) if match else None
    
    # Helper to convert period + clock to game time
    def get_game_time(period, clock):
        """Convert period and clock to total game time in minutes"""
        if pd.isna(period) or pd.isna(clock):
            return None
        period_num = int(period)
        clock_str = str(clock).replace(':', '.')
        try:
            minutes_rem = float(clock_str)
        except:
            return None
        return (period_num - 1) * 12 + (12 - minutes_rem)
    
    # Extract player involvement from PBP
    player_stats_pbp = {}
    
    for _, play_row in df_pbp.iterrows():
        txt = play_row.get('txt', '')
        if pd.isna(txt):
            continue
        
        player = parse_player_from_play(txt)
        if not player:
            continue
            
        if player not in player_stats_pbp:
            player_stats_pbp[player] = {
                'clutch_fp': 0.0,
                'total_touches': 0,
                'clutch_touches': 0,
                'shot_attempts': 0,
                'shot_makes': 0,
                'three_attempts': 0,
                'three_makes': 0,
                'ft_attempts': 0,
                'ft_makes': 0,
                'assists': 0,
                'turnovers': 0,
                'rebounds': 0,
                'blocks': 0,
                'steals': 0,
                'clutch_plays': 0,
                'scoring_times': [],  # Track when they scored
                'q1_score': 0, 'q2_score': 0, 'q3_score': 0, 'q4_score': 0,
                'substitutions': 0,  # How many times they were subbed
                'first_sub_time': None,  # When first subbed out (if at all)
            }
        
        stats = player_stats_pbp[player]
        
        # Get game time for temporal analysis
        game_time = get_game_time(play_row.get('prd'), play_row.get('clck'))
        period = int(play_row.get('prd', 0))
        
        # Clutch time (last 5 minutes of 4th quarter or overtime)
        is_clutch = (period >= 4 and 
                    pd.notna(play_row.get('clck', '')) and
                    str(play_row.get('clck', '')).replace(':', '.') >= '5.00')
        
        # Track substitutions
        if 'enters the game' in str(txt).lower() and player in txt:
            stats['substitutions'] += 1
            if stats['first_sub_time'] is None and pd.notna(game_time):
                stats['first_sub_time'] = game_time
        
        # Count touch involvement
        if any(word in str(txt).lower() for word in ['makes', 'misses', 'assists', 'rebound', 'turnover', 'steals', 'blocks']):
            stats['total_touches'] += 1
            if is_clutch:
                stats['clutch_touches'] += 1
        
        # Parse specific actions
        txt_lower = str(txt).lower()
        
        if 'makes' in txt_lower and 'assists' not in txt_lower:
            stats['shot_makes'] += 1
            # Track scoring times for temporal distribution
            if pd.notna(game_time):
                stats['scoring_times'].append(game_time)
            
            # Track quarter distribution
            if period == 1:
                stats['q1_score'] += 1
            elif period == 2:
                stats['q2_score'] += 1
            elif period == 3:
                stats['q3_score'] += 1
            elif period >= 4:
                stats['q4_score'] += 1
            
            if 'three' in txt_lower:
                stats['three_makes'] += 1
                if is_clutch:
                    stats['clutch_fp'] += 3.5  # 3 pts + 0.5 bonus
            elif 'free throw' in txt_lower:
                stats['ft_makes'] += 1
                if is_clutch:
                    stats['clutch_fp'] += 1.0
            else:
                if is_clutch:
                    stats['clutch_fp'] += 2.0
        
        if 'misses' in txt_lower:
            stats['shot_attempts'] += 1
            if 'three' in txt_lower:
                stats['three_attempts'] += 1
        
        if 'free throw' in txt_lower:
            stats['ft_attempts'] += 1
        
        if 'assists' in txt_lower:
            stats['assists'] += 1
            if is_clutch:
                stats['clutch_fp'] += 1.5
        
        if 'turnover' in txt_lower:
            stats['turnovers'] += 1
            if is_clutch:
                stats['clutch_fp'] -= 0.5
        
        if 'rebound' in txt_lower and player in txt:
            stats['rebounds'] += 1
            if is_clutch:
                stats['clutch_fp'] += 1.25
        
        if 'blocks' in txt_lower or 'block' in txt_lower:
            stats['blocks'] += 1
            if is_clutch:
                stats['clutch_fp'] += 2.0
        
        if 'steals' in txt_lower or 'steal' in txt_lower:
            stats['steals'] += 1
            if is_clutch:
                stats['clutch_fp'] += 2.0
        
        if is_clutch:
            stats['clutch_plays'] += 1
    
    # Extract numeric stats from boxscore
    features = []
    for _, row in df_bs.iterrows():
        player_name = row['playerName']
        
        # Parse stats from boxscore
        pts = row.get('pts', 0) if pd.notna(row.get('pts')) else 0
        ast = row.get('ast', 0) if pd.notna(row.get('ast')) else 0
        reb = row.get('reb', 0) if pd.notna(row.get('reb')) else 0
        stl = row.get('stl', 0) if pd.notna(row.get('stl')) else 0
        blk = row.get('blk', 0) if pd.notna(row.get('blk')) else 0
        to = row.get('to', 0) if pd.notna(row.get('to')) else 0
        
        # Parse "M-N" format stats
        fg_made = parse_stat(row.get('fg', '0-0'))
        ft_made = parse_stat(row.get('ft', '0-0'))
        three_made = parse_stat(row.get('3pt', '0-0'))
        
        # Calculate fantasy points (DraftKings scoring)
        fp = (pts * 1.0) + (three_made * 0.5) + (reb * 1.25) + (ast * 1.5) + (stl * 2.0) + (blk * 2.0) - (to * 0.5)
        
        # Get minutes
        min_played = row.get('min', 0) if pd.notna(row.get('min')) else 0
        
        # Calculate Fantasy Per Minute (FPM)
        fpm = fp / min_played if min_played > 0 else 0
        
        # Calculate usage (simplified as touches/usages from assists + turnovers + FGA)
        # We'll use a simple proxy: (PTS + AST + TO) / minutes as a usage indicator
        usage_rate = (pts + ast + to) / min_played if min_played > 0 else 0
        
        # Get PBP-specific stats
        pbp_stats = player_stats_pbp.get(player_name, {})
        
        # Clutch performance (last 5 min of Q4 + OT)
        clutch_fp = pbp_stats.get('clutch_fp', 0.0)
        clutch_ratio = clutch_fp / fp if fp > 0 else 0.0
        
        # Touches and involvement
        total_touches = pbp_stats.get('total_touches', 0)
        touches_per_min = total_touches / min_played if min_played > 0 else 0.0
        
        # Shot selection
        shot_attempts = pbp_stats.get('shot_attempts', 0)
        three_rate = pbp_stats.get('three_attempts', 0) / shot_attempts if shot_attempts > 0 else 0.0
        
        # Temporal scoring distribution
        scoring_times = pbp_stats.get('scoring_times', [])
        total_baskets = len(scoring_times)
        
        # Calculate scoring frequency (baskets per minute)
        scoring_frequency = total_baskets / min_played if min_played > 0 else 0.0
        
        # Calculate scoring consistency (std dev of time between baskets)
        # Lower std dev = more consistent = higher score
        if len(scoring_times) > 1:
            sorted_times = sorted(scoring_times)
            time_intervals = [sorted_times[i+1] - sorted_times[i] for i in range(len(sorted_times)-1)]
            std_dev = np.std(time_intervals) if len(time_intervals) > 0 else 0
            scoring_consistency = 1.0 / (1.0 + std_dev) if std_dev > 0 else 1.0
        else:
            scoring_consistency = 0.0
        
        # Quarter distribution (which quarters they score most in)
        q1_score = pbp_stats.get('q1_score', 0)
        q2_score = pbp_stats.get('q2_score', 0)
        q3_score = pbp_stats.get('q3_score', 0)
        q4_score = pbp_stats.get('q4_score', 0)
        q_scores = [q1_score, q2_score, q3_score, q4_score]
        dominant_quarter = q_scores.index(max(q_scores)) + 1 if max(q_scores) > 0 else 0
        
        # Late game emphasis (Q4 scoring ratio)
        late_game_emphasis = q4_score / total_baskets if total_baskets > 0 else 0.0
        
        # Substitution patterns
        sub_count = pbp_stats.get('substitutions', 0)
        first_sub_time = pbp_stats.get('first_sub_time', 0)
        starter_or_stuffer = 1 if first_sub_time is None or first_sub_time > 10 else 0  # Started or early sub
        
        # For momentum, we'd need multiple games - set to 0 for now
        momentum = 0.0
        
        features.append({
            'Player': player_name,
            'GameID': game_id,
            'FPM': fpm,
            'Usage': usage_rate,
            'Momentum': momentum,
            'FP': fp,
            'Minutes': min_played,
            # PBP-specific features
            'ClutchFP': clutch_fp,
            'ClutchRatio': clutch_ratio,
            'TouchesPerMin': touches_per_min,
            'ThreeRate': three_rate,
            # Temporal & distribution features
            'ScoringFrequency': scoring_frequency,
            'ScoringConsistency': scoring_consistency,
            'DominantQuarter': dominant_quarter,
            'LateGameEmphasis': late_game_emphasis,
            'Substitutions': sub_count,
            'StarterOrStuffer': starter_or_stuffer,
        })
    
    return pd.DataFrame(features)

# Manual mapping of game IDs to dates - will be auto-populated when scraping
# TODO: Extract dates automatically from ESPN game data when scraping
GAME_DATE_MAPPING = {
    '401809964': '2025-10-22',  # CHA vs WSH
    # Add more mappings here as games are scraped
    # Format: 'game_id': 'YYYY-MM-DD'
}

def extract_game_date_from_boxscore(boxscore_file: Path) -> Optional[str]:
    """Extract game date from boxscore file"""
    game_id = boxscore_file.stem.replace('_boxscore_players', '')
    
    # Check manual mapping first
    if game_id in GAME_DATE_MAPPING:
        return GAME_DATE_MAPPING[game_id]
    
    # Try to read JSON file if it exists
    json_file = SCRAPPER_DIR / f"{game_id}_boxscore.json"
    
    if json_file.exists():
        import json
        with open(json_file) as f:
            data = json.load(f)
            # Try to extract date from gameInfo
            game_info = data.get('gameInfo', {})
            if 'dtTm' in game_info:
                # Convert ESPN date format to YYYY-MM-DD (handle timezone)
                dt_str = game_info['dtTm']
                # ESPN format is typically like "2025-10-30T00:00:00Z" (UTC)
                if 'T' in dt_str:
                    from datetime import datetime
                    try:
                        # Parse UTC time and convert to local date
                        dt_utc = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                        # ESPN games are typically in EST/EDT, so convert for accuracy
                        # But for simplicity, just use the date part (ESPN stores midnight UTC = previous day local)
                        # If time is early morning UTC (0-6), it's likely previous day local
                        if dt_utc.hour < 6:
                            from datetime import timedelta
                            local_date = (dt_utc - timedelta(days=1)).date()
                            return local_date.strftime('%Y-%m-%d')
                        return dt_str.split('T')[0]
                    except:
                        return dt_str.split('T')[0]
    
    return None

def load_existing_game_data(game_id: str) -> Optional[pd.DataFrame]:
    """Load existing boxscore and PBP data for a game ID"""
    pbp_file = SCRAPPER_DIR / f"{game_id}_pbp.csv"
    boxscore_file = SCRAPPER_DIR / f"{game_id}_boxscore_players.csv"
    
    if not pbp_file.exists() or not boxscore_file.exists():
        return None
    
    try:
        return extract_pbp_features(pbp_file, boxscore_file, game_id)
    except Exception as e:
        logger.error(f"Error loading game {game_id}: {e}")
        return None

def find_games_for_date(slate_date: str, teams: list = None) -> list:
    """
    Find game IDs that correspond to a slate date.
    Uses schedule scraper to find games for teams on that date.
    
    Args:
        slate_date: Date string (YYYY-MM-DD)
        teams: Optional list of team abbreviations (e.g., ['ATL', 'BOS'])
               If None, tries to extract from linestar data
    
    Returns:
        List of game IDs matching the slate date
    """
    from datetime import datetime
    
    # First check existing files
    existing_game_ids = []
    game_files = list(SCRAPPER_DIR.glob("*_boxscore_players.csv"))
    
    for file in game_files:
        game_id = file.stem.replace('_boxscore_players', '')
        game_date = extract_game_date_from_boxscore(file)
        if game_date == slate_date:
            existing_game_ids.append(game_id)
    
    # We'll still check the schedule to find any missing games, but include existing ones
    if existing_game_ids:
        logger.info(f"Found {len(existing_game_ids)} existing games for {slate_date}: {existing_game_ids}")
    else:
        logger.info(f"No existing games found for {slate_date}, attempting to find games to scrape")
    
    # Get teams if not provided
    if teams is None:
        try:
            from utils.data_loader import load_linestar_data
            linestar_df = load_linestar_data(slate_date, normalize=False)
            if 'Team' in linestar_df.columns:
                teams = linestar_df['Team'].unique().tolist()
                logger.info(f"Found {len(teams)} teams from linestar data: {teams}")
            else:
                logger.warning(f"Could not extract teams from linestar data for {slate_date}")
                return []
        except Exception as e:
            logger.warning(f"Could not load linestar data for {slate_date}: {e}")
            return []
    
    if not teams:
        logger.warning(f"No teams provided or found for {slate_date}")
        return []
    
    # Convert date to season year (ESPN uses the year the season ends)
    # For 2025-26 season (starts Oct 2025), ESPN uses seasonYear: 2026
    date_obj = datetime.strptime(slate_date, '%Y-%m-%d')
    if date_obj.month >= 10:
        season_year = date_obj.year + 1  # Oct-Dec: next year's season
    else:
        season_year = date_obj.year  # Jan-Sep: current year's season
    
    # Use schedule scraper to find game IDs for each team
    from scrapper.schedule import get_completed_game_ids
    from scrapper.schedule import TEAM_ABBREV_MAP
    
    all_game_ids = set()
    
    # Team name mapping (handle common variations)
    team_map = {
        'CHA': 'charlotte-hornets',
        'WAS': 'washington-wizards',
        'WSH': 'washington-wizards',
        'BKN': 'brooklyn-nets',
        'NY': 'new-york-knicks',
        'NYK': 'new-york-knicks',
        'PHI': 'philadelphia-76ers',
        'TOR': 'toronto-raptors',
        'BOS': 'boston-celtics',
        'CLE': 'cleveland-cavaliers',
        'DET': 'detroit-pistons',
        'IND': 'indiana-pacers',
        'MIL': 'milwaukee-bucks',
        'CHI': 'chicago-bulls',
        'ATL': 'atlanta-hawks',
        'MIA': 'miami-heat',
        'ORL': 'orlando-magic',
        'CHA': 'charlotte-hornets',
        'WAS': 'washington-wizards',
        'DEN': 'denver-nuggets',
        'MIN': 'minnesota-timberwolves',
        'OKC': 'oklahoma-city-thunder',
        'POR': 'portland-trail-blazers',
        'UTA': 'utah-jazz',
        'GSW': 'golden-state-warriors',
        'GS': 'golden-state-warriors',
        'LAC': 'los-angeles-clippers',
        'LAL': 'los-angeles-lakers',
        'PHX': 'phoenix-suns',
        'SAC': 'sacramento-kings',
        'DAL': 'dallas-mavericks',
        'HOU': 'houston-rockets',
        'MEM': 'memphis-grizzlies',
        'NO': 'new-orleans-pelicans',
        'SA': 'san-antonio-spurs',
    }
    
    for team_abbrev in teams:
        # Convert team abbrev to ESPN team name
        team_name = team_map.get(team_abbrev, team_abbrev.lower().replace(' ', '-'))
        
        # Get completed game IDs for this team (with dates from schedule page)
        game_data = get_completed_game_ids(team_name, season_year=season_year, 
                                           game_type='regular', target_date=slate_date)
        
        for game_id, game_date in game_data:
            # game_date might be None if not extracted from schedule
            if game_date == slate_date:
                all_game_ids.add(game_id)
            elif game_date is None:
                # Date not found in schedule - check if we have it locally, or scrape to verify
                boxscore_file = SCRAPPER_DIR / f"{game_id}_boxscore_players.csv"
                if boxscore_file.exists():
                    local_date = extract_game_date_from_boxscore(boxscore_file)
                    if local_date == slate_date:
                        all_game_ids.add(game_id)
                else:
                    # Add to potential list - we'll verify after scraping
                    all_game_ids.add(game_id)
        
        if game_data:
            matched = sum(1 for _, date in game_data if date == slate_date)
            logger.info(f"Found {len(game_data)} games for {team_abbrev}, {matched} match date {slate_date}")
    
    # Try ESPN scoreboard page as a more reliable source for finding all games on a date
    # This is better than checking individual team schedules
    logger.info(f"Checking ESPN scoreboard for {slate_date}...")
    from scrapper.schedule import get_games_from_scoreboard
    scoreboard_games = get_games_from_scoreboard(slate_date, season_year=season_year)
    
    if scoreboard_games:
        logger.info(f"Scoreboard found {len(scoreboard_games)} potential games for {slate_date}")
        for game_id, game_date in scoreboard_games:
            # Verify date by checking local file if it exists
            boxscore_file = SCRAPPER_DIR / f"{game_id}_boxscore_players.csv"
            if boxscore_file.exists():
                # Trust the local file date (more accurate)
                local_date = extract_game_date_from_boxscore(boxscore_file)
                if local_date == slate_date:
                    all_game_ids.add(game_id)
                    logger.info(f"Added game {game_id} from scoreboard (verified date: {local_date})")
                elif local_date:
                    logger.debug(f"Skipping game {game_id} from scoreboard (local date: {local_date}, expected: {slate_date})")
            else:
                # File doesn't exist - trust scoreboard date (will verify after scraping)
                if game_date == slate_date or game_date is None:
                    all_game_ids.add(game_id)
                    logger.info(f"Added game {game_id} from scoreboard (will verify date after scraping)")
    
    # If we have teams but only found 1 game, try to find the missing game by checking team pairs
    # This handles cases where a team's schedule page doesn't show all games
    if len(all_game_ids) == 1 and len(teams) >= 2:
        logger.info(f"Only found 1 game for {len(teams)} teams, checking for additional games...")
        # Get all games for all teams (without date filter) to find missing games
        all_team_games = set()
        for team_abbrev in teams:
            team_name = team_map.get(team_abbrev, team_abbrev.lower().replace(' ', '-'))
            # Get games without date filter
            game_data_all = get_completed_game_ids(team_name, season_year=season_year, 
                                                   game_type='regular', target_date=None)
            for game_id, game_date in game_data_all:
                all_team_games.add((game_id, game_date))
        
        # Check these games for matching dates
        for game_id, game_date in all_team_games:
            if game_id in all_game_ids:
                continue  # Already found
            if game_date == slate_date:
                all_game_ids.add(game_id)
                logger.info(f"Found additional game {game_id} for {slate_date}")
            elif game_date is None:
                # Try to verify by checking local file or scraping
                boxscore_file = SCRAPPER_DIR / f"{game_id}_boxscore_players.csv"
                if boxscore_file.exists():
                    local_date = extract_game_date_from_boxscore(boxscore_file)
                    if local_date == slate_date:
                        all_game_ids.add(game_id)
                        logger.info(f"Found additional game {game_id} for {slate_date} (from local file)")
    
    # Add any existing games we found
    all_game_ids.update(existing_game_ids)
    
    logger.info(f"Found {len(all_game_ids)} total games for {slate_date} (existing: {len(existing_game_ids)}, new: {len(all_game_ids) - len(existing_game_ids)})")
    
    # Final check: If we have multiple teams but only 1 game, there might be missing games
    # Try to find them by checking all existing scraped games for that date
    if len(teams) >= 2 and len(all_game_ids) == 1:
        logger.warning(f"Only found 1 game for {len(teams)} teams on {slate_date}. Checking all scraped games...")
        for boxscore_file in SCRAPPER_DIR.glob("*_boxscore_players.csv"):
            game_id = boxscore_file.stem.replace('_boxscore_players', '')
            if game_id in all_game_ids:
                continue
            game_date = extract_game_date_from_boxscore(boxscore_file)
            if game_date == slate_date:
                # Check if this game involves any of our teams
                try:
                    import pandas as pd
                    df_bs = pd.read_csv(boxscore_file)
                    if 'team' in df_bs.columns:
                        game_teams = df_bs['team'].str.upper().unique().tolist()
                        if any(team in game_teams for team in teams):
                            all_game_ids.add(game_id)
                            logger.info(f"Found additional game {game_id} for {slate_date} (teams: {game_teams})")
                except Exception as e:
                    logger.debug(f"Could not check game {game_id}: {e}")
    
    logger.info(f"Final: {len(all_game_ids)} games for {slate_date}")
    return list(all_game_ids)

def scrape_games(slate_date: str, teams: list = None, force_rescrape: bool = False) -> Optional[pd.DataFrame]:
    """
    Load or scrape play-by-play data for games on a slate
    
    Args:
        slate_date: Date string for slate (YYYY-MM-DD format)
        teams: Optional list of teams to scrape (team abbreviations)
        force_rescrape: If True, re-scrape even if data exists
    
    Returns:
        DataFrame with Player and extracted features or None if unavailable
    """
    # Check if we already have processed features for this date
    features_file = PROCESSED_DIR / f"pbp_features_{slate_date}.csv"
    if features_file.exists() and not force_rescrape:
        logger.info(f"Loading existing PBP features for {slate_date}")
        return pd.read_csv(features_file)
    
    # Find games for this slate (will scrape schedule if needed)
    game_ids = find_games_for_date(slate_date, teams=teams)
    
    if not game_ids:
        logger.warning(f"No game data found for slate {slate_date}")
        return None
    
    # Load or scrape each game
    all_features = []
    verified_game_ids = []
    
    for game_id in game_ids:
        # Check if game exists and verify date
        boxscore_file = SCRAPPER_DIR / f"{game_id}_boxscore_players.csv"
        pbp_file = SCRAPPER_DIR / f"{game_id}_pbp.csv"
        
        if boxscore_file.exists() and pbp_file.exists():
            # Verify the game is actually for this date
            game_date = extract_game_date_from_boxscore(boxscore_file)
            if game_date == slate_date:
                game_features = load_existing_game_data(game_id)
                if game_features is not None:
                    all_features.append(game_features)
                    verified_game_ids.append(game_id)
                else:
                    logger.warning(f"Could not load features for existing game {game_id}")
            elif game_date:
                # Wrong date, skip this game
                logger.debug(f"Skipping game {game_id} (date: {game_date}, expected: {slate_date})")
        else:
            # Game doesn't exist - scrape it
            logger.info(f"Scraping game {game_id} for {slate_date} (missing files)")
            try:
                results = scrape_and_save_game(game_id)
                if results.get('pbp') and results.get('boxscore'):
                    # Re-check files after scraping
                    if boxscore_file.exists() and pbp_file.exists():
                        # Verify date after scraping
                        game_date = extract_game_date_from_boxscore(boxscore_file)
                        if game_date == slate_date or game_date is None:  # None if date extraction failed
                            game_features = load_existing_game_data(game_id)
                            if game_features is not None:
                                all_features.append(game_features)
                                verified_game_ids.append(game_id)
                                logger.info(f"Successfully scraped and processed game {game_id}")
                            else:
                                logger.warning(f"Scraped game {game_id} but could not extract features")
                        else:
                            logger.warning(f"Scraped game {game_id} but date mismatch (got {game_date}, expected {slate_date})")
                    else:
                        logger.warning(f"Scraped game {game_id} but files not found after scraping")
                else:
                    logger.warning(f"Failed to scrape game {game_id}: {results}")
            except Exception as e:
                logger.error(f"Error scraping game {game_id}: {e}")
    
    if not all_features:
        logger.warning(f"No valid game features found for {slate_date} (checked {len(game_ids)} games)")
        return None
    
    logger.info(f"Processed {len(verified_game_ids)} games for {slate_date}: {verified_game_ids}")
    
    # Aggregate features across games
    df_all = pd.concat(all_features, ignore_index=True)
    
    # Group by player and aggregate features
    df_agg = df_all.groupby('Player').agg({
        'FPM': 'mean',
        'Usage': 'mean',
        'Momentum': 'mean',
        'FP': 'mean',
        'Minutes': 'mean',
        'ClutchFP': 'mean',
        'ClutchRatio': 'mean',
        'TouchesPerMin': 'mean',
        'ThreeRate': 'mean',
        'ScoringFrequency': 'mean',
        'ScoringConsistency': 'mean',
        'DominantQuarter': lambda x: int(np.round(x.mean())),  # Mode would be better but mean works
        'LateGameEmphasis': 'mean',
        'Substitutions': 'mean',
        'StarterOrStuffer': lambda x: int(np.round(x.mean())),  # Binary - take mean and round
    }).reset_index()
    
    # Save aggregated features
    PROCESSED_DIR.mkdir(exist_ok=True)
    df_agg.to_csv(features_file, index=False)
    logger.info(f"Saved PBP features for {slate_date}")
    
    return df_agg


def match_players_to_linestar(scrapper_players_df: pd.DataFrame, 
                              linestar_players_list: list,
                              threshold: float = 0.85) -> pd.DataFrame:
    """
    Match scrapper player names to linestar names using fuzzy matching.
    Handles cases like 'Alex Sarr' vs 'Alexandre Sarr'.
    
    Args:
        scrapper_players_df: DataFrame with Player column from scrapper
        linestar_players_list: List of player names from linestar
        threshold: Fuzzy match threshold (0.0-1.0)
    
    Returns:
        DataFrame with matched Player names
    """
    from utils.name_matcher import match_player_name
    
    df = scrapper_players_df.copy()
    
    # Match each scrapper player to linestar
    df['Player'] = df['Player'].apply(
        lambda x: match_player_name(x, linestar_players_list, threshold)
    )
    
    return df

