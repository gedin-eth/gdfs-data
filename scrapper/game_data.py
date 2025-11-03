"""ESPN game data scraper - PBP, boxscore, recap"""
import requests
import re
import json
import csv
from pathlib import Path
from typing import Dict, Optional, List

DATA_DIR = Path(__file__).parent.parent / 'data' / 'scrapper'
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _get_game_json(game_id: str, endpoint: str) -> Optional[Dict]:
    """Get JSON data from ESPN game endpoint"""
    url = f'https://www.espn.com/nba/game/_/gameId/{game_id}/{endpoint}'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        html = r.text
        
        # Try multiple patterns to find JSON (ESPN uses both window.__espnfitt__ and window['__espnfitt__'])
        patterns = [
            r"window\['__espnfitt__'\]\s*=\s*({.+});",
            r'window\.__espnfitt__\s*=\s*({.+});',
            r"window\[\"__espnfitt__\"\]\s*=\s*({.+});",
        ]
        
        for pattern in patterns:
            json_match = re.search(pattern, html, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return None

def scrape_play_by_play(game_id: str) -> Optional[Dict]:
    """Scrape play-by-play data for a game, organized by quarters"""
    data = _get_game_json(game_id, 'play-by-play')
    if not data:
        return None
    
    try:
        gamepackage = data.get('page', {}).get('content', {}).get('gamepackage', {})
        
        # ESPN uses 'plys' key for plays array
        plays_all = gamepackage.get('plys', [])
        
        # Organize plays by quarter/period
        plays_by_quarter = {}
        
        for play in plays_all:
            period_num = play.get('prd', play.get('period', 0))  # ESPN uses 'prd'
            quarter_key = f"Q{period_num}"
            
            if quarter_key not in plays_by_quarter:
                plays_by_quarter[quarter_key] = {
                    'period': period_num,
                    'plays': []
                }
            plays_by_quarter[quarter_key]['plays'].append(play)
        
        return {
            'gameId': game_id,
            'playsByQuarter': plays_by_quarter,
            'allPlays': plays_all,
            'totalPlays': len(plays_all)
        }
    except Exception as e:
        # If parsing fails, return raw data
        return {
            'gameId': game_id,
            'raw': gamepackage,
            'error': str(e)
        }

def scrape_boxscore(game_id: str) -> Optional[Dict]:
    """Scrape boxscore data for a game"""
    # Use /boxscore/_/gameId/ endpoint which has full player stats
    url = f'https://www.espn.com/nba/boxscore/_/gameId/{game_id}'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        html = r.text
        
        # Extract JSON
        patterns = [
            r"window\['__espnfitt__'\]\s*=\s*({.+});",
            r'window\.__espnfitt__\s*=\s*({.+});',
        ]
        
        for pattern in patterns:
            json_match = re.search(pattern, html, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                    gamepackage = data.get('page', {}).get('content', {}).get('gamepackage', {})
                    
                    return {
                        'gameId': game_id,
                        'gameLeaders': gamepackage.get('gmLdrs', {}),
                        'teamStats': gamepackage.get('meta', {}).get('tmStats', {}),
                        'gameInfo': gamepackage.get('gmInfo', {}),
                        'raw': gamepackage
                    }
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return None

def scrape_recap(game_id: str) -> Optional[Dict]:
    """Scrape recap data for a game"""
    data = _get_game_json(game_id, 'recap')
    if not data:
        return None
    
    try:
        gamepackage = data.get('page', {}).get('content', {}).get('gamepackage', {})
        
        # Extract recap/story data
        return {
            'gameId': game_id,
            'gameStory': gamepackage.get('gmStry', {}),
            'gameInfo': gamepackage.get('gmInfo', {}),
            'raw': gamepackage
        }
    except Exception:
        return None

def save_pbp_csv(game_id: str, plays: List[Dict]):
    """Save play-by-play data as CSV"""
    if not plays:
        return
    
    file_path = DATA_DIR / f"{game_id}_pbp.csv"
    
    # Get all unique keys from plays
    all_keys = set()
    for play in plays:
        all_keys.update(play.keys())
    
    # Common fields we want first
    field_order = ['id', 'prd', 'clck', 'txt', 'tm', 'team', 'hmScr', 'awyScr', 'sq', 'plyTypId']
    # Add rest of fields
    other_fields = sorted([k for k in all_keys if k not in field_order])
    fieldnames = [f for f in field_order if f in all_keys] + other_fields
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for play in plays:
            # Flatten nested dicts to strings
            row = {}
            for key in fieldnames:
                val = play.get(key)
                if isinstance(val, dict):
                    row[key] = json.dumps(val)
                elif isinstance(val, list):
                    row[key] = json.dumps(val)
                else:
                    row[key] = val
            writer.writerow(row)

def save_boxscore_csv(game_id: str, boxscore_data: Dict):
    """Save boxscore data as CSV tables"""
    raw = boxscore_data.get('raw', {})
    
    # Extract player stats from bxscr (boxscore) array
    player_stats = []
    
    if 'bxscr' in raw and isinstance(raw['bxscr'], list):
        bxscr = raw['bxscr']
        
        for team_box in bxscr:
            team_info = team_box.get('tm', {})
            team_abbrev = team_info.get('abbrev', '')
            team_name = team_info.get('displayName', team_info.get('shortDisplayName', ''))
            is_home = team_info.get('isHome', False)
            
            stats_entries = team_box.get('stats', [])
            
            for entry in stats_entries:
                # Each entry can have multiple athletes (starters, bench, etc.)
                athletes = entry.get('athlts', [])
                stat_keys = entry.get('keys', [])
                stat_labels = entry.get('lbls', [])
                entry_type = entry.get('type', '')
                
                for athlete_data in athletes:
                    athlete = athlete_data.get('athlt', {})
                    stats = athlete_data.get('stats', [])
                    
                    # Create player row
                    player_row = {
                        'team': team_abbrev,
                        'teamName': team_name,
                        'isHome': is_home,
                        'playerId': athlete.get('id', ''),
                        'playerName': athlete.get('dspNm', athlete.get('shrtNm', '')),
                        'jersey': athlete.get('jersey', ''),
                        'entryType': entry_type,
                    }
                    
                    # Add stat values using labels as column names
                    for i, label in enumerate(stat_labels):
                        if i < len(stats):
                            # Clean label for CSV header
                            clean_label = label.lower().replace(' ', '_').replace('%', 'pct')
                            player_row[clean_label] = stats[i]
                    
                    player_stats.append(player_row)
    
    # Save player stats CSV
    if player_stats:
        file_path = DATA_DIR / f"{game_id}_boxscore_players.csv"
        all_keys = set()
        for player in player_stats:
            all_keys.update(player.keys())
        
        # Order columns: team/player info first, then stats
        info_cols = ['team', 'teamName', 'isHome', 'playerId', 'playerName', 'jersey', 'entryType']
        stat_cols = sorted([k for k in all_keys if k not in info_cols])
        fieldnames = info_cols + stat_cols
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for player in player_stats:
                writer.writerow(player)

def save_recap_text(game_id: str, recap_data: Dict, boxscore_data: Optional[Dict] = None):
    """Save recap as plain text with game info"""
    file_path = DATA_DIR / f"{game_id}_recap.txt"
    
    text_parts = []
    
    # Extract text from game story
    if 'gameStory' in recap_data:
        story = recap_data['gameStory']
        if 'hdln' in story:
            text_parts.append(story['hdln'])
        if 'desc' in story:
            text_parts.append(story['desc'])
    
    # Add relevant game info from boxscore or recap
    game_info = boxscore_data.get('gameInfo', {}) if boxscore_data else recap_data.get('gameInfo', {})
    
    if game_info:
        info_parts = []
        # Add relevant info like location, date, etc.
        if 'loc' in game_info:
            info_parts.append(f"Location: {game_info['loc']}")
        if 'dtTm' in game_info:
            info_parts.append(f"Date: {game_info['dtTm']}")
        if 'attnd' in game_info:
            info_parts.append(f"Attendance: {game_info['attnd']}")
        if 'refs' in game_info and isinstance(game_info['refs'], list):
            ref_names = []
            for ref in game_info['refs']:
                if isinstance(ref, dict):
                    name = ref.get('dspNm', ref.get('name', ''))
                    if name:
                        ref_names.append(name)
            if ref_names:
                info_parts.append(f"Officials: {', '.join(ref_names)}")
        
        if info_parts:
            text_parts.append('\n'.join(info_parts))
    
    # Join and save
    text = '\n\n'.join(text_parts)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

def save_game_data(game_id: str, data_type: str, data: Dict):
    """Save scraped data to JSON file"""
    file_path = DATA_DIR / f"{game_id}_{data_type}.json"
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def scrape_and_save_game(game_id: str) -> Dict[str, bool]:
    """Scrape all data for a game and save to files"""
    results = {}
    
    # Scrape PBP
    pbp = scrape_play_by_play(game_id)
    if pbp:
        # Save as CSV only
        if 'allPlays' in pbp:
            save_pbp_csv(game_id, pbp['allPlays'])
            results['pbp'] = True
        else:
            results['pbp'] = False
    else:
        results['pbp'] = False
    
    # Scrape boxscore
    boxscore = scrape_boxscore(game_id)
    if boxscore:
        # Save as CSV
        save_boxscore_csv(game_id, boxscore)
        # Also save JSON for metadata (e.g., game date)
        save_game_data(game_id, 'boxscore', boxscore)
        results['boxscore'] = True
    else:
        results['boxscore'] = False
    
    # Recap scraping removed - not needed
    results['recap'] = False
    
    return results

