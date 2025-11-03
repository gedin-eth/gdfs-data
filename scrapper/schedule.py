"""ESPN schedule scraper - get completed game IDs"""
import requests
import re
from typing import List, Optional
from datetime import datetime

# Team name to abbreviation mapping (for teams that don't follow standard pattern)
TEAM_ABBREV_MAP = {
    'brooklyn-nets': 'bkn',
    'nets': 'bkn',
    'golden-state-warriors': 'gs',
    'warriors': 'gs',
    'los-angeles-clippers': 'lac',
    'clippers': 'lac',
    'los-angeles-lakers': 'lal',
    'lakers': 'lal',
    'la-lakers': 'lal',
    'new-orleans-pelicans': 'no',
    'pelicans': 'no',
    'new-york-knicks': 'ny',
    'knicks': 'ny',
    'oklahoma-city-thunder': 'okc',
    'thunder': 'okc',
    'san-antonio-spurs': 'sa',
    'spurs': 'sa',
    'utah-jazz': 'utah',
    'jazz': 'utah',
}

def _get_team_url(team_name: str) -> Optional[str]:
    """Get correct ESPN schedule URL for a team by trying different formats"""
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    
    # Try mapping first
    if team_name.lower() in TEAM_ABBREV_MAP:
        abbrev = TEAM_ABBREV_MAP[team_name.lower()]
        url = f"https://www.espn.com/nba/team/schedule/_/name/{abbrev}/{team_name}"
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            return url
    
    # Try standard format (first 3 chars of first word)
    abbrev = team_name.split('-')[0][:3] if '-' in team_name else team_name[:3]
    url = f"https://www.espn.com/nba/team/schedule/_/name/{abbrev}/{team_name}"
    r = requests.get(url, headers=headers, timeout=5)
    if r.status_code == 200:
        return url
    
    # Try last word abbreviation
    if '-' in team_name:
        abbrev = team_name.split('-')[-1][:3]
        url = f"https://www.espn.com/nba/team/schedule/_/name/{abbrev}/{team_name}"
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            return url
    
    return None

def get_completed_game_ids(team_name: str, season_year: int = None, 
                          game_type: str = 'regular', target_date: str = None) -> List[tuple]:
    """
    Get completed game IDs for a team from ESPN schedule
    
    Args:
        team_name: Team name (e.g., 'boston-celtics')
        season_year: Filter by season year (e.g., 2026 for 2025-26 season). If None, returns all.
        game_type: 'regular', 'preseason', or 'playoff'. Default 'regular'.
        target_date: Optional date string (YYYY-MM-DD) to filter games
    
    Returns:
        List of (game_id, game_date) tuples, where game_date is YYYY-MM-DD or None if not found
    """
    url = _get_team_url(team_name)
    if not url:
        return []
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        html = r.text
        
        # Map game_type to ESPN abbreviation
        type_map = {'regular': 'reg', 'preseason': 'pre', 'playoff': 'post'}
        type_filter = type_map.get(game_type, 'reg')
        
        # Find all game IDs and check their context for filters
        all_game_ids = set(re.findall(r'/gameId/(\d+)', html))
        completed_ids = []
        
        for game_id in all_game_ids:
            # Find each occurrence of this game ID
            for match in re.finditer(rf'/gameId/{game_id}', html):
                # Get context around this occurrence (1500 chars before and after)
                start = max(0, match.start() - 1500)
                end = min(len(html), match.end() + 1500)
                context = html[start:end]
                
                # Check filters: state="post" (completed), completed=true, season year, game type
                is_post = '"state":"post"' in context
                is_completed = '"completed":true' in context
                has_result = '"result":{' in context
                
                # Season year filter
                year_match = True
                if season_year:
                    year_match = f'"seasonYear":{season_year}' in context
                
                # Game type filter
                type_match = f'"abbreviation":"{type_filter}"' in context
                
                # Only include if all filters match
                if is_post and is_completed and has_result and year_match and type_match:
                    # Try to extract date from context
                    game_date = None
                    # Look for date patterns like "2025-10-21" or "dtTm":"2025-10-21T..."
                    date_patterns = [
                        r'"dtTm":"(\d{4}-\d{2}-\d{2})',
                        r'"date":"(\d{4}-\d{2}-\d{2})',
                        r'(\d{4}-\d{2}-\d{2})',  # Fallback: any date-like string
                    ]
                    for pattern in date_patterns:
                        date_match = re.search(pattern, context)
                        if date_match:
                            game_date = date_match.group(1)
                            break
                    
                    # Filter by target_date if provided
                    if target_date and game_date and game_date != target_date:
                        continue
                    
                    completed_ids.append((game_id, game_date))
                    break  # Found valid match, no need to check other occurrences
        
        return completed_ids
    except Exception:
        return []

def get_games_from_scoreboard(date_str: str, season_year: int = None) -> List[tuple]:
    """
    Get all game IDs for a specific date from ESPN scoreboard page.
    This is more reliable than checking individual team schedules.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        season_year: Optional season year filter
    
    Returns:
        List of (game_id, game_date) tuples, where game_date should match date_str
    """
    try:
        # Convert YYYY-MM-DD to YYYYMMDD format for ESPN
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        date_formatted = date_obj.strftime('%Y%m%d')
        
        # ESPN scoreboard URL
        url = f'https://www.espn.com/nba/scoreboard/_/date/{date_formatted}'
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        html = r.text
        
        # Try to extract JSON data first (more reliable)
        json_data = None
        patterns = [
            r"window\['__espnfitt__'\]\s*=\s*({.+});",
            r'window\.__espnfitt__\s*=\s*({.+});',
            r"window\[\"__espnfitt__\"\]\s*=\s*({.+});",
        ]
        for pattern in patterns:
            json_match = re.search(pattern, html, re.DOTALL)
            if json_match:
                try:
                    import json
                    json_data = json.loads(json_match.group(1))
                    break
                except json.JSONDecodeError:
                    continue
        
        games = []
        
        # Method 1: Extract from JSON if available
        if json_data:
            try:
                # Navigate JSON structure to find games
                page_data = json_data.get('page', {})
                content = page_data.get('content', {})
                scoreboard = content.get('scoreboard', {})
                
                # Try different possible keys (ESPN uses 'evts' in scoreboard)
                events = scoreboard.get('evts', [])  # ESPN uses 'evts'
                if not events:
                    events = scoreboard.get('events', [])
                if not events:
                    events = scoreboard.get('games', [])
                if not events:
                    # Try at top level
                    events = json_data.get('events', [])
                if not events:
                    events = json_data.get('evts', [])
                
                for event in events:
                    game_id = str(event.get('id', event.get('gameId', '')))
                    if not game_id or not game_id.isdigit():
                        continue
                    
                    # Check if completed
                    status = event.get('status', {})
                    state = status.get('state', '')
                    is_final = state == 'post' or status.get('completed', False) or 'FINAL' in str(status).upper()
                    
                    # Get date
                    game_info = event.get('gameInfo', {})
                    game_date = None
                    if 'dtTm' in game_info:
                        dt_str = str(game_info['dtTm'])
                        if 'T' in dt_str:
                            game_date = dt_str.split('T')[0]
                    
                    # Include if completed and date matches
                    if is_final:
                        # Only include if date matches (or if we couldn't extract date, trust the URL query)
                        if game_date is None:
                            # Couldn't extract date - include it since we're querying by date
                            games.append((game_id, date_str))
                        elif game_date == date_str:
                            # Date matches exactly
                            games.append((game_id, game_date))
                        # Otherwise skip - date doesn't match
            except Exception as e:
                logging.getLogger(__name__).debug(f"Error parsing JSON from scoreboard: {e}")
        
        # Method 2: Fallback to regex if JSON parsing failed or found no games
        if not games:
            # Find all game IDs in the scoreboard page
            all_game_ids = set(re.findall(r'/gameId/(\d+)', html))
            
            for game_id in all_game_ids:
                # Find context around each game ID
                for match in re.finditer(rf'/gameId/{game_id}', html):
                    # Get context around this occurrence (3000 chars before and after for better coverage)
                    start = max(0, match.start() - 3000)
                    end = min(len(html), match.end() + 3000)
                    context = html[start:end]
                    
                    # Check if game is completed (be more lenient with checks)
                    is_post = '"state":"post"' in context or 'state":"post"' in context
                    is_completed = '"completed":true' in context or 'completed":true' in context
                    has_result = '"result":{' in context or 'FINAL' in context.upper() or 'Final' in context
                    is_final = is_post or is_completed or has_result
                    
                    # Extract date from context if available
                    game_date = None
                    date_patterns = [
                        r'"dtTm":"(\d{4}-\d{2}-\d{2})',
                        r'"date":"(\d{4}-\d{2}-\d{2})',
                        r'(\d{4}-\d{2}-\d{2})',
                    ]
                    for pattern in date_patterns:
                        date_match = re.search(pattern, context)
                        if date_match:
                            game_date = date_match.group(1)
                            break
                    
                    # If no date found, use target date (we're querying by date)
                    if game_date is None:
                        game_date = date_str
                    
                    # Season year filter (more lenient)
                    year_match = True
                    if season_year:
                        year_match = (f'"seasonYear":{season_year}' in context or 
                                    f'seasonYear":{season_year}' in context or
                                    str(season_year) in context)
                    
                    # Include completed games
                    if is_final and year_match:
                        games.append((game_id, game_date))
                        break  # Found valid match for this game ID
        
        # Deduplicate while preserving order
        seen = set()
        unique_games = []
        for game_id, game_date in games:
            if game_id not in seen:
                seen.add(game_id)
                unique_games.append((game_id, game_date))
        
        return unique_games
    except Exception as e:
        # Log error but don't fail completely
        import logging
        logging.getLogger(__name__).warning(f"Error scraping scoreboard for {date_str}: {e}")
        return []

