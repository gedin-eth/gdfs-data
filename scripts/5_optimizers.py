import sys
import pandas as pd
import numpy as np
from pathlib import Path
from ortools.sat.python import cp_model
import argparse
import random

try:
    from pydfs_lineup_optimizer import Site, Sport, get_optimizer
    PYDFS_AVAILABLE = True
except ImportError:
    PYDFS_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)
PREDICTIONS_DIR = Path(__file__).parent.parent / 'predictions'
OUTPUT_DIR = Path(__file__).parent.parent / 'optimized_lineups'
OUTPUT_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)

def load_slate_data(date_str):
    input_file = DATA_DIR / f'input_{date_str}.csv'
    if not input_file.exists():
        raise FileNotFoundError(f'Input file not found: {input_file}')
    
    df = pd.read_csv(input_file)
    df['Date'] = date_str
    
    pred_files = list(PREDICTIONS_DIR.glob(f'predictions_*_{date_str}.csv'))
    for pred_file in pred_files:
        pred_df = pd.read_csv(pred_file)
        target_col = pred_file.stem.replace(f'predictions_', '').replace(f'_{date_str}', '')
        df = df.merge(pred_df, left_on='Name', right_on='Name', how='left', suffixes=('', '_pred'))
        if f'{target_col}_predicted' in df.columns:
            df = df.rename(columns={f'{target_col}_predicted': f'{target_col}_pred'})
    
    df = df.fillna(0)
    
    unique_teams = df['Team'].nunique()
    game_count = unique_teams / 2
    
    avg_total = df['VegasTotals'].mean() if 'VegasTotals' in df.columns else 230
    
    return df, game_count, avg_total

def get_context_weights(game_count, avg_total):
    if game_count <= 4:
        w_own, w_clutch, w_score, w_fpm = 0.40, 0.30, 0.20, 0.10
    elif game_count <= 7:
        w_own, w_clutch, w_score, w_fpm = 0.25, 0.25, 0.30, 0.20
    else:
        w_own, w_clutch, w_score, w_fpm = 0.15, 0.15, 0.35, 0.35
    
    if avg_total > 235:
        w_own += 0.10
        w_fpm += 0.05
        w_clutch -= 0.10
    elif avg_total < 225:
        w_own -= 0.05
        w_fpm += 0.05
    
    total = w_own + w_clutch + w_score + w_fpm
    return w_own/total, w_clutch/total, w_score/total, w_fpm/total

def calculate_player_scores(df, w_own, w_clutch, w_score, w_fpm):
    score_col = 'Actual Score_pred' if 'Actual Score_pred' in df.columns else 'Projected'
    own_col = 'Ownership_pred' if 'Ownership_pred' in df.columns else 'ProjOwn'
    clutch_col = 'ClutchFP_pred' if 'ClutchFP_pred' in df.columns else 'ClutchFP'
    fpm_col = 'FPM_pred' if 'FPM_pred' in df.columns else 'FPM'
    
    for col in [score_col, own_col, clutch_col, fpm_col]:
        if col not in df.columns:
            df[col] = 0
    
    max_score = df[score_col].max() if df[score_col].max() > 0 else 1
    max_own = df[own_col].max() if df[own_col].max() > 0 else 1
    max_clutch = df[clutch_col].max() if df[clutch_col].max() > 0 else 1
    max_fpm = df[fpm_col].max() if df[fpm_col].max() > 0 else 1
    
    df['composite_score'] = (
        w_score * (df[score_col] / max_score) +
        w_own * (df[own_col] / max_own) +
        w_clutch * (df[clutch_col] / max_clutch) +
        w_fpm * (df[fpm_col] / max_fpm)
    )
    
    return df

def cp_sat_optimizer(df, num_lineups=1, diversity=0.0):
    players = df.index.tolist()
    positions = {'PG': [], 'SG': [], 'SF': [], 'PF': [], 'C': []}
    for i in players:
        pos = str(df.loc[i, 'Position']).upper()
        if '/' in pos:
            for p in pos.split('/'):
                p = p.strip()
                if p in positions:
                    positions[p].append(i)
        elif pos in positions:
            positions[pos].append(i)
    
    g_positions = positions['PG'] + positions['SG']
    f_positions = positions['SF'] + positions['PF']
    teams_list = df['Team'].unique()
    
    lineups = []
    for n in range(num_lineups):
        model = cp_model.CpModel()
        player_vars = {i: model.NewBoolVar(f'player_{i}') for i in players}
        
        salary = model.NewIntVar(0, 50000, 'salary')
        model.Add(salary == sum(df.loc[i, 'Salary'] * player_vars[i] for i in players))
        model.Add(salary <= 50000)
        model.Add(sum(player_vars[i] for i in players) == 8)
        model.Add(sum(player_vars[i] for i in positions['PG']) >= 1)
        model.Add(sum(player_vars[i] for i in positions['SG']) >= 1)
        model.Add(sum(player_vars[i] for i in positions['SF']) >= 1)
        model.Add(sum(player_vars[i] for i in positions['PF']) >= 1)
        model.Add(sum(player_vars[i] for i in positions['C']) >= 1)
        model.Add(sum(player_vars[i] for i in g_positions) >= 2)
        model.Add(sum(player_vars[i] for i in f_positions) >= 2)
        
        team_used = {t: model.NewBoolVar(f'team_{t}') for t in teams_list}
        for t in teams_list:
            team_players = df[df['Team'] == t].index.tolist()
            model.Add(sum(player_vars[i] for i in team_players) >= 1).OnlyEnforceIf(team_used[t])
            model.Add(sum(player_vars[i] for i in team_players) == 0).OnlyEnforceIf(team_used[t].Not())
        model.Add(sum(team_used[t] for t in teams_list) >= 2)
        
        if n > 0:
            for prev in lineups:
                model.Add(sum(player_vars[i] for i in prev) <= 7)
        
        score = sum(df.loc[i, 'composite_score'] * player_vars[i] for i in players)
        if diversity > 0:
            noise_vals = {i: random.uniform(-diversity, diversity) for i in players}
            noise = sum(noise_vals[i] * player_vars[i] for i in players)
            model.Maximize(score + noise)
        else:
            model.Maximize(score)
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 0.5
        solver.parameters.num_search_workers = 4
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            lineup = [i for i in players if solver.Value(player_vars[i]) == 1]
            lineups.append(lineup)
        else:
            break
    
    return lineups

def greedy_optimizer(df, num_lineups=1, max_iterations=100, timeout=5.0, salary_flex=0.05):
    import time
    
    lineups = []
    
    def get_positions(pos_str):
        pos_str = str(pos_str).upper()
        if '/' in pos_str:
            return [p.strip() for p in pos_str.split('/')]
        return [pos_str]
    
    def try_build_lineup(df_sorted, used_players, max_salary, strategy='composite'):
        lineup = []
        required_pos = {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1}
        flexible_pos = {'G': 1, 'F': 1, 'Util': 1}
        salary_used = 0
        teams = set()
        
        for phase in ['required', 'flexible']:
            if len(lineup) == 8:
                break
            
            df_avail = df_sorted[~df_sorted.index.isin(used_players)].copy()
            df_avail = df_avail[df_avail['composite_score'] > 0]
            
            if strategy == 'value':
                df_avail['value'] = df_avail['composite_score'] / (df_avail['Salary'] / 1000 + 1)
                df_sorted_phase = df_avail.sort_values('value', ascending=False)
            elif strategy == 'random':
                df_sorted_phase = df_avail.sample(frac=1.0).sort_values('composite_score', ascending=False)
            else:
                df_sorted_phase = df_avail.sort_values('composite_score', ascending=False)
            
            for idx, player in df_sorted_phase.iterrows():
                if salary_used + player['Salary'] > max_salary:
                    continue
                
                player_positions = get_positions(player['Position'])
                pos_type = None
                
                if phase == 'required':
                    for pos in player_positions:
                        if pos in required_pos and required_pos[pos] > 0:
                            pos_type = pos
                            break
                else:
                    if ('PG' in player_positions or 'SG' in player_positions) and flexible_pos['G'] > 0:
                        pos_type = 'G'
                    elif ('SF' in player_positions or 'PF' in player_positions) and flexible_pos['F'] > 0:
                        pos_type = 'F'
                    elif flexible_pos['Util'] > 0:
                        pos_type = 'Util'
                
                if pos_type:
                    lineup.append(idx)
                    used_players.add(idx)
                    salary_used += player['Salary']
                    teams.add(player['Team'])
                    
                    if phase == 'required':
                        required_pos[pos_type] -= 1
                    else:
                        flexible_pos[pos_type] -= 1
                    
                    if len(lineup) == 8:
                        break
        
        return lineup, len(lineup) == 8 and len(teams) >= 2
    
    for n in range(num_lineups):
        lineup = None
        used_players = set()
        if n > 0 and len(lineups) > 0:
            used_players.update(lineups[-1])
        
        start_time = time.time()
        strategies = ['composite', 'value', 'random']
        
        for iteration in range(max_iterations):
            if time.time() - start_time > timeout:
                break
            
            strategy = strategies[iteration % len(strategies)]
            max_salary = int(50000 * (1 + salary_flex * min(iteration // 3, 2)))
            
            df_sorted = df.sort_values('composite_score', ascending=False)
            if iteration > 0 and iteration % 10 == 0:
                df_sorted = df.sort_values('composite_score', ascending=False).sample(frac=0.8).sort_values('composite_score', ascending=False)
            
            lineup, success = try_build_lineup(df_sorted, used_players.copy(), max_salary, strategy)
            
            if success:
                lineups.append(lineup)
                break
        
        if lineup is None or len(lineup) < 8:
            break
    
    return lineups

def pydfs_optimizer(df, num_lineups=1):
    if not PYDFS_AVAILABLE:
        logger.warning('pydfs-lineup-optimizer not installed, skipping')
        return []
    
    try:
        from pydfs_lineup_optimizer.player import Player
        
        optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASKETBALL)
        players_list = []
        
        for idx, player in df.iterrows():
            pos = str(player['Position']).upper()
            if '/' in pos:
                positions = [p.strip() for p in pos.split('/')]
            else:
                positions = [pos]
            
            name_parts = str(player['Name']).split(' ', 1)
            first_name = name_parts[0] if len(name_parts) > 0 else ''
            last_name = name_parts[1] if len(name_parts) > 1 else ''
            player_id = str(player['Name']).replace(' ', '_')
            
            score = player.get('composite_score', 0) * 1000
            p = Player(
                player_id=player_id,
                first_name=first_name,
                last_name=last_name,
                positions=positions,
                team=player['Team'],
                salary=player['Salary'],
                fppg=score
            )
            players_list.append(p)
        
        optimizer.load_players(players_list)
        
        lineups = []
        for lineup in optimizer.optimize(n=num_lineups):
            lineup_indices = []
            for player in lineup.lineup:
                player_idx = df[df['Name'] == player.full_name].index
                if len(player_idx) > 0:
                    lineup_indices.append(player_idx[0])
            
            if len(lineup_indices) == 8:
                lineups.append(lineup_indices)
        
        return lineups
    except Exception as e:
        logger.error(f'pydfs optimizer error: {e}')
        import traceback
        logger.error(traceback.format_exc())
        return []

def validate_lineup(df, lineup_indices):
    players = df.loc[lineup_indices]
    if len(players) != 8:
        return False, f"Wrong player count: {len(players)}"
    if players['Salary'].sum() > 50000:
        return False, f"Salary over: ${players['Salary'].sum()}"
    if players['Team'].nunique() < 2:
        return False, f"Not enough teams: {players['Team'].nunique()}"
    
    positions = {'PG': 0, 'SG': 0, 'SF': 0, 'PF': 0, 'C': 0}
    for pos in players['Position']:
        pos_str = str(pos).upper()
        if '/' in pos_str:
            for p in pos_str.split('/'):
                p = p.strip()
                if p in positions:
                    positions[p] += 1
        elif pos_str in positions:
            positions[pos_str] += 1
    
    g_count = positions['PG'] + positions['SG']
    f_count = positions['SF'] + positions['PF']
    
    if positions['PG'] < 1 or positions['SG'] < 1 or positions['SF'] < 1 or positions['PF'] < 1 or positions['C'] < 1:
        return False, f"Missing required positions: {positions}"
    if g_count < 2:
        return False, f"Not enough G positions: {g_count}"
    if f_count < 2:
        return False, f"Not enough F positions: {f_count}"
    
    return True, "Valid"

def format_lineup(df, lineup_indices):
    players = df.loc[lineup_indices]
    return {
        'players': players['Name'].tolist(),
        'positions': players['Position'].tolist(),
        'teams': players['Team'].tolist(),
        'salary': players['Salary'].sum(),
        'total_score': players['composite_score'].sum() if 'composite_score' in players.columns else 0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, help='Date (YYYYMMDD)')
    parser.add_argument('--num_lineups', type=int, default=10)
    parser.add_argument('--optimizers', nargs='+', default=['cp_sat'], choices=['cp_sat', 'greedy', 'pydfs'])
    parser.add_argument('--diversity', type=float, default=0.05)
    parser.add_argument('--greedy_iterations', type=int, default=100, help='Max iterations for greedy optimizer')
    parser.add_argument('--greedy_timeout', type=float, default=5.0, help='Timeout per lineup in seconds for greedy')
    parser.add_argument('--greedy_salary_flex', type=float, default=0.05, help='Salary flexibility for greedy (0.05 = 5% overage allowed)')
    args = parser.parse_args()
    
    df, game_count, avg_total = load_slate_data(args.date)
    w_own, w_clutch, w_score, w_fpm = get_context_weights(game_count, avg_total)
    df = calculate_player_scores(df, w_own, w_clutch, w_score, w_fpm)
    df = df.reset_index(drop=True)
    
    all_results = {}
    
    if 'cp_sat' in args.optimizers:
        logger.info(f'Running CP-SAT optimizer...')
        lineups = cp_sat_optimizer(df, args.num_lineups, args.diversity)
        valid_lineups = []
        for l in lineups:
            is_valid, msg = validate_lineup(df, l)
            if is_valid:
                valid_lineups.append(l)
        all_results['cp_sat'] = [format_lineup(df, l) for l in valid_lineups]
        logger.info(f'Generated {len(valid_lineups)} CP-SAT lineups')
    
    if 'greedy' in args.optimizers:
        logger.info(f'Running Greedy optimizer (iterations={args.greedy_iterations}, timeout={args.greedy_timeout}s, salary_flex={args.greedy_salary_flex})...')
        lineups = greedy_optimizer(df, args.num_lineups, args.greedy_iterations, args.greedy_timeout, args.greedy_salary_flex)
        valid_lineups = []
        for l in lineups:
            is_valid, msg = validate_lineup(df, l)
            if is_valid:
                valid_lineups.append(l)
            else:
                logger.debug(f'Invalid greedy lineup: {msg}')
        all_results['greedy'] = [format_lineup(df, l) for l in valid_lineups]
        logger.info(f'Generated {len(valid_lineups)} Greedy lineups')
    
    if 'pydfs' in args.optimizers:
        logger.info(f'Running pydfs optimizer...')
        lineups = pydfs_optimizer(df, args.num_lineups)
        valid_lineups = []
        for l in lineups:
            is_valid, msg = validate_lineup(df, l)
            if is_valid:
                valid_lineups.append(l)
            else:
                logger.debug(f'Invalid pydfs lineup: {msg}')
        all_results['pydfs'] = [format_lineup(df, l) for l in valid_lineups]
        logger.info(f'Generated {len(valid_lineups)} pydfs lineups')
    
    for opt_name, lineups in all_results.items():
        output_file = OUTPUT_DIR / f'{opt_name}_{args.date}_{len(lineups)}.csv'
        results = []
        for i, lineup in enumerate(lineups):
            results.append({
                'Lineup': i+1,
                'Players': ', '.join(lineup['players']),
                'Salary': lineup['salary'],
                'Score': lineup['total_score']
            })
        pd.DataFrame(results).to_csv(output_file, index=False)
        logger.info(f'Saved {len(lineups)} lineups to {output_file}')

if __name__ == '__main__':
    main()

