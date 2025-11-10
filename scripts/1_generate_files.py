import pandas as pd
from pathlib import Path

linestar_dir = Path('linestar')
data_dir = Path('data')
results_dir = Path('results')

for csv_file in linestar_dir.glob('ls_*.csv'):
    date_str = csv_file.stem.replace('ls_', '')
    df = pd.read_csv(csv_file, index_col=False)
    
    input_df = df.drop(columns=['LineStarId', 'DFS_ID', 'Scored', 'IsOverridden'], errors='ignore')
    input_df.to_csv(data_dir / f'input_{date_str}.csv', index=False)
    
    results_df = pd.DataFrame({
        'Player Name': df['Name'],
        'Actual Score': df['Scored'],
        'Ownership': ''
    })
    with open(data_dir / f'results_{date_str}.csv', 'w') as f:
        f.write('# Header section - to be filled in later\n')
        f.write('# Date,Slate Info,etc.\n\n')
    results_df.to_csv(data_dir / f'results_{date_str}.csv', mode='a', index=False)

for csv_file in results_dir.glob('dk_*.csv'):
    date_str = csv_file.stem.replace('dk_', '')
    dk_df = pd.read_csv(csv_file, low_memory=False)
    
    lineups = dk_df[['Rank', 'Points', 'Lineup']].dropna(subset=['Points'])
    players = dk_df[['Player', '%Drafted']].dropna().drop_duplicates('Player').set_index('Player')
    
    top_score = lineups['Points'].max()
    top_3 = lineups.nsmallest(3, 'Rank')
    cash_score = lineups['Points'].quantile(0.5)
    
    results_file = data_dir / f'results_{date_str}.csv'
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                lines = f.readlines()
                header_count = sum(1 for line in lines[:10] if line.strip().startswith('#'))
                skip_rows = header_count + (1 if lines[header_count].strip() == '' else 0)
            
            results_df = pd.read_csv(results_file, skiprows=skip_rows, dtype={'Ownership': str})
            if 'Player Name' not in results_df.columns:
                print(f"Warning: {results_file} missing 'Player Name' column, skipping")
                continue
        except Exception as e:
            print(f"Warning: Could not parse {results_file}: {e}, skipping")
            continue
        def get_ownership(name):
            try:
                if name in players.index:
                    return str(players.loc[name, '%Drafted']).replace('%', '')
            except:
                pass
            return ''
        
        if 'Ownership' in results_df.columns:
            existing_ownership = results_df['Ownership'].fillna('').astype(str)
            new_ownership = results_df['Player Name'].apply(get_ownership)
            results_df['Ownership'] = new_ownership.where(new_ownership != '', existing_ownership)
        else:
            results_df['Ownership'] = results_df['Player Name'].apply(get_ownership)
        
        results_df['Ownership'] = results_df['Ownership'].astype(str).replace('nan', '')
        
        with open(results_file, 'r') as f:
            existing_header = ''.join([line for line in f.readlines()[:10] if line.strip().startswith('#')])
        
        with open(results_file, 'w') as f:
            f.write(f'# Top Lineup Score: {top_score}\n')
            f.write(f'# Cash Score: {cash_score:.2f}\n')
            f.write('# Top 3 Lineups:\n')
            for _, row in top_3.iterrows():
                f.write(f'# Rank {row["Rank"]}: {row["Points"]} - {row["Lineup"]}\n')
            f.write('\n')
        results_df.to_csv(results_file, mode='a', index=False, na_rep='')

