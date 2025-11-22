import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_PATH = r"C:\Projects\Octagon_AI\data\ufc_stats.csv"
MODEL_FILE = 'octagon_brain.pkl'

# --- HELPER FUNCTIONS ---
def time_to_minutes(t):
    try:
        mins, secs = str(t).split(':')
        return int(mins) + int(secs) / 60.0
    except:
        return np.nan

def get_new_elo(rating_winner, rating_loser):
    k = 32
    expected_winner = 1 / (1 + 10 ** ((rating_loser - rating_winner) / 400))
    expected_loser = 1 / (1 + 10 ** ((rating_winner - rating_loser) / 400))
    new_winner = rating_winner + k * (1 - expected_winner)
    new_loser = rating_loser + k * (0 - expected_loser)
    return new_winner, new_loser

def update_fighter_history(history, name, stats, date, won_fight):
    if name not in history:
        history[name] = {
            'n_fights': 0, 
            'stats': {k: 0.0 for k in stats.keys()},
            'streak': 0,
            'first_fight_date': date,
            'elo': 1500.0  # Initialize ELO
        }
    
    h = history[name]
    n = h['n_fights']
    
    if date and (h['first_fight_date'] is None or date < h['first_fight_date']):
        h['first_fight_date'] = date

    for k, v in stats.items():
        current_val = v if pd.notna(v) else 0.0
        h['stats'][k] = (h['stats'][k] * n + current_val) / (n + 1)
    
    if won_fight:
        h['streak'] = h['streak'] + 1 if h['streak'] >= 0 else 1
    else:
        h['streak'] = h['streak'] - 1 if h['streak'] <= 0 else -1
        
    h['n_fights'] += 1

# --- LOAD & CLEAN ---
print("Loading data...")
df = pd.read_csv(DATA_PATH)

if 'fight_date' in df.columns:
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    df = df.sort_values('fight_date')
else:
    df = df.sort_values('id')

df['fighter'] = df['fighter'].astype(str).str.strip().str.lower().str.replace('"', '', regex=False)
df['winner'] = df['winner'].astype(str).str.strip().str.lower().str.replace('"', '', regex=False)
df['winner'] = df['winner'].replace({'win': 'w', 'winner': 'w', 'loss': 'l', 'loser': 'l', 'draw': 'd', 'nc': 'nc'})
df = df[df['winner'].isin(['w', 'l'])].copy()

numeric_cols = [
    'knockdowns', 'significant_strikes_landed', 'significant_strikes_attempted',
    'total_strikes_landed', 'total_strikes_attempted', 'takedown_successful', 
    'takedown_attempted', 'submission_attempt', 'reversals', 'head_landed', 
    'body_landed', 'leg_landed', 'distance_landed', 'clinch_landed', 'ground_landed'
]
for col in numeric_cols:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

df['fight_time_minutes'] = df['time'].apply(time_to_minutes)
t = df['fight_time_minutes'].replace(0, 1)

df['sig_strikes_per_min'] = df['significant_strikes_landed'] / t
df['takedowns_per_min'] = df['takedown_successful'] / t
df['sig_strike_acc'] = df['significant_strikes_landed'] / (df['significant_strikes_attempted'] + 1)
df['takedown_acc'] = df['takedown_successful'] / (df['takedown_attempted'] + 1)

total_landed = df['total_strikes_landed'].replace(0, 1)
df['head_hunting_pct'] = df['head_landed'] / total_landed
df['grappler_pct'] = df['ground_landed'] / total_landed
df['leg_kicker_pct'] = df['leg_landed'] / total_landed

track_stats = numeric_cols + [
    'sig_strikes_per_min', 'takedowns_per_min', 'sig_strike_acc', 'takedown_acc',
    'head_hunting_pct', 'grappler_pct', 'leg_kicker_pct'
]

# --- HISTORY BUILD ---
print("Training ELO Engine...")
fighter_history = {} 
model_rows = []

for f_id in df['id'].unique():
    fight_df = df[df['id'] == f_id]
    if len(fight_df) != 2: continue
        
    f1 = fight_df.iloc[0]
    f2 = fight_df.iloc[1]
    name1, name2 = f1['fighter'], f2['fighter']
    date = f1['fight_date'] if 'fight_date' in f1 else None

    # Initialize if new
    if name1 not in fighter_history:
        update_fighter_history(fighter_history, name1, {k:0 for k in track_stats}, date, False)
        fighter_history[name1]['n_fights'] = 0 # Reset counter created by init
    if name2 not in fighter_history:
        update_fighter_history(fighter_history, name2, {k:0 for k in track_stats}, date, False)
        fighter_history[name2]['n_fights'] = 0

    h1 = fighter_history[name1]
    h2 = fighter_history[name2]
    
    # Build Row
    c_age1 = (date - h1['first_fight_date']).days / 365.25 if date and h1['first_fight_date'] else 0
    c_age2 = (date - h2['first_fight_date']).days / 365.25 if date and h2['first_fight_date'] else 0

    def build_row(stats_a, stats_b, age_a, age_b):
        d = {}
        for k in track_stats:
            d[f"diff_{k}"] = stats_a['stats'][k] - stats_b['stats'][k]
        d['diff_career_years'] = age_a - age_b
        d['diff_streak'] = stats_a['streak'] - stats_b['streak']
        d['diff_experience_fights'] = stats_a['n_fights'] - stats_b['n_fights']
        d['diff_elo'] = stats_a['elo'] - stats_b['elo']
        return d

    f1_won = 1 if f1['winner'] == 'w' else 0
    model_rows.append({**build_row(h1, h2, c_age1, c_age2), 'target': f1_won})
    model_rows.append({**build_row(h2, h1, c_age2, c_age1), 'target': 1 - f1_won})

    # Update
    s1 = {k: f1.get(k, 0) for k in track_stats}
    s2 = {k: f2.get(k, 0) for k in track_stats}
    update_fighter_history(fighter_history, name1, s1, date, f1['winner']=='w')
    update_fighter_history(fighter_history, name2, s2, date, f2['winner']=='w')
    
    # ELO Update
    if f1_won:
        n1, n2 = get_new_elo(h1['elo'], h2['elo'])
    else:
        n2, n1 = get_new_elo(h2['elo'], h1['elo'])
    h1['elo'], h2['elo'] = n1, n2

model_df = pd.DataFrame(model_rows)
features = [c for c in model_df.columns if c not in ['target']]
X = model_df[features]
y = model_df['target']

# --- TRAIN ---
print(f"Training on {len(model_df)} rows...")
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', xgb.XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03))
])
pipe.fit(X, y)

# --- SAVE ---
print(f"Saving brain to {MODEL_FILE}...")
package = {
    'pipe': pipe,
    'history': fighter_history,
    'track_stats': track_stats,
    'features': features
}
joblib.dump(package, MODEL_FILE)
print("✅ DONE. Now run 'py predict_loader.py'")