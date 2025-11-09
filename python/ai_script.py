import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load UFC dataset
df = pd.read_csv("C:\\Projects\\Octagon_AI\\data\\ufc_stats.csv")

# Clean fighter names
df['fighter'] = df['fighter'].str.strip().str.lower().str.replace('"', '', regex=False)
df['winner'] = df['winner'].str.strip().str.lower().str.replace('"', '', regex=False)

# Keep only fights with a winner
df = df[df['winner'].notnull()]

# Build fight-level dataset
rows = []
for fight_id in df['id'].unique():
    fight_df = df[df['id'] == fight_id]
    if len(fight_df) != 2:
        continue  # skip incomplete fights

    f1 = fight_df.iloc[0]
    f2 = fight_df.iloc[1]

    # Correct target assignment: 1 if fighter 1 actually won
    target = 1 if f1['winner'] == 'w' else 0

    row = {
        'f1_knockdowns': f1['knockdowns'],
        'f1_sig_strikes_landed': f1['significant_strikes_landed'],
        'f1_sig_strikes_attempted': f1['significant_strikes_attempted'],
        'f1_sig_strikes_rate': f1['significant_strikes_rate'],
        'f1_total_strikes_landed': f1['total_strikes_landed'],
        'f1_total_strikes_attempted': f1['total_strikes_attempted'],
        'f1_takedowns': f1['takedown_successful'],
        'f1_takedown_attempted': f1['takedown_attempted'],
        'f1_takedown_rate': f1['takedown_rate'],
        'f1_submissions': f1['submission_attempt'],
        'f1_reversals': f1['reversals'],
        'f1_head_landed': f1['head_landed'],
        'f1_body_landed': f1['body_landed'],
        'f1_leg_landed': f1['leg_landed'],
        'f1_distance_landed': f1['distance_landed'],
        'f1_clinch_landed': f1['clinch_landed'],
        'f1_ground_landed': f1['ground_landed'],

        'f2_knockdowns': f2['knockdowns'],
        'f2_sig_strikes_landed': f2['significant_strikes_landed'],
        'f2_sig_strikes_attempted': f2['significant_strikes_attempted'],
        'f2_sig_strikes_rate': f2['significant_strikes_rate'],
        'f2_total_strikes_landed': f2['total_strikes_landed'],
        'f2_total_strikes_attempted': f2['total_strikes_attempted'],
        'f2_takedowns': f2['takedown_successful'],
        'f2_takedown_attempted': f2['takedown_attempted'],
        'f2_takedown_rate': f2['takedown_rate'],
        'f2_submissions': f2['submission_attempt'],
        'f2_reversals': f2['reversals'],
        'f2_head_landed': f2['head_landed'],
        'f2_body_landed': f2['body_landed'],
        'f2_leg_landed': f2['leg_landed'],
        'f2_distance_landed': f2['distance_landed'],
        'f2_clinch_landed': f2['clinch_landed'],
        'f2_ground_landed': f2['ground_landed'],
        'target': target
    }

    rows.append(row)

df = df[df['winner'].str.lower().isin(['w', 'l'])]
def time_to_minutes(t):
    try:
        mins, secs = str(t).split(':')
        return int(mins) + int(secs)/60
    except:
        return np.nan
numeric_cols = [
    'knockdowns', 'significant_strikes_landed', 'significant_strikes_attempted',
    'significant_strikes_rate', 'total_strikes_landed', 'total_strikes_attempted',
    'takedown_successful', 'takedown_attempted', 'takedown_rate',
    'submission_attempt', 'reversals', 'head_landed', 'body_landed',
    'leg_landed', 'distance_landed', 'clinch_landed', 'ground_landed'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['fight_time_minutes'] = df['time'].apply(time_to_minutes)
df['sig_strikes_landed_per_min'] = df['significant_strikes_landed'] / df['fight_time_minutes']
df['takedowns_per_min'] = df['takedown_successful'] / df['fight_time_minutes']
df['sig_strike_accuracy'] = df['significant_strikes_landed'] / (df['significant_strikes_attempted'] + 1)
df['takedown_accuracy'] = df['takedown_successful'] / (df['takedown_attempted'] + 1)



model_df = pd.DataFrame(rows)

# Check target distribution
print("Target distribution:\n", model_df['target'].value_counts(normalize=True))

# Split features and target
X = model_df.drop(columns='target')
y = model_df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()  # handle class imbalance
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
xgb.plot_importance(model, max_num_features=20)
plt.show()
# Feature columns
feature_cols = X.columns.tolist()

# Aggregate per-fighter stats (excluding the fight itself)
fighter_stats = df.groupby('fighter').agg({
    'knockdowns': 'mean',
    'significant_strikes_landed': 'mean',
    'significant_strikes_attempted': 'mean',
    'significant_strikes_rate': 'mean',
    'total_strikes_landed': 'mean',
    'total_strikes_attempted': 'mean',
    'takedown_successful': 'mean',
    'takedown_attempted': 'mean',
    'takedown_rate': 'mean',
    'submission_attempt': 'mean',
    'reversals': 'mean',
    'head_landed': 'mean',
    'body_landed': 'mean',
    'leg_landed': 'mean',
    'distance_landed': 'mean',
    'clinch_landed': 'mean',
    'ground_landed': 'mean'
}).reset_index()

def predict_or_verify_fight(f1_name, f2_name):
    """
    Unified fight prediction function.

    - If the fight has occurred in the dataset, returns actual winner + stats.
    - If the fight is hypothetical, predicts winner using fighter stats.
    
    Returns a dictionary with:
        'winner': predicted or actual winner
        'probability_f1_win': predicted probability for f1 (if hypothetical)
        'probability_f2_win': predicted probability for f2 (if hypothetical)
        'fight_type': 'actual' or 'hypothetical'
    """
    f1_name_clean = f1_name.strip().lower()
    f2_name_clean = f2_name.strip().lower()

    # Check if both fighters exist in dataset stats
    if f1_name_clean not in fighter_stats['fighter'].values:
        raise ValueError(f"No stats found for fighter: {f1_name}")
    if f2_name_clean not in fighter_stats['fighter'].values:
        raise ValueError(f"No stats found for fighter: {f2_name}")

    # Check if fight exists in dataset
    fights = df.groupby('id').filter(lambda x: set([f1_name_clean, f2_name_clean]).issubset(set(x['fighter'])))
    if not fights.empty:
        # Get the latest fight
        fight_id = fights['id'].max()
        fight = df[df['id'] == fight_id].copy()
        # Determine actual winner
        if 'w' in fight['winner'].values:
            actual_winner = fight.loc[fight['winner'] == 'w', 'fighter'].iloc[0]
        else:
            # Infer winner if 'l' labels exist
            loser = fight.loc[fight['winner'] == 'l', 'fighter'].iloc[0]
            actual_winner = f2_name if loser == f1_name else f1_name

        return {
            'winner': actual_winner,
            'probability_f1_win': None,
            'probability_f2_win': None,
            'fight_type': 'actual',
            'fight_id': fight_id
        }

    # Hypothetical fight: predict using stats
    f1 = fighter_stats[fighter_stats['fighter'] == f1_name_clean].iloc[0]
    f2 = fighter_stats[fighter_stats['fighter'] == f2_name_clean].iloc[0]

    f1_features = {
        'f1_knockdowns': f1['knockdowns'],
        'f1_sig_strikes_landed': f1['significant_strikes_landed'],
        'f1_sig_strikes_attempted': f1['significant_strikes_attempted'],
        'f1_sig_strikes_rate': f1['significant_strikes_rate'],
        'f1_total_strikes_landed': f1['total_strikes_landed'],
        'f1_total_strikes_attempted': f1['total_strikes_attempted'],
        'f1_takedowns': f1['takedown_successful'],
        'f1_takedown_attempted': f1['takedown_attempted'],
        'f1_takedown_rate': f1['takedown_rate'],
        'f1_submissions': f1['submission_attempt'],
        'f1_reversals': f1['reversals'],
        'f1_head_landed': f1['head_landed'],
        'f1_body_landed': f1['body_landed'],
        'f1_leg_landed': f1['leg_landed'],
        'f1_distance_landed': f1['distance_landed'],
        'f1_clinch_landed': f1['clinch_landed'],
        'f1_ground_landed': f1['ground_landed']
    }

    f2_features = {
        'f2_knockdowns': f2['knockdowns'],
        'f2_sig_strikes_landed': f2['significant_strikes_landed'],
        'f2_sig_strikes_attempted': f2['significant_strikes_attempted'],
        'f2_sig_strikes_rate': f2['significant_strikes_rate'],
        'f2_total_strikes_landed': f2['total_strikes_landed'],
        'f2_total_strikes_attempted': f2['total_strikes_attempted'],
        'f2_takedowns': f2['takedown_successful'],
        'f2_takedown_attempted': f2['takedown_attempted'],
        'f2_takedown_rate': f2['takedown_rate'],
        'f2_submissions': f2['submission_attempt'],
        'f2_reversals': f2['reversals'],
        'f2_head_landed': f2['head_landed'],
        'f2_body_landed': f2['body_landed'],
        'f2_leg_landed': f2['leg_landed'],
        'f2_distance_landed': f2['distance_landed'],
        'f2_clinch_landed': f2['clinch_landed'],
        'f2_ground_landed': f2['ground_landed']
    }

    X_new = pd.DataFrame([{**f1_features, **f2_features}], columns=feature_cols)
    pred_class = model.predict(X_new)[0]
    pred_proba = model.predict_proba(X_new)[0]

    return {
        'winner': f1_name if pred_class == 1 else f2_name,
        'probability_f1_win': float(pred_proba[1]),
        'probability_f2_win': float(pred_proba[0]),
        'fight_type': 'hypothetical',
        'fight_id': None
    }

# Verify prediction with actual fight
def verify_prediction(f1_name, f2_name):
    f1_name = f1_name.strip().lower()
    f2_name = f2_name.strip().lower()

    fights = df.groupby('id').filter(lambda x: set([f1_name, f2_name]).issubset(set(x['fighter'])))
    if fights.empty:
        print("No fight found between these fighters.")
        return

    fight_id = fights['id'].max()
    fight = df[df['id'] == fight_id].copy()
    actual_winner = fight['fighter'].iloc[0]
    print(f"\n--- Fight ID {fight_id} ---")
    print(fight[['fighter', 'winner']])
    print(f"Actual winner: {actual_winner}")

# Example usage
prediction = predict_or_verify_fight("Jack Della Maddalena", "Islam Makhachev")
print("Predicted fight outcome:", prediction)
verify_prediction("Jack Della Maddalena", "Islam Makhachev")

# 1) Show how many fights each fighter has in the dataset
for name in ["islam makhachev", "jack della maddalena"]:
    fights_count = df[df['fighter'] == name].shape[0]
    print(f"{name}: fights rows in df = {fights_count}")

# 2) Show aggregated stats we use for hypotheticals
print("\nFighter stats rows (aggregated):")
print(fighter_stats.set_index('fighter').loc[["islam makhachev","jack della maddalena"]].T)

# 3) Show the exact X_new vector used for the prediction
res_f1 = fighter_stats[fighter_stats['fighter']=='islam makhachev'].iloc[0]
res_f2 = fighter_stats[fighter_stats['fighter']=='jack della maddalena'].iloc[0]

f1_features = { f"f1_{c.split('_',1)[1]}": res_f1[c] if c in res_f1.index else None 
               for c in res_f1.index if c != 'fighter'}
f2_features = { f"f2_{c.split('_',1)[1]}": res_f2[c] if c in res_f2.index else None 
               for c in res_f2.index if c != 'fighter'}

X_new_debug = pd.DataFrame([{**f1_features, **f2_features}], columns=feature_cols)
print("\nX_new used for prediction:")
print(X_new_debug.T)

# 4) Show model probabilities + feature importance
print("\nModel predict_proba:", model.predict_proba(X_new_debug)[0])
print("\nTop feature importances:")
fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(fi.head(20))
