import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import seaborn as sns

def dataset_overview(df):
    print("Dataset Overview:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nStatistical Summary:")
    print(df.describe())

# Load UFC dataset
df = pd.read_csv("C:\\Projects\\Octagon_AI\\data\\ufc_stats.csv")  # replace with your file path
df['fighter'] = df['fighter'].str.strip().str.lower().str.replace('"', '', regex=False)
df['winner'] = df['winner'].str.strip().str.lower().str.replace('"', '', regex=False)

# Assuming your dataset is called `df`
# Keep only fights where winner is known
df = df[df['winner'].notnull()]

# Get unique fight IDs
fight_ids = df['id'].unique()

# Strip spaces and lowercase names
df['fighter'] = df['fighter'].str.strip().str.lower()
df['winner'] = df['winner'].str.strip().str.lower()

rows = []

for fight in df['id'].unique():
    fight_df = df[df['id'] == fight]
    
    if len(fight_df) != 2:
        continue  # skip incomplete fights

    f1 = fight_df.iloc[0]
    f2 = fight_df.iloc[1]

    # target: 1 if f1 won
    target = 1 if f1['winner'] == 'w' else 0

    row = {
        'f1_knockdowns': f1['knockdowns'],
        'f1_sig_strikes_landed': f1['significant_strikes_landed'],
        'f1_total_strikes_landed': f1['total_strikes_landed'],
        'f1_takedowns': f1['takedown_successful'],
        'f1_submissions': f1['submission_attempt'],
        'f2_knockdowns': f2['knockdowns'],
        'f2_sig_strikes_landed': f2['significant_strikes_landed'],
        'f2_total_strikes_landed': f2['total_strikes_landed'],
        'f2_takedowns': f2['takedown_successful'],
        'f2_submissions': f2['submission_attempt'],
        'target': target
    }

    rows.append(row)

model_df = pd.DataFrame(rows)

# Check target distribution
print(model_df['target'].value_counts())

# Features
X = model_df.drop(columns='target')
y = model_df['target']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost
model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
#plt.show()
xgb.plot_importance(model)
#plt.show()

# Assuming your trained model is called `model` and feature columns are:
feature_cols = [
    'f1_knockdowns', 'f1_sig_strikes_landed', 'f1_total_strikes_landed',
    'f1_takedowns', 'f1_submissions',
    'f2_knockdowns', 'f2_sig_strikes_landed', 'f2_total_strikes_landed',
    'f2_takedowns', 'f2_submissions'
]

def predict_fight(f1_stats: dict, f2_stats: dict):
    """
    Predicts the outcome of a fight between two fighters.

    f1_stats: dict of fighter 1 stats (keys must match your features: f1_*)
    f2_stats: dict of fighter 2 stats (keys must match your features: f2_*)
    
    Returns a dict with predicted winner and probabilities.
    """
    # Combine stats into a single dataframe row
    data = {**f1_stats, **f2_stats}
    X_new = pd.DataFrame([data], columns=feature_cols)
    
    # Predict class
    pred_class = model.predict(X_new)[0]
    
    # Predict probabilities
    pred_proba = model.predict_proba(X_new)[0]
    
    result = {
        'winner': 'f1' if pred_class == 1 else 'f2',
        'probability_f1_win': np.float32(pred_proba[1]),
        'probability_f2_win': np.float32(pred_proba[0])
    }
    return result

fighter_stats = df.groupby('fighter').agg({
    'knockdowns': 'mean',
    'significant_strikes_landed': 'mean',
    'total_strikes_landed': 'mean',
    'takedown_successful': 'mean',
    'submission_attempt': 'mean'
}).reset_index()

def get_fighter_stats(name):
    name = name.strip().lower()
    stats = fighter_stats[fighter_stats['fighter'] == name]
    if stats.empty:
        raise ValueError(f"No stats found for fighter: {name}")
    return {
        'knockdowns': float(stats['knockdowns'].iloc[0]),
        'sig_strikes_landed': float(stats['significant_strikes_landed'].iloc[0]),
        'total_strikes_landed': float(stats['total_strikes_landed'].iloc[0]),
        'takedowns': float(stats['takedown_successful'].iloc[0]),
        'submissions': float(stats['submission_attempt'].iloc[0])
    }

def predict_fight_by_names(f1_name, f2_name):
    f1_stats = get_fighter_stats(f1_name)
    f2_stats = get_fighter_stats(f2_name)

    # Convert to model input format
    f1_features = {f"f1_{k}": v for k, v in f1_stats.items()}
    f2_features = {f"f2_{k}": v for k, v in f2_stats.items()}

    return predict_fight(f1_features, f2_features)

def verify_prediction(f1_name, f2_name):
    f1_name = f1_name.strip().lower()
    f2_name = f2_name.strip().lower()
    
    # Filter rows where both fighters are involved
    fight_ids = df.groupby('id').filter(lambda x: set([f1_name, f2_name]).issubset(set(x['fighter'])) )['id'].unique()
    
    if len(fight_ids) == 0:
        print("❌ No fight found between these fighters.")
        return
    elif len(fight_ids) > 1:
        print(f"⚠️ Multiple fights found: {fight_ids}. Using the latest one.")
    
    fight_id = sorted(fight_ids)[-1]  # ensure we get the highest (latest) ID
    fight = df[df['id'] == fight_id].drop_duplicates(subset='fighter').copy()

    # Normalize winner column
    fight['winner'] = fight['winner'].str.lower().str.strip()

    print(f"\n--- Verifying Fight ID {fight_id} ---")
    print(fight[['fighter', 'winner']])

    if 'w' in fight['winner'].values:
        actual_winner = fight.loc[fight['winner'] == 'w', 'fighter'].iloc[0]
        print(f"✅ Actual winner: {actual_winner}")
    elif 'l' in fight['winner'].values:
        loser = fight.loc[fight['winner'] == 'l', 'fighter'].iloc[0]
        actual_winner = f2_name if loser == f1_name else f1_name
        print(f"✅ Inferred winner: {actual_winner}")
    else:
        print("❌ Could not determine winner for this fight.")




f1_example = {
    'f1_knockdowns': 3,
    'f1_sig_strikes_landed': 50,
    'f1_total_strikes_landed': 80,
    'f1_takedowns': 3,
    'f1_submissions': 1
}

f2_example = {
    'f2_knockdowns': 0,
    'f2_sig_strikes_landed': 10,
    'f2_total_strikes_landed': 20,
    'f2_takedowns': 0,
    'f2_submissions': 0
}

#prediction = predict_fight(f1_example, f2_example)
prediction = predict_fight_by_names("Jon Jones", "Alexander Gustafsson")
#prediction = predict_fight_by_names("Johnny Walker", "Zhang Mingyang")
print(prediction)
verify_prediction("Jon Jones", "Alexander Gustafsson")