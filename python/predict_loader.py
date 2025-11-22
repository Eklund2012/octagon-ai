import joblib
import pandas as pd
import numpy as np
import os
import difflib # Tool for finding similar strings

# --- CONFIGURATION ---
MODEL_FILE = 'octagon_brain.pkl'

def load_brain():
    """Loads the saved AI model and history."""
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: '{MODEL_FILE}' not found.")
        print("   Run your training script first to generate the brain file!")
        exit()
    
    print(f"🧠 Loading {MODEL_FILE}...")
    return joblib.load(MODEL_FILE)

def find_similar_names(input_name, available_names):
    """
    If exact match fails, this looks for similar spelling.
    Returns a list of top 3 closest matches.
    """
    # normalize input
    clean_input = input_name.strip().lower()
    
    # Get close matches (cutoff=0.5 means 50% similar)
    matches = difflib.get_close_matches(clean_input, available_names, n=3, cutoff=0.4)
    return matches

def predict_matchup(brain, f1_name, f2_name):
    """Uses the loaded brain to predict a fight."""
    
    # Unpack the brain
    pipe = brain['pipe']
    history = brain['history']
    track_stats = brain['track_stats']
    features = brain['features']
    
    # Normalize names (strip spaces, lowercase)
    f1n = f1_name.strip().lower().replace('"', '')
    f2n = f2_name.strip().lower().replace('"', '')
    
    # 1. CHECK IF FIGHTERS EXIST
    # We get a list of all known fighters in our brain
    known_fighters = list(history.keys())
    
    if f1n not in history:
        print(f"❌ Error: Fighter '{f1_name}' not found.")
        # Try to find suggestions
        suggestions = find_similar_names(f1n, known_fighters)
        if suggestions:
            print(f"   Did you mean: {', '.join([s.title() for s in suggestions])}?")
        return

    if f2n not in history:
        print(f"❌ Error: Fighter '{f2_name}' not found.")
        suggestions = find_similar_names(f2n, known_fighters)
        if suggestions:
            print(f"   Did you mean: {', '.join([s.title() for s in suggestions])}?")
        return

    h1 = history[f1n]
    h2 = history[f2n]
    
    # 2. CALCULATE STATS (Simulating "Today")
    today = pd.to_datetime('today')
    
    c_age1 = (today - h1['first_fight_date']).days / 365.25 if h1['first_fight_date'] else 0
    c_age2 = (today - h2['first_fight_date']).days / 365.25 if h2['first_fight_date'] else 0
    
    # Build the differential row
    row = {}
    for k in track_stats:
        row[f"diff_{k}"] = h1['stats'][k] - h2['stats'][k]
    
    row['diff_career_years'] = c_age1 - c_age2
    row['diff_streak'] = h1['streak'] - h2['streak']
    row['diff_experience_fights'] = h1['n_fights'] - h2['n_fights']
    row['diff_elo'] = h1['elo'] - h2['elo']
    
    # Create DataFrame
    X_new = pd.DataFrame([row], columns=features)
    
    # 3. PREDICT PROBABILITY
    proba = pipe.predict_proba(X_new)[0]
    f1_prob = proba[1]
    
    def prob_to_american(p):
        if p < 0.5: return int((100/p) - 100)
        else: return int(-100 / ((1/p) - 1))

    if f1_prob > 0.5:
        winner, win_prob, fair_odds = f1_name, f1_prob, prob_to_american(f1_prob)
    else:
        winner, win_prob, fair_odds = f2_name, 1 - f1_prob, prob_to_american(1 - f1_prob)

    # --- REPORT ---
    print(f"\n{'='*45}") 
    print(f"🤖 MATCHUP: {f1_name} vs {f2_name}")
    print(f"{'='*45}")
    print(f"🏆 AI Pick:       {winner}")
    print(f"📊 Confidence:    {win_prob:.1%}")
    print(f"💰 Fair Odds:     {fair_odds}")
    print(f"{'-'*45}")
    print(f"⚡ ELO Gap:       {row['diff_elo']:.0f} points")
    print(f"🔥 Streak Gap:    {row['diff_streak']}")
    print(f"⏳ Exp Gap:       {row['diff_career_years']:.1f} years")
    print(f"{'='*45}\n")

# --- MAIN LOOP ---
if __name__ == "__main__":
    brain = load_brain()
    
    print("✅ System Ready! Enter fighter names to predict.")
    print("   (Type 'exit' to quit)")
    
    while True:
        try:
            f1 = input("Fighter 1: ")
            if f1.lower() == 'exit': break
            f2 = input("Fighter 2: ")
            if f2.lower() == 'exit': break
            
            predict_matchup(brain, f1, f2)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")