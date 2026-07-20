import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/activation_model.pkl")

# Current user profile
user = {
    "age": 25,
    "weight": 70,
    "force": 40,
    "speed": 15,
    "duration": 10,
    "emg_rms": 0.62
}

best_score = -1
best_resistance = None

# Try different resistance values
for resistance in range(10, 61):

    sample = pd.DataFrame([{
        "age": user["age"],
        "weight": user["weight"],
        "resistance": resistance,
        "force": user["force"],
        "speed": user["speed"],
        "duration": user["duration"],
        "emg_rms": user["emg_rms"]
    }])

    score = model.predict(sample)[0]

    if score > best_score:
        best_score = score
        best_resistance = resistance

print("\n===== AI Recommendation =====")
print(f"Recommended Resistance: {best_resistance}")
print(f"Expected Activation Score: {best_score:.2f}")