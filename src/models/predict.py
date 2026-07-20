import joblib
import pandas as pd

model = joblib.load("models/activation_model.pkl")

sample = pd.DataFrame([{
    "age": 25,
    "weight": 70,
    "resistance": 30,
    "force": 40,
    "speed": 15,
    "duration": 10,
    "emg_rms": 0.62
}])

prediction = model.predict(sample)

print(f"Predicted Activation Score: {prediction[0]:.2f}")