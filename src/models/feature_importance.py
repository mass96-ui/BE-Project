import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load(
    "models/emg_activation_model.pkl"
)

df = pd.read_csv(
    "data/processed/emg_feature_dataset.csv"
)

X = df.drop(
    "activation_score",
    axis=1
)

importance = model.feature_importances_

plt.bar(
    X.columns,
    importance
)

plt.title(
    "EMG Feature Importance"
)

plt.ylabel(
    "Importance"
)

plt.savefig(
    "models/feature_importance.png"
)

print("Feature Importance Saved")