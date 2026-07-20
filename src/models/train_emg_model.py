import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

df = pd.read_csv(
    "data/processed/emg_feature_dataset.csv"
)

X = df[["rms", "mav", "zcr", "wl"]]

y = df["activation_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("MAE :", mean_absolute_error(y_test, pred))
print("R2  :", r2_score(y_test, pred))

joblib.dump(
    model,
    "models/emg_activation_model.pkl"
)

print("EMG Model Saved")