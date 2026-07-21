import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    xgb_available = False

df = pd.read_csv("data/processed/emg_feature_dataset.csv")

X = df.drop("activation_score", axis=1)
y = df["activation_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

results = []

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

pred = lr.predict(X_test)

results.append([
    "Linear Regression",
    r2_score(y_test, pred)
])

# Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

results.append([
    "Random Forest",
    r2_score(y_test, pred)
])

# XGBoost
if xgb_available:

    xgb = XGBRegressor(
        n_estimators=200,
        random_state=42
    )

    xgb.fit(X_train, y_train)

    pred = xgb.predict(X_test)

    results.append([
        "XGBoost",
        r2_score(y_test, pred)
    ])

result_df = pd.DataFrame(
    results,
    columns=["Model", "R2"]
)

print(result_df)
result_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("Comparison Saved")
import matplotlib.pyplot as plt

result_df.plot(
    x="Model",
    y="R2",
    kind="bar"
)

plt.title("Model Comparison")
plt.ylabel("R2 Score")
plt.tight_layout()

plt.savefig("models/model_comparison.png")

print("Graph Saved")