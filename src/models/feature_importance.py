import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/raw/soleus_dataset.csv")

# Load trained model
model = joblib.load("models/activation_model.pkl")

# Get feature names
features = df.drop("activation_score", axis=1).columns

# Get feature importance
importance = model.feature_importances_

# Create dataframe
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort descending
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance Ranking:\n")
print(importance_df)

# Plot graph
plt.figure(figsize=(8,5))
plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.title("Feature Importance for Soleus Activation Prediction")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "models/feature_importance.png"
)

plt.show()