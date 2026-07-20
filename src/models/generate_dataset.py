import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 5000

age = np.random.randint(18, 70, n_samples)

weight = np.random.randint(45, 100, n_samples)

resistance = np.random.randint(10, 60, n_samples)

force = resistance + np.random.normal(0, 5, n_samples)

speed = np.random.randint(10, 30, n_samples)

duration = np.random.randint(5, 30, n_samples)

emg_rms = (
    0.2
    + resistance * 0.01
    + force * 0.005
    + np.random.normal(0, 0.05, n_samples)
)

activation_score = (
    emg_rms * 80
    + resistance * 0.3
    - speed * 0.2
    + np.random.normal(0, 3, n_samples)
)

activation_score = np.clip(activation_score, 0, 100)

df = pd.DataFrame({
    "age": age,
    "weight": weight,
    "resistance": resistance,
    "force": force,
    "speed": speed,
    "duration": duration,
    "emg_rms": emg_rms,
    "activation_score": activation_score
})

df.to_csv(
    "data/raw/soleus_dataset.csv",
    index=False
)

print("Dataset created successfully")
print(df.head())