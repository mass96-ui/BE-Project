import numpy as np
import pandas as pd

from emg_features import (
    calculate_rms,
    calculate_mav,
    calculate_zcr,
    calculate_waveform_length
)

np.random.seed(42)

records = []

for _ in range(3000):

    activation_level = np.random.randint(20, 100)

    amplitude = activation_level / 100

    signal = np.random.normal(
        0,
        amplitude,
        200
    )

    rms = calculate_rms(signal)
    mav = calculate_mav(signal)
    zcr = calculate_zcr(signal)
    wl = calculate_waveform_length(signal)

    records.append([
        rms,
        mav,
        zcr,
        wl,
        activation_level
    ])

df = pd.DataFrame(
    records,
    columns=[
        "rms",
        "mav",
        "zcr",
        "wl",
        "activation_score"
    ]
)

df.to_csv(
    "data/processed/emg_feature_dataset.csv",
    index=False
)

print("Dataset Created Successfully")
print(df.head())