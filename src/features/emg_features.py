import numpy as np


def calculate_rms(signal):
    """
    Root Mean Square (RMS)
    """
    signal = np.array(signal)
    return np.sqrt(np.mean(signal ** 2))


def calculate_mav(signal):
    """
    Mean Absolute Value (MAV)
    """
    signal = np.array(signal)
    return np.mean(np.abs(signal))


def calculate_zcr(signal):
    """
    Zero Crossing Rate (ZCR)
    """
    signal = np.array(signal)

    crossings = np.where(
        np.diff(np.sign(signal))
    )[0]

    return len(crossings)


def calculate_waveform_length(signal):
    """
    Waveform Length (WL)
    """
    signal = np.array(signal)

    return np.sum(
        np.abs(np.diff(signal))
    )


if __name__ == "__main__":

    # Sample EMG signal
    sample_signal = [
        0.12,
        -0.15,
        0.20,
        -0.18,
        0.25,
        -0.10,
        0.30
    ]

    print("RMS :", calculate_rms(sample_signal))
    print("MAV :", calculate_mav(sample_signal))
    print("ZCR :", calculate_zcr(sample_signal))
    print("WL  :", calculate_waveform_length(sample_signal))