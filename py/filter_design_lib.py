import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import firwin, freqz
from dataclasses import dataclass

# -----------------------------------------------------
@dataclass(frozen=True)
class FIRSpec:
    sample_rate:    float
    passband_edge:  float
    stopband_edge:  float
    attenuation_db: float

# -----------------------------------------------------
def design_lowpass_fir(
    spec:       FIRSpec,
    output_pkl: str | None = None
) -> np.ndarray:
    
    transition_width = spec.stopband_edge - spec.passband_edge

    num_taps, beta = signal.kaiserord(
        spec.attenuation_db,
        transition_width / (spec.sample_rate / 2)
    )

    taps = firwin(
        numtaps=num_taps,
        cutoff=spec.passband_edge,
        fs=spec.sample_rate,
        pass_zero=True,
        window=("kaiser", beta)
    )

    if output_pkl is not None:
        with open(output_pkl, "wb") as f:
            pickle.dump(taps, f)

    return taps

# -----------------------------------------------------
def plot_frequency_response(
    taps: np.ndarray,
    spec: FIRSpec
) -> None:
    
    freq, response = freqz(
        b=taps,
        fs=spec.sample_rate
    )

    plt.figure(figsize=(10, 5))
    plt.plot(freq, 20 * np.log10(abs(response)), 'b')

    plt.axvline(spec.passband_edge, color="green", linestyle="--", label="Passband Edge")
    plt.axvline(spec.stopband_edge, color="red", linestyle="--", label="Stopband Edge")
    plt.axhline(-spec.attenuation_db, color="black", linestyle="--", label="Attentuation Target")

    plt.title("Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.grid(True)
    plt.legend()
    plt.show()

# -----------------------------------------------------
def plot_impulse_response(
    taps: np.ndarray
) -> None:

    plt.figure(figsize=(10, 4))
    plt.stem(taps, basefmt="C0-")

    plt.title("Impulse Response")
    plt.xlabel("Tap Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

