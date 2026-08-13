from scipy.io import wavfile
from scipy import signal
import numpy as np
import pickle

# ignoring non-data chunk WavFileWarning
from scipy.io.wavfile import WavFileWarning
import warnings
warnings.simplefilter('ignore', WavFileWarning)

# -----------------------------------------------------

fs, data = wavfile.read("../wav/apt.2021-11-27T10_42_34_142.wav")

i_data = data[:,0]
q_data = data[:,1]

iq_data = i_data + 1j * q_data


# BLOCK 1 ---------------------------------------------
# channel select FIR and normalization

with open("channel_select_fir.pkl", "rb") as f:
    c_select_taps = pickle.load(f)

f_data = signal.lfilter(c_select_taps, [1.0], iq_data)

# trim the startup transient
f_data = f_data[76:] # numtaps - 1 = 76

# normalization
max_abs_val = np.max(np.abs(f_data))
fnorm_data = f_data / max_abs_val

# verification
print(f"\nblock one | channel select FIR & normalization\n---")
print("max of data absolute values after normalization, 1 expected")
print(f"{np.max(np.abs(fnorm_data))}")


# BLOCK 2 ---------------------------------------------
# fm demodulation and unwrapping phase angle +- 3.14

phase = np.angle(fnorm_data)
phase_diff = np.diff(phase)

# handle unwrapping
wrapped_data = np.mod((phase_diff + np.pi), 2 * np.pi) - np.pi

# verification
print(f"\nblock two | fm demodulation & unwrapping of phase\n---")
print("min and max of unwrapped phase angle data, +-3.14 expected")
print(f"min: {np.min(wrapped_data)} | max: {np.max(wrapped_data)}")
