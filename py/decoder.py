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

signal_data = signal.lfilter(c_select_taps, [1.0], iq_data)

# trim the startup transient
signal_data = signal_data[len(c_select_taps) - 1:]

# normalization
signal_data /= np.max(np.abs(signal_data))

# verification
print(f"\nblock one | channel select FIR & normalization\n---")
print("max of data absolute values after normalization, 1 expected")
print(f"{np.max(np.abs(signal_data))}")


# BLOCK 2 ---------------------------------------------
# fm demodulation and unwrapping phase angle +- 3.14

phase = np.angle(signal_data)
signal_data = np.diff(phase)

# handle unwrapping
signal_data = np.mod((signal_data + np.pi), 2 * np.pi) - np.pi

# verification
print(f"\nblock two | fm demodulation & unwrapping of phase\n---")
print("min and max of unwrapped phase angle data, +-3.14 expected")
print(f"min: {np.min(signal_data)} | max: {np.max(signal_data)}")

# BLOCK 3 ---------------------------------------------
# anti-aliasing FIR, decimation
print(f"\nblock three | anti-aliasing FIR, decimating by 5\n---")

with open("anti_alias_fir.pkl", "rb") as f:
    anti_alias_taps = pickle.load(f)

signal_data = signal.lfilter(anti_alias_taps, [1.0], signal_data)

# trim the startup transient
signal_data = signal_data[len(anti_alias_taps) - 1:]

print(f"length of signal, before decimation: {len(signal_data)}")

# decimate by 5
signal_data = signal_data[::5];

print(f"length of signal, after decimation:  {len(signal_data)}")

# BLOCK 4 ---------------------------------------------
# rectification and smoothing FIR
print(f"\nblock four | retification and smoothing FIR\n---")

# rectify
signal_data = np.abs(signal_data)
print("rectification, no negative values expected")
print(f"min: {np.min(signal_data)}")

with open("smoothing_fir.pkl", "rb") as f:
    smoothing_taps = pickle.load(f)

signal_data = signal.lfilter(smoothing_taps, [1.0], signal_data)

# trim the startup transient
signal_data = signal_data[len(smoothing_taps) - 1:]

# normalization, 0-255 for pixel brightness
signal_data -= np.min(signal_data)
signal_data /= np.max(signal_data)
signal_data *= 255
