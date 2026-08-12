from scipy.io import wavfile
from scipy import signal
import numpy as np
import pickle

fs, data = wavfile.read("../wav/apt.2021-11-27T10_42_34_142.wav")

i_data = data[:,0]
q_data = data[:,1]

iq_data = i_data + 1j * q_data

# channel select taps
with open("channel_select_fir.pkl", "rb") as f:
    c_select_taps = pickle.load(f)

f_data = signal.lfilter(c_select_taps, [1.0], iq_data)

# normalization
max_abs_val = np.max(np.abs(f_data))
fnorm_data = f_data / max_abs_val
