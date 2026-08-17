import math
import numpy as np

from scipy.signal import resample_poly
from scipy.io import wavfile

def resample(path: str, target_rate: int = 62400, output_path: str | None = None) -> np.ndarray:
    fs, data = wavfile.read(path)

    i_data = data[:, 0]
    q_data = data[:, 1]

    iq_data = i_data + 1j * q_data

    gcd = math.gcd(fs, target_rate)
    L = target_rate // gcd
    M = fs // gcd

    resampled = resample_poly(iq_data, L, M)

    # debug verification
    expected_len = len(iq_data) * L / M
    print(f"resampled {path}: {fs} Hz -> {target_rate} Hz (L={L}, M={M})")
    print(f"  {len(iq_data)} -> {len(resampled)} samples (expected ~{expected_len:.0f})")

    if output_path is not None:
        # split complex back into two real channels, stack into (N, 2)
        i_out = resampled.real.astype(np.float32)
        q_out = resampled.imag.astype(np.float32)
        out_data = np.stack([i_out, q_out], axis=1)

        wavfile.write(output_path, target_rate, out_data)
        print(f"  wrote {output_path} ({target_rate} Hz, float32, stereo)")

    return resampled


if __name__ == "__main__":
    resampled_1 = resample(
        "/Users/andrewngo/Codebase/noaa-apt-decoder-artix7/wav/apt.2021-11-27T10_42_34_142.wav",
        output_path="/Users/andrewngo/Codebase/noaa-apt-decoder-artix7/wav/apt.2021-11-27T10_42_34_142_62400hz.wav"
    )
    print(len(resampled_1))