import numpy as np
import streamlit as st
from numba import jit, njit


@jit(forceobj=True)
def _dft(waves, fs):
    """
    Discrete Fourier Transform.
    """
    wave = waves.reshape(1, -1)
    N = len(waves)
    n = np.arange(0, N).reshape(1, -1)
    k = np.arange(0, N).reshape(-1, 1)

    # DFT
    scalar = -2j * np.pi / N
    matrix = k * n
    mat = scalar * matrix
    e = np.exp(mat)

    dft_sum = np.sum(wave * e, axis=1)
    wave_amplitude = np.absolute(dft_sum)
    wave_dft = {
        "Amplitude": np.array_split(wave_amplitude.flatten(), 2)[0],
        "Frequency": np.array_split((n * fs / N).flatten(), 2)[0],
    }
    return wave_dft


@jit(forceobj=True)
def _filtered_waves(raw_waves, theta, r_pol, r_zero, fs, N):
    coef = np.array(
        [
            1,
            -2 * r_zero * np.cos(theta),
            r_zero ** 2,
            2 * r_pol * np.cos(theta),
            -(r_pol ** 2),
        ]
    ).reshape(-1, 1)

    x = np.concatenate((np.array([raw_waves[0], raw_waves[0]]), raw_waves))
    y = np.zeros(len(raw_waves) + 2)
    for index in range(len(raw_waves)):
        x_mat = np.array(
            [x[index + 2], x[index + 1], x[index], y[index + 1], y[index]]
        ).reshape(1, -1)
        y[index + 2] = np.dot(x_mat, coef)
    t = np.arange(0, N) / fs
    return {"Filtered Waves": y[2:], "Time": t}


@jit(forceobj=True)
def _filter_omega(theta, r_pol, r_zero, fs, N):
    n = np.arange(0, N).reshape(1, -1)
    f = n * fs / N
    omega = 2 * np.pi * n / N
    coef = np.array([[1], [2]])
    e_mat = np.exp(-1j * omega * coef)

    # Zero
    zero_coef = np.array([-2 * r_zero * np.cos(theta), r_zero ** 2]).reshape(1, -1)
    zero = 1 + np.dot(zero_coef, e_mat)

    # Pole
    pole_coef = np.array([-2 * r_pol * np.cos(theta), r_pol ** 2]).reshape(1, -1)
    pole = 1 + np.dot(pole_coef, e_mat)

    h_omega = zero / pole
    h_omega_abs = np.absolute(h_omega)
    return {
        "Gain": np.array_split(h_omega_abs.flatten(), 2)[0],
        "Frequency": np.array_split(f.flatten(), 2)[0],
    }


class ZeroPoleFilter:
    def __init__(self):
        pass

    def wave_gen(self, wave_state):
        """
        Generate a waveform of a given frequency and amplitude.
        """
        freq = np.empty(3)
        amp = np.empty(3)

        for key, values in wave_state.items():
            for num in range(1, 4):
                if key == "Wave " + str(num):
                    freq[num - 1] = values["Frequency"]
                    amp[num - 1] = values["Amplitude"]

            if key == "Sampling Frequency":
                fs = values
            elif key == "Duration":
                duration = values
            elif key == "Standard Deviation":
                scale = values
            elif key == "Mean":
                loc = values

        freq = freq.reshape(-1, 1, 1)
        amp = amp.reshape(-1, 1, 1)
        t = np.linspace(0.0, duration, int(duration * fs)).reshape(1, -1)
        noise = np.random.normal(loc, scale, int(duration * fs)).reshape(1, -1)
        waves = {}
        waves["Waves"] = (
            np.sum(amp * np.sin(2 * np.pi * freq * t), axis=0) + noise
        ).flatten()
        waves["t"] = t.flatten()
        return waves

    def dft(self, waves, fs):
        return _dft(waves, fs)

    def filter_waves(self, raw_waves, filter_state, fs):
        """
        Filter the raw waves with a given filter state.
        """
        # Get Filtered Waves
        theta = filter_state["Cutoff Frequency"] * 2 * np.pi / fs
        r_pol = filter_state["Pole Radius"]
        r_zero = filter_state["Zero Radius"]
        filtered_waves = _filtered_waves(
            raw_waves, theta, r_pol, r_zero, fs, len(raw_waves)
        )

        # Get Filtered Waves DFT
        filtered_waves_dft = _dft(filtered_waves["Filtered Waves"], fs)

        # Get Pole Zero Diagram
        pole_zero = {
            "X Axis": [
                r_zero * np.cos(theta),
                r_zero * np.cos(theta),
                r_pol * np.cos(theta),
                r_pol * np.cos(theta),
            ],
            "Y Axis": [
                r_zero * np.sin(theta),
                -r_zero * np.sin(theta),
                r_pol * np.sin(theta),
                -r_pol * np.sin(theta),
            ],
            "Type": ["white", "white", "white", "white"],
        }

        # Get H(omega)
        filter_omega = _filter_omega(theta, r_pol, r_zero, fs, len(raw_waves))

        return filtered_waves, filtered_waves_dft, pole_zero, filter_omega
