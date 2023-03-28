import numpy as np


class DiscreteFourier:

    def __init__(self, signal):
        self.signal = signal
        self.N = len(signal)
        self.fourier = np.zeros(self.N, dtype=np.complex_)  # initialize array of 0 with complex type

    def naive_transform(self, signal, N):
        for k in range(N):
            for n in range(N):
                self.fourier[k] += signal[n] * np.exp(((-2j * np.pi) / N) * (k * n))
        return self.fourier



