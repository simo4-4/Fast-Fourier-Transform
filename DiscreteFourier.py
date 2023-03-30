import numpy as np


def naive_transform(signal):
    N = len(signal)
    fourier = np.zeros(N, dtype=np.complex_)

    for k in range(N):
        for n in range(N):
            fourier[k] += signal[n] * np.exp((-2j * np.pi * k * n) / N)
    return fourier


def FFT(signal):
    N = len(signal)

    if N <= 4:
        return naive_transform(signal)
    else:
        X_even = FFT(signal[::2])
        X_odd = FFT(signal[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)

        X = np.concatenate([X_even + factor[:int(N / 2)] * X_odd, X_even + factor[int(N / 2):] * X_odd])
        return X


#[[1,2,3],[3,4,5],[3,6,3]]
def FFT_2D(signal):
    N = len(signal)

    if N == 1:
        return FFT(signal[0])
    else:
        factor = np.exp()
        return X
