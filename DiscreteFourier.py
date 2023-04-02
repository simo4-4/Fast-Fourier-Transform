import numpy as np


def naive_transform(signal):
    N = len(signal)
    fourier = np.zeros(N, dtype=np.complex_)

    for k in range(N):
        for n in range(N):
            fourier[k] += signal[n] * np.exp((-2j * np.pi * k * n) / N)
    return fourier


def naive_inverse(signal):
    N = len(signal)
    fourier = np.zeros(N, dtype=np.complex_)

    for k in range(N):
        for n in range(N):
            fourier[k] += signal[n] * np.exp((2j * np.pi * k * n) / N)
    return fourier / N


def FFT_1D(signal):
    N = len(signal)

    # Threshold of 4 returns the fastest result
    if N <= 4:
        return naive_transform(signal)
    else:
        X_even = FFT_1D(signal[::2])
        X_odd = FFT_1D(signal[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)

        X = np.concatenate([X_even + factor[:int(N / 2)] * X_odd, X_even + factor[int(N / 2):] * X_odd])
        return X


def inverse_inter(signal):
    N = len(signal)

    if N <= 1:
        return signal
    else:
        X_even = inverse_inter(signal[::2])
        X_odd = inverse_inter(signal[1::2])
        factor = np.exp(2j * np.pi * np.arange(N) / N)

        X = np.concatenate([X_even + factor[:int(N / 2)] * X_odd, X_even + factor[int(N / 2):] * X_odd])

        return X


def FFT_1D_inverse(signal):
    return inverse_inter(signal) / len(signal)


# pad a 2D signal with zeros to the next power of 2
def pad_signal(signal):
    # find the next power of 2
    n = 2 ** (signal.shape[0] - 1).bit_length()
    m = 2 ** (signal.shape[1] - 1).bit_length()

    # create a new array with the new dimensions
    padded_signal = np.zeros((n, m), dtype=np.complex_)

    # copy the old array into the new one
    padded_signal[:signal.shape[0], :signal.shape[1]] = signal

    return padded_signal


# remove the padding from a 2D signal
def remove_padding(original_signal, padded_signal):
    return padded_signal[:original_signal.shape[0], :original_signal.shape[1]]


def FFT_2D(matrix: np.ndarray):
    matrix_padded = pad_signal(matrix)
    output = np.zeros(matrix_padded.shape, dtype=np.complex_)

    # Take 1D FTT of Rows
    for i, row in enumerate(matrix_padded):
        output[i] = FFT_1D(row)

    # Take 1D FTT of Columns
    # To access the column we just transpose our matrix
    for i, column in enumerate(output.T):
        output.T[i] = FFT_1D(column)

    return remove_padding(matrix, output)


def FFT_2D_numpy(matrix: np.ndarray):
    matrix_padded = pad_signal(matrix)
    output = np.zeros(matrix_padded.shape, dtype=np.complex_)

    # Take 1D FTT of Rows
    for i, row in enumerate(matrix_padded):
        output[i] = np.fft.fft(row)

    # Take 1D FTT of Columns
    # To access the column we just transpose our matrix
    for i, column in enumerate(output.T):
        output.T[i] = np.fft.fft(column)

    return remove_padding(matrix, output)


def FFT_2D_inverse(matrix: np.ndarray):
    matrix_padded = pad_signal(matrix)
    output = np.zeros(matrix_padded.shape, dtype=np.complex_)

    # Take 1D FTT of Rows
    for i, row in enumerate(matrix_padded):
        output[i] = FFT_1D_inverse(row)

    # Take 1D FTT of Columns
    # To access the column we just transpose our matrix
    for i, column in enumerate(output.T):
        output.T[i] = FFT_1D_inverse(column)

    return remove_padding(matrix, output)


def denoise(fft, threshold=0.9):
    threshold = (1-threshold) * np.max(fft)
    fft[abs(fft) < threshold] = 0
    return fft
