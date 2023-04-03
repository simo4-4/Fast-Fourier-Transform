import numpy as np


def assert_power_of_2(n: int):
    assert (n & (n - 1)) == 0, "Signal must have a length that is a power of 2"


def DFT_1D_naive(signal: np.ndarray) -> np.ndarray:
    """
    Naive implementation of the DFT
    :param signal: 1D signal
    :return: 1D DFT of the signal
    """
    N = len(signal)

    fourier = np.zeros(N, dtype=np.complex_)

    for k in range(N):
        for n in range(N):
            fourier[k] += signal[n] * np.exp((-2j * np.pi * k * n) / N)

    return fourier


def IDFT_1D_naive(signal: np.ndarray) -> np.ndarray:
    """
    Naive implementation of the inverse DFT
    :param signal: 1D signal
    :return: 1D inverse DFT of the signal
    """
    N = len(signal)
    fourier = np.zeros(N, dtype=np.complex_)

    for k in range(N):
        for n in range(N):
            fourier[k] += signal[n] * np.exp((2j * np.pi * k * n) / N)
    return fourier / N


def FFT_1D(signal: np.ndarray, threshold=4) -> np.ndarray:
    """
    Fast implementation of the DFT
    1D cooley-tukey fast fourier transform divide and conquer algorithm
    Runs in O(n log n) time
    :param signal: 1D signal
    :param threshold: threshold of when to use the naive implementation
    :return: 1D DFT of the signal
    """
    N = len(signal)
    assert_power_of_2(N)

    if N <= threshold:
        return DFT_1D_naive(signal)
    else:
        X_even = FFT_1D(signal[::2], threshold)
        X_odd = FFT_1D(signal[1::2], threshold)
        factor = np.exp(-2j * np.pi * np.arange(N) / N)

        N_half = int(N / 2)
        left = X_even + factor[:N_half] * X_odd
        right = X_even + factor[N_half:] * X_odd
        return np.concatenate([left, right])


def IFFT_1D(signal: np.ndarray) -> np.ndarray:
    """
    Fast implementation of the inverse DFT
    :param signal: 1D signal
    :return: 1D inverse DFT of the signal
    """
    def ifft(sig: np.ndarray):
        N = len(sig)
        assert_power_of_2(N)

        if N <= 1:
            return sig.copy()
        else:
            X_even = ifft(sig[::2])
            X_odd = ifft(sig[1::2])
            factor = np.exp(2j * np.pi * np.arange(N) / N)

            N_half = int(N / 2)
            left = X_even + factor[:N_half] * X_odd
            right = X_even + factor[N_half:] * X_odd
            return np.concatenate([left, right])

    return ifft(signal) / len(signal)


def DFT_2D_naive(matrix: np.ndarray) -> np.ndarray:
    """
    Naive implementation of the 2D DFT
    :param matrix: 2D signal
    :return: 2D DFT of the signal
    """

    N, M = matrix.shape

    # Vectorized implementation of the 2D DFT (slightly faster than the single 4 nested loops implementation)
    k, l, n, m = np.meshgrid(np.arange(N), np.arange(M), np.arange(N), np.arange(M), indexing='ij')
    fourier = np.sum(matrix * np.exp((-2j * np.pi * (k * n / N + l * m / M))), axis=(2, 3))

    return fourier


def FFT_2D(matrix: np.ndarray, threshold=4) -> np.ndarray:
    """
    Fast implementation of the 2D DFT
    :param matrix: 2D signal, expects a square matrix N x N where N is a power of 2
    :param threshold: threshold to use the naive implementation
    :return: 2D DFT of the signal
    """
    assert_power_of_2(matrix.shape[0])
    assert_power_of_2(matrix.shape[1])
    output = np.zeros(matrix.shape, dtype=np.complex_)

    # Take 1D FTT of Rows
    for i, row in enumerate(matrix):
        output[i] = FFT_1D(row, threshold)

    # Take 1D FTT of Columns
    # To access the column we just transpose our matrix
    for i, column in enumerate(output.T):
        output.T[i] = FFT_1D(column, threshold)

    return output


def IFFT_2D(matrix: np.ndarray) -> np.ndarray:
    """
    Fast implementation of the 2D inverse DFT
    :param matrix: 2D signal
    :return: 2D inverse DFT of the signal
    """
    assert_power_of_2(matrix.shape[0])
    assert_power_of_2(matrix.shape[1])
    output = np.zeros(matrix.shape, dtype=np.complex_)

    # Take 1D FTT of Rows
    for i, row in enumerate(matrix):
        output[i] = IFFT_1D(row)

    # Take 1D FTT of Columns
    # To access the column we just transpose our matrix
    for i, column in enumerate(output.T):
        output.T[i] = IFFT_1D(column)

    return output


def pad_signal(signal: np.ndarray) -> np.ndarray:
    """
    Pad a 2D signal with zeros to the next power of 2
    :param signal: 2D signal
    :return: 2D signal padded with zeros
    """
    # find the next power of 2
    rows, cols = signal.shape
    n = 2 ** (rows - 1).bit_length()
    m = 2 ** (cols - 1).bit_length()

    assert_power_of_2(n)
    assert_power_of_2(m)

    # create a new array with the new dimensions
    padded_signal = np.zeros((n, m), dtype=np.complex_)

    # copy the old array into the new one
    padded_signal[:rows, :cols] = signal

    return padded_signal


def remove_padding(original_signal: np.ndarray, padded_signal: np.ndarray) -> np.ndarray:
    """
    Remove the padding from a 2D signal
    :param original_signal: original 2D signal
    :param padded_signal: the original 2D signal padded with zeros
    :return: 2D signal the same size as the original signal without the padding
    """
    return padded_signal[:original_signal.shape[0], :original_signal.shape[1]]


def denoise(signal: np.ndarray, threshold=0.1) -> np.ndarray:
    """
    Denoise a signal by removing the values below a certain threshold
    :param signal:
    :param threshold: threshold for the values to keep
    :return: denoised signal
    """

    signal = signal.copy()
    rows, cols = signal.shape

    row_low_freq = int(rows * threshold)
    row_high_freq = int(rows * (1 - threshold))
    signal[row_low_freq:row_high_freq, :] = 0
    # print("Removing rows {} to {}".format(row_low_freq, row_high_freq))

    col_low_freq = int(cols * threshold)
    col_high_freq = int(cols * (1 - threshold))
    signal[:, col_low_freq:col_high_freq] = 0
    # print("Removing columns {} to {}".format(col_low_freq, col_high_freq))

    total = rows * cols
    total_zeros = np.count_nonzero(signal == 0)
    total_non_zeros = total - total_zeros
    print(f"Denoising with threshold {threshold}:")
    print(f"Number of non-zero values: {total_non_zeros} out of {total}.")
    print(f"Percentage of non-zero values: {total_non_zeros / total * 100}%")
    print()

    return signal


def compress(signal: np.ndarray, compression_level: float) -> np.ndarray:
    """
    Compress a signal by keeping all the coefficients of very low frequencies as well as a fraction of the largest
    coefficients from higher frequencies
    :param signal:
    :param compression_level: fraction of the largest coefficients to keep
    :return: compressed signal
    """

    initial_zeros = np.count_nonzero(signal == 0)

    compliment_level = 1 - compression_level
    low_freq = np.percentile(signal, compliment_level * 50)
    high_freq = np.percentile(signal, 100 - compliment_level * 50)

    compressed = np.where((signal <= low_freq) | (signal >= high_freq), signal, 0)

    total = compressed.size
    total_zeros = np.count_nonzero(compressed == 0) - initial_zeros
    total_non_zeros = total - total_zeros
    print(f"For compression level {compression_level * 100}%:")
    print(f"Number of non-zero values: {total_non_zeros} out of {total}.")
    print(f"Percentage of non-zero values: {total_non_zeros / total * 100}%")
    print()

    return compressed












