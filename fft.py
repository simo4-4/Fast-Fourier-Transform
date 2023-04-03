import DiscreteFourier as df
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import image as mpimg
import pprint
import argparse


def mode_1(matrix: np.ndarray):
    """
    2D FFT of an image
    :param matrix: the image to be transformed
    """
    padding = df.pad_signal(matrix)
    fft = df.FFT_2D(padding)

    # show original and fft
    plt.subplot(1, 2, 1)
    plt.imshow(matrix, cmap='gray')
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(fft), cmap='gray', norm=LogNorm())
    plt.title("FFT")

    plt.show()


def mode_2(matrix: np.ndarray):
    padding = df.pad_signal(matrix)
    fft = df.FFT_2D(padding)

    fft = df.denoise(fft, 0.1)
    inverse = df.IFFT_2D(fft)
    inverse = df.remove_padding(matrix, inverse)

    plt.subplot(1, 2, 1)
    plt.imshow(matrix, cmap='gray')
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(inverse), cmap='gray')
    plt.title("Denoised Image")

    plt.show()


def mode_2_testing(matrix: np.ndarray):
    padding = df.pad_signal(matrix)
    fft = df.FFT_2D(padding)

    denoise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45]

    # Show original image
    plt.figure(figsize=(8, 8))
    plt.title("Denoised Images")

    plt.subplot(3, 3, 1)
    plt.imshow(matrix, cmap='gray')
    plt.title("Original Image")

    for i, denoise_level in enumerate(denoise_levels):
        fft_denoised = df.denoise(fft, denoise_level)
        inverse = df.IFFT_2D(fft_denoised)
        inverse = df.remove_padding(matrix, inverse)

        plt.subplot(3, 3, i + 2)
        plt.imshow(np.abs(inverse), cmap='gray')
        plt.title(f"D Level: {denoise_level}")

    plt.show()


def mode_3(matrix: np.ndarray):
    padding = df.pad_signal(matrix)
    fft = df.FFT_2D(padding)

    compression_levels = [0.0, 0.1, 0.25, 0.5, 0.70, 0.95]

    plt.figure(figsize=(12, 8))

    for i, compression_level in enumerate(compression_levels):
        fft_compressed = df.compress(fft, compression_level)
        inverse = df.IFFT_2D(fft_compressed)
        inverse = df.remove_padding(matrix, inverse)

        # Save fft_compressed non-zero coefficients to file
        np.savetxt(f"results/compressed_files/fft_compressed_{compression_level}.txt"
                   , fft_compressed[fft_compressed != 0])

        plt.subplot(2, 3, i + 1)
        plt.imshow(np.abs(inverse), cmap='gray')
        plt.title(f"Compression Level: {compression_level * 100}%")

    plt.show()


def mode_4():
    """
    Compare the 2D DFT and 2D FFT
    Print the time taken for each function to run for each image size
    """
    sizes: list[int] = [2 ** x for x in range(2, 7)]
    ft_function: list[tuple] = [
        (df.DFT_2D_naive, "2D DFT Naive"),
        (df.FFT_2D, "2D FFT"),
        (np.fft.fft2, "Numpy FFT"),
    ]
    runs_per_function = 5

    data: dict[str, dict[int, list[float]]] = {}
    for (_, function_name) in ft_function:
        data[function_name] = {}

    for i, size in enumerate(sizes):
        # print(f"Image size: {size}x{size}")
        for j, (function, function_name) in enumerate(ft_function):
            # print(f"Running {function_name}")

            time_taken = []
            for k in range(runs_per_function):
                start = time.time()
                function(np.random.rand(size, size))
                end = time.time()

                time_taken.append(end - start)
                # print(f"Run {k}: Time taken: {end - start:.8f}")

            data[function_name][size] = time_taken
            # print(f"Average time taken: {np.mean(time_taken):.8f}")
        # print()

    plot_data(data, sizes)


def plot_data(data: dict, keys: list[int]):
    """
    Plot the data
    :param data: the data to plot
    :param keys: the keys to use for the x-axis
    """
    # pprint.pprint(data)

    means: dict[str, dict] = {}
    std_devs: dict[str, dict] = {}
    variance: dict[str, dict] = {}
    for function_name, function_data in data.items():
        if function_name not in means:
            means[function_name] = {}
        if function_name not in std_devs:
            std_devs[function_name] = {}
        if function_name not in variance:
            variance[function_name] = {}
        for size, times in function_data.items():
            means[function_name][size] = np.mean(times)
            std_devs[function_name][size] = np.std(times)
            variance[function_name][size] = np.var(times)

    # Print
    for function_name in data.keys():
        print(f"Function: {function_name}")
        print(f"{'SIZE':>5} {'MEAN':>10} {'STDDEV':>10} {'VARIANCE':>20}")
        for size in keys:
            print(f"{size:>5} "
                  f"{means[function_name][size]:>10.8f} "
                  f"{std_devs[function_name][size]:>10.10f} "
                  f"{variance[function_name][size]:>10.20f} ")

    # Point plot with error bars for standard deviation with all functions
    for function_name, function_data in means.items():
        y = [function_data[key] for key in keys]
        y_error = [std_devs[function_name][key] for key in keys]
        plt.errorbar(keys, y, yerr=y_error, label=function_name, marker='o', capsize=5, capthick=1)

    plt.xlabel("Image Size (pixels)")
    plt.ylabel("Time Taken (s)")
    plt.title("Time taken for 2D Naive DFT, 2D FFT and Numpy FFT")
    plt.legend()
    plt.savefig("results/naive-vs-fft-vs-numpy.png")
    plt.show()

    # Point plot with error bars for standard deviation with 2D FFT and Numpy FFT
    for function_name, function_data in means.items():
        if function_name == "2D DFT Naive":
            continue

        y = [function_data[key] for key in keys]
        y_error = [std_devs[function_name][key] for key in keys]
        plt.errorbar(keys, y, yerr=y_error, label=function_name, marker='o', capsize=5, capthick=1)

    plt.xlabel("Image Size (pixels)")
    plt.ylabel("Time Taken (s)")
    plt.title("Time taken for 2D FFT and Numpy FFT")
    plt.legend()
    plt.savefig("results/fft-vs-numpy.png")
    plt.show()



def main():
    """
    Main function
    Parse command line arguments and call the appropriate function

    Usage:
        python fft.py [-m mode] [-i image]
    """

    parser = argparse.ArgumentParser(description="Perform 2D FFT on an image")
    parser.add_argument("-m", metavar="mode", type=int, default=1, choices=[1, 2, 3, 4],
                        help="The mode to run the program in")
    parser.add_argument("-i", metavar="image", type=str,
                        default="moonlanding.png", help="The image to perform the 2D FFT on")

    args = parser.parse_args()
    mode = args.m
    image = args.i

    matrix = mpimg.imread(image)
    if mode == 1:
        mode_1(matrix)
    elif mode == 2:
        mode_2_testing(matrix)
    elif mode == 3:
        mode_3(matrix)
    elif mode == 4:
        mode_4()


if __name__ == '__main__':
    main()
