import DiscreteFourier as Df
import time
import numpy as np
import math


# function to compare values in an array and print values that are not equal to each other
def compare_arrays(array1, array2):
    for i in range(len(array1)):
        if math.floor(array1[i]) != math.floor(array2[i]):
            print(f"Index: {i} \t Array1: {array1[i]} \t Array2: {array2[i]}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Discrete Fourier Transform\n\n')
    signal = [3, 9, 11, 13, 15, 17]

    signal = np.random.randint(0, 10, 1024)

    tic = time.perf_counter()
    naive = Df.naive_transform(signal)
    toc = time.perf_counter()
    print(f"Naive in  {toc - tic:0.4f} seconds")


    tic = time.perf_counter()
    berkeley = Df.FFT(signal)
    toc = time.perf_counter()
    print(f"Berkeley in  {toc - tic:0.4f} seconds \n\n")

