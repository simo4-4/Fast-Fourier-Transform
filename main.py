import DiscreteFourier as df
from PIL import Image
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Discrete Fourier Transform\n\n')
    signal = [1, 0, -1, 0]
    signal = [0, 1, 0, -1]
    signal = [-1,2,3,0]
    N = len(signal)
    fourier = df.DiscreteFourier(signal)
    dft = fourier.naive_transform(signal, N)
    print(dft)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
