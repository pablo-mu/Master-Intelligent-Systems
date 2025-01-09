from PIL import Image
import scipy.signal.windows
from scipy.ndimage import filters
import numpy.fft as fft
import numpy as np
import timeit
import statistics
import matplotlib.pyplot as plt
import math as math
import glob
import os
import sys
import visualPercepUtils as vpu

def FT(im):
    return fft.fftshift(fft.fft2(im))

def IFT(ft):
    return fft.ifft2(fft.ifftshift(ft))
def avgFilter(filterSize):
    mask = np.ones((filterSize, filterSize))
    return mask/np.sum(mask)

def averageFilterSpace(im, filterSize):
    return filters.convolve(im, avgFilter(filterSize))

def averageFilterFrequency(im, filterSize):
    filterMask = avgFilter(filterSize)  # the usually small mask
    filterBig = np.zeros_like(im, dtype=float)  # as large as the image (dtype is important here!)

    # Now, place filter (the "small" filter mask) at the center of the "big" filter

    ## First, get sizes
    w, h = filterMask.shape
    w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2  # half width and height of the "big" mask

    ## Then, paste the small mask at the center using the sizes computed before as an aid
    filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterMask

    # FFT of the big filter
    filterBig = fft.ifftshift(filterBig)  # shift origin at upper-left corner

    # Finally, IFT of the element-wise product of the FT's
    return np.absolute(IFT(FT(im) * FT(filterBig)))

if __name__ == "__main__":
    im = np.array(Image.open('./imgs-P3/einstein.jpg').convert('L'))
    sizes = [1,3,7,15,31,51,75,101]
    mean_times_space = []
    mean_times_frequency = []
    deviation_times = []
    k=0
    for i in sizes:
        # Number of repetitions for timing
        number_of_repeats = 10
        # Execute the code and record execution times
        execution_times = timeit.repeat(lambda: averageFilterSpace(im,i), setup="pass", repeat=5, number=number_of_repeats)
        # Calculate mean and standard deviation
        mean_times_space.append(statistics.stdev(execution_times))
        execution_times = timeit.repeat(lambda: averageFilterFrequency(im,i), setup="pass", repeat=5,number=number_of_repeats)
        mean_times_frequency.append(statistics.stdev(execution_times))
        #deviation_times.append(statistics.stdev(execution_times))
        k+=1
        print(k)
    a = plt.plot([1,3,7,15,31,51,101,201], mean_times_space, label='Space Filter')
    b = plt.plot([1, 3, 7, 15, 31, 51, 101, 201], mean_times_frequency, label='Frequency Filter')
    plt.ylabel('Deviation times of execution')
    plt.xlabel('Filter Size')
    plt.legend(['Space Filter', 'Frequency Filter'])
    plt.show()
