from PIL import Image
import scipy.signal.windows
from scipy.ndimage import filters
import numpy.fft as fft
import numpy as np
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


def testaverageFT(im, params=None):
    filterSize = params['filterSize']
    filterMask = avgFilter(filterSize)  # the usually small mask
    filterBig = np.zeros_like(im, dtype=float)

    ## First, get sizes
    w, h = filterMask.shape
    w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2  # half width and height of the "big" mask

    ## Then, paste the small mask at the center using the sizes computed before as an aid
    filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterMask
    # FFT of the big filter
    filterBig = FT(fft.ifftshift(filterBig))
    #phase = np.angle(filterBig)
    magnitude = np.absolute(filterBig)
    im2 = FT(im)
    im2 = im2 * filterBig
    #im_filt_phase = np.angle(im2)
    im_filt_magnitude = np.absolute(im2)
    return [magnitude]


# -----------------
# Test image files
# -----------------
path_input = './imgs-P3/'
path_output = './imgs-out-P3/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.pgm")
else:
    files = [path_input + 'einstein.jpg']  # lena255, habas, mimbre

# --------------------
# Tests to perform
# --------------------
bAllTests = True
if bAllTests:
    tests = ['testaverageFT']
else:
    tests = ['testaverageFT']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testaverageFT': 'Average Filter Transform of Fourier'
             }

bSaveResultImgs = False

testsUsingPIL = []


def doTests(size):
    print("Testing on", files)
    for imfile in files:
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array
        for test in tests:
            if test is "testaverageFT":
                params = {}
                params['filterSize'] = size
                subTitle = ":|FT(M)|"

            if test in testsUsingPIL:
                outs_pil = eval(test)(im_pil, params)
                outs_np = vpu.pil2np(outs_pil)
            else:
                # apply test to given image and given parameters
                outs_np = eval(test)(im, params)
            print("# images", len(outs_np))
            print(len(outs_np))

            vpu.showInGrid(outs_np, title=nameTests[test] + subTitle)


if __name__ == "__main__":
    for i in [3,5,7,15,21,31,41]:
        doTests(i)