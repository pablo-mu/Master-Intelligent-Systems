from PIL import Image
import scipy.signal.windows
from scipy.ndimage import filters
import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob

import visualPercepUtils as vpu


def FT(im):
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
    return fft.fftshift(fft.fft2(im))

def IFT(ft):
    return fft.ifft2(fft.ifftshift(ft))  # assumes ft is shifted and therefore reverses the shift before IFT
def test(im,im2,params =None):
    res = params*FT(im)+(1-params)*FT(im2)
    return IFT(res)


def dotest():
    im = np.array(Image.open('./imgs-P3/stp1.gif').convert('L'))
    im2 = np.array(Image.open('./imgs-P3/stp2.gif').convert('L'))
    params = np.linspace(0,1,9)
    print(len(params))
    im_shape = np.array(np.shape(im2))
    print(im_shape)
    im_res = np.zeros(np.append(im_shape,len(params)))
    print(np.shape(im_res))
    i=0
    for param in params:
        im_res[:,:,i] = test(im,im2,param)
        plt.imshow(im_res[:,:,i], cmap = 'gray')
        plt.title(f"λ = {param}")
        plt.show()
        i = i+1
    #c=[]
    #for i in range(len(params)):
    #    c.append(im_res[:,:,i])

    #vpu.showInGrid(c, title='hola')


if __name__ == "__main__":
    dotest()