import scipy.signal.windows
from PIL import Image
from scipy.ndimage import filters
from scipy.signal import medfilt2d
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob

def quotientImage(im, sigma):
    filtered_im = filters.gaussian_filter(im, sigma)
    quotient_image = np.divide(im, filtered_im)
    return quotient_image


image_lena = np.array(Image.open('./imgs-P2/lena512.pgm'))
quotient_1 = quotientImage(image_lena,1)
plt.figure(1)
plt.imshow(quotient_1,cmap='gray',interpolation=None)
plt.show()

quotient_3 = quotientImage(image_lena,3)
plt.figure(3)
plt.imshow(quotient_3,cmap='gray',interpolation=None)
plt.show()

quotient_5 = quotientImage(image_lena,5)
plt.figure(5)
plt.imshow(quotient_5,cmap='gray',interpolation=None)
plt.show()

quotient_7 = quotientImage(image_lena,7)
plt.figure(7)
plt.imshow(quotient_7,cmap='gray',interpolation=None)
plt.show()

quotient_9 = quotientImage(image_lena,9)
plt.figure(9)
plt.imshow(quotient_9,cmap='gray',interpolation=None)
plt.show()

quotient_15 = quotientImage(image_lena,15)
plt.figure(15)
plt.imshow(quotient_15,cmap='gray',interpolation=None)
plt.show()

    #Image.fromarray(im1).save('./imgs-P2/lenaquotient1.pgm')
    #im3 = quotientImage(im, 3)
    #Image.fromarray(im3).save('./imgs-P2/lenaquotient3.pgm')
    #im5 = quotientImage(im, 5)
    #Image.fromarray(im5).save('./imgs-P2/lenaquotient5.pgm')
    #im7 = quotientImage(im, 7)
    #Image.fromarray(im7).save('./imgs-P2/lenaquotient7.pgm')