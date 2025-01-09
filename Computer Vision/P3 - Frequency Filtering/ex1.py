from PIL import Image
from scipy.ndimage import filters
import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import glob

def FT(im):
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
    return fft.fftshift(fft.fft2(im))

def IFT(ft):
    return fft.ifft2(fft.ifftshift(ft))  # assumes ft is shifted and therefore reverses the shift before IFT

def testFT(im, params=None):
    ft = FT(im)
    #print(ft.shape)
    phase = np.angle(ft)
    magnitude = np.log(np.absolute(ft))
    bMagnitude = True
    if bMagnitude:
        im2 = np.absolute(IFT(ft))  # IFT consists of complex number. When applied to real-valued data the imaginary part should be zero, but not exactly for numerical precision issues
    else:
        im2 = np.real(IFT(ft)) # with just the module we can't appreciate the effect of a shift in the signal (e.g. if we use fftshift but not ifftshift, or viceversa)
        # Important: one case where np.real() is appropriate but np.absolute() is not is where the sign in the output is relevant
    return [magnitude, phase, im2]

if __name__ == "__main__":
    im = np.array(Image.open('./imgs-P3/habas.pgm'))
    [im_mag, im_phase, im_ft] = testFT(im)
    print(np.max(im_mag))
    print(np.min(im_mag))
    magnitude_values = im_mag.flatten()
    # Create a boxplot
    plt.boxplot(magnitude_values)
    plt.title('Boxplot of Magnitude Values')
    plt.ylabel('Magnitude')
    plt.show()

    # Flatten the phase spectrum into a 1D array
    phase_values = im_phase.flatten()
    plt.hist(phase_values, bins=50, range=(-np.pi, np.pi), density=True)
    plt.title('Histogram of Phase Values')
    plt.xlabel('Phase Value (radians)')
    plt.ylabel('Frequency')
    # Set custom tick locations and labels for x-axis
    custom_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    custom_labels = ['-π', '-π/2', '0', 'π/2', 'π']
    plt.xticks(custom_ticks, custom_labels)
    plt.show()