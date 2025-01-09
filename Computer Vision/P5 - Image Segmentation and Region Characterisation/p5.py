 #!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import glob
import sys
from skimage.color import label2rgb


from skimage import measure

from skimage.morphology import disk, square, closing, opening # for the mathematically morphology part

sys.path.append("../../p1/code")
import visualPercepUtils as vpu

bStudentVersion=True
if not bStudentVersion:
    import p5e

def testOtsu(im, params=None):
    nbins = 256
    th = threshold_otsu(im)
    hist = np.histogram(im.flatten(), bins=nbins, range=[0, 255])[0]
    return [th, im > th, hist]  # threshold, binarized image (using such threshold), and image histogram


def fillGaps(im, params=None):
    binIm = im > threshold_otsu(im)
    sElem = disk(params['closing-radius'])
    return [binIm, closing(binIm, sElem)]

# Don't worry about this function
def removeSmallRegions(im, params=None):
    binIm = im > threshold_otsu(im)
    sElem = disk(params['opening-radius'])
    return [binIm, opening(binIm, sElem)]

# Don't worry about this function
def fillGapsThenRemoveSmallRegions(im, params=None):
    binIm, closeIm = fillGaps(im, params)  # first, fill gaps
    sElem = disk(params['opening-radius'])
    return [binIm, opening(closeIm, sElem)]

def labelConnectedComponents(im, params=None):
    binIm = im > threshold_otsu(im, params)
    binImProc = fillGapsThenRemoveSmallRegions(im, params)[1]
    return [binIm, binImProc,
            measure.label(binIm, background=0), measure.label(binImProc, background=0)]

def coins(im,params = None):
    binIm = im > threshold_otsu(im, params)
    binImProc = fillGapsThenRemoveSmallRegions(im, params)[1]
    binImProc_mes = measure.label(binImProc, background=0)
    binImProc_mes = is_coin(binImProc_mes)
    label_im_mes = measure.label(binIm, background = 0)
    label_im_mes = is_coin(label_im_mes)
    return [binIm, binImProc,label2rgb(label_im_mes, colors= [[1,0,0],[0,1,0]]), label2rgb(binImProc_mes, colors= [[1,0,0],[0,1,0]])]

def dif_coins(im,params = None):
    binIm = im > threshold_otsu(im, params)
    binImProc = fillGapsThenRemoveSmallRegions(im, params)[1]
    binImProc_mes = measure.label(binImProc, background=0)
    binImProc_mes = which_coin(binImProc_mes)
    label_im_mes = measure.label(binIm, background = 0)
    label_im_mes = which_coin(label_im_mes)
    return [binIm, binImProc,label2rgb(label_im_mes, image = im, colors= [[1,0,0],[0,1,0], [0,0,1]]), label2rgb(binImProc_mes, image = im, colors= [[1,0,0],[0,1,0],[0,0,1]])]
def is_coin(im):
    regions = measure.regionprops(im)
    for region in regions:
        if region.perimeter > 0 and abs(region.area / region.perimeter - region.axis_minor_length / 4) < 10 and abs(region.area / region.perimeter - region.axis_major_length / 4) < 10:
            im[im == region.label] = 2
        else:
            if region.label != 0:
                im[im == region.label] = 1
    return im

def which_coin(im):
    regions = measure.regionprops(im)
    for region in regions:
        if region.perimeter > 0 and abs(region.area / region.perimeter - region.axis_minor_length / 4) < 10 and abs(region.area / region.perimeter - region.axis_major_length / 4) < 10:
            if abs(region.axis_minor_length - 38.32) < 5 or abs(region.axis_major_length - 38.32) <5:
                im[im == region.label] = 3
            else:
                im[im == region.label] = 2
        else:
            if region.label != 0:
                im[im == region.label] = 1
    return im
def reportPropertiesRegions(labelIm,title):
    print("* * "+title)
    regions = measure.regionprops(labelIm)
    for r, region in enumerate(regions):  # enumerate() is often handy: it provides both the index and the element
        print("Region", r + 1, "(label", str(region.label) + ")")
        print("\t area: ", region.area)
        print("\t perimeter: ", round(region.perimeter, 1))
        print("\t Radius", region.axis_minor_length)# show only one decimal place
        if abs(region.area/region.perimeter - region.axis_minor_length/4) < 10 or abs(region.area/region.perimeter - region.axis_major_length/4) < 10:
            print("This region is a coin.")

# -----------------
# Test image files
# -----------------
path_input = './imgs-P5/'
path_output = './imgs-out-P5/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.p??")
else:
    files = [path_input + 'monedas2.pgm']

# --------------------
# Tests to perform
# --------------------
bAllTests = True
if bAllTests:
    tests = ['testOtsu', 'labelConnectedComponents', 'coins', 'dif_coins']
else:
    tests = ['fillGaps']
    tests = ['labelConnectedComponents']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testOtsu': "thresholding with Otsu's method",
             'labelConnectedComponents': 'Labelling conected components',
             'coins': 'Coin detection',
             'dif_coins': 'Different Coin Detection'}

myThresh = 180  # use your own value here
diskSizeForClosing = 2  # don't worry about this
diskSizeForOpening = 5  # don't worry about this

def doTests():
    print("Testing ", tests, "on", files)
    nFiles = len(files)
    nFig = None
    for i, imfile in enumerate(files):
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array

        for test in tests:
            title = nameTests[test]
            print(imfile, test)
            if test is "testOtsu":
                params = {}
            elif test is "labelConnectedComponents":
                params = {}
                params['closing-radius'] = diskSizeForClosing
                params['opening-radius'] = diskSizeForOpening
                subtitles = ["original image", "binarized image", "Processed binary image", "Connected components", "Connected componentes from processed binary image"]
            elif test is "coins":
                params = {}
                params['closing-radius'] = diskSizeForClosing
                params['opening-radius'] = diskSizeForOpening
                subtitles = ["original image", "binarized image", "Processed binary image", "Connected components", "Connected componentes from processed binary image"]
            elif test is "dif_coins":
                params = {}
                params['closing-radius'] = diskSizeForClosing
                params['opening-radius'] = diskSizeForOpening
                subtitles = ["original image", "binarized image", "Processed binary image", "Connected components", "Connected componentes from processed binary image"]



            outs_np = eval(test)(im, params)

            if test is "testOtsu":
                outs_np_plot = [outs_np[2]] + [outs_np[1]] + [im > myThresh]
                subtitles = ["original image", "Histogram", "Otsu with threshold=" + str(outs_np[0]),
                             "My threshold: " + str(myThresh)]
                m = n = 2
            else:
                outs_np_plot = outs_np
            print(len(outs_np_plot))
            vpu.showInGrid([im] + outs_np_plot, m=m, n=n, title=title, subtitles=subtitles)
            if test is 'labelConnectedComponents':
                plt.figure()
                labelImOriginalBinaryImage = outs_np_plot[2]
                labelImProcessedBinaryImage = outs_np_plot[3]
                vpu.showImWithColorMap(labelImOriginalBinaryImage,'jet') # the default color map, 'spectral', does not work in lab computers
                plt.show(block=True)
                titleForBinaryImg = "From unprocessed binary image"
                titleForProcesImg = "From filtered binary image"
                reportPropertiesRegions(labelIm=labelImOriginalBinaryImage,title=titleForBinaryImg)
                reportPropertiesRegions(labelIm=labelImProcessedBinaryImage,title=titleForProcesImg)

                if not bStudentVersion:
                    p5e.displayImageWithCoins(im,labelIm=labelImOriginalBinaryImage,title=titleForBinaryImg)
                    p5e.displayImageWithCoins(im,labelIm=labelImProcessedBinaryImage,title=titleForProcesImg)

    plt.show(block=True)  # show pending plots


if __name__ == "__main__":
    doTests()
