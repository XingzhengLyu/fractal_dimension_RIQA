# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 13:15:32 2021

@author: Xingzheng Lyu
"""
import numpy as np
import pylab as pl 
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = '14'
import matplotlib.patches as patches
from skimage.morphology import skeletonize
import skimage.io as io


# From https://github.com/jankaWIS/fractal_dimension_analysis/blob/main/fractal_analysis_fxns.py
def boxcount(Z, k):
    """
    returns a count of squares of size kxk in which there are both colours (black and white), ie. the sum of numbers
    in those squares is not 0 or k^2
    Z: np.array, matrix to be checked, needs to be 2D
    k: int, size of a square
    """
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)  # jumps by powers of 2 squares

    # We count non-empty (0) and non-full boxes (k*k)
    return len(np.where((S > 0) & (S < k * k))[0])

def fractal_dimension(Z, threshold=0.9):
    """
    calculate fractal dimension of an object in an array defined to be above certain threshold as a count of squares
    with both black and white pixels for a sequence of square sizes. The dimension is the a coefficient to a poly fit
    to log(count) vs log(size) as defined in the sources.
    :param Z: np.array, must be 2D
    :param threshold: float, a thr to distinguish background from foreground and pick up the shape, originally from
    (0, 1) for a scaled arr but can be any number, generates boolean array
    :return: coefficients to the poly fit, fractal dimension of a shape in the given arr
    """
    # Only for 2d image
    assert (len(Z.shape) == 2)

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

# Modified From https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/fractal_dimension.py
def plot_fd2(Z,img):
    sizes = 512,256,128, 64, 32, 16
    subfigs = ['(a) Retinal Field Definition','(b) Box Size:','(c) Box Size:','(d) Box Size:','(e) Box Size:','(f) Box Size:']#subfigs = ['(a):','(b)','(c)','(d)','(e)']
    xmin, xmax = 0, Z.shape[1]
    ymin, ymax = 0, Z.shape[0]
    plt.figure(figsize=(7, 5))

    for i, size in enumerate(sizes):
        ax = plt.subplot(2, 3, i+1, frameon=False)
        if i==0:
            ax.imshow(img)
        else:
            ax.imshow(1-Z, plt.cm.gray, interpolation="bicubic", vmin=0, vmax=1,
                      extent=[xmin, xmax, ymin, ymax], origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        if i>0:
            for y in range(Z.shape[0]//size+1):
                for x in range(Z.shape[1]//size+1):
                    s = (Z[y*size:(y+1)*size, x*size:(x+1)*size] > 0.25).sum()
                    if s > 0 and s < size*size:
                        rect = patches.Rectangle(
                            (x*size, Z.shape[0]-1-(y+1)*size),
                            width=size, height=size,
                            linewidth=.5, edgecolor='.25',
                            facecolor='.75', alpha=.5)
                        ax.add_patch(rect)
            ax.set_xlabel(subfigs[i]+' '+ str(size))
        else:
            ax.set_xlabel(subfigs[i])
    plt.tight_layout()
    plt.savefig("fractal-dimension2.png", dpi=300)
    plt.show()
    
def fd_vessel(filepath,maculaDir=None,plot_process=0,use_skeleton=0):
    vessel = pl.imread(filepath)
    if vessel.ndim==3:
        vessel = vessel[:,:,0]
    vessel = vessel>0
    
    if maculaDir:
        mask = pl.imread(maculaDir)
        if mask.ndim==3:
            mask = mask[:,:,0]
        mask = mask>0
        vessel[mask==0] = 0
        
    if use_skeleton:
        vessel = skeletonize(vessel, method='lee')
        io.imsave('skeleton.png',vessel*255)
    fd = fractal_dimension(vessel)
    
    if plot_process==1:
        img = pl.imread(r"F:\workspace2\vessel_gan\public-model\test/qudarant.jpg")
        plot_fd2(vessel,img)
    return fd


filepath = '' # fullpath to binary vessel image
fd = fd_vessel(filepath, plot_process=0)
