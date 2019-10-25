'''This is the comet version wtih MPI enabled'''
'''This will do it iteratively for all viewpoints of a certain exposure'''
#https://stackoverflow.com/questions/27698604/what-do-the-different-values-of-the-kind-argument-mean-in-scipy-interpolate-inte --> how to interpret convolution

import matplotlib
matplotlib.use('agg')

import seaborn as sns
sns.set_style("dark")

import numpy as np
from mangadap.proc.templatelibrary import TemplateLibrary
from mangadap.util.pixelmask import SpectralPixelMask
from mangadap.par.emissionlinedb import EmissionLineDB
from mangadap.proc.ppxffit import PPXFFit
from mangadap.proc.stellarcontinuummodel import StellarContinuumModelBitMask
from mangadap.proc.spatialbinning import VoronoiBinningPar
from mangadap.proc.spatialbinning import VoronoiBinning

from matplotlib import pyplot as plt 
from scipy import interpolate, fftpack
from astropy import constants as con
from astropy import units as un

from matplotlib.animation import FuncAnimation

import astropy.io.fits as pyfits
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.colors import LogNorm

import os
import numpy.ma as ma

import sys
import scipy

import astropy

from scipy import ndimage
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import iterate_structure
from scipy.ndimage.filters import maximum_filter
import scipy.optimize as opt

from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

import statmorph



from time import clock
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance, cKDTree
from scipy import ndimage


def detect_peaks(image):
    """                                                                                                                                                                                                
    Takes an image and detect the peaks using the local maximum filter.                                                                                                                                
    Returns a boolean mask of the peaks (i.e. 1 when                                                                                                                                                   
    the pixel's value is the neighborhood maximum, 0 otherwise)                                                                                                                                        
    """

    # define an 8-connected neighborhood                                                                                                                                                               
    struct = generate_binary_structure(2,1)

    neighborhood = iterate_structure(struct, 10).astype(bool)

    #apply the local maximum filter; all pixel of maximal value                                                                                                                                        
    #in their neighborhood are set to 1                                                                                                                                                                
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are                                                                                                                                                
    #looking for, but also the background.                                                                                                                                                             
    #In order to isolate the peaks we must remove the background from the mask.                                                                                                                        


    #we create the mask of the background                                                                                                                                                              
    background = (image==0)

    #a little technicality: we must erode the background in order to                                                                                                                                   
    #successfully subtract it form local_max, otherwise a line will                                                                                                                                    
    #appear along the background border (artifact of the local maximum filter)                                                                                                                         
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks,                                                                                                                                                  
    #by removing the background from the local_max mask (xor operation)                                                                                                                                
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

def redistribute_voronoi(bins, x, y, xnode, ynode, mask_hex, noise, counts, vel, vel_e, sig, sig_e, mask_list):

    '''First, make a list of repeated elements'''
    records_array = bins
    vals, inverse, count = np.unique(records_array, return_inverse=True,
                              return_counts=True)

    idx_vals_repeated = np.where(count > 1)[0]
    vals_repeated = vals[idx_vals_repeated]

    rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
    _, inverse_rows = np.unique(rows, return_index=True)
    res = np.split(cols, inverse_rows[1:])
    '''Somehow stack all spaxels that have the same value for bins'''
    '''Somehow apply the entire thing like a segmentation map to mask_hex'''
    searchbins=np.linspace(0,np.max(bins), np.max(bins)+1)
    searchval_list=[]

    vel_2d = np.zeros((np.shape(mask_hex)[1], np.shape(mask_hex)[2]))
    vel_e_2d = np.zeros((np.shape(mask_hex)[1], np.shape(mask_hex)[2]))
    sig_2d = np.zeros((np.shape(mask_hex)[1], np.shape(mask_hex)[2]))
    sig_e_2d = np.zeros((np.shape(mask_hex)[1], np.shape(mask_hex)[2]))


    #I would say actually just search through all of res and if the same bin as in mask_list exists, 
    #fill everything with those values
    for j in range(np.shape(mask_list)[0]):#mask_list is actually the list of bin values of the fit ones
        '''First go through and just assign things their normal value'''
        ii = np.where(bins == mask_list[j])
        for p in range(len(ii[0])):
            #fill em all:
            index = ii[0][p]
            vel_2d[x[index],y[index]]=vel[j]
            vel_e_2d[x[index],y[index]]=vel_e[j]
            sig_2d[x[index],y[index]]=sig[j]
            sig_e_2d[x[index],y[index]]=sig_e[j]
        
        
    return vel_2d, vel_e_2d, sig_2d, sig_e_2d

def remap(bins, x, y, xnode, ynode, mask_hex, noise, counts, noise_no_s):#bins is the voronoi bins
    #This will help remap the voronoi bins and stack for input to ppxf
    vals, inverse, count = np.unique(bins, return_inverse=True,
                              return_counts=True)

    idx_vals_repeated = np.where(count > 1)[0]
    vals_repeated = vals[idx_vals_repeated]

    rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
    _, inverse_rows = np.unique(rows, return_index=True)
    res = np.split(cols, inverse_rows[1:])#res ends up making a list of bin indices of things that are repeated in bins
    #this does not include non-repeats
    mask_list=[]
    for j in range(np.shape(res)[0]):
        indices_in_bin=res[j]
        for i in range(len(indices_in_bin)):
            if i==0:
                continue
            else:
                mask_list.append(indices_in_bin[i])
    mask_list=np.sort(mask_list)#this is a list of the spaxels which are shared
    
    '''Somehow stack all spaxels that have the same value for bins'''
    '''Somehow apply the entire thing like a segmentation map to mask_hex'''
    searchbins=np.linspace(0,np.max(bins), np.max(bins)+1)#Now the bins are listed in order
    bins_2d= np.zeros((np.shape(mask_hex)[1], np.shape(mask_hex)[2]))
    bins_2d_plot= np.zeros((np.shape(mask_hex)[1], np.shape(mask_hex)[2]))
    
    bb_hex= np.zeros((np.shape(mask_hex)[1], np.shape(mask_hex)[2]))
    rnd = np.argsort(np.random.random(xnode.size))

    mu, sigma = 0, 1
    searchval_list=[]
    list_spec=[]
    list_noise=[]
    list_bins=[]


    for j in range(len(bins)):#so for all the different bins
        searchval = bins[j]

        try:                                                                                                                                                                                    
            index = searchval_list.index(searchval)#this exists if it has already been used   
            continue
        except ValueError:
            ii = np.where(bins == searchval)#find the index of the bin, but some bins are repeated, so there might be multiple indices (ii)
            #This is an array of indices
            if len(ii[0]) > 1:
                #print('long ii[0]', ii[0], type(ii[0]))
                ii_list=[]
                for p in range(len(ii[0])):
                    ii_list.append(ii[0][p])#bins.tolist().index(ii[0][p]))
                ii=ii_list
                    
            else:
                ii = ii[0]#bins.tolist().index(ii[0])
        if isinstance(ii,list):
            mask_hex_list=[]                                                                                                                                                                     
            percent_e=[]     

            for p in range(len(ii)):

                index = ii[p]
                bin_value = bins[index]
                bins_2d[x[index],y[index]] = bin_value
                bins_2d_plot[x[index],y[index]] = rnd[bin_value]
                searchval_list.append(bin_value)
                
                mask_hex_list.append(mask_hex[:,x[index],y[index]])                                                                                                                              
                percent_e.append((noise_no_s[:,x[index],y[index]]**2/mask_hex[:,x[index],y[index]]**2))                                                                                               
                                                                                                                                                                                                     
            new_mask_hex = np.array(np.median(mask_hex_list, axis=0))#np.apply_over_axes(np.median, mask_hex[:,x[ii],y[ii]], (1,2))                                                              
            s=np.random.normal(mu,sigma,len(new_mask_hex))                                                                                                                                   
            n_bins = len(ii)
            covariance = 1#for now because I don't understand covariance (1+1.62*math.log(n_bins))
            new_noise = covariance * np.array(np.sqrt(np.sum(percent_e, axis=0))*np.median(mask_hex_list, axis=0))#*s#np.sqrt(np.apply_over_axes(np.sum, (noise[:,x[ii],y[ii]]/mask_hex[:,x[ii],y[ii]])**2, (1\,2)))           
            


            '''The only one you assign value to is the first one'''
            for p in range(len(ii)):
               if p ==0:
                   
                   list_spec.append(np.reshape(new_mask_hex,np.shape(new_mask_hex)[0]))
                   list_noise.append(np.reshape(new_noise,np.shape(new_noise)[0]))
                   list_bins.append(bins[ii[0]])
               else:
                   mask_hex[:,x[ii[p]],y[ii[p]]] = new_mask_hex
                   noise[:,x[ii[p]],y[ii[p]]] = new_noise
                   bb_hex[x[ii[p]],y[ii[p]]] = np.sum(new_mask_hex)
            mask_hex[:,x[ii[0]],y[ii[0]]] = new_mask_hex
            noise[:,x[ii[0]],y[ii[0]]] = new_noise
            bb_hex[x[ii[0]],y[ii[0]]] = np.sum(new_mask_hex)
            
        else:
            bin_value = bins[ii]#what if ii is longer than 1?
            new_mask_hex = mask_hex[:,x[ii],y[ii]]
            s=np.random.normal(mu,sigma,(len(new_mask_hex),1))
            new_noise = noise_no_s[:,x[ii],y[ii]]#was s*noise
            searchval_list.append(bin_value)
            mask_hex[:,x[ii],y[ii]] = new_mask_hex
            bb_hex[x[ii],y[ii]] = np.sum(new_mask_hex)
            noise[:,x[ii],y[ii]] = new_noise#a random number between -1 and 1                                    
            bins_2d[x[ii],y[ii]] = bin_value
            bins_2d_plot[x[ii],y[ii]] = rnd[bin_value]
            list_spec.append(np.reshape(new_mask_hex,np.shape(new_mask_hex)[0]))
            list_noise.append(np.reshape(new_noise,np.shape(new_noise)[0]))
            list_bins.append(bin_value)
    return mask_hex, noise, bins_2d, res, bins_2d_plot, mask_list, list_spec, list_noise, list_bins, bb_hex
#----------------------------------------------------------------------------

def sn_func(index, signal=None, noise=None):
    """
    Default function to calculate the S/N of a bin with spaxels "index".

    The Voronoi binning algorithm does not require this function to have a
    specific form and this default one can be changed by the user if needed
    by passing a different function as

        ... = voronoi_2d_binning(..., sn_func=sn_func)

    The S/N returned by sn_func() does not need to be an analytic
    function of S and N.

    There is also no need for sn_func() to return the actual S/N.
    Instead sn_func() could return any quantity the user needs to equalize.

    For example sn_func() could be a procedure which uses ppxf to measure
    the velocity dispersion from the coadded spectrum of spaxels "index"
    and returns the relative error in the dispersion.

    Of course an analytic approximation of S/N, like the one below,
    speeds up the calculation.

    :param index: integer vector of length N containing the indices of
        the spaxels for which the combined S/N has to be returned.
        The indices refer to elements of the vectors signal and noise.
    :param signal: vector of length M>N with the signal of all spaxels.
    :param noise: vector of length M>N with the noise of all spaxels.
    :return: scalar S/N or another quantity that needs to be equalized.
    """

    sn = np.sum(signal[index])/np.sqrt(np.sum(noise[index]**2))

    # The following commented line illustrates, as an example, how one
    # would include the effect of spatial covariance using the empirical
    # Eq.(1) from http://adsabs.harvard.edu/abs/2015A%26A...576A.135G
    # Note however that the formula is not accurate for large bins.
    #
    sn /= 1 + 1.62*np.log10(index.size)#changed to the approximation from Law et al. 2016

    return  sn

#----------------------------------------------------------------------

def voronoi_tessellation(x, y, xnode, ynode, scale):
    """
    Computes (Weighted) Voronoi Tessellation of the pixels grid

    """
    if scale[0] == 1:  # non-weighted VT
        tree = cKDTree(np.column_stack([xnode, ynode]))
        classe = tree.query(np.column_stack([x, y]))[1]
    else:
        if x.size < 1e4:
            classe = np.argmin(((x[:, None] - xnode)**2 + (y[:, None] - ynode)**2)/scale**2, axis=1)
        else:  # use for loop to reduce memory usage
            classe = np.zeros(x.size, dtype=int)
            for j, (xj, yj) in enumerate(zip(x, y)):
                classe[j] = np.argmin(((xj - xnode)**2 + (yj - ynode)**2)/scale**2)

    return classe

#----------------------------------------------------------------------

def _roundness(x, y, pixelSize):
    """
    Implements equation (5) of Cappellari & Copin (2003)

    """
    n = x.size
    equivalentRadius = np.sqrt(n/np.pi)*pixelSize
    xBar, yBar = np.mean(x), np.mean(y)  # Geometric centroid here!
    maxDistance = np.sqrt(np.max((x - xBar)**2 + (y - yBar)**2))
    roundness = maxDistance/equivalentRadius - 1.

    return roundness

#----------------------------------------------------------------------

def _accretion(x, y, signal, noise, targetSN, pixelsize, quiet, sn_func):
    """
    Implements steps (i)-(v) in section 5.1 of Cappellari & Copin (2003)

    """
    n = x.size
    classe = np.zeros(n, dtype=int)  # will contain the bin number of each given pixel
    good = np.zeros(n, dtype=bool)   # will contain 1 if the bin has been accepted as good

    # For each point, find the distance to all other points and select the minimum.
    # This is a robust but slow way of determining the pixel size of unbinned data.
    #
    if pixelsize is None:
        if x.size < 1e4:
            pixelsize = np.min(distance.pdist(np.column_stack([x, y])))
        else:
            raise ValueError("Dataset is large: Provide `pixelsize`")

    currentBin = np.argmax(signal/noise)  # Start from the pixel with highest S/N
    SN = sn_func(currentBin, signal, noise)

    # Rough estimate of the expected final bins number.
    # This value is only used to give an idea of the expected
    # remaining computation time when binning very big dataset.
    #
    w = signal/noise < targetSN
    maxnum = int(np.sum((signal[w]/noise[w])**2)/targetSN**2 + np.sum(~w))

    # The first bin will be assigned CLASS = 1
    # With N pixels there will be at most N bins
    #
    for ind in range(1, n+1):

        if not quiet:
            print(ind, ' / ', maxnum)

        classe[currentBin] = ind  # Here currentBin is still made of one pixel
        xBar, yBar = x[currentBin], y[currentBin]    # Centroid of one pixels

        while True:

            if np.all(classe):
                break  # Stops if all pixels are binned

            # Find the unbinned pixel closest to the centroid of the current bin
            #
            unBinned = np.flatnonzero(classe == 0)
            k = np.argmin((x[unBinned] - xBar)**2 + (y[unBinned] - yBar)**2)

            # (1) Find the distance from the closest pixel to the current bin
            #
            minDist = np.min((x[currentBin] - x[unBinned[k]])**2 + (y[currentBin] - y[unBinned[k]])**2)

            # (2) Estimate the `roundness' of the POSSIBLE new bin
            #
            nextBin = np.append(currentBin, unBinned[k])
            roundness = _roundness(x[nextBin], y[nextBin], pixelsize)

            # (3) Compute the S/N one would obtain by adding
            # the CANDIDATE pixel to the current bin
            #
            SNOld = SN
            SN = sn_func(nextBin, signal, noise)

            # Test whether (1) the CANDIDATE pixel is connected to the
            # current bin, (2) whether the POSSIBLE new bin is round enough
            # and (3) whether the resulting S/N would get closer to targetSN
            #
            if (np.sqrt(minDist) > 1.2*pixelsize or roundness > 0.3
                or abs(SN - targetSN) > abs(SNOld - targetSN) or SNOld > SN):
                if SNOld > 0.8*targetSN:
                    good[currentBin] = 1
                break

            # If all the above 3 tests are negative then accept the CANDIDATE
            # pixel, add it to the current bin, and continue accreting pixels
            #
            classe[unBinned[k]] = ind
            currentBin = nextBin

            # Update the centroid of the current bin
            #
            xBar, yBar = np.mean(x[currentBin]), np.mean(y[currentBin])

        # Get the centroid of all the binned pixels
        #
        binned = classe > 0
        if np.all(binned):
            break  # Stop if all pixels are binned
        xBar, yBar = np.mean(x[binned]), np.mean(y[binned])

        # Find the closest unbinned pixel to the centroid of all
        # the binned pixels, and start a new bin from that pixel.
        #
        unBinned = np.flatnonzero(classe == 0)
        if sn_func(unBinned, signal, noise) < targetSN:
            break  # Stops if the remaining pixels do not have enough capacity
        k = np.argmin((x[unBinned] - xBar)**2 + (y[unBinned] - yBar)**2)
        currentBin = unBinned[k]    # The bin is initially made of one pixel
        SN = sn_func(currentBin, signal, noise)

    classe *= good  # Set to zero all bins that did not reach the target S/N

    return classe, pixelsize

#----------------------------------------------------------------------------

def _reassign_bad_bins(classe, x, y):
    """
    Implements steps (vi)-(vii) in section 5.1 of Cappellari & Copin (2003)

    """
    # Find the centroid of all successful bins.
    # CLASS = 0 are unbinned pixels which are excluded.
    #
    good = np.unique(classe[classe > 0])
    xnode = ndimage.mean(x, labels=classe, index=good)
    ynode = ndimage.mean(y, labels=classe, index=good)

    # Reassign pixels of bins with S/N < targetSN
    # to the closest centroid of a good bin
    #
    bad = classe == 0
    index = voronoi_tessellation(x[bad], y[bad], xnode, ynode, [1])
    classe[bad] = good[index]

    # Recompute all centroids of the reassigned bins.
    # These will be used as starting points for the CVT.
    #
    good = np.unique(classe)
    xnode = ndimage.mean(x, labels=classe, index=good)
    ynode = ndimage.mean(y, labels=classe, index=good)

    return xnode, ynode

#----------------------------------------------------------------------------

def _cvt_equal_mass(x, y, signal, noise, xnode, ynode, pixelsize, quiet, sn_func, wvt):
    """
    Implements the modified Lloyd algorithm
    in section 4.1 of Cappellari & Copin (2003).

    NB: When the keyword WVT is set this routine includes
    the modification proposed by Diehl & Statler (2006).

    """
    dens2 = (signal/noise)**4     # See beginning of section 4.1 of CC03
    scale = np.ones_like(xnode)   # Start with the same scale length for all bins

    for it in range(1, xnode.size):  # Do at most xnode.size iterations

        xnode_old, ynode_old = xnode.copy(), ynode.copy()
        classe = voronoi_tessellation(x, y, xnode, ynode, scale)

        # Computes centroids of the bins, weighted by dens**2.
        # Exponent 2 on the density produces equal-mass Voronoi bins.
        # The geometric centroids are computed if WVT keyword is set.
        #
        good = np.unique(classe)
        if wvt:
            for k in good:
                index = np.flatnonzero(classe == k)   # Find subscripts of pixels in bin k.
                xnode[k], ynode[k] = np.mean(x[index]), np.mean(y[index])
                sn = sn_func(index, signal, noise)
                scale[k] = np.sqrt(index.size/sn)  # Eq. (4) of Diehl & Statler (2006)
        else:
            mass = ndimage.sum(dens2, labels=classe, index=good)
            xnode = ndimage.sum(x*dens2, labels=classe, index=good)/mass
            ynode = ndimage.sum(y*dens2, labels=classe, index=good)/mass

        diff2 = np.sum((xnode - xnode_old)**2 + (ynode - ynode_old)**2)
        diff = np.sqrt(diff2)/pixelsize

        if not quiet:
            print('Iter: %4i  Diff: %.4g' % (it, diff))

        if diff < 0.1:
            break

    # If coordinates have changed, re-compute (Weighted) Voronoi Tessellation of the pixels grid
    #
    if diff > 0:
        classe = voronoi_tessellation(x, y, xnode, ynode, scale)
        good = np.unique(classe)  # Check for zero-size Voronoi bins

    # Only return the generators and scales of the nonzero Voronoi bins

    return xnode[good], ynode[good], scale[good], it

#-----------------------------------------------------------------------

def _compute_useful_bin_quantities(x, y, signal, noise, xnode, ynode, scale, sn_func):
    """
    Recomputes (Weighted) Voronoi Tessellation of the pixels grid to make sure
    that the class number corresponds to the proper Voronoi generator.
    This is done to take into account possible zero-size Voronoi bins
    in output from the previous CVT (or WVT).

    """
    # classe will contain the bin number of each given pixel
    classe = voronoi_tessellation(x, y, xnode, ynode, scale)

    # At the end of the computation evaluate the bin luminosity-weighted
    # centroids (xbar, ybar) and the corresponding final S/N of each bin.
    #
    good = np.unique(classe)
    xbar = ndimage.mean(x, labels=classe, index=good)
    ybar = ndimage.mean(y, labels=classe, index=good)
    area = np.bincount(classe)
    sn = np.empty_like(xnode)
    for k in good:
        index = np.flatnonzero(classe == k)   # index of pixels in bin k.
        sn[k] = sn_func(index, signal, noise)

    return classe, xbar, ybar, sn, area

#-----------------------------------------------------------------------

def _display_pixels(x, y, counts, pixelsize):
    """
    Display pixels at coordinates (x, y) coloured with "counts".
    This routine is fast but not fully general as it assumes the spaxels
    are on a regular grid. This needs not be the case for Voronoi binning.

    """
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    nx = int(round((xmax - xmin)/pixelsize) + 1)
    ny = int(round((ymax - ymin)/pixelsize) + 1)
    img = np.full((nx, ny), np.nan)  # use nan for missing data
    j = np.round((x - xmin)/pixelsize).astype(int)
    k = np.round((y - ymin)/pixelsize).astype(int)
    img[j, k] = counts

    plt.imshow((img), interpolation='nearest', cmap='plasma',
               extent=[xmin - pixelsize/2, xmax + pixelsize/2,
                       ymin - pixelsize/2, ymax + pixelsize/2])
    plt.colorbar()
#----------------------------------------------------------------------

def voronoi_2d_binning(x, y, signal, noise, targetSN, cvt=True,
                         pixelsize=None, plot=True, quiet=True,
                         sn_func=None, wvt=True):
    """
    PURPOSE:
          Perform adaptive spatial binning of Integral-Field Spectroscopic
          (IFS) data to reach a chosen constant signal-to-noise ratio per bin.
          This method is required for the proper analysis of IFS
          observations, but can also be used for standard photometric
          imagery or any other two-dimensional data.
          This program precisely implements the algorithm described in
          section 5.1 of the reference below.

    EXPLANATION:
          Further information on VORONOI_2D_BINNING algorithm can be found in
          Cappellari M., Copin Y., 2003, MNRAS, 342, 345

    CALLING SEQUENCE:

        binNum, xBin, yBin, xBar, yBar, sn, nPixels, scale = \
            voronoi_2d_binning(x, y, signal, noise, targetSN,
                               cvt=True, pixelsize=None, plot=True,
                               quiet=True, sn_func=None, wvt=True)

    """
    # This is the main program that has to be called from external programs.
    # It simply calls in sequence the different steps of the algorithms
    # and optionally plots the results at the end of the calculation.

    assert x.size == y.size == signal.size == noise.size, \
        'Input vectors (x, y, signal, noise) must have the same size'
    assert np.all((noise > 0) & np.isfinite(noise)), \
        'NOISE must be positive and finite'

    if sn_func is None:
        sn_func = _sn_func

    # Perform basic tests to catch common input errors
    #
    if sn_func(np.flatnonzero(noise > 0), signal, noise) < targetSN:
        raise ValueError("""Not enough S/N in the whole set of pixels.
            Many pixels may have noise but virtually no signal.
            They should not be included in the set to bin,
            or the pixels should be optimally weighted.
            See Cappellari & Copin (2003, Sec.2.1) and README file.""")
    if np.min(signal/noise) > targetSN:
        return 0, 0, 0, 0, 0, 0, 0, 0
        #raise ValueError('All pixels have enough S/N and binning is not needed')

    t1 = clock()
    if not quiet:
        print('Bin-accretion...')
    classe, pixelsize = _accretion(
        x, y, signal, noise, targetSN, pixelsize, quiet, sn_func)
    if not quiet:
        print(np.max(classe), ' initial bins.')
        print('Reassign bad bins...')
    xnode, ynode = _reassign_bad_bins(classe, x, y)
    if not quiet:
        print(xnode.size, ' good bins.')
    t2 = clock()
    if cvt:
        if not quiet:
            print('Modified Lloyd algorithm...')
        xnode, ynode, scale, it = _cvt_equal_mass(
            x, y, signal, noise, xnode, ynode, pixelsize, quiet, sn_func, wvt)
        if not quiet:
            print(it - 1, ' iterations.')
    else:
        scale = np.ones_like(xnode)
    classe, xBar, yBar, sn, area = _compute_useful_bin_quantities(
        x, y, signal, noise, xnode, ynode, scale, sn_func)
    single = area == 1
    t3 = clock()
    if not quiet:
        print('Unbinned pixels: ', np.sum(single), ' / ', x.size)
        print('Fractional S/N scatter (%):', np.std(sn[~single] - targetSN, ddof=1)/targetSN*100)
        print('Elapsed time accretion: %.2f seconds' % (t2 - t1))
        print('Elapsed time optimization: %.2f seconds' % (t3 - t2))

    if plot:
        plt.clf()
        plt.subplot(211)
        rnd = np.argsort(np.random.random(xnode.size))  # Randomize bin colors
        _display_pixels(x, y, rnd[classe], pixelsize)
        plt.plot(xnode, ynode, '+w', scalex=False, scaley=False) # do not rescale after imshow()
        plt.xlabel('R (arcsec)')
        plt.ylabel('R (arcsec)')
        plt.title('Map of Voronoi bins')

        plt.subplot(212)
        rad = np.sqrt(xBar**2 + yBar**2)  # Use centroids, NOT generators
        plt.plot(np.sqrt(x**2 + y**2), signal/noise, ',k')
        if np.any(single):
            plt.plot(rad[single], sn[single], 'xb', label='Not binned')
        plt.plot(rad[~single], sn[~single], 'or', label='Voronoi bins')
        plt.xlabel('R (arcsec)')
        plt.ylabel('Bin S/N')
        plt.axis([np.min(rad), np.max(rad), 0, np.max(sn)*1.05])  # x0, x1, y0, y1
        plt.axhline(targetSN)
        plt.legend()

    return classe, xnode, ynode, xBar, yBar, sn, area, scale

#----------------------------------------------------------------------------



def determine_fiber(arcs_totes, size_a):
    if arcs_totes > 32.5:
        n_fibers=127
        dia = 32.5/size_a
    else:
        if arcs_totes > 27.5:
            n_fibers=91
            dia = 27.5/size_a
        else:
            if arcs_totes > 22.5:
                n_fibers=61
                dia=22.5/size_a
            else:
                if arcs_totes > 17.5:
                    n_fibers=37
                    dia=17.5/size_a
                else:
                    n_fibers=19
                    dia = 12.5/size_a
    #print('arcs_totes going in', arcs_totes)
    #print('size of a spaxel in arcs', size_a)
    #print(n_fibers,'number of fibers', 'overall diameter', dia)
    
    return n_fibers, dia

def map_to_coords(data, size):
    x_list=[]
    y_list=[]
    
    
    for i in range(size):
        for j in range(size):
            x_list.append(i)
            y_list.append(j)


    return x_list, y_list

def mask_map(size, dia, x_list, y_list, size_a):
    '''Now, make a mask'''
    '''
    Each pixel is 0.5"
    We know the map is already centered
    Try sending the whole thing into kinemetry
    XBIN = x_cen-x; x_cen is size/2
    '''

    '''
    Argh really not looking forward to possibly constructing a hexagonal mask myself
    '''

    import math
    from matplotlib import path
    diff_x=((dia-0.5))/math.tan(math.radians(60))-(size/2-(dia-0.5)/2)
    diff_here=((dia-0.5/size_a)/2)/math.tan(math.radians(60))
    radius=(dia-0.5/size_a)/2


    poly_verts = path.Path([ (size/2,size/2-(dia-0.5)/2), (size/2+(dia-0.5)/2,((dia-0.5))/math.tan(math.radians(60))),
                            (size/2+(dia-0.5)/2,size/2+(dia-0.5)/2-diff_x), (size/2, size/2+(dia-0.5)/2),
                            (size/2-(dia-0.5)/2,size/2+(dia-0.5)/2-diff_x),
                           (size/2-(dia-0.5)/2,((dia-0.5))/math.tan(math.radians(60)))])

    poly_verts = path.Path([ (size/2,size/2-radius), (size/2+radius,size/2-radius+diff_here),
                            (size/2+radius,size/2+radius-diff_here), (size/2, size/2+radius),
                            (size/2-radius,size/2+radius-diff_here),
                           (size/2-radius,size/2-radius+diff_here)])




    
    xy=np.column_stack((x_list,y_list))
    #xy=np.column_stack((size, size))
    inside=poly_verts.contains_points(xy)

    return inside


def xys(data, data_noise, inside, size):
    x_list=[]
    y_list=[]
    signal_list=[]
    noise_list=[]
    k=0


    for i in range(size):
        for j in range(size):
            if inside[k] ==True:
                #print(type(data_noise[i,j]))
                if isinstance(data_noise[i,j], float):
                    x_list.append(i)
                    y_list.append(j)
                    signal_list.append(data[i,j])
                    noise_list.append(data_noise[i,j])
            k+=1
    return x_list, y_list, signal_list, noise_list



def fit_2_gaussian(x_1,y_1,x_2,y_2, data):
    # Create x and y indices
    data=np.flipud(data)
    x = np.linspace(0, 299, 300)
    y = np.linspace(0, 299, 300)
    x, y = np.meshgrid(x, y)
    

    # add some noise to the data and try to fit the data generated beforehand
    initial_guess = (20,x_1,y_1,7,7,0,10,20,x_2,y_2,7,7,0)#these are good guesses for the units of surface brightness
    data=data.ravel()
    
   
    
    try:
        popt, pcov = opt.curve_fit(twoD_two_Gaussian, (x, y), data, p0=initial_guess)
        fit='yes'
    except RuntimeError:
        popt=[0,0,0,0,0,0,0,0,0,0,0,0,0]
        fit='no'
        #flag for if the fit failed
 
    
    return popt[1], popt[2], popt[8], popt[9], popt[0], popt[7], np.sqrt(popt[3]**2+popt[4]**2), np.sqrt(popt[10]**2+popt[11]**2), fit 

def twoD_two_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset,
                     amplitude_2, xo_2, yo_2, sigma_x_2, sigma_y_2, theta_2):
    (x, y) = xdata_tuple 
    xo = float(xo)
    yo = float(yo)   
    xo_2 = float(xo_2)
    yo_2 = float(yo_2)  
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    a_2 = (np.cos(theta_2)**2)/(2*sigma_x_2**2) + (np.sin(theta_2)**2)/(2*sigma_y_2**2)
    b_2 = -(np.sin(2*theta_2))/(4*sigma_x_2**2) + (np.sin(2*theta_2))/(4*sigma_y_2**2)
    c_2 = (np.sin(theta_2)**2)/(2*sigma_x_2**2) + (np.cos(theta_2)**2)/(2*sigma_y_2**2)
    
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))+ amplitude_2*np.exp( - (a_2*((x-xo_2)**2) + 2*b_2*(x-xo_2)*(y-yo_2) 
                            + c_2*((y-yo_2)**2)))
    
    return g.ravel()
def determine_coords(img):
    
    '''Apply a 10x10 kernal to the image to filter out noise (its basically a low pass filter)
    to smooth things out'''
    kernel = np.ones((10,10))
    

    lp = ndimage.convolve(img, kernel)#was result
    
    
    '''Okay here is where you can filter out the really low stuff
    (anything below 20% of the max is eliminated so that we can detect the peak pixel)'''
    
    max_value=(lp.max())
    low = np.where(lp < 0.2*max_value)
    
   
    lp[low] = 0
    
    
    
    
    
    
    '''Detects brightest peaks in an image (can detect more than 1)'''
    indices = np.where(detect_peaks(lp) == 1)#was hp_lp_sharp
    
    number_of_sols=len(indices[0])
    
    
    try:
        return indices[0][0],indices[0][-1],indices[1][0],indices[1][-1], lp, number_of_sols
    except IndexError:
        #if there are no peaks this means the simulation was somehow cut off and
        #starting with returning zeros will flag the entire procedure to continue
        #without further ado
        return 0,0,0,0,lp,number_of_sols


def detect_peaks(image):
    """                                                                                                                                                                                                         
    Takes an image and detect the peaks using the local maximum filter.                                                                                                                                         
    Returns a boolean mask of the peaks (i.e. 1 when                                                                                                                                                            
    the pixel's value is the neighborhood maximum, 0 otherwise)                                                                                                                                                 
    """

    # define an 8-connected neighborhood                                                                                                                                                                        
    struct = generate_binary_structure(2,1)

    neighborhood = iterate_structure(struct, 10).astype(bool)

    #apply the local maximum filter; all pixel of maximal value                                                                                                                                                 
    #in their neighborhood are set to 1                                                                                                                                                                         
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are                                                                                                                                                         
    #looking for, but also the background.                                                                                                                                                                      
    #In order to isolate the peaks we must remove the background from the mask.                                                                                                                                 


    #we create the mask of the background                                                                                                                                                                       
    background = (image==0)

    #a little technicality: we must erode the background in order to                                                                                                                                            
    #successfully subtract it form local_max, otherwise a line will                                                                                                                                             
    #appear along the background border (artifact of the local maximum filter)                                                                                                                                  
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks,                                                                                                                                                           
    #by removing the background from the local_max mask (xor operation)                                                                                                                                         
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

def determine_brighter(img, x, y, x2, y2, pix, redshift):
    kpc_arcmin=cosmo.kpc_proper_per_arcmin(redshift)#insert the redshift to get the kpc/arcmin scaling

    
    ap_size=(3*(kpc_arcmin.value/60))/pix ###3 arcsec diameter * (kpc/arcsec)   / (kpc/pix) -->
    #This is now in units of pixels
    
    '''step 1: define the circular aperture'''
    positions = [(x, y), (x2, y2)]
    from photutils import CircularAperture,aperture_photometry
    apertures = CircularAperture(positions, ap_size)
    phot_table = aperture_photometry(img, apertures)
    total_light_1=phot_table['aperture_sum'][0]
    total_light_2=phot_table['aperture_sum'][1]

    
    
    masks = apertures.to_mask(method='center')
    mask = masks[0]

    image = mask.to_image(shape=((np.shape(img)[0], np.shape(img)[0])))
    return total_light_1, total_light_2


def clip_image(ins, pixelscale, redshift, xcen, ycen):
 
    kpc_arcmin=cosmo.kpc_proper_per_arcmin(redshift)#insert the redshift  
    #print(kpc_arcmin.value/60, 'kpc per arcsec')
    '''Divide the pixelscale (kpc/pix) by kpc/arcsec to get arcsec
    size of pixels'''
    size_a=pixelscale/(kpc_arcmin.value/60)
    #print('size a in image', size_a)
    num_pix_half=int(17/size_a)
    '''32.5" per diameter is the largest IFU'''
    #print('number of pixels across in image', num_pix_half*2)
    
    
    
    
    if xcen-num_pix_half < 0 or ycen-num_pix_half < 0 or xcen+num_pix_half > 300 or ycen+num_pix_half > 300:
        print('Outside of the box')
        clipped=0
        tag='no'
    
    else:
        #build clipped and centered image
        clipped=(ins[xcen-num_pix_half:xcen+num_pix_half,ycen-num_pix_half:ycen+num_pix_half])
        tag='yes'

 
    
    return clipped, size_a, num_pix_half, tag, xcen, ycen


'''Now I have to convert the units of LAURAS sims into nanomaggies and AB mags (mags of DR7 and DR13)
'''
def nanomags(z, pixscale, camera_data, view, number, cimg):

    c = 299792.458*1000#to get into m/s



    pixelscale=pixscale
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70,Om0=0.3)
    d_A = cosmo.comoving_distance(z).value/(1+z)
    #here's a good review of all the different distances cosmology distances:
    #http://www.astro.ufl.edu/~guzman/ast7939/projects/project01.html

    #Convert from specific intensity units (W/m/m^2/sr) from SUNRISE to Janskies (W/Hz/m^2): 
    Janskies=np.array(10**(26)*camera_data*(pixelscale/(1000*d_A))**2*np.pi*((6185.2*10**(-10))**2/c), dtype='>f4')
    #this 1.35e-6 comes from the arcsin(R_sky/Distance to object)
    #the answer needs to be in radians

    #J=10^-26 W/m^2/Hz, so units of flux density
    #reference site: http://www.cv.nrao.edu/course/astr534/Brightness.html
    #We need to go from a spectral brightness (I_nu) which is in m units
    #To a flux density (S_nu) which is in units of Janskies (W/m^2/Hz)

    
    Janskies_bright = Janskies*100**(1/5)

    nanomaggy=Janskies_bright/(3.631*10**(-6))
    '''Now convert into the correct background:)'''

    #nanomaggies and stuff: (Finkbeiner et al. 2004)

    #first, convert to counts (dn) using the average value of cimg from the SDSS frame images
    #dn=img/cimg+simg
    counts=(nanomaggy)/cimg#+simg
    return counts

def convolve_rebin_image(number, z, pixscale, view, counts, size, sigma_im, sky_mean, sky_std, gain, darkvar, simg):
    #  prep=convolve_rebin_image(myr,e,z,pixelscale,view)
    pixelscale = pixscale
    kpc_arcmin=cosmo.kpc_proper_per_arcmin(z)

    sigma=sigma_im/2.355#this is MaNGA spatial scale (different than SDSS spatial scale)
    kernel_sigma_pix = (sigma*(kpc_arcmin.value/60))/pixscale
    gaussian_2D_kernel = Gaussian2DKernel(kernel_sigma_pix)
    result = (convolve(counts,gaussian_2D_kernel))
    
    factor = (pixelscale/(kpc_arcmin.value/60))/0.5
    
    rebin = scipy.ndimage.zoom(result, factor, order=0)
    
    '''Now, introduce noise'''
    sky_resids_mine_counts=np.random.normal(sky_mean, sky_std, np.shape(rebin))#0.77, 3.73, np.shape(rebin))#0.331132,5.63218,np.shape(nanomaggy))    
    #gain=3.85#4.735                                                                                                                        
    #darkvar=3.875#1.1966                                                                                                                   
    #simg=63.711#121.19590411 #average background value (counts) pre background subraction (used to calculate poisson error)                
    noisy_counts = sky_resids_mine_counts+rebin
    counts_bg=rebin+simg                                                                                                          
    sigma_counts=np.sqrt(counts_bg/gain+darkvar)                                                                                           
    plt.clf()
    fig=plt.figure()
    ax0 = fig.add_subplot(321)
    im0 = ax0.imshow(result, cmap='afmhot_r')
    plt.colorbar(im0)
    ax0.set_title('Convolved')

    ax1 = fig.add_subplot(322)
    im1 = ax1.imshow(rebin, cmap='afmhot_r')
    plt.colorbar(im1)
    ax1.set_title('Rebinned')
    
    ax2 = fig.add_subplot(323)
    im2 = ax2.imshow(noisy_counts, cmap='afmhot_r')
    plt.colorbar(im2)
    ax2.set_title('Noise Added')

    ax3 = fig.add_subplot(324)
    im3 = ax3.imshow(sigma_counts, cmap='afmhot_r')
    plt.colorbar(im3)
    ax3.set_title('Error Image')
    
    ax4 = fig.add_subplot(325)
    im4 = ax4.imshow(noisy_counts/sigma_counts, cmap='jet')
    plt.colorbar(im4)
    ax4.set_title('g-band S/N per spaxel')
    plt.savefig('figs/g_band_image.png')

    
    return noisy_counts, noisy_counts/sigma_counts#S, S/N
    



def get_effective_radius(view, myr, run, image, pixelscale, z):

    x_cen=0
    y_cen=0


    b=determine_coords(image)
    #this determines the locations of the galaxies                                                                                                                                                          

    if b[0]==0:
        #if the first index from determine_coords is zero, the galaxy is out of the image                                                                                                                   
        #and we can continue and skip this particular image (sometimes other viewpoints of                                                                                                                  
        #the same snapshot are in the frame so don't get rid of an entire snapshot)                                                                                                                         
        return 0, 0, 1, 0, 0, 0, 0







    low_pass=b[4]
    num_sol=b[5]

    '''Now, fit a couple 2D gaussians if there are 2 brightest pixels, otherwise                                                                                                                            
    fit only one 2D gaussian. The output of fit_2_gaussians will be the positions of these                                                                                                                  
    maxima'''

    if num_sol==1:
        #this is if there's only really one solution because the bulges are too close together                                                                                                              
        #fit a 2D gaussian to center the previous guess of peak pixels using the entire surface                                                                                                             
        #brightness profile                                                                                                                                                                                 
        c=fit_2_gaussian(b[1],np.shape(image)[0]-b[0],b[1],np.shape(image)[0]-b[0],low_pass)
        if c[8]=='no':
             c=fit_2_gaussian(b[2],np.shape(image)[0]-b[0],b[3],np.shape(image)[0]-b[1],low_pass)

    else:
        c=fit_2_gaussian(b[2],np.shape(image)[0]-b[0],b[3],np.shape(image)[0]-b[1],low_pass)



    if c[4] > c[5]:
        '''this means point 1 is brighter'''
        in_x = c[1]
        in_y = c[0]
        in_2_x = c[3]
        in_2_y = c[2]


    if c[5] > c[4]:
        '''point 2 is the brighter source'''
        in_x = c[3]
        in_y = c[2]
        in_2_x = c[1]
        in_2_y = c[0]




    '''Now place a aperture over each center and figure out which is brighter overall'''
    try:
        c_final=determine_brighter(image,  in_y,np.shape(image)[0]-in_x,  in_2_y, np.shape(image)[0]-in_2_x, pixelscale, z)
    except UnboundLocalError:
        return 0, 0, 1, 0, 0, 0, 0
    #if c_final[2]=='yes':                                                                                                                                                                                  
    #    return 0, 0, 1, 0                                                                                                                                                                                  

    if c_final[0] > c_final[1]:
        #Clip the image in the SDSS imaging size around                                                                                                                                                     
        d=clip_image(image, pixelscale, z, int(np.shape(image)[0]-in_x), int(in_y))
        #this means the first aperture (center) is indeed brighter                                                                                                                                          
    else:
        d=clip_image(image, pixelscale, z, int(np.shape(image)[0]-in_2_x), int(in_2_y))

    if d[3]=='no':
        #this means the cutout is outside the image --> bad                                                                                                                                                 
        return 0, 0, 1, 0, 0, 0, 0

    x_cen=d[4]
    y_cen=d[5]

    if x_cen==0 or y_cen==0:
        flag=1
    else:
        flag=0

    e=nanomags(z, pixelscale, d[0],view, myr, 0.005005)

    #Size is determined by what?                                                                                                                                                                            
    #arc_size of                                                                                                                                                                                            
    #return clipped, size_a, num_pix_half, tag, xcen, ycen                                                                                                                                 v                 
    kpc_arcmin=cosmo.kpc_proper_per_arcmin(z)
    size_a = pixelscale/(kpc_arcmin.value/60)#now we're in "/pix                                                                                                                                            
    pix_size = d[2]
    size = int(pix_size*size_a) #now we're in units of " of one side of the thing                                                                                                                           

    prep=convolve_rebin_image(myr,z,pixelscale,view, e, size, 1.43, 0.33, 5.63, 4.735, 1.1966, 121.1959)
    #def convolve_rebin_image(number, z, pixscale, view, counts, size):                                                                                                                                     
    '''The second extension here is the S_N in r-band'''
    plt.clf()


    masked_S_N = ma.masked_where(prep[0] < 1, prep[1])
    g_band_signal = ma.masked_where(prep[0] < 1, prep[0])
    masked_S_N = ma.masked_where(np.abs(masked_S_N) < 1, masked_S_N)




    '''Now use statmorph to get the half light radius'''
    import photutils
    import scipy.ndimage as ndi
    threshold = photutils.detect_threshold(g_band_signal, error=g_band_signal/masked_S_N, snr=10)#, snr=1.5)
    npixels = 5  # minimum number of connected pixels
    segm = photutils.detect_sources(g_band_signal, threshold, npixels)

    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label

    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float > 0.5

    source_morphs = statmorph.source_morphology(g_band_signal, segmap, weightmap=masked_S_N)#, gain=gain, psf=psf)

    try:
        morph = source_morphs[0]
 
    except IndexError:
        threshold = photutils.detect_threshold(g_band_signal, error=g_band_signal/masked_S_N, snr=100)#, snr=1.5)                                                                                           
        npixels = 5  # minimum number of connected pixels                                                                                                                                                  
        segm = photutils.detect_sources(g_band_signal, threshold, npixels)

        label = np.argmax(segm.areas) + 1
        segmap = segm.data == label

        segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
        segmap = segmap_float > 0.5

        source_morphs = statmorph.source_morphology(g_band_signal, segmap, weightmap=masked_S_N)#, gain=gain, psf=psf)            
        try:
            morph = source_morphs[0]
        except IndexError:
            threshold = photutils.detect_threshold(g_band_signal, error=g_band_signal/masked_S_N, snr=1)
                                                                                                                             
            npixels = 5  # minimum number of connected pixels                                                               
                                                                                                                                
            segm = photutils.detect_sources(g_band_signal, threshold, npixels)

            label = np.argmax(segm.areas) + 1
            segmap = segm.data == label

            segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
            segmap = segmap_float > 0.5

            source_morphs = statmorph.source_morphology(g_band_signal, segmap, weightmap=masked_S_N)#, gain=gain, psf=psf)   
            try:
                morph = source_morphs[0]
            except IndexError:
                segmap = np.ones((np.shape(g_band_signal)[0], np.shape(g_band_signal)[1]))
                source_morphs = statmorph.source_morphology(g_band_signal, segmap, weightmap=masked_S_N)
                morph = source_morphs[0]
    return x_cen, y_cen, flag, d[2], g_band_signal, masked_S_N, morph.rhalf_ellip, morph.sersic_ellip, morph.sersic_theta#g_band_signal/masked_S_N      

def get_center(view, myr, run, image, pixelscale, z):
    
    x_cen=0
    y_cen=0
    
    
    b=determine_coords(image)
    #this determines the locations of the galaxies

    if b[0]==0:
        #if the first index from determine_coords is zero, the galaxy is out of the image
        #and we can continue and skip this particular image (sometimes other viewpoints of
        #the same snapshot are in the frame so don't get rid of an entire snapshot)
        return 0, 0, 1, 0







    low_pass=b[4]
    num_sol=b[5]

    '''Now, fit a couple 2D gaussians if there are 2 brightest pixels, otherwise
    fit only one 2D gaussian. The output of fit_2_gaussians will be the positions of these
    maxima'''

    if num_sol==1:
        #this is if there's only really one solution because the bulges are too close together
        #fit a 2D gaussian to center the previous guess of peak pixels using the entire surface
        #brightness profile
        c=fit_2_gaussian(b[1],np.shape(image)[0]-b[0],b[1],np.shape(image)[0]-b[0],low_pass)
        if c[8]=='no':
             c=fit_2_gaussian(b[2],np.shape(image)[0]-b[0],b[3],np.shape(image)[0]-b[1],low_pass)

    else:
        c=fit_2_gaussian(b[2],np.shape(image)[0]-b[0],b[3],np.shape(image)[0]-b[1],low_pass)



    if c[4] > c[5]:
        '''this means point 1 is brighter'''
        in_x = c[1]
        in_y = c[0]
        in_2_x = c[3]
        in_2_y = c[2]


    if c[5] > c[4]:
        '''point 2 is the brighter source'''
        in_x = c[3]
        in_y = c[2]
        in_2_x = c[1]
        in_2_y = c[0]




    '''Now place a aperture over each center and figure out which is brighter overall'''
    try:
        
        c_final=determine_brighter(image,  in_y,np.shape(image)[0]-in_x,  in_2_y, np.shape(image)[0]-in_2_x, pixelscale, z)
    except UnboundLocalError:
        return 0, 0, 1, 0
    #if c_final[2]=='yes':
    #    return 0, 0, 1, 0

    if c_final[0] > c_final[1]:
        #Clip the image in the SDSS imaging size around 
        d=clip_image(image, pixelscale, z, int(np.shape(image)[0]-in_x), int(in_y))
        #this means the first aperture (center) is indeed brighter
    else:
        d=clip_image(image, pixelscale, z, int(np.shape(image)[0]-in_2_x), int(in_2_y))

    if d[3]=='no':
        #this means the cutout is outside the image --> bad
        return 0, 0, 1, 0

    x_cen=d[4]
    y_cen=d[5]
    
    if x_cen==0 or y_cen==0:
        flag=1
    else:
        flag=0

    e=nanomags(z, pixelscale, d[0],view, myr, 0.003813)
        
    #Size is determined by what?                                                                                                       
    #arc_size of                                                                                                                       
    #return clipped, size_a, num_pix_half, tag, xcen, ycen
    kpc_arcmin=cosmo.kpc_proper_per_arcmin(z)
    size_a = pixelscale/(kpc_arcmin.value/60)#now we're in "/pix                                                                       
    pix_size = d[2]
    size = int(pix_size*size_a) #now we're in units of " of one side of the thing 

    prep=convolve_rebin_image(myr,z,pixelscale,view, e, size, 1.61, 0.77, 3.73, 3.85, 3.875, 63.711)
    #def convolve_rebin_image(number, z, pixscale, view, counts, size):
    '''The second extension here is the S_N in r-band'''
    plt.clf()
    
    fig=plt.figure()
    ax0=fig.add_subplot(211)
    im0=ax0.imshow(prep[0], norm=matplotlib.colors.LogNorm(), cmap='afmhot_r')
    ax0.set_title('Counts')
    plt.colorbar(im0)
    
    masked_S_N = ma.masked_where(prep[0] < 1, prep[1])
    g_band_signal = ma.masked_where(prep[0] < 1, prep[0])
    g_band_signal = ma.masked_where(np.abs(masked_S_N) < 1, g_band_signal)
    masked_S_N = ma.masked_where(np.abs(masked_S_N) < 1, masked_S_N)
    
    

    ax1=fig.add_subplot(212)
    im1=ax1.imshow(masked_S_N, norm=matplotlib.colors.LogNorm(), cmap='afmhot_r')
    ax1.set_title('S/N')
    plt.colorbar(im1)
    plt.savefig('figs/S_N_g_band_size_check.png')
    
    return x_cen, y_cen, flag, d[2], g_band_signal, masked_S_N#g_band_signal/masked_S_N



def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def redshift_to_Newtonian_velocity(v, redshift, ivar=False):
    if ivar:
        return v*numpy.square(1.0+redshift)
    return (v-astropy.constants.c.to('km/s').value*redshift)/(1.0+redshift)


c = con.c.to(un.km/un.s).value



#Load the MaNGA wavelengths array shape (4563, )
manga_wave = np.load('manga_wavelengths_AA.npy')

run='fg3_m12'

print('beginning to select snapshot')

if run=='fg3_m12':
    myr_list=[170,180,185, 190, 195, 205, 210, 220, 225, 230, 240, 250, 260,265,275,285, 295,305,311, 315,320,    5,10,20,30,40,60,80,100,120,140,160]
    #myr_list=[305,311,315,320]
    #myr_list=[195]
    #myr_list=[225,250,260,265,315]
    myr_list=[5,10,20,30,40,60,80,100,120,275]
    myr_list=[250, 295, 315]
    myr_run = [205]
if run=='fg3_m12_agnx0':
    myr_list=[195,205,210,220]
myr = 205

z=0.03

prefix='mcrx_205_agnx0.fits'
prefix_im='/Users/beckynevin/CfA_Code/Kinematics_SUNRISE/images/q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_205.fits'

mcrx=pyfits.open(str(prefix))
im=pyfits.open(str(prefix_im))
size=183
factor=2


views=[1,2,3,4,5,6]
views = [1]
for i in range(len(views)):
        view = views[i]
        filename = 'cubes_ppxf/stellar_kinematics_SCATTER_emline_'+str(run)+'_'+str(myr)+'_'+str(view)+'.fits'
        try:
            testfile = pyfits.open(filename)
            #continue
        except FileNotFoundError:
            nothing = 1

        image = im['CAMERA'+str(view)+'-BROADBAND'].data
        pixelscale = im['CAMERA'+str(view)+'-BROADBAND'].header['CD1_1']

        try:
            broadband = image[64]#[65]#[65]
            broadband_r = image[65]
        except IndexError:
            #its these:
            broadband = image[1]#[2]
            broadband_r = image[2]

        '''Step 1 is to use the imaging to pull a g-band image and its error image'''
        '''The product of this is a S/N image that we will use to make sure the average g-band S/N for the spectrum scales'''
    
        mid_r = get_effective_radius(view,myr,run,broadband_r,pixelscale,z)

        half_light_radius = mid_r[6]

        mid=get_center(view,myr,run, broadband, pixelscale, z)

    
        mid_x=mid[0]
        mid_y=mid[1]
        if mid[2]==1:
            '''This means that its flagged'''
            continue


        '''Now I need to convolve, rebin, and add noise to the spectra'''


#What is nonscatter vs scatter? --> 18+view is NONSCATTER
        try:
            CAMERA0=mcrx[25+view]#25
        except IndexError:
            continue
        Ang=np.array([x[0]*10**(10) for x in mcrx[5].data])
        #so I think this is in air wavelengths, we need to convert to vacuum
        s = 10**4/Ang
        n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.000159740894897 / (38.92568793293 - s**2)
        Ang_vac = Ang*n

    
    
   
    

        pixelscale=CAMERA0.header['CD1_1']
        kpc_arcmin=cosmo.kpc_proper_per_arcmin(0.03)

    
        #clip to same size as your g-band image:
        try:
            window=CAMERA0.data[:,int(mid_x-mid[3]):int(mid_x+mid[3]),int(mid_y-mid[3]):int(mid_y+mid[3])]
        except TypeError:
            continue
    
        Ang = Ang_vac
    #first, interpolate to manga data
        from scipy.interpolate import griddata


    #first take care of the spatial direction by convolving and rebinning

        out = window



        hdul = pyfits.HDUList()
        hdul.append(pyfits.PrimaryHDU())

        hdul.append(pyfits.ImageHDU(data=np.sum(out,axis=0)))
        hdul.writeto('cubes_ppxf/image_SCATTER_emline_'+str(run)+'_'+str(myr)+'_'+str(view)+'.fits', clobber='True')



    
    ##the kernel needs to be the same dimensions, just constant across all wavelengths
        sigma = 2.5/2.355
        kernel_sigma_pix = (sigma*(kpc_arcmin.value/60))/pixelscale
                                                                                                                                  
        gaussian_2D_kernel = Gaussian2DKernel(kernel_sigma_pix)
    #do i need to apply this at every wavelength?
    #maybe I can use replicate to make a 3D kernel instead lol
        gaussian_3D_kernel = np.repeat(gaussian_2D_kernel.array[np.newaxis,:,:], np.shape(out)[0], axis=0)
        gaussian_3D_kernel = gaussian_3D_kernel#/np.sum(gaussian_3D_kernel)
        from scipy import signal

    #Try remaking that first dimension to have some shape
    
        delta = []
        for p in range(len(Ang)-1):
            delta.append(Ang[p+1]-Ang[p])

        Ang_step = np.median(delta)#this is the size of each step in Ang, which averages around 0.3-0.35AA
    #sigma = 72 km s is the median resolution; v/c = delta lam / lam
    #1.67AA
    #This is 3.5AA on average
        sigma_pix = 0.001#1/Ang_step#3.5/Ang_step#gives the sigma of the convolution in pixels
        from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

        '''What if we tried making sigma_pix a function of wavelength'''
        '''Actually the way to do this is to first interpolate onto a new wavelength scale and
        then to do the same width convolution'''
        '''So things need to be finer sampled (linearly) starting from the longest wavelengths'''
    
        '''Build a new array'''
        new_Ang=[]
        for p in range(10*len(Ang)):
            if p==0:
                starting_lambda=Ang[0]
            else:
                starting_lambda=starting_lambda+starting_lambda/2000
            new_Ang.append(starting_lambda)
            if starting_lambda > Ang[-1]:
                break


        '''Now remap onto this new weird angstrom scale before you convolve :/'''
        input_all_conv = np.reshape(out, (out.shape[0],out.shape[1]*out.shape[2]))
        input_all_conv = input_all_conv.T

        min_index=np.where(new_Ang==find_nearest(new_Ang,Ang[0]))[0][0]+1
        max_index=np.where(new_Ang==find_nearest(new_Ang,Ang[-1]))[0][0]-1
        new_Ang_wave=new_Ang[min_index:max_index-1]
        int_func = interpolate.interp1d(Ang, input_all_conv)
        int_flux = int_func(new_Ang_wave)
        int_flux = int_flux.T

        ready_conv = np.reshape(int_flux, (int_flux.shape[0], out.shape[1], out.shape[2]))
    

    
        gauss_1 = Gaussian1DKernel(sigma_pix)
        norm_1D = gauss_1.array/np.max(gauss_1.array)

        xs = np.linspace(0, len(gauss_1.array)-1, len(gauss_1.array))
    #So now we have a normalized kernel to multiply every dimension of gaussian_3D_kernel by
    #new = gaussian_3D_kernel*norm_1D[:,np.newaxis,np.newaxis]
        new_3D = np.zeros((len(gauss_1.array),np.shape(gaussian_3D_kernel)[1],np.shape(gaussian_3D_kernel)[1]))
        for o in range(np.shape(gaussian_3D_kernel)[1]):
            for h in range(np.shape(gaussian_3D_kernel)[1]):
                new_3D[:,o,h] = norm_1D*gaussian_3D_kernel[0,o,h]#instead of norm_1D used to be gauss_1.array

    

        conv = signal.fftconvolve(ready_conv, new_3D, mode='same')#kernel was gaussian_3D_kernel
    #conv = ready_conv
        zoom_factor = (pixelscale/(kpc_arcmin.value/60))/0.5
    
    #This rebins to a 0.5" spaxel scale
        rebin = scipy.ndimage.zoom(conv, (1, zoom_factor, zoom_factor), order=0)


    #Now rebin spectrall (do I need to convolve?)
        input_all = np.reshape(rebin, (rebin.shape[0],rebin.shape[1]*rebin.shape[2]))
        input_all = input_all.T

        min_index=np.where(manga_wave==find_nearest(manga_wave,new_Ang_wave[0]))[0][0]+1
        max_index=np.where(manga_wave==find_nearest(manga_wave,new_Ang_wave[-1]))[0][0]-1
        cut_manga_wave=manga_wave[min_index:max_index-1]
        int_func = interpolate.interp1d(new_Ang_wave, input_all)
        int_flux = int_func(cut_manga_wave)
        int_flux = int_flux.T

        back = np.reshape(int_flux, (int_flux.shape[0], rebin.shape[1], rebin.shape[2]))
    

    #now you need to add typical noise to the spectrum
        typ_noise = np.load('average_S_N.npy')[min_index:max_index-1]


    #Now, make a mask after minning and apply it to the analysis
    
        try:
            mask_rebin = np.ma.array(back, mask = np.tile(mid[5].mask, (np.shape(back)[0],1)))
        except np.ma.core.MaskError:
            continue

    
        '''Now do a hexagon mask'''
    #So this is the extent of the galaxy in the image prior to spatially rebinning
    #with the old spaxel/pixelscale

        size_a=pixelscale/(kpc_arcmin.value/60)
        arcs_totes = 1.5*2*half_light_radius*size_a#*0.5#size_a
    
        fiber = determine_fiber(arcs_totes, size_a)#0.5)#size_a)
        size = np.shape(back)[1]

        coords=map_to_coords(mask_rebin, size)
        inside = mask_map(size,fiber[1], coords[0], coords[1], size_a)#0.5)
    
        mask_rebin_hex = np.ma.array(mask_rebin, mask = ~np.tile(inside, (back.shape[0],1)))
        mask_rebin_hex_saved = mask_rebin_hex
        mask_rebin_saved = mask_rebin

        try:
            mask_r_band = np.ma.array(mid_r[4], mask = ~inside)
        except np.ma.core.MaskError:
            continue
#numpy.ma.core.MaskError: Mask and data not compatible: data size is 1, mask size is 4489.


        mask_g_band_2D = np.ma.array(mid[4], mask = ~inside)
        mask_g_band_noise_2D = np.ma.array(mid[4]/mid[5], mask = ~inside)



        coords_voronoi = xys(mask_g_band_2D, mask_g_band_noise_2D, inside, size)
    

        normalized_noise = typ_noise/np.median(typ_noise)
    #gaussian array
        mu, sigma = 0 , 1

        add_noise_array = np.zeros((len(cut_manga_wave),np.shape(mask_rebin_hex)[1],np.shape(mask_rebin_hex)[2]))
        s_noise_array = np.zeros((len(cut_manga_wave),np.shape(mask_rebin_hex)[1],np.shape(mask_rebin_hex)[2]))
        troubleshoot_noise = np.zeros((len(cut_manga_wave),np.shape(mask_rebin_hex)[1],np.shape(mask_rebin_hex)[2]))
        for u in range(np.shape(mask_rebin_hex)[1]):
            for l in range(np.shape(mask_rebin_hex)[2]):
                s=np.random.normal(mu,sigma,len(normalized_noise))
    #now multiply it by the mid[4] value
                troubleshoot_noise[:,u,l] = mid[5][u,l]
                s_noise_array[:,u,l] = ((mask_rebin_hex[:,u,l])/(mid[5][u,l]*normalized_noise))
                add_noise_array[:,u,l] = s*(mask_rebin_hex[:,u,l]/(mid[5][u,l]*normalized_noise))
            #normalized noise is average S/N with a max of 1
            #mid[5] is the g-band S/N at that spaxel
        add_noise_array = np.ma.array(add_noise_array, mask = np.tile(mid[5].mask, (add_noise_array.shape[0],1)))
        s_noise_array = np.ma.array(s_noise_array, mask = np.tile(mid[5].mask, (s_noise_array.shape[0],1)))

    

    

        output_vor, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
            np.array(coords_voronoi[0]), np.array(coords_voronoi[1]), np.array(coords_voronoi[2]), np.array(coords_voronoi[3]), 10, plot=0, quiet=1, sn_func=sn_func)

        if type(output_vor) == int:
            input_all = np.reshape(mask_rebin_hex_saved,(np.shape(mask_rebin_hex_saved)[0], np.shape(mask_rebin_hex_saved)[1]*np.shape(mask_rebin_hex_saved)[2]))
            input_all = input_all.T
            input_noise = np.reshape(s_noise_array,(np.shape(mask_rebin_hex_saved)[0], np.shape(mask_rebin_hex_saved)[1]*np.shape(mask_rebin_hex_saved)[2]))
            input_noise = input_noise.T
       
            '''then do a bunch of things'''
        else:
    

            '''Now its time to figure out how to deal with the Voronoi bins'''
            '''We need to remap the xs and ys back onto 2D maps'''
            print('og input to remapped shape', np.shape(mask_rebin_hex))
            remapped =remap(output_vor, np.array(coords_voronoi[0]), np.array(coords_voronoi[1]), xNode, yNode, mask_rebin_hex,add_noise_array, np.array(coords_voronoi[2]), s_noise_array)

        
    
        
            cmap_here='plasma'
            plt.clf()
            fig=plt.figure()
            ax0=fig.add_subplot(331)
            #this will be the spatial dimension
            im0 = ax0.imshow(abs(out[int(len(Ang)/2),:,:]), norm=matplotlib.colors.LogNorm(), cmap=cmap_here)
            plt.colorbar(im0)
            ax0.set_title('Simulated Galaxy')
            ax0.set_yticklabels([])
            ax0.set_xticklabels([])

            ax1 = fig.add_subplot(313)
            ax1.plot(Ang, np.apply_over_axes(np.mean,out, (1,2))[:,0,0]+0.5, label='Simulated Spectrum', lw=0.5)
            ax1.plot(new_Ang_wave, np.apply_over_axes(np.mean,conv, (1,2))[:,0,0], label='Convolved', lw=0.5)
            mu, sigma = 0,1
    
            ax1.plot(cut_manga_wave, np.apply_over_axes(np.mean,mask_rebin_hex+add_noise_array,(1,2))[:,0,0]-0.5, label='Convolved and Rebinned', lw=0.5)
            ax1.plot(cut_manga_wave, np.apply_over_axes(np.mean,add_noise_array,(1,2))[:,0,0], label='Noise', lw=0.5)

            plt.xlim([3500,4500])
            plt.xlabel('Wavelength [$\AA$]')
            plt.ylabel('Relative Flux')
            plt.legend(loc='upper center', ncol=4,framealpha=1, frameon=True)#, bbox_to_anchor=(1.2, 0.5))


            ax2 = fig.add_subplot(332)
            im2 = ax2.imshow(conv[int(len(new_Ang_wave)/2),:,:], norm=matplotlib.colors.LogNorm(), cmap=cmap_here)
            ax2.set_title('3D Convolved')
            plt.colorbar(im2)
        
    
            ax4 = fig.add_subplot(333)
            im4 = ax4.imshow(mask_rebin_saved[int(len(cut_manga_wave)/2),:,:], norm=matplotlib.colors.LogNorm(), cmap=cmap_here)
            plt.colorbar(im4)
            ax4.set_title('Rebinned and Masked by S/N')



            ax5 = fig.add_subplot(334)
            im5 = ax5.imshow(mask_rebin_hex_saved[int(len(cut_manga_wave)/2),:,:], cmap='plasma', norm=matplotlib.colors.LogNorm())
            plt.colorbar(im5)
            ax5.set_title('Hexagonal Footprint Masked')

            masked_noise = ma.masked_where(np.isnan(remapped[0][int(len(cut_manga_wave)/2),:,:]), remapped[1][int(len(cut_manga_wave)/2),:,:])

            masked_noise = ma.masked_where(masked_noise==0, masked_noise)
            ax6 = fig.add_subplot(335)
            im6 = ax6.imshow(masked_noise, cmap='RdBu', norm=matplotlib.colors.LogNorm())
            plt.colorbar(im6)
            ax6.set_title('Add Noise')
      
            vor_masked=ma.masked_where(remapped[4]==0, remapped[4])
            ax7 = fig.add_subplot(336)
            im7 = ax7.imshow(vor_masked, cmap='jet')
            plt.colorbar(im7)#, fraction=0.046, pad=0.04)
            ax7.set_title('Voronoi Bins')
            ax7.set_yticklabels([])
            ax7.set_xticklabels([])
            plt.tight_layout()
            plt.savefig('cubes_ppxf/description_remapped_check_SCATTER_emline_'+str(run)+'_'+str(myr)+'_'+str(view)+'.png', dpi=1000)
        



            '''find the spaxel of maximum brightness, this is the AGN light'''
        
            plt.clf()
            summed_specs = np.sum(out,axis=0)
            coords = np.argwhere(summed_specs.max() == summed_specs)
        #max_coord = np.where(summed_specs = np.max(summed_specs))
        #print('max coords', coords)
            plt.plot(Ang, out[:,coords[0][0],coords[0][1]], label='Simulated Spectrum', lw=0.5)

            plt.savefig('figs/brightest_spaxel_'+str(run)+'_'+str(myr)+'_'+str(view)+'.png', dpi=1000)





            input_all = np.array(remapped[6])
    


            input_noise = np.array(remapped[7])#was np.array()



        tpl = TemplateLibrary('MILESHC',
                        match_to_drp_resolution=True,#was False
                        velscale_ratio=1,  #1: this would be 0 if your fake spectra didn't have the same velocity scale as MaNGA
                        spectral_step=1e-4,
                        log=True,hardcopy=False)
                        #directory_path='.',
                        #processed_file='mileshc.fits',
                        #clobber=True)



# Instantiate the python PPXF object that actually does the fitting
        contbm = StellarContinuumModelBitMask()#EmissionLineModelBitMask()#StellarContinuumModelBitMask()#EmissionLineModelBitMask()#StellarContinuumModelBitMask()
        ppxf = PPXFFit(contbm)

        # Provide a guess redshift and guess velocity dispersion for PPXF.
        # My fake spectra aren't redshifted so I give it the smallest redshift
        # that doesn't throw an error - this was found through trial and error!
        # The dispersion I know because I already smoothed the spectra to have
        # the same dispersion as the minimum detectable by MaNGA (see above).
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

        method_save='ELPEXT'
        redshifted_template = [x for x in tpl['WAVE'].data]
        
        print('these were the shapes', np.shape(input_all))
        print('error', np.shape(input_noise))
        print('shape before voronoi rebinning', np.shape(s_noise_array))
        
        # Optionally search for a subtracted cube if it exists
        try:
            print('trying AGN SUB')
            AGN_sub_data = pyfits.open('subtracted/cube_'+str(run)+'_'+str(myr)+'_'+str(view)+'.fits')
            AGN_sub = AGN_sub_data[0].data
            #cube_fg3_m12_205_0.fits
            print('shape of AGN_sub', np.shape(AGN_sub))
            AGN = 'yes'
            #input_all = AGN_sub.reshape(np.shape(AGN_sub)[0], np.shape(AGN_sub)[1]*np.shape(AGN_sub)[2])
            #input_all = input_all.T
            #print('final shape', np.shape(input_all))
            #input_noise = 0.01*input_all
            #input_noise = s_noise_array
            #input_noise = np.reshape(s_noise_array,(np.shape(mask_rebin_hex_saved)[0], np.shape(mask_rebin_hex_saved)[1]*np.shape(mask_rebin_hex_saved)[2]))
            #input_noise = input_noise.T
            #print('shape after rebinning', np.shape(input_noise))
            print('input shape to remapped', np.shape(AGN_sub), np.shape(s_noise_array))
            remapped =remap(output_vor, np.array(coords_voronoi[0]), np.array(coords_voronoi[1]), xNode, yNode, AGN_sub,add_noise_array, np.array(coords_voronoi[2]), s_noise_array)


            input_all = np.array(remapped[6])

            print('input_all shape', np.shape(input_all))



            input_noise = np.array(remapped[7])#was np.array() 
            
        
        except FileNotFoundError:
            AGN = 'no'


        nspec = np.shape(input_all)[0]#.shape[0]                                                                                                                         \                                                                                                     

        input_redshift = 0.0000001#-0.000275                                                                                                                             \                                                                                                     

        guess_redshift = np.full(nspec, input_redshift, dtype=float)
        guess_dispersion = np.full(nspec, 100, dtype=float)

        sc_pixel_mask = SpectralPixelMask(emldb=EmissionLineDB('ELPEXT'))#waverange=[5100,6300])                                                                                                                                                                               


        
        model_wave, model_flux, model_mask, model_par  = ppxf.fit(tpl_wave=tpl['WAVE'].data.copy(), tpl_flux=tpl['FLUX'].data.copy(), obj_wave=cut_manga_wave, obj_flux=input_all,obj_ferr=input_noise,  guess_redshift=guess_redshift, guess_dispersion=guess_dispersion, iteration_mode='no_global_wrej',velscale_ratio=1, mask=sc_pixel_mask, degree=8,  moments=2, quiet=True)
        
    
    
        new_vel=PPXFFit.convert_velocity(model_par['KIN'][:,0], model_par['KINERR'][:,0])
        new_vel_kin=redshift_to_Newtonian_velocity(new_vel[0], input_redshift)

        if type(output_vor) == int:
            kins=np.reshape(new_vel_kin,(np.shape(mask_rebin_hex_saved)[1],np.shape(mask_rebin_hex_saved)[2]))
            kins_err=np.reshape(model_par['KINERR'][:,0],(np.shape(mask_rebin_hex_saved)[1],np.shape(mask_rebin_hex_saved)[2]))
            sigma=np.reshape(model_par['KIN'][:,1],(np.shape(mask_rebin_hex_saved)[1],np.shape(mask_rebin_hex_saved)[2]))
            sigma_err=np.reshape(model_par['KINERR'][:,1],(np.shape(mask_rebin_hex_saved)[1],np.shape(mask_rebin_hex_saved)[2]))

        else:


            '''Find a way to give the masked pixels the values of their starting ones'''
            redist = redistribute_voronoi(output_vor, np.array(coords_voronoi[0]), np.array(coords_voronoi[1]), xNode, yNode, mask_rebin_hex,add_noise_array, np.array(coords_voronoi[2]), new_vel_kin, new_vel[1], model_par['KIN'][:,1], model_par['KINERR'][:,1], remapped[8])#the last one is an array list of indices for things that share the same bin number, so things need to be reshapped according tho this


            kins=redist[0]#np.reshape(redist[0],(spectrum.shape[1],spectrum.shape[2]))
            kins_err=redist[1]#np.reshape(redist[1],(spectrum.shape[1],spectrum.shape[2]))
            sigma=redist[2]#np.reshape(redist[2],(spectrum.shape[1],spectrum.shape[2]))
            sigma_err=redist[3]#np.reshape(redist[3],(spectrum.shape[1],spectrum.shape[2]))

    #kins=ma.masked_where(abs(kins)>2000, kins)
    #kins_err=ma.masked_where(abs(kins)>2000, kins_err)
    


     

        
        hdul = pyfits.HDUList()
        hdul.append(pyfits.PrimaryHDU())
        hdr = hdul[0].header
        hdr['REFF'] = str(half_light_radius)
        hdr['ellip'] = str(mid_r[7])
        hdr['PA_img'] = str(mid_r[8])
        hdr['METHOD'] = method_save
        hdul.append(pyfits.ImageHDU(data=kins))
        hdul.append(pyfits.ImageHDU(data=kins_err))
        hdul.append(pyfits.ImageHDU(data=sigma))
        hdul.append(pyfits.ImageHDU(data=sigma_err))
        hdul.append(pyfits.ImageHDU(data=mask_r_band.filled()))
        if AGN =='yes':
            hdul.writeto('cubes_ppxf/stellar_kinematics_SCATTER_emline_AGN_sub_'+str(run)+'_'+str(myr)+'_'+str(view)+'.fits', clobber='True')

        else:
            hdul.writeto('cubes_ppxf/stellar_kinematics_SCATTER_emline_'+str(run)+'_'+str(myr)+'_'+str(view)+'.fits', clobber='True')

