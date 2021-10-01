import scipy.ndimage.morphology as morph
import scipy.ndimage as nd
from scipy.signal import savgol_coeffs
import numpy as np
from astropy.stats import mad_std
from astropy.convolution import convolve, Gaussian2DKernel
import scipy.stats as ss
from spectral_cube import SpectralCube
import warnings


import astropy.wcs as wcs
import astropy.units as u
from astropy.io import fits
from functools import reduce

np.seterr(divide='ignore', invalid='ignore')

mad_to_std_fac = 1.482602218505602



def nchan_thresh_mask(cube, thresh=5., nchan=2):
    """
    Make mask from a cube applying a threshold across some number of
    consecutive channels.

    Parameters:

    -----------

    cube : np.array

        A cube in the same units as threshold.

    Keywords:
    ---------

    thresh : float

        Threshold for inclusion in mask if.

     nchan : int

        Number of consecutive channels at thresh required for a detection. Default: 2

    """

    # TBD Error checking on types, dimensionality, etc.

    mask = np.greater_equal(cube, thresh,
                            where=(~np.isnan(cube)),
                            out=np.full(cube.shape, False, dtype=bool))

    kernel = np.ones(nchan, dtype=np.bool)
    kernel = kernel[:, np.newaxis, np.newaxis]

    mask = morph.binary_opening(mask, kernel)

    return(mask)


def reject_small_regions(mask, min_volume=0, min_area=0):
    """
    Remove small regions from a mask. Small can be defined in either
    volume or area.

    Parameters:

    -----------

    mask : np.array
        A mask.

    Keywords:
    ---------

    minvolume : int    
        Minimum volume in pixels. Default 0.

    minarea : int
        Minimum area in pixels. Default 0.

    """

    # TBD Error checking on types, dimensionality, etc.

    # Blob color the cube and loop over regions

    regions, regct = nd.label(mask)
    objslices = nd.find_objects(regions)

    for ii, thisslice in enumerate(objslices):
        subcube = regions[thisslice]

        if min_volume > 0:
            volume = np.sum(subcube == (ii+1))
            if volume < min_volume:
                mask[regions == (ii+1)] = False

        if min_area > 0:
            if mask.ndim == 3:
                area = np.sum(np.any(subcube == (ii+1), axis=0))
            if mask.ndim == 2:
                area = np.sum(np.any(subcube == (ii+1)))

            if area < min_area:
                mask[regions == (ii+1)] = False

    return(mask)


def grow_mask(mask, iters_xy=0, iters_v=0, constraint=None):
    """
    Grow a boolean mask via dilation in the spectral (v) dimension,
    spatial (xy) dimension, or into a constraint.

    Logic:

    (Case I) If iters_xy and/or iters_v suppplied:

    1. Dilate original mask by iters_xy

    2. Dilate original mask by iters_v

    3. Take the union of the two new masks.

    4. Apply the constraint.

    (Case II) Only a constraint is supplied:

    1. Include all regions of the constraint that include an element
    of the original mask.

    Parameters:

    -----------

    mask : np.array
        A mask.

    Keywords:
    ---------

    iters_xy : int    
        Number of iterations of expansion in spatial dimensions.

    iters_v : int    
        Number of iterations of expansion in spectral dimension.

    constraint : np.array that can be broadcast to mask
        Another mask to use as a constraint.

    """

    # TBD Error checking on types, dimensionality, etc.

    if iters_xy > 0:

        # Generate the structure to dilate by
        struct = morph.iterate_structure(
            morph.generate_binary_structure(2, 1), iters_xy)

        # Fill in a third dimension if needed
        if mask.ndim == 3:
            struct = struct[np.newaxis, :, :]

        if iters_v > 0:
            mask_xy = morph.binary_dilation(mask, struct)
        else:
            mask = morph.binary_dilation(mask, struct)

    if iters_v > 0:

        struct = np.ones(iters_v, dtype=np.bool)
        struct = struct[:, np.newaxis, np.newaxis]

        if iters_xy > 0:
            mask_v = morph.binary_dilation(mask, struct)
        else:
            mask = morph.binary_dilation(mask, struct)

    if iters_v > 0 and iters_xy > 0:
        mask = np.logical_or(mask_v, mask_xy)

    if (iters_v > 0 or iters_xy > 0) and (constraint is not None):
        mask = np.logical_and(mask, constraint)

    if (iters_v == 0 and iters_xy == 0) and (constraint is not None):

        # blob color regions in the constraint
        regions, regct = nd.label(constraint)

        # get a list of all region assignments that have a True value
        # in the original mask
        good_regions = np.unique(regions[mask])

        # create a new mask that includes only these good new regions
        mask = np.zeros_like(mask, dtype=np.bool)
        for hit in good_regions:
            mask[regions == hit] = True

    return(mask)


def cprops_mask(data, noise=None,
                hi_thresh=5, hi_nchan=2,
                lo_thresh=None, lo_nchan=None,
                min_pix=None, min_area=None,
                min_beams=None, ppbeam=None,
                grow_xy=None, grow_v=None,
                prior_hi=None,
                prior_lo=None,
                invert=False):
    """
    Standard CPROPS masking recipe.

    Parameters:

    -----------

    data : np.array

        Original data

    noise : np.array

        Estimate of the amplitude of the noise at every position in the data
        (or an array that will broadcast to that under division).

    Keywords:
    ---------

    hi_thresh : float
        Threshold for detection (in units of sigma).  Default: 5

    hi_nchan : int
        Number of consecutive channels needed for detection.  Default: 2

    lo_thresh : float
        Threshold for inclusion in mask if connected to a hi_thresh
        detection. Default: None

    lo_nchan : int
        Number of consecutive channels at lo_thresh required for a detection
        if connected to a hi_thresh detection. Default: None

    min_pix : int
        Number of pixels required for a detection.  Default: None

    min_area : int
        Minimum number of pixels required in area projection for a detection.
        Default: None

    min_beams : float
        Minimum number of beams for a pixel to be rejected.  Overrides min_pix.
        Default: None

    ppbeam : float
        Number of pixels per beam.  Required if min_beams is set

    grow_xy : int
        Number of iterations to grow the final mask in the xy plane.

    grow_v : int
        Number of iterations to grow the final mask in velocity 

    prior_hi : np. array
        Mask that will be applied to the high significance mask before
        any expansion.

    prior_lo : np. array
        Mask that will be applied to the low significance mask.

    invert : bool
        If True, invert the data before applying masking.

        Used for assessing the number of false positives given masking
        criteria. Default: False.

    """

    # TBD error checking, dimensions, types, etc.

    if noise is None:
        warnings.warn("No noise estimate supplied. Taking noise to be unity.")
        noise = 1.0

    # Recase the cube into a signal-to-noise cube
    signif = data / noise

    # If requested, invert the data
    if invert:
        signif *= -1

    # Create a the core mask

    hi_mask = nchan_thresh_mask(
        signif, thresh=hi_thresh, nchan=hi_nchan)

    # If requested, reject small regions from the mask
    if ((min_beams is not None)
        or (min_pix is not None)
            or (min_area is not None)):

        if min_pix is None:
            min_pix = 0

        if min_area is None:
            min_area = 0

        if min_beams is not None:
            assert ppbeam is not None
            min_pix = min_beams * ppbeam

        hi_mask = reject_small_regions(
            hi_mask, min_volume=min_pix, min_area=min_area)

    # If a prior is supplied for the high significane mask, apply it

    if prior_hi is not None:
        hi_mask *= prior_hi

    # If supplied, make a lower significance mask and expand into it

    if (lo_thresh is not None) and (lo_nchan is not None):
        lo_mask = nchan_thresh_mask(
            signif, thresh=lo_thresh, nchan=lo_nchan)

        if prior_lo is not None:
            lo_mask *= prior_lo

        # Now expand the original mask into the lower mask
        mask = grow_mask(hi_mask, constraint=lo_mask)
    else:
        mask = hi_mask
        
    # If requested, grow the mask in xy and v directions. Sequential
    # calls mean that the xy is applied then the v.

    if grow_xy is not None:
        mask = grow_mask(mask, iters_xy=grow_xy)

    if grow_v is not None:
        mask = grow_mask(mask, iters_v=grow_v)

    return(mask)


def join_masks(orig_mask_in, new_mask_in,
               order='bilinear', operation='or',
               outfile=None,
               thresh=0.5,
               ):
    """
    Reproject and combine a new mask 

    Parameters:

    -----------

    orig_mask_in : string or SpectralCube
        
        The original mask.

    new_mask_in : string or SpectralCube

        The new mask

    Keywords:
    ---------

    order : string
        Order of interpolation. Passed to spectral cube.

    operation : string
        method to combine the masks 'or' or 'and' 

    outfile : string
        Filename where the mask will be written. The mask is also
        returned.

    thresh : float
        Floating point value above which the mask is considered
        true. Relevant because of interpolation. Default 0.5 .

    """

    # TBD - check for two dimensional case

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Read the data
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    if type(orig_mask_in) is str:
        orig_mask = SpectralCube.read(orig_mask_in)
    elif type(orig_mask_in) is SpectralCube:
        orig_mask = orig_mask_in
    else:
        warnings.warn('Unrecognized input type for orig_mask_in')
        raise NotImplementedError

    # Enable large operations
    orig_mask.allow_huge_operations = True

    if type(new_mask_in) is str:
        new_mask = SpectralCube.read(new_mask_in)
    elif type(new_mask_in) is SpectralCube:
        new_mask = new_mask_in
    else:
        warnings.warn('Unrecognized input type for new_mask_in')
        raise NotImplementedError

    # Enable large operations
    new_mask.allow_huge_operations = True

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Reproject the new mask onto the original WCS
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    if order == 'fast_nearest_neighbor':

        warnings.warn('Joining masks with nearest neighbor coordinate lookup')
        # Grab WCS out of template mask and map to other mask
        x, y, _ = new_mask.wcs.wcs_world2pix(
            *(orig_mask.world[0, :, :][::-1]), 0)
        _, _, z = new_mask.wcs.wcs_world2pix(
            *(orig_mask.world[:, 0, 0][::-1]), 0)

        x = np.rint(x).astype(np.int)
        y = np.rint(y).astype(np.int)
        z = np.rint(z).astype(np.int)
        new_mask_data = np.array(
            new_mask.filled_data[:].value > thresh, dtype=np.bool)

        # Create new mask
        new_mask_vals = np.zeros(orig_mask.shape, dtype=np.bool)
        # Find all values that are in bounds
        inbounds = reduce((lambda A, B: np.logical_and(A, B)),
                          [(z[:, np.newaxis, np.newaxis] >= 0),
                           (z[:, np.newaxis, np.newaxis] < new_mask.shape[0]),
                           (y[np.newaxis, :, :] >= 0),
                           (y[np.newaxis, :, :] < new_mask.shape[1]),
                           (x[np.newaxis, :, :] >= 0),
                           (x[np.newaxis, :, :] < new_mask.shape[2])])
#        inbounds = np.logical_and.reduce(
        # Look 'em up in the new_mask
        new_mask_vals[inbounds] = new_mask_data[(z[:, np.newaxis, np.newaxis]*np.ones(inbounds.shape, dtype=np.int))[inbounds],
                                                (y[np.newaxis, :, :]*np.ones(inbounds.shape,
                                                                             dtype=np.int))[inbounds],
                                                (x[np.newaxis, :, :]*np.ones(inbounds.shape, dtype=np.int))[inbounds]]

    else:

        warnings.warn('Joining masks with reprojection')
        new_mask = new_mask.reproject(orig_mask.header, order=order)
        new_mask = new_mask.spectral_interpolate(orig_mask.spectral_axis)
        new_mask_vals = np.array(new_mask.filled_data[:].value > thresh,
                                 dtype=np.bool)

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Combine
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    if operation.strip().lower() == 'or':
        mask = np.logical_or(np.array(orig_mask.filled_data[:].value > thresh,
                                      dtype=np.bool),
                             new_mask_vals)
    elif operation.strip().lower() == 'and':
        mask = np.logical_and(np.array(orig_mask.filled_data[:].value > thresh,
                                       dtype=np.bool),
                              new_mask_vals)
    elif operation.strip().lower() == 'sum':
        mask = (orig_mask.filled_data[:].value) + new_mask_vals
    else:
        warnings.warn('Unrecognized operation. Not "and" or "or".')
        raise NotImplementedError

    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    # Write to disk, return output, etc.
    # &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    mask = SpectralCube(mask.astype(np.int), wcs=orig_mask.wcs,
                        header=orig_mask.header,
                        meta={'BUNIT': ' ', 'BTYPE': 'Mask'})

    # Write to disk, if desired
    if outfile is not None:
        hdu = fits.PrimaryHDU(np.array(mask.filled_data[:], dtype=np.uint8),
                              header=mask.header)
        hdu.writeto(outfile, overwrite=overwrite)

    return(mask)

