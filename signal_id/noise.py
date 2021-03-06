import logging
import scipy.ndimage.morphology as morph
import scipy.ndimage as nd
from scipy.signal import savgol_coeffs
import numpy as np
from astropy.stats import mad_std
from astropy.convolution import convolve, Gaussian2DKernel
import scipy.stats as ss

from spectral_cube import SpectralCube
import astropy.wcs as wcs
import astropy.units as u
# from pipelineVersion import version as pipeVer
from astropy.io import fits

import warnings

from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

np.seterr(divide='ignore', invalid='ignore')

mad_to_std_fac = 1.482602218505602

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def mad_zero_centered(data, mask=None):
    """
    Estimates the noise in a data set using the median absolute
    deviation. Assumes a normal distribution and that the data are
    centered on zero. Excludes any masked regions.
    
    Parameters:
    -----------
    
    data : np.array

        Array of data (floats)
    
    Keywords:
    ---------
    
    mask : np.bool

        Boolean array with True indicating where data can be used in
        the noise estimate. (i.e., True is noise). If none is supplied
        a trivial mask is estimated.

    """

    # TBD: Add input error checking

    # Select finite data and exclude not-a-number
    where_data_valid = np.logical_and(~np.isnan(data), np.isfinite(data))

    # Make a trivial mask (to simplify the code) if none has been supplied
    if mask is None:
        mask = np.isfinite(data)

    # Check that we have enough data to estimate the noise
    nData = mask.sum()
    if nData == 0:
        logger.info('No data in mask. Returning NaN.')
        return(np.nan)

    # Estimate the significance threshold likely associated with
    # false positives based on the size of the data.

    sig_false = ss.norm.isf(0.5 / nData)

    # Make a first estimate of the noise based on the negatives.

    data_lt_zero = np.logical_and(
        np.less(data, 0,
                where=where_data_valid,
                out=np.full(where_data_valid.shape,
                            False, dtype=bool)), mask)

    mad1 = mad_to_std_fac * np.abs(np.median(data[data_lt_zero]))

    # Make a second estimate now including the positives less than
    # the false-positive threshold.

    data_lt_mad1 = np.logical_and(
        np.less(data, (sig_false * mad1),
                where=where_data_valid,
                out=np.full(where_data_valid.shape,
                            False, dtype=bool)), mask)
    mad2 = mad_to_std_fac * np.abs(np.median(np.abs(data[data_lt_mad1])))

    # Return this second estimate

    return(mad2)


def noise_cube(datain, mask=None,
               nThresh=30, iterations=1,
               do_map=True, do_spec=True,
               box=None, spec_box=None,
               bandpass_smooth_window=None,
               bandpass_smooth_order=3,
               oversample_boundary=False,
               return_spectral_cube=False):
    """

    Makes an empirical estimate of the noise in a cube assuming that
    it is normally distributed about zero. Treats the spatial and
    spectral dimensions as separable.
    
    Parameters:
    -----------
    
    data : np.array
        Array of data (floats)
    
    Keywords:
    ---------
    
    mask : np.bool

        Boolean array with False indicating where data can be 
        used in the noise estimate. (i.e., True is signal)
    
    do_map : np.bool
    
        Estimate spatial variations in the noise. Default is True. If
        set to False, all locations in a plane have the same noise
        estimate.

    do_spec : np.bool
    
        Estimate spectral variations in the noise. Default is True. If
        set to False, all channels in a spectrum have the same noise
        estimate.

    box : int

        Spatial size of the box over which noise is calculated in
        pixels.  Default: no box, every pixel gets its own noise
        estimte.
    
    spec_box : int

        Spectral size of the box overwhich the noise is calculated.
        Default: no box, each channel gets its own noise estimate.
    
    nThresh : int
        Minimum number of data to be used in an individual noise estimate.
    
    iterations : int
        Number of times to iterate the noise solution to force Gaussian 
        statistics.  Default: no iterations.
    
    bandpass_smooth_window : int
        Number of channels used in bandpass smoothing kernel.  Defaults to 
        nChan / 4 where nChan number of channels.  Set to zero to suppress 
        smoothing. Uses Savitzky-Golay smoothing
        
    bandpass_smooth_order : int
        Polynomial order used in smoothing kernel.  Defaults to 3.
    
    """

    if type(datain) is SpectralCube:
        data = datain.filled_data[:].value
    elif type(datain) is np.ndarray:
        data = datain
    else:
        warning.warn("Input data must be ndarray or SpectralCube")
        raise ValueError
    
    # Create a mask that identifies voxels to be fitting the noise

    noisemask = np.isfinite(data)
    if mask is not None:
        noisemask[mask] = False

    # Default the spatial step size to individual pixels

    step = 1
    halfbox = step // 2

    # If the user has supplied a spatial box size, recast this into a
    # step size that critically samples the box and a halfbox size
    # used for convenience.

    if box is not None:
        step = np.floor(box/2.5).astype(np.int)
        halfbox = int(box // 2)

    # Include all pixels adjacent to the spatial
    # boundary of the data as set by NaNs
    boundary = np.all(np.isnan(data), axis=0)

    if oversample_boundary:
        struct = nd.generate_binary_structure(2, 1)
        struct = nd.iterate_structure(struct, halfbox)
        rind = np.logical_xor(nd.binary_dilation(boundary, struct),
                              boundary)
        extray, extrax = np.where(rind)
    else:
        extray, extrax = None, None

    # If the user has supplied a spectral box size, use this to
    # calculate a spectral step size.

    if spec_box is not None:
        spec_step = np.floor(spec_box / 2).astype(np.int)
        boxv = int(spec_box // 2)
    else:
        boxv = 0

    # Default the bandpass smoothing window

    if bandpass_smooth_window is None:
        bandpass_smooth_window = 2 * (data.shape[0] // 8) + 1

    # Initialize output to be used in the case of iterative
    # estimation.

    noise_cube_out = np.ones_like(data)

    # Iterate

    for ii in np.arange(iterations):

        if not do_map:

            # If spatial variations are turned off then estimate a
            # single value and fill the noise map with this value.

            noise_value = mad_zero_centered(data, mask=noisemask)
            noise_map = np.zeros(data.shape[1:]) + noise_value

        else:

            # Initialize map to be full of not-a-numbers
            noise_map = np.zeros(data.shape[1:]) + np.nan

            # Make a noise map

            xx = np.arange(data.shape[2])
            yy = np.arange(data.shape[1])

            # Sample starting at halfbox and stepping by step. In the
            # individual pixel limit this just samples every spectrum.

            xsamps = xx[halfbox::step]
            ysamps = yy[halfbox::step]
            xsampsf = (xsamps[np.newaxis, :]
                       * (np.ones_like(ysamps))[:, np.newaxis]).flatten()
            ysampsf = (ysamps[:, np.newaxis] * np.ones_like(xsamps)).flatten()

            for x, y in zip(xsampsf, ysampsf):
                # Extract a minicube and associated mask from the cube

                minicube = data[:, (y-halfbox):(y+halfbox+1),
                                (x-halfbox):(x+halfbox+1)]
                minicube_mask = noisemask[:, (y-halfbox):(y+halfbox+1),
                                          (x-halfbox):(x+halfbox+1)]

                # If we have enough data, fit a noise value for this entry

                if np.sum(minicube_mask) > nThresh:
                    noise_map[y, x] = mad_zero_centered(minicube,
                                                        mask=minicube_mask)

            if extrax is not None and extray is not None:
                for x, y in zip(extrax, extray):

                    minicube = data[:, (y-halfbox):(y+halfbox+1),
                                    (x-halfbox):(x+halfbox+1)]
                    minicube_mask = noisemask[:, (y-halfbox):(y+halfbox+1),
                                              (x-halfbox):(x+halfbox+1)]

                    if np.sum(minicube_mask) > nThresh:
                        noise_map[y, x] = mad_zero_centered(minicube,
                                                            mask=minicube_mask)

            noise_map[boundary] = np.nan

            # If we are using a box size greater than an individual pixel
            # interpolate to fill in the noise map.

            if halfbox > 0:

                # Note the location of data, this is the location
                # where we want to fill in noise values.
                data_footprint = np.any(np.isfinite(data), axis=0)

                # Generate a smoothing kernel based on the box size.
                kernel = Gaussian2DKernel(box / np.sqrt(8 * np.log(2)))

                # Make a weight map to be used in the convolution, in
                # this weight map locations with measured values have
                # unity weight. This without measured values have zero
                # weight.

                # wt_map = np.isfinite(noise_map).astype(np.float)
                # wt_map[boundary] = np.nan
                # Take an average weighted by the kernel at each
                # location.
                # noise_map[np.isnan(noise_map)] = 0.0
                # y, x = np.where(np.isfinite(noise_map))
                # import scipy.interpolate as interp
                # func = interp.interp2d(x, y, noise_map[y, x], kind='cubic')

                noise_map = convolve(noise_map, kernel, boundary='extend')

                # yy, xx = np.indices(noise_map.shape)
                # noise_map_beta = interp.griddata((y, x), noise_map[y,x],
                #                                  (yy, xx), method='cubic')
                # noise_map_beta = func(yy, xx)
                # noise_map_beta[boundary] = np.nan
                # noise_map = noise_map_beta
                # wt_map = convolve(wt_map, kernel, boundary='extend')

                # noise_map /= wt_map

                # Set the noise map to not-a-number outside the data
                # footprint.

                noise_map[~data_footprint] = np.nan
        # Initialize spectrum

        noise_spec = np.zeros(data.shape[0]) + np.nan

        if not do_spec:

            # If spectral variations are turned off then assume that
            # the noise_map describes all channels of the cube.

            pass

        else:

            # Loop over channels

            zz = np.arange(data.shape[0])
            for z in zz:

                # Idententify the range of channels to be considered
                # in this estimate.

                lowz = np.clip(z - boxv, 0, data.shape[0])
                hiz = np.clip(z + boxv + 1, 0, data.shape[0])

                # Extract a slab from the cube and normalize it by the
                # noise map. Now any measured noise variations are
                # relative to those in the noise map.

                slab = data[lowz:hiz, :, :] / noise_map[np.newaxis, :, :]
                slab_mask = noisemask[lowz:hiz, :, :]
                noise_spec[z] = mad_zero_centered(slab, mask=slab_mask)

            # Smooth the spectral variations in the noise.

            if bandpass_smooth_window > 0:

                # Initialize a Savitzky-Golay filter then run it over
                # the noise spectrum.

                kernel = savgol_coeffs(int(bandpass_smooth_window),
                                       int(bandpass_smooth_order))

                baddata = np.isnan(noise_spec)
                noise_spec = convolve(noise_spec, kernel,
                                      nan_treatment='interpolate',
                                      boundary='extend')
                noise_spec[baddata] = np.nan

                # Make sure that the noise spectrum is normalized by
                # setting the median to one.

                noise_spec /= np.nanmedian(noise_spec)

        # Combine the spatial and spectral variations into a
        # three-dimensional noise estimate.

        noise_cube = np.ones_like(data)
        noise_cube *= (noise_map[np.newaxis, :]
                       * noise_spec[:, np.newaxis, np.newaxis])

        if iterations == 1:
            return(noise_cube)
        else:

            # If iterating, normalize the data by the current noise
            # estimate and scale the current noise cube by the new
            # estimate.

            data = data / noise_cube
            noise_cube_out *= noise_cube

    # If iterating return the iterated noise cube.
    if return_spectral_cube and type(datain) is SpectralCube:
        return(SpectralCube(u.Quantity(noise_cube_out, datain.unit), 
                            datain.wcs, header=datain.header))
    else:
        return(noise_cube_out)
