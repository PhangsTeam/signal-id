import numpy as np
import scipy.ndimage as nd
import astropy.units as u
from spectral_cube import SpectralCube
import astropy.wcs as wcs


def channel_shift(x, ChanShift, replace_nan=True, min_weight=0.2):
    # Shift an array of spectra (x) by a set number of Channels (array)
    # x = np.atleast_2d(x)
    xshape = x.shape
    if x.ndim == 1:
        x.shape += (1,)
        
    ChanShift = np.atleast_1d(ChanShift)
    nan_mask = np.isnan(x)

    if np.any(nan_mask) and replace_nan:
        wt = 1 - nan_mask
        shiftwt = channel_shift(wt, ChanShift)
        shiftwt.shape = x.shape
        x = np.nan_to_num(x)
        wtresult=True
    else:
        wtresult=False
        
    ftx = np.fft.fft(x, axis=0)
    m = np.fft.fftfreq(x.shape[0])
    phase = np.exp(-2 * np.pi * m[:, np.newaxis]
                   * 1j * ChanShift[np.newaxis, :])
    x2 = np.real(np.fft.ifft(ftx * phase, axis=0))
    x2.shape = x.shape

    if wtresult:
        x2 /= shiftwt
        x2[shiftwt < min_weight] = np.nan

    x2.shape = xshape
    return(x2)


def shuffle_cube(DataCube, centroid_map, chunk=1000):
    """
    Shuffles cube so that the velocity appearing in the centroid_map is set 
    to the middle channel and velocity centroid.
    
    Parameters
    ----------
    DataCube : SpectralCube
        The original spectral cube with spatial dimensions Nx, Ny and spectral dimension Nv
    centroid_map : 2D numpy.ndarray
        A 2D map of the centroid velocities for the lines to stack of dimensions Nx, Ny.
        Note that DataCube and Centroid map must have equivalent (but not necessarily equal) 
        spectral units (e.g., km/s and m/s)
    
    Keywords
    --------
    chunk : int
        Number of data points to include in a chunk for processing.
    
    Returns
    -------
    OutCube : np.array
        Output SpectralCube cube shuffled so that the emission is at 0.
 
    """

    spaxis = DataCube.spectral_axis
    y, x = np.where(np.isfinite(centroid_map))
    centroid_map = centroid_map.to(spaxis.unit)
    v0 = spaxis[len(spaxis) // 2]
    newspaxis = spaxis - v0
    relative_channel = np.arange(len(spaxis)) - (len(spaxis) // 2)
    centroids = centroid_map[y, x]
    sortindex = np.argsort(spaxis)
    shift = -1 * np.interp(centroids, spaxis[sortindex],
                           np.array(relative_channel[sortindex], dtype=np.float))
    NewCube = np.empty(DataCube.shape)
    NewCube.fill(np.nan)
    nchunk = (len(x) // chunk)
    for thisx, thisy, thisshift in zip(np.array_split(x, nchunk),
                                       np.array_split(y, nchunk),
                                       np.array_split(shift, nchunk)):
        spectrum = DataCube.filled_data[:, thisy, thisx].value
        baddata = ~np.isfinite(spectrum)
        shifted_spectrum = channel_shift(np.nan_to_num(spectrum),
                                         np.atleast_1d(thisshift))
        shifted_mask = channel_shift(baddata, np.atleast_1d(thisshift))
        shifted_spectrum[baddata > 0] = np.nan
        NewCube[:, thisy, thisx] = shifted_spectrum
    hdr = DataCube.header
    hdr['CRVAL3'] = 0.0
    hdr['CRPIX3'] = len(spaxis) // 2 + 1
    newwcs = wcs.WCS(hdr)
    return(SpectralCube(NewCube, newwcs, header=hdr))


def bin_by_mask(DataCube, mask, centroid_map,
                return_weights=False, weight_map=None):
    """
    Bin a data cube by a label mask, aligning the data to a common centroid.  Returns an array.

    Parameters
    ----------
    DataCube : SpectralCube
        The original spectral cube with spatial dimensions Nx, Ny and spectral dimension Nv
    Mask : 2D numpy.ndarray
        A 2D map containing boolean values with True indicate where the spectra should be aggregated.
    centroid_map : 2D numpy.ndarray
        A 2D map of the centroid velocities for the lines to stack of dimensions Nx, Ny.
        Note that DataCube and Centroid map must have equivalent spectral units (e.g., km/s)
    weight_map : 2D numpy.ndarray
        Map containing the weight values to be used in averaging
    Returns
    -------
    Spectrum : np.array
        Spectrum of average over mask.
    """
    spaxis = DataCube.spectral_axis.value
    y, x = np.where(mask)
    v0 = spaxis[len(spaxis) // 2] * DataCube.spectral_axis.unit
    relative_channel = np.arange(len(spaxis)) - (len(spaxis) // 2)
    centroids = centroid_map[y, x].to(DataCube.spectral_axis.unit).value
    sortindex = np.argsort(spaxis)
    shift = -1 * np.interp(centroids, spaxis[sortindex],
                           np.array(relative_channel[sortindex],
                                    dtype=np.float))
    spectra = DataCube.filled_data[:, y, x].value
    shifted_spectra = channel_shift(spectra, shift)
    if weight_map is not None:
        wts = weight_map[y, x]
    else:
        wts = np.ones_like(shift)
    accum_spectrum = np.nansum(wts[np.newaxis, :]
                               * shifted_spectra, axis=1) / np.nansum(wts)
    shifted_spaxis = (DataCube.spectral_axis - v0)
    if return_weights:
        return(accum_spectrum, shifted_spaxis, np.nansum(wts))
    else:
        return(accum_spectrum, shifted_spaxis)


def bin_by_label(DataCube, LabelMap, centroid_map,
                 weight_map=None, return_weights=False,
                 background_labels=[0]):
    """
    Bin a data cube by a label mask, aligning the data to a common centroid.

    Parameters
    ----------
    DataCube : SpectralCube
        The original spectral cube with spatial dimensions Nx, Ny and spectral dimension Nv
    LabelMap : 2D numpy.ndarray
        A 2D map containing integer labels for each pixel into objects defining the stacking.
    centroid_map : 2D numpy.ndarray
        A 2D map of the centroid velocities for the lines to stack of dimensions Nx, Ny.
        Note that DataCube and Centroid map must have equivalent spectral units to DataCube
    background_labels : list
        List of values in the label map that correspond to background objects and should not
        be processed with the stacking. 
    return_weights : bool
        Return the sum of weights for each spectrum (equals number of contributing spectra for unweighted map)

    Returns
    -------
    output_list : list of dict
        List of dict where each entry contains the stacked spectrum for a given label
    unique_labels = array of unique labels in same order as output list

    """
    UniqLabels = np.unique(LabelMap)
    output_list = []
    for ThisLabel in UniqLabels:
        if ThisLabel not in background_labels:
            if return_weights:
                thisspec, spaxis, wts = bin_by_mask(DataCube,
                                                    (LabelMap == ThisLabel),
                                                    centroid_map,
                                                    return_weights=True,
                                                    weight_map=weight_map)
            else:
                thisspec, spaxis = bin_by_mask(DataCube,
                                               (LabelMap == ThisLabel),
                                               centroid_map,
                                               weight_map=weight_map)
                wts = None
            output_list += [{'label': ThisLabel,
                             'spectrum': thisspec,
                             'spectral_axis': spaxis,
                             'weights': wts}]
    return(output_list, UniqLabels)
