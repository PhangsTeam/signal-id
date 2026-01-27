from spectral_cube import SpectralCube
from radio_beam import Beam
import astropy.units as u
import numpy as np
from astropy.io import fits
import os
import sys
import warnings

def common_cube(filename, 
                target_rms=None,
                target_resolution=None,
                datadir='',
                outdir='',
                outname='common.fits',
                cube=None,
                rmsname=None,
                rmsmap=None,
                distance=None,
                overwrite=True,
                return_cube=False,
                huge=True):
        """
        Convolves and adds noise to a data cube to arrive at 
        a target linear resolution and noise level
        
        Parameters:
        
        filename : str
            File name of spectral data cube to process
    
        outname : str
            File name for output cube
            
        target_rms : astropy.Quantity
            Target RMS level in units of cube brightness
            
        target_resolution : astropy.Quantity
            Target resolution expressed as either length (linear 
            resolution) or angle (angular resolution) 
            
        datadir : str
            Directory for input data
        
        outdir : str
            Directory for output data
        
        rmsname : str
            Name for input RMS data cube
        
        rmsmap : np.ndarray
            numpy array of noise values at positions corresponding to 
            the spectral cube.
        
        distance : astropy.Quantity
            Distance to target; required if a linear target 
            resolution is requested
            
        overwrite : bool
            Overwrite output products if they already exist?
        
        return_cube : bool
            Return a SpectralCube?
        """
    
        if cube is None:
            cube = SpectralCube.read(os.path.join(datadir, filename))
    
        cube.allow_huge_operations = huge
        orig_beam_area = cube.beam.sr
        target_beam = None
        if target_resolution is not None:
            if target_resolution.unit.is_equivalent(u.pc):
                if distance is None:
                    warnings.warn("A distance with units must be provided to reach a target linear resolution")
                    raise ValueError
        
                target_beam_size = (target_resolution / distance).to(
                    u.dimensionless_unscaled) * u.rad
            elif target_resolution.is_equivalent(u.rad):
                target_beam_size = target_resolution
            else:
                warnings.warn("target_resolution must be either linear distance or angle")
                raise ValueError
            
            target_beam = Beam(major=target_beam_size,
                               minor=target_beam_size, pa=0 * u.deg)
            if target_beam > cube.beam:
                cube = cube.convolve_to(target_beam)
            else:
                warnings.warn('Target linear resolution unreachable.')

        if target_rms is not None:
            if target_beam is None:
                target_beam = cube.beam
            noisevals = np.random.randn(*cube.shape)
            null_beam = Beam(major=1e-7 * u.deg, minor=1e-7 * u.deg,
                             pa=0 * u.deg)
            noisevals[~np.isfinite(cube)] = np.nan
            noisecube = SpectralCube(noisevals, cube.wcs,
                                     header=cube.header, beam=null_beam,
                                     meta={'BUNIT': 'K'})
            noisecube = noisecube.with_mask(np.isfinite(cube))
            noisecube.allow_huge_operations = huge
            area_ratio = (orig_beam_area / target_beam.sr).to(
                u.dimensionless_unscaled).value
            noisecube = noisecube.convolve_to(target_beam)

            if (rmsname is not None) and (rmsmap is None):
                rmsmap = fits.getdata(os.path.join(datadir,
                                                   rmsname)) * u.K
            if rmsmap is None:
                warnings.warn("One of rmsmap or rmsname must be provided for noise homogenization")
                raise FileNotFoundError

            rmsamplitude = rmsmap * area_ratio**0.5
            if np.all(rmsamplitude > target_rms):
                warnings.warn("All noise values larger than target value")
            else:
                addamplitude = (target_rms**2 - rmsamplitude**2)
                addamplitude[addamplitude < 0] = 0.0
                addamplitude.shape = addamplitude.shape
                noisevals = noisecube.filled_data[:]
                noisevals /= np.nanstd(noisevals)
                noisevals *= (addamplitude**0.5)
                cube.allow_huge_operations = huge
                cube += noisevals
        if type(outname) is str:
            cube.write(outdir + outname,
                       overwrite=overwrite)
        if return_cube:
            return(cube)
        else:
            return(True)
