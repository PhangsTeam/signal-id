from phangsPipeline import scMaskingRoutines as scm
from spectral_cube import SpectralCube
from radio_beam import Beam
from galaxies import Galaxy
import astropy.units as u
import numpy as np
from astropy.io import fits
from phangs import PhangsGalaxy

import sys
sys.path.append('/mnt/space/erosolow/phangs_imaging_scripts')


def homogenize(targres=60,
               galnames=None,
               outdir=None,
               targetrms=None,
               **kwargs):
    if galnames is None:
        galnames = ['ngc0628', 'ngc1637', 'ngc2903', 'ngc3511',
                    'ngc3521', 'ngc3621',
                    'ngc3627', 'ngc4826', 'ngc5068', 'ngc5643']
    if targetrms is None:
        targetrms = 0.1 * (60 / targres) * u.K
    if outdir is None:
        outdir = '/mnt/work/erosolow/phangs/v3p4_{0}pc/'.format(targres)
    for gal in galnames:
        common_cube(gal, resolution=targres * u.pc,
                    targetrms=targetrms,
                    huge=True,
                    suffix='_{0}pc'.format(targres),
                    outdir=outdir,
                    **kwargs)


def buildnative(galnames=None,
                outdir=None,
                targetrms=None,
                **kwargs):
    if galnames is None:
        galnames = ['ngc4826',
                    'ngc5068',
                    'ngc0628',
                    'ngc2903',
                    'ngc3621',
                    'ngc3521',
                    'ngc3511',
                    'ngc5643',
                    'ngc1637',
                    'ngc3627']
    if outdir is None:
        outdir = '/mnt/work/erosolow/phangs/v3p4_native/'
    for gal in galnames:
        native_cube(gal,
                    huge=True,
                    suffix='_native',
                    outdir=outdir,
                    **kwargs)


def native_cube(galname,
                datadir='/mnt/bigdata/PHANGS/Archive/ALMA/delivery_v3p4/cubes/',
                outdir='/mnt/work/erosolow/phangs/v3p4_native/',
                suffix='_native',
                huge=True):
    g = PhangsGalaxy(galname)
    print("Processing {0}".format(g.name))
    cubename = datadir + galname + '_12m+7m+tp_co21_pbcorr_round_k.fits'
    rmsname = cubename.replace('co21_pbcorr', 'co21_noise_pbcorr')
    cube = SpectralCube.read(cubename)
    cube.allow_huge_operations = huge
    orig_beam_area = cube.beam.sr
    cube.write(outdir + galname + '_co21' + suffix + '.fits',
               overwrite=True)
    rms = scm.noise_cube(cube.filled_data[:].value, box=15,
                         spec_box=5, iterations=3,
                         bandpass_smooth_window=21)
    rmscube = SpectralCube(rms, wcs=cube.wcs,
                           header=cube.header,
                           meta={'BUNIT': 'K'})
    rmscube.write(outdir + galname + '_co21_noise'
                  + suffix + '.fits', overwrite=True)
    mask = scm.simple_mask(cube.filled_data[:].value, rms,
                           hi_thresh=4, hi_nchan=3,
                           lo_thresh=2, lo_nchan=2)
    maskcube = SpectralCube(mask.astype(np.int8), wcs=cube.wcs,
                            header=cube.header)
    maskcube.write(outdir + galname + '_co21_signalmask'
                   + suffix + '.fits', overwrite=True)


def common_cube(galname,
                datadir='/mnt/bigdata/PHANGS/Archive/ALMA/delivery_v3p4/cubes/',
                outdir='',
                outname=None,
                suffix='_60pc',
                rmsroot='_co21_noise',
                maskroot='_co21_signalmask',
                huge=False,
                resolution=None,
                cubename=None,
                cube=None,
                rmsname=None,
                rmsmap=None,
                makerms=True,
                makemask=True,
                targetrms=None):
      g = PhangsGalaxy(galname)
       # print("Processing {0}".format(g.name))

       if outname is None:
            outname = galname + '_co21' + suffix + '.fits',

        if cube is None:
            if cubename is None:
                cubename = (datadir
                            + galname
                            + '_12m+7m+tp_co21_pbcorr_round_k.fits')

            cube = SpectralCube.read(cubename)

        cube.allow_huge_operations = huge
        orig_beam_area = cube.beam.sr
        target_beam = None
        if resolution is not None:
            target_beam_size = (resolution / g.distance).to(
                u.dimensionless_unscaled) * u.rad
            target_beam = Beam(major=target_beam_size,
                               minor=target_beam_size, pa=0 * u.deg)
            if target_beam > cube.beam:
                cube = cube.convolve_to(target_beam)
            else:
                return()
        if targetrms is not None:
            if target_beam is None:
                target_beam = cube.beam
            noisevals = np.random.randn(*cube.shape)
            null_beam = Beam(major=1e-7 * u.deg, minor=1e-7 * u.deg,
                             pa=0 * u.deg)
            noisecube = SpectralCube(noisevals, cube.wcs,
                                     mask=(cube > -100*u.K),
                                     header=cube.header, beam=null_beam,
                                     meta={'BUNIT': 'K'})
            noisecube.allow_huge_operations = True
            area_ratio = (orig_beam_area / target_beam.sr).to(
                u.dimensionless_unscaled).value
            noisecube = noisecube.convolve_to(target_beam)

            if (rmsname is not None) and (rmsmap is None):
                rmsmap = fits.getdata(rmsname) * u.K
            if (rmsname is None) and (rmsmap is None):
                rmsname = cubename.replace('co21_pbcorr', 'co21_noise_pbcorr')
                rmsmap = fits.getdata(rmsname) * u.K

            rmsamplitude = rmsmap * area_ratio**0.5
            if np.all(rmsamplitude > targetrms):
                return()
            addamplitude = (targetrms**2 - rmsamplitude**2)
            addamplitude[addamplitude < 0] = 0.0
            addamplitude.shape = addamplitude.shape
            noisevals = noisecube.filled_data[:]
            noisevals /= np.std(noisevals)
            noisevals *= (addamplitude**0.5)
            cube.allow_huge_operations = True
            cube += noisevals

        cube.write(outdir + outname,
                   overwrite=True)
        if makerms:
            rms = scm.noise_cube(cube.filled_data[:].value, box=15,
                                 spec_box=5, iterations=3,
                                 bandpass_smooth_window=21)
            rmscube = SpectralCube(rms, wcs=cube.wcs,
                                   header=cube.header,
                                   meta={'BUNIT': 'K'})
            rmscube.write(outdir + galname + rmsroot
                          + suffix + '.fits', overwrite=True)
            if makemask:
                mask = scm.simple_mask(cube.filled_data[:].value, rms,
                                       hi_thresh=4, hi_nchan=3,
                                       lo_thresh=2, lo_nchan=2)
                maskcube = SpectralCube(mask.astype(np.int8), wcs=cube.wcs,
                                        header=cube.header)
                maskcube.write(outdir + galname + maskroot
                               + suffix + '.fits', overwrite=True)


def build_replicates(filename,
                     maskname=None,
                     rmsname=None,
                     galname=None,
                     rmsroot='_noise',
                     maskroot='_signalmask',
                     datadir='',
                     outdir='',
                     n_replicates=100,
                     mask_kwargs=None,
                     noise_kwargs=None, **kwargs):
    """
    Builds a set of replicates out of a given data cube
    Parameters
    ----------
    filename : str
        Name of file to be processed
    Keywords
    --------
    maskname : str 
        Name of mask file (assumed to be FITS)
    rmsname : str
        Name of RMS file to be used (FITS)
    galname : str
        Canonical PHANGS galaxy name of target
    
    rmsroot : str
        Filename tag used to identfiy output RMS files
    maskroot : str
        Filename tag used to identify mask files
    datadir : str
        Directory where cube, mask, and rms files are found
    outdir : str
        Directory where replicate cubes, masks, and RMS are stored
    n_replicates : int
        Number of cube replicates
    
    mask_kwargs, noise_kwargs : dict
        Keyword dictionary passed to noise generation routines
    """
    cube = SpectralCube.read(datadir + filename)
    origmask = cube.mask.include()

    if (maskname is None) or (rmsname is None):
        mask, rms = scm.recipe_phangs_mask(cube,
                                           mask_kwargs=mask_kwargs,
                                           noise_kwargs=noise_kwargs,
                                           return_rms=True)
    else:
        mask = SpectralCube.read(datadir + maskname)
        rms = SpectralCube.read(datadir + rmsname)

    if galname is None:
        galname = (filename.split('_'))[0]

    g = PhangsGalaxy(galname)
    maskarray = np.array(mask.filled_data[:].value, dtype=np.bool)
    nonoise = cube.filled_data[:]
    nonoise[~maskarray] = 0 * cube.unit
    nonoise = SpectralCube(nonoise, cube.wcs, header=cube.header)
    nonoise = nonoise.with_mask(origmask)
    nonoise.write(outdir + galname + '_original.fits')
    rms = u.Quantity(np.zeros_like(nonoise), cube.unit)

    numzeros = np.ceil(np.log10(n_replicates))
    for i in range(n_replicates):
        ctr = str(i).zfill(int(numzeros))
        thisname = galname + '_replicate_{0}.fits'.format(ctr)
        common_cube(galname,
                    cube=nonoise,
                    rmsmap=rms,
                    outname=thisname,
                    suffix='_{0}'.format(ctr),
                    outdir=outdir, **kwargs)


def test_replicates():
    build_replicates('ngc4826_12m+7m+tp_co21_pbcorr_round_k.fits',
                     datadir='/mnt/bigdata/PHANGS/Archive/ALMA/delivery_v3p4/cubes/',
                     rmsname='ngc4826_12m+7m+tp_co21_noise_pbcorr_round_k.fits',
                     maskname='ngc4826_12m+7m+tp_co21_signalmask.fits',
                     outdir='/mnt/work/erosolow/phangs/replicates/',
                     rmsroot='_noise',
                     maskroot='_signalmask',
                     n_replicates=3,
                     targetrms=0.1 * u.K)
