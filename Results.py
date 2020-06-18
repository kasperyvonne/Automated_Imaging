import numpy as np
import subprocess
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
from tqdm import tqdm
from pathlib import Path
import os
import glob
import pandas as pd
import astropy.units as u
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox

def get_rms(image):
    '''
    IMAGE: image data of the image.fits

    Example: get_rms(fits.open(str(path_to_data))[0])
    Calculates the rms (root mean square) in all four corners of the image with some distance to the source.
    '''
    x0=int(image.header['CRPIX1'])-75
    x1=int(image.header['CRPIX1'])+75
    y0=int(image.header['CRPIX2'])-75
    y1=int(image.header['CRPIX2'])+75
    
    image_data = image.data[0,0,:,:]
    lower_left = np.sqrt(np.mean(np.square(image_data[0:x0,0:y0])))
    lower_right = np.sqrt(np.mean(np.square(image_data[x1:image.header['NAXIS1'],0:y0])))
    upper_left = np.sqrt(np.mean(np.square(image_data[0:x0,y1:image.header['NAXIS2']])))
    upper_right = np.sqrt(np.mean(np.square(image_data[x1:image.header['NAXIS2'],y1:image.header['NAXIS2']])))
    x = np.array([lower_left,lower_right,upper_left,upper_right])
    return(np.mean(x)) 

def get_dynamicrange(path_to_image):
    im = fits.open(str(path_to_image))[0]
    im1 = im.data[0,0,:,:]
    rms = get_rms(im)
    peak = im1.max()
    return (peak/rms)
def with_beam_and_mask(path_to_image, sourcename, rms, sigma, out_path):    
    image = fits.open(path_to_image)[0]

    im1 = image.data[0,0,:,:]
    header = image.header
    name1 = str(Path(path_to_image).stem).split('-')[0]
    epoch = name1.split('.u.')[-1]
    
    #print('Peak Flux=', im1.max())  
    mask = np.zeros_like(im1)
    mask[im1>(sigma*rms)]=1
    newmask= mask*im1
    cumulativeflux=np.sum(newmask)
    #print('cumulative flux:',cumulativeflux)
    newmask[newmask==0]=-10
    #print(newmask)
    x_n_pixel = header['NAXIS1']

    x_inc = (header['CDELT1'] * u.degree).to(u.mas) 
    x_ref_value = (header['CRVAL1'] * u.degree).to(u.mas)
    y_n_pixel = header['NAXIS2']


    y_inc = (header['CDELT2'] * u.degree).to(u.mas)
    y_ref_value = (header['CRVAL2'] * u.degree).to(u.mas)

    x_ref_pixel = header['CRPIX1']
    y_ref_pixel = header['CRPIX2']

    x = np.linspace(x_ref_pixel * -x_inc, (x_n_pixel - x_ref_pixel) * (x_inc), x_n_pixel)
    y = np.linspace(y_ref_pixel * -y_inc, (y_n_pixel - y_ref_pixel) * y_inc, y_n_pixel)

    bmin = (header['BMIN'] * u.degree).to(u.mas)
    bmaj =  (header['BMAJ'] * u.degree).to(u.mas)

    #print(x_ref_value,y_ref_value)
    #plt.rcParams.update({'font.size': 22})
    #plt.figure(figsize=(14,10)) 
    #plt.figure(fontsize = 12)
    fig, ax = plt.subplots(figsize=(8,6))
    im = plt.pcolormesh(x,y,newmask,
                    cmap='hot',
                    norm = colors.SymLogNorm(linthresh=0.0001, 
                                            linscale=0.0001, vmin= im1.min(), vmax=im1.max()) )
    #print('The min value:',im1.min())   
        
    #cbar = plt.colorbar(im, ticks = [-0.001, 0, 0.001, 0.01], label='Flux in Jy/beam')
    cbar = plt.colorbar(im, ticks = [(-1e-3, 0, 1e-3, 1e-2)] , label='Flux in Jy/beam')
    plt.ylim([-10, 20])
    plt.xlim([10, -20])
    #plt.yticks([450, 500, 600, 700])
    nice_year=str(epoch).split('_')[0]
    nice_month=str(epoch).split('_')[1]
    nice_day=str(epoch).split('_')[2]
    plt.xlabel('Relativ R.A. in mas')
    plt.ylabel('Relativ Declination in mas')
    plt.title(str(sourcename)+', '+'Epoch: '+str(nice_day)+'.'+str(nice_month)+'.'+str(nice_year))
    
    
    box = AnchoredAuxTransformBox(ax.transData, loc='lower left', frameon = False)
    el = Ellipse((0, 0), width=bmin.value, height=bmaj.value, angle = -header['BPA'],color='white',ec='k',alpha =0.75)  
    box.drawing_area.add_artist(el)

    ax.add_artist(box)
    #flagged = '250km'
    textstr = '\n'.join((
    r'rms=$%.6f$ Jy/beam' % (rms),
    r'cut at=$%.0f$$\times$ rms' % (sigma),
    r'peak flux=$%.4f$ Jy/beam'%(im1.max()),
    r'cumulative flux=$%.4f$ Jy/beam' % (cumulativeflux)))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.75)
    
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


    #cbar.ax.set_ylabel('Flux in Jy/beam')
    plt.tight_layout()
    plt.savefig(str(out_path)+str(name1)+'_cut_at_'+str(sigma)+'.png')
    plt.clf()
    fig.clf()
def with_beam_and_text(path_to_image, sourcename, DyRa, out_path):    
    image = fits.open(path_to_image)[0]

    im1 = image.data[0,0,:,:]
    header = image.header
    name1 = str(Path(path_to_image).stem).split('-')[0]
    epoch = name1.split('.u.')[-1]
    
  
    scale = np.logspace(np.log10(np.absolute(im1.min())), np.log10(im1.max()), 5)
    #print('The contour lines are at:', scale)
    x_n_pixel = header['NAXIS1']

    x_inc = (header['CDELT1'] * u.degree).to(u.mas) 
    x_ref_value = (header['CRVAL1'] * u.degree).to(u.mas)
    y_n_pixel = header['NAXIS2']


    y_inc = (header['CDELT2'] * u.degree).to(u.mas)
    y_ref_value = (header['CRVAL2'] * u.degree).to(u.mas)

    x_ref_pixel = header['CRPIX1']
    y_ref_pixel = header['CRPIX2']

    x = np.linspace(x_ref_pixel * -x_inc, (x_n_pixel - x_ref_pixel) * (x_inc), x_n_pixel)
    y = np.linspace(y_ref_pixel * -y_inc, (y_n_pixel - y_ref_pixel) * y_inc, y_n_pixel)

    bmin = (header['BMIN'] * u.degree).to(u.mas)
    bmaj =  (header['BMAJ'] * u.degree).to(u.mas)

    #print(x_ref_value,y_ref_value)
    #plt.rcParams.update({'font.size': 22})
    #plt.figure(figsize=(14,10)) 
    #plt.figure(fontsize = 12)
    fig, ax = plt.subplots(figsize=(8,6))
    im = plt.pcolormesh(x,y,im1,
                    cmap='hot',
                    norm = colors.SymLogNorm(linthresh=0.0001, 
                                            linscale=0.0001, vmin= im1.min(), vmax=im1.max()) )
    #print('The min value:',im1.min())   
    cont = plt.contour(x,y, im1,
                scale,
                colors='w',linewidths = 1, linestyles='solid',
                norm = colors.SymLogNorm(linthresh=0.0001, 
                                       linscale=0.0001, vmin= -0.001345288, vmax=0.09435418))
        
    #cbar = plt.colorbar(im, ticks = [-0.001, 0, 0.001, 0.01], label='Flux in Jy/beam')
    cbar = plt.colorbar(im, ticks = [(-1e-3, 0, 1e-3, 1e-2)] , label='Flux in Jy/beam')
    plt.ylim([-10, 20])
    plt.xlim([10, -20])
    #plt.yticks([450, 500, 600, 700])
    nice_year=str(epoch).split('_')[0]
    nice_month=str(epoch).split('_')[1]
    nice_day=str(epoch).split('_')[2]
    plt.xlabel('Relativ R.A. in mas')
    plt.ylabel('Relativ Declination in mas')
    plt.title(str(sourcename)+', '+'Epoch: '+str(nice_day)+'.'+str(nice_month)+'.'+str(nice_year))
    
    
    box = AnchoredAuxTransformBox(ax.transData, loc='lower left', frameon = False)
    el = Ellipse((0, 0), width=bmin.value, height=bmaj.value, angle = -header['BPA'],color='white',ec='k',alpha =0.75)  
    box.drawing_area.add_artist(el)

    ax.add_artist(box)
    textstr = r'Dynamic Range=$%.2f$' % (DyRa)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.75)
    
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


    #cbar.ax.set_ylabel('Flux in Jy/beam')
    plt.tight_layout()
    plt.savefig(str(out_path)+str(name1)+'_uncut.png' )
    plt.clf()    
    fig.clf()
def get_dyra(path_to_hdf):
    df = pd.read_hdf(path_to_hdf, 'df')
    sorted_df = df.sort_values(by=['DR'], axis = 1).iloc[:,-1:]
    dyra=sorted_df.loc['DR',:].tolist()[0]
    return dyra

def get_params(path_to_hdf, out_dict):
    df = pd.read_hdf(path_to_hdf, 'df')
    sorted_df = df.sort_values(by=['DR'], axis = 1).iloc[:,-1:]
    with open(str(out_dict), 'w') as file:
        file.write(sorted_df.to_latex())

def get_paramset(path_to_hdf, out_dict):
    data = pd.read_hdf(path_to_hdf, 'df')
    param_set = pd.read_hdf(path_to_hdf,'ParamSets')
    with open(str(out_dict), 'w') as file:
        file.write(param_set.to_latex(index=True,float_format="%.4f",header=False,na_rep=''))
    
def run_WSClean(path_to_hdf, epoch):
    df = pd.read_hdf(path_to_hdf, 'df')
    sorted_df = df.sort_values(by=['DR'], axis = 1).iloc[:,-1:]
    xs=int(sorted_df.loc['xsize'][0])
    ys=int(sorted_df.loc['ysize'][0])
    scale=np.round(sorted_df.loc['scale'][0],1)
    niter=int(sorted_df.loc['niter'][0])
    mgain=np.round(sorted_df.loc['mgain'][0],4)
    gain=np.round(sorted_df.loc['gain'][0],4)
    automask=np.round(sorted_df.loc['auto_mask'][0],4)
    autothresh=np.round(sorted_df.loc['auto_thresh'][0],4)
    weight=np.round(sorted_df.loc['weight'][0],4)
    scalebias=sorted_df.loc['scale_bias'][0]

    data_path = str(Path(path_to_hdf).parent.parent.parent)+'/data/measurement_sets/0149+710.u.'+str(epoch)+'.ms'
    out_dict = str(Path(path_to_hdf).parent.parent.parent)+'/BestEpochsLocal/'
    out_dict2 = str(Path(path_to_hdf).parent.parent.parent)+'/BestEpochsLocal/'+str(epoch)
    #print(data_path)
    #print(out_dict)

    command = 'wsclean '
    command = command \
    +'-quiet ' \
    +'-j '+str(2)+' ' \
    +'-multiscale ' \
    +' -size ' + str(xs) + ' ' + str(ys) \
    + ' -scale ' + str(scale) + 'masec' \
    + ' -niter ' + str(niter) \
    + ' -mgain ' + str(mgain) \
    + ' -gain ' + str(gain) \
    + ' -auto-mask ' + str(automask) \
    + ' -weight briggs ' + str(weight) \
    + ' -multiscale-scale-bias ' + str(scalebias) \
    + ' -auto-threshold ' + str(autothresh) \
    + ' -name ' + str(out_dict2)+'/'+str(epoch)  \
    + ' ' + str(data_path)
    
    
    if os.path.exists(out_dict)==False:
       subprocess.call(["mkdir", str(out_dict)])

    if os.path.exists(out_dict2)==False:
       subprocess.call(["mkdir", str(out_dict2)])

    subprocess.run(command, shell=True, timeout = 300)
    print('Cleaning done :) ')
    return str(out_dict2)+'/'+str(epoch)
def run_WSClean_with_beam(path_to_hdf, epoch, beamsize):

    df = pd.read_hdf(path_to_hdf, 'df')
    sorted_df = df.sort_values(by=['DR'], axis = 1).iloc[:,-1:]
    xs=int(sorted_df.loc['xsize'][0])
    ys=int(sorted_df.loc['ysize'][0])
    scale=np.round(sorted_df.loc['scale'][0],1)
    niter=int(sorted_df.loc['niter'][0])
    mgain=np.round(sorted_df.loc['mgain'][0],4)
    gain=np.round(sorted_df.loc['gain'][0],4)
    automask=np.round(sorted_df.loc['auto_mask'][0],4)
    autothresh=np.round(sorted_df.loc['auto_thresh'][0],4)
    weight=np.round(sorted_df.loc['weight'][0],4)
    scalebias=sorted_df.loc['scale_bias'][0]

    data_path = str(Path(path_to_hdf).parent.parent.parent)+'/data/measurement_sets/0149+710.u.'+str(epoch)+'.ms'
    out_dict = str(Path(path_to_hdf).parent.parent.parent)+'/BestEpochSameBeam/'
    out_dict2 = str(Path(path_to_hdf).parent.parent.parent)+'/BestEpochSameBeam/'+str(epoch)
    #print(data_path)
    #print(out_dict)

    command = 'wsclean '
    command = command \
    +'-quiet ' \
    +'-j '+str(2)+' ' \
    +'-multiscale ' \
    +'-beam-size '+ str(beamsize) \
    +' -size ' + str(xs) + ' ' + str(ys) \
    + ' -scale ' + str(scale) + 'masec' \
    + ' -niter ' + str(niter) \
    + ' -mgain ' + str(mgain) \
    + ' -gain ' + str(gain) \
    + ' -auto-mask ' + str(automask) \
    + ' -weight briggs ' + str(weight) \
    + ' -multiscale-scale-bias ' + str(scalebias) \
    + ' -auto-threshold ' + str(autothresh) \
    + ' -name ' + str(out_dict2)+'/'+str(epoch)  \
    + ' ' + str(data_path)
    
    
    if os.path.exists(out_dict)==False:
       subprocess.call(["mkdir", str(out_dict)])

    if os.path.exists(out_dict2)==False:
       subprocess.call(["mkdir", str(out_dict2)])

    subprocess.run(command, shell=True, timeout = 300)
    print('Cleaning done :) ')
    return str(out_dict2)+'/'+str(epoch)

def get_beam_info(path_to_image):
    '''
    Get the bmaj and bmin values in masec from the fits header
    '''
    image = fits.open(path_to_image)[0]

    im1 = image.data[0,0,:,:]
    header = image.header
    
    bmin = ((header['BMIN'] * u.degree).to(u.mas)).value
    bmaj = ((header['BMAJ'] * u.degree).to(u.mas)).value
    bangle =header['BPA']
    
    return(bmin, bmaj, bangle)

 

def main():
    
    path_to_grid = '/home/yvonne/Schreibtisch/TXS_FUCKING_final'
    path_2017_11_18=str(path_to_grid)+'/results_0149+710.u.2017_11_18/MRun_2/data_TXS0149+710.h5'
    path_2017_01_28=str(path_to_grid)+'/results_0149+710.u.2017_01_28/MRun_2/data_TXS0149+710.h5'
    path_2017_04_22=str(path_to_grid)+'/results_0149+710.u.2017_04_22/MRun_2/data_TXS0149+710.h5'
    path_2017_06_17=str(path_to_grid)+'/results_0149+710.u.2017_06_17/MRun_2/data_TXS0149+710.h5'
    path_2018_02_02=str(path_to_grid)+'/results_0149+710.u.2018_02_02/MRun_2/data_TXS0149+710.h5'
    path_2018_10_06=str(path_to_grid)+'/results_0149+710.u.2018_10_06/MRun_2/data_TXS0149+710.h5'
    path_2019_07_19=str(path_to_grid)+'/results_0149+710.u.2019_07_19/MRun_2/data_TXS0149+710.h5'
    
    #test_path ='/home/yvonne/Dokumente/MA/TXS_final_grid/BestEpochsLocal/2019_07_19/2019_07_19-image.fits'
    #with_beam_and_mask(test_path, sourcename, get_rms(fits.open(str(test_path))[0]), 10)

    get_paramset(str(path_to_grid)+'/results_0149+710.u.2017_11_18/MRun_0/data_TXS0149+710.h5', str(path_to_grid)+'/IntialParamGrid.txt' )
    print("The inital parametergrid is saved as a latexfile.")
    path_list=[path_2017_11_18, path_2017_01_28, path_2017_04_22,
        path_2017_06_17, path_2018_02_02, path_2018_10_06, path_2019_07_19]
    sourcename='TXS0149+710'
    beam_list=np.zeros((len(path_list),2))
    i=0
    for path in path_list:
        epoch = str(Path(path).parent.parent).split('.u.')[-1]
        print('starting with:',epoch)
        #creates a new dict with localy cleaned images
        path_to_fits = run_WSClean(path, epoch)
        path_to_image = str(path_to_fits)+'-image.fits'
        #plots images with and without mask  
        out = '/home/yvonne/Schreibtisch/Ergebnisse_15_06-17_06/Plots/'
        with_beam_and_text(path_to_image, sourcename, get_dyra(path), out)
        with_beam_and_mask(path_to_image, sourcename, get_rms(fits.open(str(path_to_image))[0]), 5, out)
        #Parameterset for every epoch as latex table
        if os.path.exists(str(path_to_grid)+'/BestParams/')==False:
            subprocess.call(["mkdir",str(path_to_grid)+'/BestParams/'])
        get_params(path, str(path_to_grid)+'/BestParams/'+str(epoch)+'_Params.txt')
        minor, major, angle = get_beam_info(path_to_image)
        beam_list[i]=[minor,major]
        i+=1
    
    beamsize= beam_list.max()*0.001
    print("Now let's do this again with a set beam at ", beamsize, "arcsec")
    for path in path_list:
        epoch = str(Path(path).parent.parent).split('.u.')[-1]
        print('starting with:',epoch)
        #creates new dict with cleaned images with the biggest beam
        path_to_fits = run_WSClean_with_beam(path, epoch, beamsize)
        path_to_image = str(path_to_fits)+'-image.fits'
        with_beam_and_text(path_to_image, sourcename, get_dynamicrange(path_to_image), '/home/yvonne/Schreibtisch/Ergebnisse_15_06-17_06/Plots/Beam_')
        with_beam_and_mask(path_to_image, sourcename, get_rms(fits.open(str(path_to_image))[0]), 5, '/home/yvonne/Schreibtisch/Ergebnisse_15_06-17_06/Plots/Beam_')
        print('done with:', epoch)

if __name__ == '__main__':
    main()
