import numpy as np
import subprocess
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
from tqdm import tqdm
from pathlib import Path
import os
import glob
from gauss_utils import gaussian_source
import pandas as pd
import astropy.units as u
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
class GaussFit:
    def __init__(self):
        self.data = []
    
    
    def gaussian(self, height, center_x_old, center_y_old, width, rotation):
        """Returns a gaussian function with the given parameters"""
        width = float(width)

        rotation = np.deg2rad(rotation)
        center_x = center_x_old * np.cos(rotation) - center_y_old * np.sin(rotation)
        center_y = center_x_old * np.sin(rotation) + center_y_old * np.cos(rotation)

        def rotgauss(x,y):
            xp = x * np.cos(rotation) - y * np.sin(rotation)
            yp = x * np.sin(rotation) + y * np.cos(rotation)
            g = height*np.exp(
                -(((center_x-xp)/width)**2+
                  ((center_y-yp)/width)**2)/2.)
            return g
        return rotgauss

    def moments(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        x, y = np.where(data == data.max())
        x = x.max()
        y = y.max()
        width = 0.5 
        height = data.max()
        return height, x, y, width, 0.0


    def fitgaussian(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = self.moments(data)
        errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(data.shape)) - data)
        p, success = leastsq(errorfunction, params)
        return np.array([p[0], p[1], p[2], p[3], p[3], p[4]])

def nice_date(epoch):
    nice_year=str(epoch).split('_')[0]
    nice_month=str(epoch).split('_')[1]
    nice_day=str(epoch).split('_')[2]
    return str(nice_day)+'.'+str(nice_month)+'.'+str(nice_year)

def fit_jet(image, n_iter):
    image = image.copy()
    fit = GaussFit()
    params = np.ones((n_iter, 6))
    residual = np.ones((n_iter, *image.shape))
    gaussian = np.ones((n_iter, *image.shape))
    
    for i in range(n_iter):
        params[i] = fit.fitgaussian(image)
        gaussian[i] = gaussian_source(image.shape[0], 1, *params[i])
        residual[i] = image - gaussian[i]
        image -= gaussian[i]
    return params, residual, gaussian

def sort_params(pa, header):
    center_x = header['CRPIX1']
    center_y = header['CRPIX2']
    dist = np.ones(pa.shape[0])
    #print('dist:',dist)
    parameter = np.ones((pa.shape[0],pa.shape[1]+1))
    #print('parameter:',parameter)
    #print(pa)
    i=0
    for p in pa:
        r = np.sqrt((center_x-p[1])**2+(center_y-p[2])**2)
        #print(r)
        p= np.append(p,r)
        parameter[i]=p
        dist[i]=r
        i+=1
    ind = np.argsort(dist)  

    return parameter[ind]
def plot_results_new_contour_only(img, header, params, epoch,sigma, rms, out):
    '''
    Takes image and params array to visualize source overlayed with fitted components.
    
    img: 2darray
        image of radio source
    params: nx6array
        holds 6 fitted params for n iterations, ordering of params:
        amplitude, x, y, sig_x, sig_y, rotation
    '''
    f, ax = plt.subplots(1, 1, figsize=(8, 6) )
    ax.set_aspect('equal')
    #pxl -> R.A. and Dec
    #scale = np.logspace(np.log10(np.absolute(img.min())), np.log10(img.max()), 5)
    bmin = (header['BMIN'] * u.degree).to(u.mas)
    bmaj =  (header['BMAJ'] * u.degree).to(u.mas)
    
    x_n_pixel = header['NAXIS1']

    x_inc = (header['CDELT1'] * u.degree).to(u.mas) 
    x_ref_value = (header['CRVAL1'] * u.degree).to(u.mas)
    y_n_pixel = header['NAXIS2']


    y_inc = (header['CDELT2'] * u.degree).to(u.mas)
    y_ref_value = (header['CRVAL2'] * u.degree).to(u.mas)

    x_ref_pixel = header['CRPIX1']
    y_ref_pixel = header['CRPIX2']
    
    x = np.linspace((x_ref_pixel * -x_inc).value, ((x_n_pixel - x_ref_pixel) * (x_inc)).value, x_n_pixel)
    y = np.linspace((y_ref_pixel * -y_inc).value, ((y_n_pixel - y_ref_pixel) * y_inc).value, y_n_pixel)
    
    x_b= (x_ref_pixel * -x_inc).value 
    x_m= ((((x_n_pixel - x_ref_pixel) * (x_inc)).value)-x_b)/x_n_pixel
    y_b= (y_ref_pixel * -y_inc).value #-51.30000000000004
    y_m= ((((y_n_pixel - y_ref_pixel) * y_inc).value)-y_b)/y_n_pixel
    
    mask = np.zeros_like(img)
    mask[img > (sigma*rms)]=1
    newmask= mask*img
    cumulativeflux=np.sum(newmask)
    #print('cumulative flux:',cumulativeflux)
    newmask[newmask==0]=-10
    
    
    #im = ax.imshow(img, origin='lower', norm=LogNorm())
    #im = plt.pcolormesh(x,y,newmask,
    #                cmap='gray',
    #                norm = colors.SymLogNorm(linthresh=0.0001, 
    #                                        linscale=0.0001, vmin= img.min(), vmax=img.max()) )
    
    scale = np.logspace(np.log10(np.absolute(img.min())), np.log10(img.max()), 10)
    cont = plt.contour(x,y, img,
                scale,
                colors='k',linewidths = 0.7, linestyles='solid',
                norm = colors.SymLogNorm(linthresh=0.0001, 
                                        linscale=0.0001,vmin= img.min(), vmax=img.max()))
        
    #cbar = plt.colorbar(im, ticks = [(-1e-3, 0, 1e-3, 1e-2)] , label='Flux in Jy/beam')
    
    plt.ylim([-5, 10])
    plt.xlim([5, -18])
    box = AnchoredAuxTransformBox(ax.transData, loc='lower left', frameon = False)
    
    el = Ellipse((0, 0), width=bmin.value, height=bmaj.value, angle = -header['BPA'],color='white',ec='k',alpha =0.75)  
    box.drawing_area.add_artist(el)
    
    ax.add_artist(box)
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.75)
    props2 = dict(boxstyle='round',edgecolor = 'none', facecolor='white', alpha=1)

    ax.text(0.75, 0.95, str(nice_date(epoch)), transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
    color_list = np.array(['#440558ff', '#423c81ff', '#2f6b8dff', '#22958bff',
                       '#9ed93aff', '#9ed93aff', '#b0dd31ff', '#c3df2eff'])
    comp_list = np.array(['C0','C1','C2','C3','C4','C5'])
    
    # add mark and ellipse for every iteration
    i = 0
    for p in params:
        xp = (y_m*p[1])+y_b
        yp = (x_m*p[2])+x_b 
        ax.text(yp+0.7,xp+0.7, '    ', fontsize=10, verticalalignment='top', bbox=props2)
        i+=1
    i=0    
    for p in params:
        xp = (y_m*p[1])+y_b
        yp = (x_m*p[2])+x_b
        #hp = 2 * np.sqrt(2 * np.log(2)) * np.abs(x_m*p[3])
        #wp = 2 * np.sqrt(2 * np.log(2)) * np.abs(x_m*p[3])
        hp = x_m*p[3]
        wp = x_m*p[3]
        ap = p[4]
        
        plt.plot(yp, xp, linestyle='None', marker="*", color=color_list[i], label='Component '+str(i))
        ellipse = Ellipse((yp,
                          xp),
                          height=hp,
                          width=wp,
                          angle=ap,
                          edgecolor=color_list[i],
                          facecolor=color_list[i],
                          alpha=0.6,
                          linewidth=1.5,
                          zorder=5,
                         )
        ax.add_artist(ellipse)
        ax.text(yp+0.7,xp+0.7, str(comp_list[i]), fontsize=10, verticalalignment='top')
        i += 1
    
    f.legend(bbox_to_anchor=(0.28, 0.9))
    plt.xlabel('Relativ R.A. in mas')
    plt.ylabel('Relativ Declination in mas')
    #plt.subplots_adjust(left = 0.35, bottom=0.35)
    f.tight_layout()
    #plt.title('the MINIMALISTICA')
    #plt.show()
    plt.savefig(out, dpi = 300)
    #plt.show()
    plt.clf()
    f.clf()

def epochs_comp_plot(comps, out):
    epoch_list = ['2017_01_28','2017_04_22','2017_06_17','2017_11_18','2018_02_02','2018_10_06','2019_07_19']
    
    color_list = np.array(['#440558ff', '#423c81ff', '#2f6b8dff', '#22958bff',
                       '#9ed93aff', '#9ed93aff', '#b0dd31ff', '#c3df2eff'])
    comp_list = np.array(['C0','C1','C2','C3','C4','C5'])
    
    
    plt.figure(figsize=(5,12))
    ax1 = plt.subplot(711)
    ax1.set_aspect('equal')
    path_comp=comps
    params = pd.read_hdf(path_comp, epoch_list[0]).to_numpy()
    i = 0
    for p in params:
        xp = p[1]
        yp = p[2]
        hp = p[3]
        wp = p[4]
        ap = p[4]

        plt.plot(yp, xp, linestyle='None', marker='*', color=color_list[i], label='Component '+str(i))
        ellipse = Ellipse((yp,
                          xp),
                          height=hp,
                          width=wp,
                          angle=ap,
                          edgecolor=color_list[i],
                          facecolor='none',
                          linewidth=1,
                          zorder=5,
                         )
        ax1.add_artist(ellipse)
        ax1.text(yp-7,xp+7, str(comp_list[i]), fontsize=10, verticalalignment='top')
        i += 1    
        
    plt.ylim(505,540)
    plt.xlim(490,580)
    plt.legend(ncol=2,bbox_to_anchor=(1, 1.8))
    plt.ylabel('Pixel')
    plt.text(495,535,nice_date(epoch_list[0]))
    plt.setp(ax1.get_xticklabels(),visible=True)
    
    ax2 = plt.subplot(712, sharex=ax1,sharey=ax1)
    ax2.set_aspect('equal')
    params = pd.read_hdf(path_comp, epoch_list[1]).to_numpy()
    i = 0
    for p in params:
        xp = p[1]
        yp = p[2]
        hp = p[3]
        wp = p[4]
        ap = p[4]

        plt.plot(yp, xp, linestyle='None', marker='*', color=color_list[i], label='Component '+str(i))
        ellipse = Ellipse((yp,
                          xp),
                          height=hp,
                          width=wp,
                          angle=ap,
                          edgecolor=color_list[i],
                          facecolor='none',
                          linewidth=1,
                          zorder=5,
                         )
        ax2.add_artist(ellipse)
        ax2.text(yp-7,xp+7, str(comp_list[i]), fontsize=10, verticalalignment='top')
        i += 1 
    plt.ylabel('Pixel')
    plt.text(495,535,nice_date(epoch_list[1]))
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3 = plt.subplot(713, sharex=ax1,sharey=ax1)
    ax3.set_aspect('equal')
    params = pd.read_hdf(path_comp, epoch_list[2]).to_numpy()
    i = 0
    for p in params:
        xp = p[1]
        yp = p[2]
        hp = p[3]
        wp = p[4]
        ap = p[4]

        plt.plot(yp, xp, linestyle='None', marker='*', color=color_list[i], label='Component '+str(i))
        ellipse = Ellipse((yp,
                          xp),
                          height=hp,
                          width=wp,
                          angle=ap,
                          edgecolor=color_list[i],
                          facecolor='none',
                          linewidth=1,
                          zorder=5,
                         )
        ax3.add_artist(ellipse)
        ax3.text(yp-7,xp+7, str(comp_list[i]), fontsize=10, verticalalignment='top')
        i += 1  
    plt.ylabel('Pixel')    
    plt.text(495,535,nice_date(epoch_list[2]))
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax4= plt.subplot(714, sharex=ax1,sharey=ax1)
    ax4.set_aspect('equal')
    params = pd.read_hdf(path_comp, epoch_list[3]).to_numpy()
    i = 0
    for p in params:
        xp = p[1]
        yp = p[2]
        hp = p[3]
        wp = p[4]
        ap = p[4]

        plt.plot(yp, xp, linestyle='None', marker='*', color=color_list[i], label='Component '+str(i))
        ellipse = Ellipse((yp,
                          xp),
                          height=hp,
                          width=wp,
                          angle=ap,
                          edgecolor=color_list[i],
                          facecolor='none',
                          linewidth=1,
                          zorder=5,
                         )
        ax4.add_artist(ellipse)
        ax4.text(yp-7,xp+7, str(comp_list[i]), fontsize=10, verticalalignment='top')
        i += 1  
    plt.ylabel('Pixel')    
    plt.text(495,535,nice_date(epoch_list[3]))
    plt.setp(ax4.get_xticklabels(), visible=False)
    
    ax5= plt.subplot(715, sharex=ax1,sharey=ax1)
    ax5.set_aspect('equal')
    params = pd.read_hdf(path_comp, epoch_list[4]).to_numpy()
    i = 0
    for p in params:
        xp = p[1]
        yp = p[2]
        hp = p[3]
        wp = p[4]
        ap = p[4]

        plt.plot(yp, xp, linestyle='None', marker='*', color=color_list[i], label='Component '+str(i))
        ellipse = Ellipse((yp,
                          xp),
                          height=hp,
                          width=wp,
                          angle=ap,
                          edgecolor=color_list[i],
                          facecolor='none',
                          linewidth=1,
                          zorder=5,
                         )
        ax5.add_artist(ellipse)
        ax5.text(yp-7,xp+7, str(comp_list[i]), fontsize=10, verticalalignment='top')
        i += 1  
    plt.ylabel('Pixel')    
    plt.text(495,535,nice_date(epoch_list[4]))   
    plt.setp(ax5.get_xticklabels(), visible=False)   
    
    ax6= plt.subplot(716, sharex=ax1,sharey=ax1)
    ax6.set_aspect('equal')
    params = pd.read_hdf(path_comp, epoch_list[5]).to_numpy()
    i = 0
    for p in params:
        xp = p[1]
        yp = p[2]
        hp = p[3]
        wp = p[4]
        ap = p[4]

        plt.plot(yp, xp, linestyle='None', marker='*', color=color_list[i], label='Component '+str(i))
        ellipse = Ellipse((yp,
                          xp),
                          height=hp,
                          width=wp,
                          angle=ap,
                          edgecolor=color_list[i],
                          facecolor='none',
                          linewidth=1,
                          zorder=5,
                         )
        ax6.add_artist(ellipse)
        ax6.text(yp-7,xp+7, str(comp_list[i]), fontsize=10, verticalalignment='top')
        i += 1  
    plt.ylabel('Pixel')    
    plt.text(495,535,nice_date(epoch_list[5]))   
    plt.setp(ax6.get_xticklabels(), visible=False)    
    
    ax7= plt.subplot(717, sharex=ax1,sharey=ax1)
    ax7.set_aspect('equal')
    params = pd.read_hdf(path_comp, epoch_list[6]).to_numpy()
    i = 0
    for p in params:
        xp = p[1]
        yp = p[2]
        hp = p[3]
        wp = p[4]
        ap = p[4]

        plt.plot(yp, xp, linestyle='None', marker='*', color=color_list[i], label='Component '+str(i))
        ellipse = Ellipse((yp,
                          xp),
                          height=hp,
                          width=wp,
                          angle=ap,
                          edgecolor=color_list[i],
                          facecolor='none',
                          linewidth=1,
                          zorder=5,
                         )
        ax7.add_artist(ellipse)
        ax7.text(yp-7,xp+7, str(comp_list[i]), fontsize=10, verticalalignment='top')
        i += 1  
    plt.ylabel('Pixel')
    plt.xlabel('Pixel') 
    plt.text(495,535,nice_date(epoch_list[6]))
    plt.setp(ax7.get_xticklabels(),visible = True)  
    plt.subplots_adjust(bottom=0.051)
    #plt.show()
    plt.savefig(out,dpi=300)
    plt.clf()
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

def df_transformation(path):
    df= pd.read_hdf(path, 'df')
    sorted_df = df.sort_values(by=['DR'], axis = 1).iloc[:,-1:]
    r = (sorted_df.drop(['path'])).transpose()
    return r

def get_params_all_epochs(path_list, out_dict):
    a = len(path_list)
    df_new = df_transformation(path_list[0])
    for i in np.arange(1,a):
        df_new = pd.concat([df_new,df_transformation(path_list[i])])
    with open(str(out_dict), 'w') as file:
        file.write(df_new.to_latex())    


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
    
    path_to_grid = '/home/yvonne/Schreibtisch/TXS24_06/'
    path_2017_11_18=str(path_to_grid)+'/results_0149+710.u.2017_11_18/MRun_2/data_TXS0149+710.h5'
    path_2017_01_28=str(path_to_grid)+'/results_0149+710.u.2017_01_28/MRun_2/data_TXS0149+710.h5'
    path_2017_04_22=str(path_to_grid)+'/results_0149+710.u.2017_04_22/MRun_2/data_TXS0149+710.h5'
    path_2017_06_17=str(path_to_grid)+'/results_0149+710.u.2017_06_17/MRun_2/data_TXS0149+710.h5'
    path_2018_02_02=str(path_to_grid)+'/results_0149+710.u.2018_02_02/MRun_2/data_TXS0149+710.h5'
    path_2018_10_06=str(path_to_grid)+'/results_0149+710.u.2018_10_06/MRun_2/data_TXS0149+710.h5'
    path_2019_07_19=str(path_to_grid)+'/results_0149+710.u.2019_07_19/MRun_2/data_TXS0149+710.h5'
    
    #test_path ='/home/yvonne/Dokumente/MA/TXS_final_grid/BestEpochsLocal/2019_07_19/2019_07_19-image.fits'
    #with_beam_and_mask(test_path, sourcename, get_rms(fits.open(str(test_path))[0]), 10)
    out = '/home/yvonne/Schreibtisch/Ergebnisse_24_06/'
    get_paramset(str(path_to_grid)+'/results_0149+710.u.2017_11_18/MRun_2/data_TXS0149+710.h5', str(out) +'/IntialParamGrid.txt' )
    print("The inital parametergrid is saved as a latexfile.")
    path_list=[path_2017_11_18, path_2017_01_28, path_2017_04_22,
        path_2017_06_17, path_2018_02_02, path_2018_10_06, path_2019_07_19]
    get_params_all_epochs(path_list, str(out)+'/allbestparam.txt' )
    print('The best parameters of every epoch are written in a text file.' )
    
    sourcename='TXS0149+710'
    beam_list=np.zeros((len(path_list),2))
    out_plot = str(out)+'Plots/'
    i=0
    '''
    for path in path_list:
        epoch = str(Path(path).parent.parent).split('.u.')[-1]
        print('starting with:',epoch)
        #creates a new dict with localy cleaned images
        path_to_fits = run_WSClean(path, epoch)
        path_to_image = str(path_to_fits)+'-image.fits'
        #plots images with and without mask  
        with_beam_and_text(path_to_image, sourcename, get_dyra(path), out_plot)
        with_beam_and_mask(path_to_image, sourcename, get_rms(fits.open(str(path_to_image))[0]), 5, out_plot)
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
        with_beam_and_text(path_to_image, sourcename, get_dynamicrange(path_to_image), str(out_plot)+'/Beam_')
        with_beam_and_mask(path_to_image, sourcename, get_rms(fits.open(str(path_to_image))[0]), 5, str(out_plot)+'/Beam_')
        print('done with:', epoch)
    

    print('Fit those Components and make nice Plots')  
    path_to_same_beam = str(path_to_grid)+'/BestEpochSameBeam'
    path_to_images_wsb =glob.glob(str(path_to_same_beam)+'/**/*-image.fits')  
    for image in path_to_images_wsb:
        epoch= str(Path(image).stem).split('-')[0]
        print("It is Party time for ", epoch)
        f1 = fits.open(image)
        img1 = f1[0].data[0, 0]
        header1 = f1[0].header
        pa1, *_ = fit_jet(img1, 5)
        sparam = sort_params(pa1, header1)
        df2 = pd.DataFrame(sparam)
        df2.to_hdf(str(out)+'/Components.h5',str(epoch))
        plot_results_new_contour_only(img1, header1, sparam, epoch, 5, get_rms(f1[0]), str(out_plot)+'/Comps'+str(Path(image).stem)+'.png')
    '''
    print('Almost there')
    epochs_comp_plot(str(out)+'/Components.h5',str(out_plot)+'/AllComps.png')     
    print('DONE! Wuhuuu!')
if __name__ == '__main__':
    main()
