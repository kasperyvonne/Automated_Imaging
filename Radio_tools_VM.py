#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('agg')
import subprocess
import os
import glob
import numpy as np
import numpy.fft
import pandas as pd
import astropy.units as u
from tqdm import tqdm
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
import multiprocessing as mp
'''

To use this you need to install 
    WSClean: https://sourceforge.net/p/wsclean/wiki/Installation/
    and Casa: https://casa.nrao.edu/casa_obtaining.shtml (Don't forget to set path in bashrc)

Preferably your folder structure should look like this /home/.../Sources/<name of source>/data/*.uvf 

--convertion from fits files to .ms files--
The measurement sets, which are requierd for WSClean, are created from every .uvf file 
and stored within in the data folder in an extra measurement_sets folder.
--run WSClean--
WSClean will be run several times per epoch with your requested parameter combinations. The best
outcomes will be stored in '.../<source_name>/best_epochs/<measurement_name>/MRun<X>_Run_Nr<Y>_<measurement_name>/'.
You could open the resulting <measurement_name>-image.fits for example with ds9 or every other fitsviewer.
--plot results--
For every epoch, there will be a results_<measurement_name> folder with subfolders for the diffrent MRuns.
Within you find a data_<source_name>.hdf file with all parameters and Dynamic Ranges for each run.
Also you will find some plots:  a plot of the best image, that is to say the image with the highest Dynamic Range, 
a plot of the best three images and their corresponding residuals, 
a plot for every paramter distribution.

Questions: yvonne.kasper@tu-dortmund.de
'''

'''
Functions to get the data to desired format
'''
def uvfits_to_ms(input_data, output_data):
    '''
    INPUT_DATA: path to radio image in fits format
    OUTPUT_DATA: path to save casa measurement set
    
    EXAMPLE: uvfits_to_ms('/home/MAXMUSTERMANN/*path to data*/sources/<source_name>/data/<file_name>.uvf',
                            '/home/MAXMUSTERMANN/*Path to data*/sources/<source_name>/data/measurement_sets/<file_name>.ms')
  
    Converts .uvf files to .ms measurement sets

    '''
    in_file = input_data
    out_file = output_data
    parent_path= Path(out_file).parent

    casa = 'casa -c '
    arg = "importuvfits(fitsfile='"+str(in_file)+"'"+", vis="+"'"+str(out_file)+"')"

    
    command = casa + '"' + arg + '"'

    if os.path.exists(str(out_file)):
        print('Measurement set already exists!')
    elif os.path.exists(str(parent_path))==False:
        subprocess.call(["mkdir", str(parent_path) ])
        subprocess.run(command, shell=True)
    else:
        subprocess.run(command, shell=True)


def all_uvf2ms(input_dir):
    '''
    INPUT_DIR: folder with your .uvf files of the source

    EXAMPLE: all_uvf2ms('/home/MAXMUSTERMANN/*path to data*//sources/<source_name>/data')
    
    Converts all .uvf files in the input dictionary to .ms files. Saves .ms files in 
    measurement_sets folder with original filename. 
    
    '''
    data_path = input_dir
    
    files= glob.glob(str(data_path)+'/*.uvf')
    for f in tqdm(files):
        name= Path(f).stem
        uvfits_to_ms(f,str(data_path)+'/measurement_sets/'+str(name)+'.ms')


def all_data2ms(data_path,data_type):
    '''
    INPUT_DIR: folder with your .data_type files of the source
    DATA_TYPE: data type of your data.
    EXAMPLE: all_data2ms('/home/MAXMUSTERMANN/*path to data*/sources/<source_name>/data', 'uvf')
    
    Converts all .data_type files in the input dictionary to .ms files, Saves .ms files in 
    measurement_sets folder with original filename. 
    
    '''
    
    files= glob.glob(str(data_path)+'/*.'+str(data_type))
    for f in tqdm(files):
        name= Path(f).stem
        uvfits_to_ms(f,str(data_path)+'/measurement_sets/'+str(name)+'.ms')

'''
Functions for actual cleaning
'''

def run_wsclean(data_path, model_name, xsize, ysize, scale, niter = 1000000,
                mgain=0.8, auto_thresh=6, gain=0.1, auto_mask=3,
                weight=0, rank_filter=3, multiscale=True,
                scale_bias=0.6, model=False, predict=False,
                ):
    '''
    DATA_PATH: path to radio data in fits format
    MODEL_NAME: name for the cleaned image and models
    XSIZE,YSIZE: desired size of the image in pixle
    SCALE: angular size of a single pixel
    ...

    Example: run_wsclean('/home/MAXMUSTERMANN/*path_to_data*/sources/<source_name>/data/measurement_sets/<file_name>.ms/',
             'name_of_measurement', 1024, 1024, 0.1)

    Runs wsclean with choosen parameters. The cleaning will be stopped after five minutes. The used parameters are saved 
    in an textfile for further juse.

    '''
    parent_path= Path(data_path).parent
    parent2_path= Path(parent_path).parent
    parent3_path= Path(parent2_path).parent
    measurement=Path(data_path).stem

    out_dict = str(parent3_path) +'/epochs/' + measurement 
    out_dict2 = str(out_dict) +'/'+ str(model_name) 

    
    command = 'wsclean '
    if multiscale is True:
        command = command \
        +'-quiet ' \
	    +'-j '+str(1)+' ' \
        +'-multiscale ' \
        +' -size ' + str(xsize) + ' ' + str(ysize) \
        + ' -scale ' + str(scale) + 'masec' \
        + ' -niter ' + str(niter) \
        + ' -mgain ' + str(mgain) \
        + ' -gain ' + str(gain) \
        + ' -auto-mask ' + str(auto_mask) \
        + ' -weight briggs ' + str(weight) \
        + ' -multiscale-scale-bias ' + str(scale_bias) \
        + ' -auto-threshold ' + str(auto_thresh) \
        + ' -name ' + str(out_dict2)+'/'+str(model_name)  \
        + ' ' + str(data_path)
    else:
        command = command \
        +'-quiet ' \
        +' -size ' + str(xsize) + ' ' + str(ysize) \
        + ' -scale ' + str(scale) + 'masec' \
        + ' -niter ' + str(niter) \
        + ' -mgain ' + str(mgain) \
        + ' -gain ' + str(gain) \
        + ' -auto-mask ' + str(auto_mask) \
        + ' -weight briggs ' + str(weight) \
        + ' -auto-threshold ' + str(auto_thresh) \
        + ' -name ' + str(out_dict2)+'/'+str(model_name)  \
        + ' ' + str(data_path) 
    #create folder for wsclean output
 
    if os.path.exists(out_dict2)==False:
       subprocess.call(["mkdir", str(out_dict2)])
    
   # run command (with timelimit of five minutes) and save textfile with parameters
    print('...WSCleaning...')

    try:
        subprocess.run(command, shell=True, timeout = 300)
        f = open(str(out_dict2)+"/" + "parameter_info.txt", "w+")
        f.write(str(xsize) + ' ' + str(ysize)+ ' ' \
                +str(scale)+ ' '  + str(niter)+ ' ' \
                +str(mgain) + ' '+ str(gain)+ ' ' \
                +str(auto_mask)+ ' ' + str(auto_thresh)+ ' ' \
                +str(scale_bias))

    except subprocess.TimeoutExpired:        
        f = open(str(out_dict2)+"/" + "parameter_info.txt", "w+")
        f.write(str(xsize) + ' ' + str(ysize)+ ' ' \
                +str(scale)+ ' '  + str(niter)+ ' ' \
                +str(mgain) + ' '+ str(gain)+ ' ' \
                +str(auto_mask)+ ' ' + str(auto_thresh)+ ' ' \
                +str(scale_bias) +' WSCLEAN DID NOT TERMINATED!')
        print(' WSCLEAN DID NOT TERMINATED!')        
        #There are certainly some more suitable data formats, but this works fore the moment.
   
def run_grid(data_path, model_name, 
            xsize, ysize, scale, 
            niter= 5000,
            mgain_min=0.6875, mgain_max=0.7250, mgain_steps=4,
            gain_min=0.04, gain_max = 0.0675, gain_steps =4,
            auto_mask_min=2, auto_mask_max =4, auto_mask_steps = 4,
            auto_thresh_min=0.2, auto_thresh_max=0.7, auto_thresh_steps = 4,
            weight_min=-1, weight_max =1, weight_step = 3, 
            rank_filter=3, 
            multiscale=True,
            scale_bias_min=0.5,scale_bias_max=0.7,scale_bias_step=4,
            model=False, 
            predict=False, run_nr =0):
            ''' 
            DATA_PATH: path to .ms file
            MODEL_NAME: name for the cleaned image and models
            XSIZE,YSIZE: desired size of the image in pixle
            SCALE: angular size of a single pixel
            PARAMETER_MAX: highest possible value for the parameter
            PARAMETER_MIN: lowest possible value for the parameter
            PARAMETER_step: amout of steps in the value space
            ...
            RUN_NR: number of MRun, accordingly for the inital grid run_nr = 0

            EXAMPLE: run_grid('/home/MAXMUSTERMANN/*path to data*/sources/<source_name>/data/measurement_sets/<file_name>.ms/',
                                'name of measurement',1024,1024,0.1)
            
            Creates an inital parametergrid out of the given values, calls run_wsclean and saves the output in the epochs
            folder. Creates results folder for each epoch with subfolders for the MRuns.
            The parametersets are saved in the data_<source_name>.hdf with key:'ParamSets'
            '''
            parent_path= Path(data_path).parent
            parent2_path= Path(parent_path).parent
            parent3_path= Path(parent2_path).parent
            
            #creation of the paramsets
            mgain = np.linspace(mgain_min, mgain_max, num = mgain_steps)
            gain = np.linspace(gain_min, gain_max, num = gain_steps)
            automask =np.linspace(auto_mask_min, auto_mask_max, num = auto_mask_steps) 
            autothresh = np.linspace(auto_thresh_min, auto_thresh_max, num = auto_mask_steps)
            weight =np.linspace(weight_min, weight_max, num =weight_step)
            scale_bias = np.linspace(scale_bias_min, scale_bias_max, num =scale_bias_step)
            i = 0
            print(model_name)
            print(mgain)
            print(gain)
            print(automask)
            print(autothresh)
            print(weight)
            print(scale_bias)
            
            #create epochs folder
            if os.path.exists(str(parent3_path)+'/epochs')==False:
                subprocess.call(["mkdir", str(parent3_path)+'/epochs/'])
            
            #creates subfolder for every epoch
            if os.path.exists(str(parent3_path)+'/epochs/'+str(model_name))==False:
                subprocess.call(["mkdir", str(parent3_path)+'/epochs/'+str(model_name)])
            
            #creates results folder for each epoch
            if os.path.exists(str(parent3_path)+'/results_'+str(model_name))==False:
                subprocess.call(["mkdir", str(parent3_path)+'/results_'+str(model_name)+'/'])
            
            #creates subfolder for the MRun
            if os.path.exists(str(parent3_path)+'/results_'+str(model_name)+'/MRun_'+str(run_nr))==False:
                subprocess.call(["mkdir", str(parent3_path)+'/results_'+str(model_name)+'/MRun_'+str(run_nr)])   
            
            for m in mgain:
                for am in automask:
                    for a in autothresh:
                        for w in weight:
                            for g in gain:
                                for s in scale_bias:
                                    run_wsclean(str(data_path),
                                                'Run_Nr'+str(i)+'_'+str(model_name),
                                                xsize, ysize, scale,
                                                niter= niter, mgain=m, gain = g,
                                                auto_mask = am, auto_thresh= a,
                                                weight = w, scale_bias = s)
                                    i=i+1
            
            #creates a dataframe with the parametersets and saves them in the hdf with key:'ParamSets'
            df = pd.DataFrame((mgain, gain, automask, autothresh, weight, scale_bias),
                                    index=['mgain', 'gain', 'auto_mask', 'auto_thresh', 'weight', 'scale_bias'])            
            
            df.to_hdf(str(parent3_path)+'/results_'+str(model_name)+'/MRun_'+str(run_nr)+'/data_'+str(Path(parent3_path).stem)+'.h5', 'ParamSets')

def run_another_grid(path_to_hdf, path_to_ms, number_paramspace, xsize =1024, ysize = 1024, scale = 0.1,
                    niter= 5000, rank_filter=3, multiscale=True, model=False, 
                    predict=False, run_nr=0):
    ''' 
    PATH_TO_HDF: path to .hdf file with data of the initial grid
    PATH_TO_MS: path to the measurement_sets
    NUMBER_PARAMSPACE: amout of values in the new paramspace
    XSIZE,YSIZE: desired size of the image in pixle
    SCALE: angular size of a single pixel
    ...
    RUN_NR: number of MRun, accordingly the number of the grid

    EXAMPLE: run_another_grid('/home/MAXMUSTERMANN/*path to data*/TXS/results_0149+710.u.2018_02_02/MRun_1/data.hdf',
                    str(Path(path_to_data).parent)+'/measurement_sets/'+str(name)+'.ms',
                    number_paramspace=3, xsize =1024, ysize = 1024, scale = 0.1,
                    niter= 5000, rank_filter=3, multiscale=True, model=False, 
                    predict=False, run_nr=2) 

    Calls get_new_paramspace(...) to create new parameter spaces for each parameter. Calls run_grid(...) to run another grid.
    '''
    #creating new parameter spaces for each parameter
    number = number_paramspace                
    mgain_min, mgain_max, mgain_steps = get_new_paramspace(path_to_hdf, 'mgain', number, run_nr=run_nr)
    gain_min, gain_max, gain_steps = get_new_paramspace(path_to_hdf, 'gain', number, run_nr=run_nr)
    auto_mask_min, auto_mask_max, auto_mask_steps = get_new_paramspace(path_to_hdf, 'auto_mask', number, run_nr=run_nr)
    auto_thresh_min, auto_thresh_max, auto_thresh_steps = get_new_paramspace(path_to_hdf, 'auto_thresh', number, run_nr=run_nr)
    weight_min, weight_max, weight_step = get_new_paramspace(path_to_hdf, 'weight', number, run_nr=run_nr)
    scale_bias_min,scale_bias_max,scale_bias_step = get_new_paramspace(path_to_hdf, 'scale_bias', number, run_nr=run_nr)

    #run new grid with run_grid
    name=Path(path_to_ms).stem
    run_grid(path_to_ms, name, xsize, ysize, scale, niter,
            mgain_min, mgain_max, mgain_steps,
            gain_min, gain_max, gain_steps,
            auto_mask_min, auto_mask_max, auto_mask_steps,
            auto_thresh_min, auto_thresh_max, auto_thresh_steps,
            weight_min, weight_max, weight_step,
            rank_filter, multiscale,
            scale_bias_min,scale_bias_max,scale_bias_step,
            model, predict,run_nr)


'''
Function to delete unnecessary data
'''

def delete_data(path_to_hdf, value, run_nr):
    '''
    PATH_TO_DATA: Path to data_{}.hdf file
    VALUE: amout of images to preserve

    Example: delete_data(/home/MAXMUSTERMANN/*path to Data*/TXS/results_0149+710.u.2018_02_02/MRun_1/data.hdf', 3, 0)
    
    Saves the best WSClean output.The amount is given by VALUE. 
    Adds the Number of MRun to the data name, to avoid loosing data in later MRuns.
    Copies the best WSClean output to BestEpochs folder. 
    Deletes everything in the epochs folder.

    '''
    #get paths to best data
    df = pd.read_hdf(path_to_hdf, key = 'df')
    sorted_df = df.sort_values(by=['DR'], axis = 1, ascending=False).iloc[:,0:value]
    listed = sorted_df.loc['path',:].tolist()

    parent_path= Path(listed[0]).parent
    parent2_path= Path(parent_path).parent
    parent3_path= Path(parent2_path).parent #../Source/<source_name>/results<...>
    parent4_path= Path(parent3_path).parent #'../Source/<source_name>
    
    #create new folder for best epochs
    if os.path.exists(str(parent4_path)+'/BestEpochs')==False:
       subprocess.call(["mkdir", str(parent4_path)+'/BestEpochs/'])

    #rename and copy best epochs 
    for listedpath in listed:
        old = str(Path(listedpath).parent)
        name = old.split('/')[-1]
        command = 'mv ' + str(Path(listedpath).parent)+'/ '+ str(Path(listedpath).parent.parent)+'/MRun'+str(run_nr)+'_'+str(name)+'/'
        subprocess.call(command, shell=True)
        command2='cp -r ' + str(Path(listedpath).parent.parent)+'/MRun'+str(run_nr)+'_'+str(name)+'/ '+ str(parent4_path)+'/BestEpochs/'
        subprocess.call(command2, shell=True)
    
    #delete everything in the epochs folder
    command3 = 'rm -rf ' +str(parent4_path)+'/epochs/*'
    subprocess.call(command3, shell=True)
'''
Functions to get specific quantities
'''

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
    
def get_dynamic_range(path_to_data):
    '''
    PATH_TO_DATA: path to the *-image.fits file

    Example: get_dynamic_range('/home/MAXMUSTERMANN/*path to data*/IC310/epochs/0313+411.u.2014_05_12/
    Run_Nr67_0313+411.u.2014_05_12/Run_Nr67_0313+411.u.2014_05_12-image.fits')

    Calculates the Dynamic Range of the image. The Dynamic Range is definded as 
    max peak flux/ rms of pic.
    '''

    data_file = fits.open(str(path_to_data))
    data =data_file[0]
    rms= get_rms(data)
    im_data = data.data[0,0,:,:]
    peak = im_data.max()  
    return peak/rms 

def get_params(path_to_data):
    '''
    PATH_TO_DATA: path to the *-image.fits

    Example: get_params(str(path_to_data)+/*-image.fits')

    Function gets all important cleaning parameter and returns them as a list
    '''
    parent = Path(path_to_data).parent
    xsize, ysize, scale, niter, mgain, gain, auto_mask, auto_thresh, scale_bias  = np.genfromtxt(str(parent)+'/parameter_info.txt', unpack=True)
    data_file = fits.open(str(path_to_data))
    data =data_file[0]
    weight_string =  data.header['WSCWEIGH']
    weight = float(weight_string.replace("Briggs'(", "").replace(')',''))
    DRange = get_dynamic_range(path_to_data)
    paramslist = (xsize, ysize, scale, niter, mgain, gain, auto_mask, auto_thresh, weight, scale_bias,  DRange, str(path_to_data))
    return paramslist

def get_all_parameters(path_to_data, run_nr=0):
    '''
    PATH_TO_DATA: Path to the folder in the epoch folder : .../epochs/name
    RUN_NR: number of MRun

    Example: get_all_parameters('/home/MAXMUSTERMANN/*path to data*/IC310/epochs/0313+411.u.2014_05_12')
    
    Collects all parameter of evey image in given data and stores them
    in an dataframe with image names as columns and parameter as row index:

    Output Example:
                            Run_Nr3156_0149+710.u.2017_01_28-image
    xsize                      1024
    ysize                      1024
    scale                       0.1
    niter                    500000
    mgain                     0.952
    gain                       0.08
    auto_mask                1.8725
    auto_thresh                3.75
    weight            Briggs'(-0.5)
    ...                         ...   
    
    '''
    im = glob.glob(str(path_to_data)+'/**/*-image.fits')
    dicti = {}
    for i in tqdm(im) :
        dicti[str(Path(i).stem)] = (get_params(i))

    df = pd.DataFrame(dicti,index=['xsize', 'ysize', 'scale', 'niter', 'mgain','gain', 'auto_mask',
                                'auto_thresh', 'weight', 'scale_bias', 'DR', 'path'])
    df = df.dropna(axis='columns')                   
    #print(df.sort_values(by=['DR'], axis = 1).iloc[:,0:3]) #first 3
    #print(df.sort_values(by=['DR'], axis = 1).iloc[:,-3:]) #last 3
    name = im[0].split('/')[-5]
    model_name= str(path_to_data).split('/')[-1]
    print(im)
    df.to_hdf(str(Path(path_to_data).parent.parent)+'/results_'+str(model_name)+'/MRun_'+str(run_nr)+'/data_'+str(name)+'.h5', 'df')
    return(df)

def get_new_paramspace(path_to_hdf, param, number, run_nr):
    '''
    PATH_TO_HDF: Path to the .hdf file with the results of the previous grid
    PARAM: The Parameter to find a new value space for
    NUMBER: Amount of values for the new space
    RUN_NR: Number of the current MRun

    Example: get_new_paramspace('/home/MAXMUSTERMANN/*path to data*/IC310/results_0313+411.u.2014_05_12/MRun_1/data_IC310.h5', 'mgain', number=3, run_nr=2)

    Creates a new parameter space for the next grid, based on the parameters of the best image in
    the previous run. 
    If there is only one  value given, this is value is used again as the only one for the next grid.
    Some parameters (weight and scale_bias) can be assuemed to be satisfactorily determined
    after two grid runs. To save computational time, these will be set to the best value after 
    two runs.   
    '''
    #get old paramsets and get rid of unused data
    ParamSets = pd.read_hdf(path_to_hdf, 'ParamSets')
    df = pd.read_hdf(path_to_hdf, 'df')
    df_new = df.transpose()
    df_short = df_new.drop(['xsize','ysize','scale','niter','path'], axis =1)    

    #get parameter value of the best image
    df_DR = df_short.sort_values(by=['DR'], axis = 0, ascending=False)
    bestDRvalues = df_DR.iloc[0]
    bestParam = bestDRvalues.loc[param]

    #if only one value is given, keep it
    if len(ParamSets.loc[param].dropna())==1:
        return(bestParam, bestParam, 1)
    
    #dont change weight and scalebias values after third MRun
    if run_nr >= 2 and param == 'weight':
        return(bestParam, bestParam, 1)
    if run_nr >= 2 and param == 'scale_bias':
        return(bestParam, bestParam, 1) 

    #get position of best value
    ParamRange = ParamSets.loc[param].values[:]
    spacing = np.abs(ParamRange[1]-ParamRange[0])
    mask = np.zeros_like(ParamRange)
    mask[ParamRange == bestParam]=1

    # weight needs to be between -1 and 1
    if param =='weight':
        if mask[0]==1: 
            end = 0
            space = np.abs((end-bestParam)/number)
            stop = end-space
            return (bestParam, stop,  number)

        elif mask[-1]==1:    
            end = 0
            space = np.abs((end-bestParam)/number)
            start = end+space
            return (start, bestParam, number)

        else :
            newSpacing = spacing/(number-(number/2))
            start= bestParam - ((number-1)/2)*newSpacing
            if start < -1: 
                start = -1
            stop= bestParam + ((number-1)/2)*newSpacing
            if stop >1:
                stop=1
            return (start, stop, number)

    #gain and mgain need to be >0
    if param == 'gain' or 'mgain':
        if mask[0]==1: 
            start= bestParam-(spacing/(2*run_nr)*number)
            if start <0:
                start =0.000001
            return (start, bestParam, number)

        elif mask[-1]==1:    
            end= bestParam+(spacing/(2*run_nr)*(number))
            if end <1:
                end =0.99999
            return (bestParam,end, number)

        else :
            newSpacing = spacing/(number-(number/2))
            start= bestParam - ((number-1)/2)*newSpacing
            stop= bestParam + ((number-1)/2)*newSpacing
            if start<0 :
                start = 0.000001
            if stop >1 :
                stop = 0.99999    
            return (start, stop, number)     

    # everything else
    # if best value is on the far left
    if mask[0]==1: 
        start= bestParam-(spacing/(2*run_nr)*number)
        return (start, bestParam, number)

    #if best value is on the far right
    elif mask[-1]==1:    
        end= bestParam+(spacing/(2*run_nr)*(number))
        return (bestParam,end, number)

    #if value is not at the edge
    else :
        newSpacing = spacing/(number-(number/2))
        start= bestParam - ((number-1)/2)*newSpacing
        stop= bestParam + ((number-1)/2)*newSpacing
        return (start, stop, number)

'''
Functions to plot data
'''
  

def get_best_pictures(path_to_hdf, sourcename, run_nr=0):
    '''
    PATH_TO_HDF: Path to .hdf file with results of the grid.
    RUN_NR: Number of MRun. Used for output path.
    SOURCENAME: Name of the Source. Used for the title.

    Creates plot with six subplots. Three best clean images with corresponding residuals.
    Coordinates are in pixel and (yet) without a beam.
    '''

    df = pd.read_hdf(path_to_hdf, 'df')
    sorted_df = df.sort_values(by=['DR'], axis = 1).iloc[:,-3:]
    listed = sorted_df.loc['path',:].tolist()
    Drange = sorted_df.loc['DR',:].tolist()
    #print(Drange)
    
    im1 = fits.open(listed[0])[0].data[0,0,:,:]
    im2 = fits.open(listed[1])[0].data[0,0,:,:]
    im3 = fits.open(listed[2])[0].data[0,0,:,:]

    name1 = Path(listed[0]).stem
    parent1 = Path(listed[0]).parent
    path_to_resi1 = str(parent1) +'/'+ str(name1.split('-')[0])+'-residual.fits' 
    name2 = Path(listed[1]).stem
    parent2 = Path(listed[1]).parent
    path_to_resi2 = str(parent2) +'/'+ str(name2.split('-')[0])+'-residual.fits' 
    name3 = Path(listed[2]).stem
    parent3 = Path(listed[2]).parent
    path_to_resi3 = str(parent3) +'/'+ str(name3.split('-')[0])+'-residual.fits' 
    print(name1)
    print(name2)
    print(name3)
    epoch = name1.split('-')[0].split('.u.')[-1]
    nice_year=str(epoch).split('_')[0]
    nice_month=str(epoch).split('_')[1]
    nice_day=str(epoch).split('_')[2]
    re1 = fits.open(path_to_resi1)[0].data[0,0,:,:]
    re2 = fits.open(path_to_resi2)[0].data[0,0,:,:]
    re3 = fits.open(path_to_resi3)[0].data[0,0,:,:]

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=[12, 9])
    fig.suptitle('Best three images of '+str(sourcename) + ', Epoch: '+str(nice_day)+'.'+str(nice_month)+'.'+str(nice_year), fontsize=20)

    ax1.set_title('DR='+ str(Drange[0]))
    ax1.set_xlabel('Pixel')
    ax1.set_ylabel('Pixel')
    plt.ylim([300, 500])
    plt.xlim([500, 300])
    plt1 = ax1.imshow(im1,origin='lower',
                        cmap='hot',
                        norm = colors.SymLogNorm(linthresh=0.0001, 
                                                linscale=0.0001, vmin= -0.001345288, vmax=0.09435418) )
    fig.colorbar(plt1, ax=ax1,  ticks = [(-1e-3, 0, 1e-3, 1e-2)] , fraction=0.046, pad=0.04, label='Flux in Jy/beam')
    
    ax2.set_title('DR='+ str(Drange[1]))
    ax2.set_xlabel('Pixel')
    ax2.set_ylabel('Pixel')
    plt.ylim([300, 500])
    plt.xlim([500, 300])
    plt2 = ax2.imshow(im2,origin='lower',
                        cmap='hot',
                        norm = colors.SymLogNorm(linthresh=0.0001, 
                                                linscale=0.0001, vmin= -0.001345288, vmax=0.09435418) )
    fig.colorbar(plt2, ax=ax2, ticks = [(-1e-3, 0, 1e-3, 1e-2)] , fraction=0.046, pad=0.04, label='Flux in Jy/beam')

    ax3.set_title('DR='+ str(Drange[2]))
    ax3.set_xlabel('Pixel')
    ax3.set_ylabel('Pixel')
    plt.ylim([300, 500])
    plt.xlim([500, 300])
    plt3 = ax3.imshow(im3,origin='lower',
                        cmap='hot',
                        norm = colors.SymLogNorm(linthresh=0.0001, 
                                                linscale=0.0001, vmin= -0.001345288, vmax=0.09435418) )
    fig.colorbar(plt3, ax=ax3,  ticks = [(-1e-3, 0, 1e-3, 1e-2)] ,fraction=0.046, pad=0.04, label='Flux in Jy/beam')

    ax4.set_title('Corresponding Residual')
    ax4.set_xlabel('Pixel')
    ax4.set_ylabel('Pixel')
    plt.ylim([300, 500])
    plt.xlim([500, 300])
    plt4 = ax4.imshow(re1,origin='lower',
                        cmap='hot',
                        norm = colors.SymLogNorm(linthresh=0.0001, 
                                                linscale=0.0001, vmin= -0.001345288, vmax=0.09435418) )
    fig.colorbar(plt4, ax=ax4, ticks = [(-1e-3, 0, 1e-3, 1e-2)] , fraction=0.046, pad=0.04, label='Flux in Jy/beam')

    ax5.set_title('Corresponding Residual')
    ax5.set_xlabel('Pixel')
    ax5.set_ylabel('Pixel')
    plt.ylim([300, 500])
    plt.xlim([500, 300])
    plt5 = ax5.imshow(re2,origin='lower',
                        cmap='hot',
                        norm = colors.SymLogNorm(linthresh=0.0001, 
                                                linscale=0.0001, vmin= -0.001345288, vmax=0.09435418) )
    fig.colorbar(plt5, ax=ax5, ticks = [(-1e-3, 0, 1e-3, 1e-2)] , fraction=0.046, pad=0.04, label='Flux in Jy/beam')

    ax6.set_title('Corresponding Residual')
    ax6.set_xlabel('Pixel')
    ax6.set_ylabel('Pixel')
    plt.ylim([300, 500])
    plt.xlim([500, 300])
    plt6 = ax6.imshow(re3,origin='lower',
                        cmap='hot',
                        norm = colors.SymLogNorm(linthresh=0.0001, 
                                                linscale=0.0001, vmin= -0.001345288, vmax=0.09435418) )
    fig.colorbar(plt6, ax=ax6, ticks = [(-1e-3, 0, 1e-3, 1e-2)] , fraction=0.046, pad=0.04, label='Flux in Jy/beam')
    plt.tight_layout()
    
    plt.savefig(str(Path(path_to_hdf).parent)+'/Best3_MRun_'+str(run_nr)+'.png')
    plt.clf()	

def get_nicest_picture(path_to_hdf, sourcename, run_nr):
    '''
    PATH_TO_HDF: Path to .hdf file with results of the grid.
    RUN_NR: Number of MRun. Used for the title and output path.
    SOURCENAME: Name of the Source. Used for the title.

    Creates a nice plot in relativ R.A. and relativ Declination (both in mas) 
    coordinates with five contourlines as logspace between min and max of data.
    '''
    df = pd.read_hdf(path_to_hdf, 'df')
    sorted_df = df.sort_values(by=['DR'], axis = 1).iloc[:,-3:]
    listed = sorted_df.loc['path',:].tolist()
    Drange = sorted_df.loc['DR',:].tolist()
    
    image = fits.open(listed[0])[0]

    im1 = image.data[0,0,:,:]
    header = image.header
    name1 = Path(listed[0]).stem
    epoch = name1.split('-')[0].split('.u.')[-1]
    
  
    scale = np.logspace(np.log10(np.absolute(im1.min())), np.log10(im1.max()), 5)

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
    fig, ax = plt.subplots(figsize=(8,6))

    im = plt.pcolormesh(x,y,im1,
                    cmap='hot',
                    norm = colors.SymLogNorm(linthresh=0.0001, 
                                            linscale=0.0001, vmin= -0.001345288, vmax=0.09435418) )
        
    cont = plt.contour(x,y, im1,
                scale,
                colors='white',linewidths = 1, linestyles='solid',
                norm = colors.SymLogNorm(linthresh=0.0001, 
                                        linscale=0.0001, vmin= -0.001345288, vmax=0.09435418))
        
    #cbar = plt.colorbar(im, ticks = [-0.001, 0, 0.001, 0.01], label='Flux in Jy/beam')
    cbar = plt.colorbar(im, ticks = [(-1e-3, 0, 1e-3, 1e-2)] , label='Flux in Jy/beam')
    plt.ylim([-15, 20])
    plt.xlim([15, -20])
    nice_year=str(epoch).split('_')[0]
    nice_month=str(epoch).split('_')[1]
    nice_day=str(epoch).split('_')[2]

    plt.xlabel('Relativ R.A. in mas')
    plt.ylabel('Relativ Declination in mas')
    plt.title(str(sourcename)+' '+'Epoch: '+str(nice_day)+'.'+str(nice_month)+'.'+str(nice_year)+' MRun '+str(run_nr))
   
    box = AnchoredAuxTransformBox(ax.transData, loc='lower left', frameon = False)
    el = Ellipse((0, 0), width=bmin.value, height=bmaj.value, angle = -header['BPA'],color='white',ec='k',alpha =0.75)  
    box.drawing_area.add_artist(el)

    ax.add_artist(box)
    #cbar.ax.set_ylabel('Flux in Jy/beam')
    plt.tight_layout()
    plt.savefig(str(Path(path_to_hdf).parent)+'/BestOne_MRun_'+str(run_nr)+'.png')
    plt.clf()

def plot_parameterplot(path_to_hdf, Parameter):
    '''
    PATH_TO_HDF: Path to the .hdf file with the results of the previous grid
    PARAMETER: The Parameter to find a new value space for

    Example: plot_parameterplot('/home/MAXMUSTERMANN/*path to data*/IC310/results_0313+411.u.2014_05_12/MRun_1/data_IC310.h5', 'mgain')

    Plots an histogram of the distribution of values in the parameterspace.
    Shows how many images are within a certain range of Dynamic Ranges when beeing cleand with a certain value.
    '''
    #get paramspace and dataframe with results
    ParamSets = pd.read_hdf(path_to_hdf, 'ParamSets')
    df = pd.read_hdf(path_to_hdf, 'df')
    df_new = df.transpose()
    df_short = df_new.drop(['xsize','ysize','scale','niter','path'], axis =1)    

    #sort by DR
    df_DR = df_short.sort_values(by=['DR'], axis = 0, ascending=False)
    bestDRvalue = df_DR['DR'].values[0]
    
    Drange2= df_DR.loc[:,'DR'].tolist()

    # find index where the DR gets less than xy% of best DR
    Drange = np.array(Drange2)
    ind1 = len(Drange[Drange>0.9*bestDRvalue])
    ind2 = len(Drange[Drange>0.5*bestDRvalue])
    ind3 = len(Drange[Drange>0.3*bestDRvalue])
    
    #fill arrays to hist      
    a = np.array([df_DR.iloc[0][str(Parameter)]])
    b = df_DR.iloc[1:ind1][str(Parameter)].values[:]
    c = df_DR.iloc[ind1:ind2][str(Parameter)].values[:]
    d = df_DR.iloc[ind2:ind3][str(Parameter)].values[:]

    space = ParamSets.loc[str(Parameter)].dropna().values[:]

    #calculate bins
    if len(space) == 1:
        bins = np.array([space[0]-space[0]/2, space[0], space[0]+space[0]/2])
        bins = np.sort(bins)
        center = (bins[:-1]+bins[1:])/2
        start = space[0]-space[0]/2 
        stop = space[0]+space[0]/2

    else:    
        start = space[0]+(space[0]-space[1])
        stop = space[-1]-(space[0]-space[1])
        #print(space)
        bins =  np.linspace(start, stop, len(space)+2)
        center = (bins[:-1]+bins[1:])/2

    newbins = np.sort(center)

    #plot
    plt.figure()
    plt.hist([a,b,c,d],newbins,ec='k', histtype='barstacked', lw=1,stacked=True, color=['#80BA26', '#6B6B6B', '#919191', '#BFBFBF'])
    plt.ylim(0.1, 1000)
    plt.xlim(start,stop)
    plt.xticks(space)
    plt.yscale('log')
    plt.xlabel(str(Parameter))
    plt.ylabel('Number of Pictures')
    plt.tight_layout()
    plt.savefig(str(Path(path_to_hdf).parent)+'/Parameterplot_'+str(Parameter)+'.pdf')
    plt.clf()
    plt.close('all')

def plot_all_parameterplots(path_to_hdf):
    '''
    PATH_TO_HDF: Path to the .hdf file with the results of the previous grid
    PARAMETER: The Parameter to find a new value space for

    Example: plot_all_parameterplots('/home/MAXMUSTERMANN/*path to data*/IC310/results_0313+411.u.2014_05_12/MRun_1/data_IC310.h5')

    Calls plot_parameterplot(...) for every varied parameter
    '''
    paramSets = pd.read_hdf(path_to_hdf, 'ParamSets')
    params = paramSets.index.values
    for param in params:
        plot_parameterplot(path_to_hdf, param)
        print(param, 'done')

'''
DO EVERYTHING
'''
def do_everything(path_to_data):
    '''
    PATH_TO_DATA: path to the .uvf files

    Example: do_everything('/home/MAXMUSTERMANN/*path to data*/IC310/data/0313+411.u.2014_05_12.uvf')

    There are things you need to change in this function to customize your grid:
    number_MRuns: number of grids that are processed consecutively
    run_grid(...): parameter space for the inital grid

    WSClean will be run several times, with the every parameter combination prepared by the grid.
    This will be done as often as you like, with a automaticaly adjusted parameter space for the grid. 
    '''
    #choose your desired amout of MRuns
    number_MRuns = 3

    name = Path(path_to_data).stem #0313+411.u.2013_05_05
    path_to_ms= str(Path(path_to_data).parent)+'/measurement_sets/'+str(name)+'.ms'
    uvfits_to_ms(path_to_data, path_to_ms)
    source = str(path_to_ms).split('/')[-4]
    sourcename = Path(path_to_data).parent.parent.stem
    run_nr = 0

    #choose the initial parameters and the parameter spaces for your grid
    run_grid(path_to_ms, name, 
        xsize=1024, ysize=1024, scale=0.1, 
        niter= 5000,
        mgain_min=0.5, mgain_max=0.95, mgain_steps=3,
        gain_min=0.05, gain_max = 0.1, gain_steps =3,
        auto_mask_min=2, auto_mask_max =3, auto_mask_steps = 3,
        auto_thresh_min=0.5 ,auto_thresh_max=4, auto_thresh_steps = 3,
        weight_min=-1, weight_max =1, weight_step = 3, 
        rank_filter=3, 
        multiscale=True,
        scale_bias_min=0.5,scale_bias_max=0.7,scale_bias_step=3,
        model=False, 
        predict=False, run_nr =run_nr)  
    path_to_epochs= str(Path(path_to_ms).parent.parent.parent)+'/epochs'  
    get_all_parameters(str(path_to_epochs)+'/'+str(name), run_nr=run_nr)
    path_to_hdf= str(Path(path_to_ms).parent.parent.parent)+'/results_'+str(name)+'/MRun_'+str(run_nr)+'/data_'+str(source)+'.h5'
    get_best_pictures(path_to_hdf, sourcename, run_nr=run_nr)
    get_nicest_picture(path_to_hdf, sourcename, run_nr=run_nr)
    plot_all_parameterplots(path_to_hdf)
    delete_data(path_to_hdf, 3, 0)
    
    for run_nr in np.linspace(1, number_MRuns-1, num=number_MRuns-1):
        run_nr = int(run_nr)
        run_another_grid(path_to_hdf, path_to_ms, number_paramspace=3, xsize =1024, ysize = 1024, scale = 0.1,
                    niter= 5000, rank_filter=3, multiscale=True, model=False, 
                    predict=False, run_nr=run_nr)           
        path_to_epochs= str(Path(path_to_ms).parent.parent.parent)+'/epochs'   
        get_all_parameters(str(path_to_epochs)+'/'+str(name), run_nr=run_nr)
        path_to_hdf= str(Path(path_to_ms).parent.parent.parent)+'/results_'+str(name)+'/MRun_'+str(run_nr)+'/data_'+str(source)+'.h5'
        get_best_pictures(path_to_hdf, sourcename, run_nr=run_nr) # maybe get sourcename out of the data.hdf 
        get_nicest_picture(path_to_hdf, sourcename, run_nr=run_nr)
        plot_all_parameterplots(path_to_hdf)
        delete_data(path_to_hdf, 3, run_nr)
        
def main():
    '''
    You need to change the path to your data.
    The epochs in '.../sources/<source_name>/data/' will be done by multiprocessing. WSClean uses one thread per epoch.
    '''
    path_to_data = '/net/big-tank/POOL/users/ykasper/sources/TXS0149+710/data'
    msets = glob.glob(str(path_to_data)+'/*.uvf')
    nepochs = len(msets)
    p = mp.Pool(processes= nepochs, maxtasksperchild =1)
    p.map(do_everything, msets)

if __name__ == '__main__':
    main()

