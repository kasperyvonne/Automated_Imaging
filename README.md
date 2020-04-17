# How to use my Radio Tools:
## Requirements
Before you can use my Radio Tools you need to install WSClean (https://arxiv.org/abs/1407.1943) and Casa. <br>

1. WSClean:<br>
     For Ubuntu and Debian it can be installed with **sudo apt-get install wsclean** or follow the instructions on <br> https://sourceforge.net/p/wsclean/wiki/Installation/.
2. Casa: <br>
    Download the current release of the code for your specific operating system on <br>
    https://casa.nrao.edu/casa_obtaining.shtml <br>
    and follow the installation instructions .

## Get your data    
1. Download your Visibility Data. These may be .uvf or .UVP or similar files.<br>
   In my case (http://www.astro.purdue.edu/MOJAVE/sourcepages/0313+411.shtml) there are eight epochs as .uvf files.
2. You may want to look at several different sources, therefore I recommend creating a folder called "sources" and another with the name of your source within the "sources" folder. In this folder you save the data files inside another folder called  "data".<br> The structure looks like this: '.../sources/<source_name>/data/<files.uvf>'.<br> This may sound unnecessary or too complicated but provides a clearly arranged structure for further use.

## Choose the script
In the **Radio_tools_VM.py** script, multiprocessing is used to clean the different epochs in your data folder simultaneously. Since WSClean gets one cpu core for cleaning, you need at least the same number of cores as you have epochs in your data folder.<br>
If you want to use this tool on your local machine use: **Radio_tools_local.py**. The epochs in your data folder will be cleaned one by one. This will take much longer than the multiprocessing way.

## Adjust the script
You need to adjust a few things in the script. 
Of course, you need to adjust the path to your data in the *main()* function and in the **do_everything(...)** function you could use the default values or change things up for your case. 
