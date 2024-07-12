
import os
import h5py
import CareyBehavior
import CareyLib
import ibllib.io.spikeglx as npx
import time
import numpy as np
import psutil
import sys
from matplotlib import pyplot as plt
import tkinter as tk
from natsort import natsorted, ns
import pandas as pd
import simpleaudio as sa
from numpy import arange, sin, pi
import random
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import gc
from matplotlib.widgets import Button
from sys import platform as sysplt
import datetime
import CareyUtils
import CareyFileProcessing
import scipy
from sklearn.linear_model import LinearRegression
import ibllib
from scipy.signal import savgol_filter
from numba import jit
import warnings
from tqdm import tqdm

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

class ExperimentalFiles:

    RAW_FILES_DIR = 'TM RAW FILES'

    TRACKING_FILES_DIR = 'TM TRACKING FILES'

    TRIAL_FILES_DIR = 'TM TRIAL FILES'

    STRIDE_FILES_DIR = 'TM STRIDE FILES'

    SESSION_FILES_DIR = 'TM SESSION FILES'

    PLOTTING_FILES_DIR = 'TM PLOTTING FILES'

    TM_EXTERNALLY_SORTED_EPHYS = 'TM EXTERNALLY SORTED EPHYS';

    TM_EPHYS_NEURON_FILES_DIR = 'TM EPHYS NEURON FILES' # ephys neuron folder

    TM_STRIDE_FILES_DIR = 'TM STRIDE DATA FILES' # stride data folder

    TM_PROCESSED_FILES_DIR = 'TM PROCESSED FILES' # Processed behavioral folder

    LOGFOLDER = 'LOGFOLDER'

    exp_folders = ['TM RAW FILES', 'TM TRACKING FILES', 'TM TRIAL FILES',
        'TM STRIDE FILES', 'TM SESSION FILES', 'TM EXTERNALLY SORTED EPHYS',
        'PROCESSING_FILES', 'LOGFOLDER']


    @staticmethod
    def getExperimentalDirectories(input_path):
        # returns the experimental root directory and the experiment
        # directory name from input path
        dirs = [];
        fidx = 0;
        exp_name = [];
        animal_name = [];
        session = [];

        exp_folders = ExperimentalFiles.exp_folders

        # check in for the experimental folder in input_path
        fidx = -1
        idx = 0
        for f in exp_folders:
            +idx
            if input_path.find(f) > 1:
                fidx = idx
                break

        if fidx < 0:
            return 0
        else:
            #print('exp dir detected')
            pass

        # get the root directory from the full dir
        fullpathfolders = input_path.split(os.path.sep)
        if sysplt=="linux" or sysplt=="linux2" or sysplt=='darwin':
            # split the folders / parts
            fpf = ExperimentalFiles.fixLinuxSplitPath(fullpathfolders)
        idx = -1
        for dd in range(len(fpf)):
            ++idx
            if fpf[dd].find(f)>=0:
                exp_dir = fpf[dd]
                break

        root_path = ExperimentalFiles.pathjoin(fullpathfolders[0:idx-1])
        raw_path =      os.path.join(root_path, \
                                       ExperimentalFiles.RAW_FILES_DIR, \
                                           fpf[dd+1])
        tracking_path = os.path.join(root_path, \
                                       ExperimentalFiles.TRACKING_FILES_DIR, \
                                           fpf[dd+1])
        trial_path =    os.path.join(root_path, \
                                       ExperimentalFiles.TRIAL_FILES_DIR, \
                                           fpf[dd+1])
        stride_path =   os.path.join(root_path, \
                                       ExperimentalFiles.STRIDE_FILES_DIR, \
                                           fpf[dd+1])
        session_path =  os.path.join(root_path, \
                                       ExperimentalFiles.SESSION_FILES_DIR, \
                                           fpf[dd+1])
        plotting_path = os.path.join(root_path, \
                                       ExperimentalFiles.PLOTTING_FILES_DIR, \
                                           fpf[dd+1])
        suffix_path =   fpf[dd+1]
        
        exp_input_path = input_path
        
        directory_type = exp_dir
            
            
        exp_dirs = {'root_path':        root_path, \
                    'raw_path':         raw_path, \
                    'tracking_path':    tracking_path, \
                    'trial_path':       trial_path, \
                    'stride_path':      stride_path, \
                    'session_path':     session_path, \
                    'plotting_path':    plotting_path, \
                    'suffix_path':      suffix_path, \
                    'exp_input_path':   exp_input_path, \
                    'directory_type':   directory_type}
        
        return exp_dirs

    def createExperimentalFileTree(input_path, writelog=1):
        for ii in range(len(ExperimentalFiles.exp_folders)):
            folder_to_create = os.path.join(input_path,
                                            ExperimentalFiles.exp_folders[ii])
            os.mkdir(folder_to_create)

        if writelog:
            this_message = "created experimental folders"
            logWriter(os.path.join(input_path,
                                   ExperimentalFiles.LOGFOLDER,
                                   "logfile.txt"),

                                    this_message)


    def fixLinuxSplitPath(fullpathfolders):
        if fullpathfolders[0]=='':
            fullpathfolders[0]='/'
        return fullpathfolders

    def pathsplit(in_path):
        # recursively splits the path given
        path_list = []
        clipped_path = in_path
        a = 'mlem'
        b = 'mlem'
        # make this detection for linux
        
        if sysplt=="linux" or sysplt=="linux2" or sysplt=='darwin':
            # all absolute paths start with a forward slash
            volmatch = ''
        else:
            # under windows this appears as 'c:'
            pass
        
        while b is not '':
            a, b = os.path.split(clipped_path)
            clipped_path = a
            path_list.insert(0,b)
            print(a + '||' + b)
            input()
        
        path_list.insert(0,a)

        return path_list

    def pathjoin(in_list):
        out_path = ''
        for ii in range(len(in_list)):
            out_path = os.path.join(out_path, in_list[ii])
            
        return out_path

class CareyNPXReader:
    # class for reading neuropixels data
    #
    # basic usage:
    # from CareyLib import *
    # cpr = CareyNPXReader
    # cpr.datadir = '/path/to/my/spikeglx/data'
    #
    #
    #
    #
    #
    def __init__(self, datadir=''):

        self.datadir      = datadir # directory where the acquisition data is
        self._apbin       = ''
        self._lfbin       = ''
        self._apmeta      = ''
        self._lfpmeta     = ''
        self._nidqbin     = ''
        self._nidqmeta    = ''
        self.split_channel_dir = '' # where individual channels will be saved to

        if os.path.isdir(datadir):
            self.findDataFiles()
    def findDataFiles(self):
        # looks for ap and lfp bin files in datadir
        # confirm datadir is a dir
        if os.path.isdir(self.datadir):
            # look for bin and meta files
            files = os.listdir(self.datadir)
            for f in files:
                if f[-7:].find('.ap.bin') > -1:
                    self._apbin = os.path.join(self.datadir, f)
                elif f[-7:].find('lf.bin') > -1:
                    self._lfbin = os.path.join(self.datadir, f)
                elif f[-7:].find('ap.meta') > -1:
                    self._apmeta = os.path.join(self.datadir, f)
                elif f[-7:].find('lf.meta') > -1:
                    self._lfmeta = os.path.join(self.datadir, f)
                elif f[-9:].find('nidq.bin') > -1:
                    self._nidqbin   = os.path.join(self.datadir, f)
                elif f[-9:].find('nidq.meta') > -1:
                    self._nidqmeta  = os.path.join(self.datadir, f)
        else:
            print('data directory wrongly specified')
    def saveSplitChannels(self, ch, verbose=0):

        if not os.path.isdir(self.datadir):
            print(str(self.__class__) + ' : define datadir')
            return

        spikereader = npx.Reader(self._apbin)
        lfpreader   = npx.Reader(self._lfbin)

        if not os.path.isdir(self.split_channel_dir):
            print(str(self.__class__) + ' : define directory for saving ' +
                              'channels in new format (split_channel_dir)')
            return




        # TODO: get expected file size, measure ram,
        # separate reading / mapping process from writing to hard drive
        # writing should iterate over each channel


        for c in ch:
            # keep track of time spent reading out each channel
            tic = time.time()

            # read _apbin
            apdata = spikereader.read_samples(0,
                                           int(spikereader.data.shape[0]-1),
                                           channels=ch)
            # read _lfbin
            lfdata = lfpreader.read_samples(0,
                                           int(lfpreader.data.shape[0]-1),
                                           channels=ch)
            # build full file locations for saving channels

            # TODO: feed function handle for building filename
            ap_file_name =  os.path.splitext(
                                os.path.basename(self._apbin))[0]
            ap_file_name = ap_file_name + '_ch_' + str(c)

            lfp_file_name = os.path.splitext(
                                os.path.basename(self._lfbin))[0]
            lfp_file_name = lfp_file_name + '_ch_' + str(c)


            ap_file_loc =   os.path.join(   self.split_channel_dir,
                                            ap_file_name )
            lfp_file_loc =   os.path.join(  self.split_channel_dir,
                                            lfp_file_name )

            # save data
            np.save(ap_file_loc, apdata[0])
            np.save(lfp_file_loc, lfdata[0])

            # clean up before next interation
            apdata = None
            lfdata = None

            ttt = time.time()
            toc = ttt-tic;
            # ---- print out time
            if verbose==1:
                print("Read and saved channel " + str(c) + "in " + str(toc) + "s")
    def splitAllChannels(self, verbose=0, memory='default', mempc=0.7, chsel='all'):

        delim = '\n'

        bits_ = 32

        # memory dictactes how much memory this function is allowed to use
        if memory=='default':
            psu = psutil.virtual_memory()
            memory = psu.available # get the gigs
            memory = memory*mempc  # use only mempc of that
        else:
            memory = memory*(np.power(2,30)) # input should be Gb

        if verbose:
            print("Using " + str(memory/(np.power(2,30))) + " of RAM")

        if not os.path.isdir(self.datadir):
            print(str(self.__class__) + ' : define datadir')
            return

        spikereader = npx.Reader(self._apbin)
        lfpreader   = npx.Reader(self._lfbin)

        if not os.path.isdir(self.split_channel_dir):
            print(str(self.__class__) + ' : define directory for saving ' +
                              'channels in new format (split_channel_dir)')
            return

        if chsel=='all':
            chsel = range(spikereader.nc)

        # based on available memory, estimate how much space we can use
        n_samples = memory/(spikereader.nc*bits_)
        n_samples = int(np.floor(n_samples))

        # now we need two for loops: 1 to iterate over time, other to
        # iterate over the channel writting

        # figure out the sample intervals
        n_segments = int(np.ceil((spikereader.ns/n_samples)))
        segment_length = int(np.round(spikereader.ns/n_segments))
        segment_length_lfp = int(np.round(lfpreader.ns/n_segments))
        segments        = np.zeros((n_segments, 2))
        segments_lfp    = np.zeros((n_segments, 2))

        # use this for loop to define segments
        for s in range(n_segments):
            segments[s,0] = s*segment_length
            segments[s,1] = (s+1)*(segment_length) # ommitting -1 because the
            # Reader ignores the last index
            segments_lfp[s,0] = s*segment_length_lfp
            segments_lfp[s,1] = (s+1)*(segment_length_lfp)

            if s==(n_segments-1):
                segments[s,1]       = spikereader.ns
                segments_lfp[s,1]   = lfpreader.ns

        # build the filenames
        aplist = []
        lflist = []

        aphandle = []
        lfhandle = []


        print("Segmenting dat into " + str(n_segments) + " segments.")


        for c in chsel:
            # TODO: feed function handle for building filename
            # we have ot write to text files to be able to append... in the
            # end of this function, after all  files are written, we'll go
            # one by one and read the txt file, write it to a npy file and
            # delete the text file. it's a mess, but seems like the only way
            # to do it
            ap_file_name =  os.path.splitext(
                                os.path.basename(self._apbin))[0] +\
                                '_ch_' + str(c) + '.npy'
            lfp_file_name = os.path.splitext(
                                os.path.basename(self._lfbin))[0] +\
                                '_ch_' + str(c) + '.npy'
            ap_file_loc =   os.path.join(   self.split_channel_dir,
                                            ap_file_name )
            lfp_file_loc =   os.path.join(  self.split_channel_dir,
                                            lfp_file_name )

            aplist.append(ap_file_loc)
            lflist.append(lfp_file_loc)

            # f_ap = open(ap_file_loc, 'ab')
            # f_lp = open(lfp_file_loc, 'ab')
            # aphandle.append( f_ap )
            # lfhandle.append( f_lp )
            f_ap = None
            f_lp = None

        for s in range(n_segments):

            tic = time.time()

            apdata = spikereader.read_samples( first_sample=int(segments[s,0]),
                                                last_sample=int(segments[s,1]),
                                    channels=np.array(range(spikereader.nc)))
            lfdata = lfpreader.read_samples(first_sample=int(segments_lfp[s,0]),
                                            last_sample=int(segments_lfp[s,1]),
                                    channels=np.array(range(lfpreader.nc)))



            for c in chsel:
                apchannel = apdata[0][:,c]
                lfchannel = lfdata[0][:,c]

                # check if channel file exists
                if os.path.isfile(aplist[c]):
                    # load the existing file
                    tmp_apch = np.load(aplist[c])
                    # concatenatethe arrays
                    apchannel = np.concatenate((tmp_apch, apchannel), axis=0)
                if os.path.isfile(lflist[c]):
                    tmp_lfch = np.load(lflist[c])
                    lfchannel = np.concatenate((tmp_lfch, lfchannel), axis=0)

                np.save(aplist[c], apchannel)
                np.save(lflist[c], lfchannel)

                # write a new delimiter char at the end to prevent index merge
                # aphandle[c].write(delim)
                # lfhandle[c].write(delim)

                # clear apchannel and lfchannel
                apchannel = None
                lfchannel = None

                tmp_apch = None
                tmp_lfch = None

            # clear apdata
            apdata = None
            lfdata = None

            ttt = time.time()
            toc = ttt-tic;
            if verbose==1:
                print("Read and saved segment " + str(s) + " in " + str(toc)\
                                                + " s")
    def extractSyncChannel(datadir, save_to_file=0, device='npx'):
        # device can be nidq instead

        NPX_SYNC_IDX    = 6
        NIDQ_SYNC_IDX   = 0

        cpr = CareyNPXReader(datadir)
        # look for npx board files and NI files
        cpr.findDataFiles()

        # if ap.bin found, look for metadata to extract sampling freq
        if device == 'npx' and os.path.exists(cpr._apbin):
            print("Found ap.bin file. Reading sync channel...")
            tic = time.time()
            npx_obj = npx.Reader(cpr._apbin)
            sync_ch = npx_obj.read_sync(slice(0, npx_obj.ns))
            sync_ch = sync_ch[:, NPX_SYNC_IDX]
            # get end time, report it
            toc = time.time() - tic

            print('Read ap.bin sync in %.0f seconds' % toc)

        if device == 'nidq' and os.path.exists(cpr._nidqbin):
            print("Found nidq.bin file. Reading sync channel...")
            tic = time.time()
            npx_obj = npx.Reader(cpr._nidqbin)

            sync_ch = npx_obj.read_sync(slice(0, npx_obj.ns))
            sync_ch = sync_ch[:, NIDQ_SYNC_IDX]
            # get end time, report it
            toc = time.time() - tic
            print('Read nidq.bin sync in %.0f seconds' % toc)

        print(sync_ch.shape)

        if isinstance(save_to_file, str):
            if not os.path.exists(os.path.split(save_to_file)[0]):
                os.makedirs(os.path.split(save_to_file)[0])
            np.save(save_to_file, sync_ch)
            print('saved sync pulses in {}'.format(save_to_file))

        return sync_ch

    def extractSyncFromCameras(videosdir, save_to_file=0, fileext='.csv', sync=2, framerate=432, sortby='trial'):
        # file={.csv, .avi}, avi not implemented
        # idx = 1, this is the frame index from column 1 (idx = 2-1=1 a la python)
        # sync= 2, this is the index for the sync pulse (idx = 3-1=2 a la python)

        # list files from videosdir
        file_list = []
        for file in os.listdir(videosdir):
            if file.endswith(fileext):
                # add file to list
                file_list.append(file)

        file_datetimes_unsorted = []
        formatString = '%Y-%m-%dT%H_%M_%S' # built to bonsai's naming convention
        idx=slice(-23,-4)

        if sortby == 'time':
            for fileidx, file in enumerate(tqdm(file_list)):
                dt = datetime.datetime.strptime(file[idx], formatString)
                file_datetimes_unsorted.append(dt)

            # sort files by date
            ordered_file_indices = np.argsort(file_datetimes_unsorted)
            # and apply the sort
            file_list_unsorted = file_list
            file_list = [file_list_unsorted[i] for i in ordered_file_indices]
            file_datetimes = [file_datetimes_unsorted[i] for i in ordered_file_indices]

        elif sortby == 'trial':
            trial_list = []
            for fileidx, file in enumerate(tqdm(file_list)):
                trial = int(file_list[fileidx].split('_')[-1][0:-4])
                trial_list.append(trial)

            ordered_trial_indices = np.argsort(trial_list)
            file_list_unsorted = file_list
            file_list = [file_list_unsorted[i] for i in ordered_trial_indices]


        # determine the number of columns from a sample csv

        # iterate through files
        for fileidx, file in enumerate(tqdm(file_list)):
            # assuming the 3 column weird thing I had:
            if fileidx==0:
                this_trial = pd.read_csv(os.path.join(videosdir, file),
                                               header=None)
                A = np.arange(this_trial.columns.shape[0]).tolist()
                column_names_csv = [str(x) for x in A]
                column_names_session = ['trial', 'filename_avi', 'filename_csv', 'frame_idx',
                                        'trialwise_idx', 'trialwise_time', 'sessionwise_time',
                                        'syncpulse']
                column_names_session = column_names_session + column_names_csv
                # frame_idx is different from trialwise_idx in that frame_idx is always continuous
                # when frames are lost, trialwise_idx will skip the number of lost frames such that
                # trialwise_idx will always be equal or higher than frame_idx. trialwise_idx will have
                # a matching time called trialwise_time

                metadata_session = pd.DataFrame(columns = column_names_session)

            this_trial = pd.read_csv(os.path.join(videosdir, file),
                                     header=None)
            this_trial_metadata = pd.DataFrame(columns = column_names_session)
            this_trial_metadata['trial'] = np.ones(this_trial.shape[0]) * (fileidx+1)
            # indexing of trials starting at 1
            if sortby == 'time':
                this_filename_avi = CareyUtils.findMatchingFile_bonsai( os.path.join(videosdir, file),
                                                                        folder='same',
                                                                        targetext='.avi',
                                                                        formatString='%Y-%m-%dT%H_%M_%S',
                                                                        timestring_indices=slice(-23, -4),
                                                                        tol=5)
            elif sortby == 'trial':
                this_filename_avi = CareyUtils.findMatchingFile_bonsai(os.path.join(videosdir, file),
                                                                       folder='same',
                                                                       targetext='.avi',
                                                                       formatString='trial')

            this_trial_metadata = this_trial_metadata.assign( filename_avi = this_filename_avi )
            this_trial_metadata = this_trial_metadata.assign(filename_csv = file)
            this_trial_metadata['frame_idx']    = np.arange(this_trial.shape[0])
            this_trial_metadata['trialwise_idx']= this_trial[1]-this_trial[1][0]
            this_trial_metadata['trialwise_time'] = this_trial_metadata['trialwise_idx'] * 1/framerate
            this_trial_metadata = this_trial_metadata.assign(sessionwise_time = 0) # placeholder
            this_trial_metadata['syncpulse'] = this_trial[2]
            this_trial_metadata['0'] = this_trial[0]
            this_trial_metadata['1'] = this_trial[1]
            this_trial_metadata['2'] = this_trial[2]

            metadata_session = metadata_session.append(this_trial_metadata, ignore_index=True)


        # save to file
        if isinstance(save_to_file, str):
            __, ext = os.path.splitext(save_to_file)
            if ext=='.pkl':
                metadata_session.to_pickle(save_to_file)
            if ext=='.csv':
                metadata_session.to_csv(save_to_file)

            print('saved sync pulses in {}'.format(save_to_file))



class CareyVis:

    MAX_SIGS = 6
    DEFAULT_WIN = 'fullscreen'

    # class to visualize (and listen to) ephys signals

    def __init__(self, data, fs, file_list=[]):

        self.figurehandle = None # this will the figure handle
        self.plothandle = None
        self.axes = None

        self.data = data
        self.fs   = fs # this may be super inneficient

        if file_list == []:
            for ii in range(len(data)):
                file_list.append('ch' + str(ii))


    def Plot(self):


        self.activech = 0

        if len(self.data) > CareyVis.MAX_SIGS:
            print('Too many files, will load {0}'.format(self.MAX_SIGS))


        # plot one signal to try out
        self.getOrSetFigure()

        # TODO: number of axes should be the same as plotted signals
        self.axes = plt.axes([0.01, 0.2, 0.98, 0.75])

        # make time vec
        tot_time = float(self.data[self.activech].shape[0])/float(
                                                self.fs[self.activech])
        self.t = np.linspace(0, tot_time,
                             self.data[self.activech].shape[0])

        self.plothandle = plt.plot(self.t, self.data[self.activech])
        plt.show()
        # a = 1

    def getOrSetFigure(self):
        if self.figurehandle == None:
            dpi, w, h = CareyVis.getScreenDPI()
            self.figurehandle = plt.figure()
            # self.figurehandle = plt.figure(figsize=(float(w/10),float(h/10)))

        ax_play = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_stop = plt.axes([0.81, 0.05, 0.1, 0.075])

        # these guys need to be part of the class otherwise they get swept out
        # by the garbage collector.
        # somewhat superficial explanation here.
        # https://stackoverflow.com/questions/13165332/python-matplotlib-button-not-working
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.parsePlay)
        self.btn_stop = Button(ax_stop, 'Stop')
        self.btn_stop.on_clicked(self.parseStop)
        plt.show()


        return self.figurehandle

    @staticmethod
    def getScreenDPI():
       root = tk.Tk()
       dpi      = int(root.winfo_fpixels('1i'))
       swidth   = int(root.winfo_screenwidth())
       sheight  = int(root.winfo_screenheight())
       root.destroy()

       return dpi, swidth, sheight

    def parsePlay(self, event):
        # get current axes, segment signal, pass it to play
        # this is a wrapper because callbacks with arguments are sort of
        # complicated

        # TODO: fetch current axes
        # TODO: only playing first channel
        # CareySound.play(self.data, self.fs)

        # 1 get xlim
        [x0, x1] = self.axes.get_xlim()
        x0_idx = int((np.abs(self.t - x0)).argmin())
        x0_idx = np.maximum(x0_idx, int(0))
        x1_idx = (np.abs(self.t - x1)).argmin()
        x1_idx = np.minimum(x1_idx, self.data[self.activech].shape[0]-1)
        plotted = self.data[self.activech][x0_idx:x1_idx]

        print('pressed play')
        print(str(x0))
        print(str(x1))
        # print(str(plotted.shape))
        CareySound.play(plotted, self.fs[0])

    def parseStop(self, event):
        CareySound.stopAll()
        print('Clicked stop')

class CareySound:

    DEF_FS = 44100

    def __init__(self):
        pass
    def play(in_array, fs):

        # in_array should be a numpy array
        # fs should be the sampling frequency

        #0: validate numpy array

        # 1 - resample to 44100, which is a playable sampling rate
        playtime = float(in_array.shape[0]) / float(fs)
        t       = np.linspace(float(0), playtime, in_array.shape[0], False)
        t_441   = np.linspace(float(0), playtime,
                              int(np.round(playtime*CareySound.DEF_FS)),
                              False)

        soundvec = np.interp(t_441, t, in_array)

        in_array = None # cleanup memory #HACKERMAN

        # 2 - normalize array between -1 and 1
        # Ensure that highest value is in 16-bit range
        soundvec = soundvec * (2**15 - 1) / np.max(np.abs(soundvec))
        soundvec = soundvec.astype(np.int16)

        # 3 - generate a play object and play it
        play_obj = sa.play_buffer(soundvec, 1, 2, CareySound.DEF_FS)
        # play_obj.wait_done()
        return play_obj

    def stopAll():
        sa.stop_all()
        # for obj in gc.get_objects():
        #     if isinstance(obj, sa.shiny.PlayObject):
        #         print('cleaning up ' + str(obj))
        #         obj.stop()

class CareyDataPicker:
    # general data loader which takes either a folder or a file in any known
    # format and loads the data into the RAM
    def __init__(self, dataloc=''):
        # dataloc should be a file or folder that exists on the drive
        # if not, GUI prompts to sleection

        ttfolder = 'choose directory of channel data files'
        ttfile   = 'select channel data files'
        dir_opt = {}
        dir_opt['initialdir'] = os.curdir
        dir_opt['mustexist'] = True

        self.file_list = []

        self.isfile = 0
        self.isfolder = 0

        self.tkroot = tk.Tk()
        file_btn = tk.Button(self.tkroot, text = 'File(s)',
                             command=self.clickedFile)
        folder_btn = tk.Button(self.tkroot, text = 'Folder',
                             command=self.clickedFolder)


        file_btn.pack()
        folder_btn.pack()

        self.tkroot.mainloop()
        self.tkroot.destroy()

        if self.isfile:
            outsel = tk.filedialog.askopenfilenames(title = ttfile)

        elif self.isfolder:
            outsel = tk.filedialog.askdirectory(title = ttfolder, **dir_opt)
        # dir_opt = {}
        # dir_opt['initialdir'] = os.curdir
        # dir_opt['mustexist'] = True

        # tt = 'choose directory of channel npy files'


        self.parseFileList(outsel)


        # TODO: optionally input metadata file 8 additional button to file or
        # folder, have a neuropixels menu


    def clickedFile(self):
        self.isfile     = 1
        self.isfolder   = 0
        print('Selected file(s)')
        self.tkroot.quit()
    def clickedFolder(self):

        self.isfile     = 0
        self.isfolder   = 1
        print('Selected folder')
        self.tkroot.quit()
    def parseFileList(self, outsel):
        if type(outsel)==tuple:
            self.file_list = list(outsel)
        elif os.path.isdir(outsel):
            # expand all data files in there
            self.file_list = natsorted(os.listdir(outsel),
                                alg=ns.PATH | ns.IGNORECASE)
class CareyDataLoader:
    # this will be just a wrapper over numpy, with the additional feature of
    # grabbing sampling frequencies
    def __init__(self, file_list=[]):
        pass

    @staticmethod
    def loadChannels(file_list):
        if file_list==[]:
            ff = CareyDataPicker()
            file_list = ff.file_list
            ff = None
        elif isinstance(file_list, list):
            # do the thing
            pass
        else:
            print("CareyDataLoader.loadChannels: I don't know what to do")

        data = []
        for ii in range(len(file_list)):
            tmp_ch = np.load(file_list[ii])
            data.append( tmp_ch )

        return data


        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtWidgets.QWidget(self)

        l = QtWidgets.QVBoxLayout(self.main_widget)
        sc = MyStaticMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        dc = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        l.addWidget(sc)
        l.addWidget(dc)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("All hail matplotlib!", 2000)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """embedding_in_qt5.py example
Copyright 2005 Florent Rougon, 2006 Darren Dale, 2015 Jens H Nielsen

This program is a simple example of a Qt5 application embedding matplotlib
canvases.

It may be used and modified with no restriction; raw copies as well as
modified versions may be distributed without limitation.

This is modified from the embedding in qt4 example to show the difference
between qt4 and qt5"""
                                )

    def setSamplingFreq(fs, file_list):
        # if it's one value, set for all
        # python is a piece of overcomplicated, convoluted shit
        # let's "only" admit that datatypes can be float, int, lists or
        # numpy arrays

        num_ch = len(file_list)

        if      type(fs) == int or type(fs) == float:
            fsarray = np.repeat(float(fs), num_ch)
        elif    type(fs) == list:
            for ii in len(fs):
                fsarray = np.asarray(fs)
        # elif    type(fs) == numpy.ndarray:
        #     pass

        return fsarray

class RotaryEncoder:
    def getDisplacement(encoder1_time, encoder1_sig):
        pass
    def getAbsoluteDisplacement(encoder1_time, encoder1_sig,
                                encoder2_time, encoder2_sig):
        pass
    def getSpeedAndDistance(encoder1, encoder2, diameter=0.25, ticks_per_revolution=88, conv_kernel_size=1501, dt=1/9000):
        # direct translation from Hugo's getSpeedAndDistance (matlab)

        encoder1 = encoder1.astype(np.int8)
        encoder2 = encoder2.astype(np.int8)

        encoder1_delayed = np.zeros(encoder1.shape)
        encoder2_delayed = np.zeros(encoder2.shape)

        encoder1_delayed[2:] = encoder1[:-2]
        encoder2_delayed[2:] = encoder2[:-2]

        direction = np.logical_xor(encoder1.astype(bool), encoder2_delayed.astype(bool))
        counter = np.logical_xor(
                        np.logical_xor(encoder1.astype(bool), encoder1_delayed.astype(bool)),
                        np.logical_xor(encoder2.astype(bool), encoder2_delayed.astype(bool))    )

        rising_edges_times, __ = CareyUtils.getRisingLoweringEdges(counter.astype(np.uint8))
        rising_edges = np.zeros(encoder1.shape)
        rising_edges[rising_edges_times] = 1

        positive_distance = np.cumsum(rising_edges * direction * (diameter*np.pi/ticks_per_revolution))
        negative_distance = np.cumsum(-(rising_edges * (1-direction)) * (diameter*np.pi/ticks_per_revolution))
        pulses = rising_edges * direction - rising_edges * (1 - direction)
        distance = pulses * (diameter * np.pi / ticks_per_revolution)
        distance_data = np.cumsum(distance)

        speed = np.convolve(distance, np.ones((conv_kernel_size)), 'same')/(conv_kernel_size*dt)

        speed_data = savgol_filter(speed, conv_kernel_size*2-1, 3)
        # speed_data = speed_data(floor(conv_kernel_size/2):end-floor(conv_kernel_size/2)-1);

        time_array = np.linspace(0, encoder1.shape[0]*dt, encoder1.shape[0])

        return speed_data, distance_data, time_array

    def getSpeedAndDistance_fromFile(nibin, ENC1_CH=1, ENC2_CH=2, TICKS_PER_REV=88, WHEEL_DIAM=0.25, KERN_PTS=1501):
        # get dt from file
        print('Reading nibin file (takes a couple minutes...)')
        nibin_reader = ibllib.io.spikeglx.Reader(nibin)
        digital_sigs = nibin_reader.read_sync_digital(slice(0, nibin_reader.ns))
        print('Estimating speed from auxiliary inputs to NI')

        encoder1 = digital_sigs[:, ENC1_CH]
        encoder2 = digital_sigs[:, ENC2_CH]

        del digital_sigs

        dt = 1 / nibin_reader.fs

        speed_data, distance_data, time_array = RotaryEncoder.getSpeedAndDistance(encoder1, encoder2,
                                                   diameter=WHEEL_DIAM, ticks_per_revolution=TICKS_PER_REV,
                                                   conv_kernel_size=KERN_PTS, dt=dt)

        return speed_data, distance_data, time_array

class NeuropixelsExperiment:
    """defined a class for an experiment on the neuropixels rig (Diogo)
    Each experiment is considered a session of acquisition in one animal"""
    def __init__(self):
        # neuropixels
        self.npx_apbin  = ''
        self.npx_apmeta = ''
        # ni daq
        self.nidq_bin    = ''
        self.nidq_meta   = ''
        # body camera
        self.behavior_videos    = '' # this one's a folder
        self.video_format       = '.avi'
        self.metadata_format    = '.csv'
        # processing directory
        self.processing_dir     = ''

    def extractSyncronizationStreams(self, extract_npx=1, extract_nidq=1, extract_behav_videos=1, sortby='trial'):
        if not self.validateFiles():
            raise ValueError('Something is wrong with the data directories or their assignment!')

        prefix = self.mouse + '_' + self.session + '_'

        # start with npx, which is the one that takes the longest
        if extract_npx:
            print('Extracting sync channel from NPX data stream at:')
            print(self.npx_apbin)
            CareyNPXReader.extractSyncChannel(os.path.split(self.npx_apbin)[0],
                                              save_to_file=os.path.join(self.processing_dir, prefix + 'apbin_sync.npy'),
                                              device='npx')
            print('Done extracting neuropixels sync signal')

        # next is the nidq
        if extract_nidq:
            print(f'Extracting sync channel from ni daq data stream at: {self.nidq_bin}')
            CareyNPXReader.extractSyncChannel(os.path.split(self.nidq_bin)[0],
                                              save_to_file=os.path.join(self.processing_dir, prefix + 'nidq_sync.npy'),
                                              device='nidq')
            print('Done extracting ni daq sync signal')


        if extract_behav_videos:
            print('Extracting sync channel from ni daq data stream at:')
            print(self.behavior_videos)
            CareyNPXReader.extractSyncFromCameras(self.behavior_videos,
                                            save_to_file=os.path.join(self.processing_dir,
                                                                      prefix + 'behavior_metadata.csv'),
                                            fileext='.csv', sync=2, framerate=432, sortby=sortby)
            print('Done extracting metadata and sync from behavioral videos')

    def validateFiles(self):
        """iterate through the file directories and confirm that they are assigned, exist and are not empty"""
        return 1

    def synchronizeDataStreams(self, verbose=1):
        """this is the function that creates a corrected time array based on npx time"""

        prefix = self.mouse + '_' + self.session + '_'

        apmeta = ibllib.io.spikeglx.read_meta_data(self.npx_apmeta)
        nidqmeta = ibllib.io.spikeglx.read_meta_data(self.nidq_meta)

        npx_sync_ch = np.load(os.path.join(self.processing_dir, prefix + 'apbin_sync.npy'))
        nidq_sync_ch = np.load(os.path.join(self.processing_dir, prefix + 'nidq_sync.npy'))

        ap_time_array = np.arange(0, apmeta['fileTimeSecs'], 1 / apmeta['imSampRate'])
        nidq_time_array = np.arange(0, nidqmeta['fileTimeSecs'], 1 / nidqmeta['niSampRate'])

        # get pulse index
        pulses_npx, __ = CareyUtils.getRisingLoweringEdges(npx_sync_ch)
        pulses_nidq, __ = CareyUtils.getRisingLoweringEdges(nidq_sync_ch)

        print(f'Found {pulses_npx.size} pulses for neuropixels and {pulses_nidq.size} pulses for NI DAQ')

        # get timing of pulses
        pulse_timing_npx = ap_time_array[pulses_npx.astype(int)]
        pulse_timing_nidq = nidq_time_array[pulses_nidq.astype(int)]

        timing_model = LinearRegression()
        timing_model.fit(pulse_timing_nidq[:, None], pulse_timing_npx)
        if verbose:
            print('initial \t m = %f, b = %f' % (timing_model.coef_, timing_model.intercept_))

        # after one hour, what's the deviation?
        delta_t = 3600  # 1 hour in seconds
        deviation_after_1h = delta_t * (timing_model.coef_ - 1)
        if verbose:
            print(f"After one hour of acquisition, npx and nidq are {deviation_after_1h} seconds apart")

        # correcting the nidq time array:
        nidq_time_array_corrected = nidq_time_array * timing_model.coef_ + timing_model.intercept_

        # checking that the timing is right
        pulse_timing_nidq_corrected = nidq_time_array_corrected[pulses_nidq.astype(int)]
        timing_model_corrected = LinearRegression()
        timing_model_corrected.fit(pulse_timing_npx[:, None], pulse_timing_nidq_corrected)
        if verbose:
            print('corrected \t m = %f, b = %f' % (timing_model_corrected.coef_, timing_model_corrected.intercept_))

        # save corrected time arrays in a file
        print(f'Saving npx time array in {os.path.join(self.processing_dir, prefix + "apbin_time_array.npy")}')
        np.save(os.path.join(self.processing_dir, prefix + 'apbin_time_array.npy'), ap_time_array)
        print(f'Saving corrected ni daq time array in {os.path.join(self.processing_dir, prefix + "apbin_time_array.npy")}')
        np.save(os.path.join(self.processing_dir, prefix + 'nidq_time_array.npy'), nidq_time_array_corrected)

        return nidq_time_array_corrected, ap_time_array

    def correctCameraTime(self, behavior_metadata_file, syncpulse_file, align_to='npx',
                          outputfile='default'):

        prefix = self.mouse + '_' + self.session + '_'

        # read appropriate sync channel
        if align_to=='npx':
            sessionwise_syncpulse = np.load(os.path.join(self.processing_dir, prefix + 'apbin_sync.npy'))
            # estimate the time array from metadata
            meta = ibllib.io.spikeglx.read_meta_data(self.npx_apmeta)
            sessionwise_time_array = np.arange(0, meta['fileTimeSecs'], 1 / meta['imSampRate'])
        elif align_to=='nidq':
            sessionwise_syncpulse = np.load(syncpulse_file)
            meta = ibllib.io.spikeglx.read_meta_data(self.nidq_meta)
            sessionwise_time_array = np.arange(0, meta['fileTimeSecs'], 1 / meta['niSampRate'])

        start_indices, end_indices, pulse_indices_by_trial = NeuropixelsExperiment.segmentTrialsFromSyncSignal(
                                                                                            sessionwise_syncpulse )

        # read camera metadata
        camera_metadata = pd.read_csv(behavior_metadata_file)
        # camera_metadata['trial'] = camera_metadata['trial'].astype(int) # TODO: fix this in the future
        cols = camera_metadata.columns
        cols = cols.to_list()
        camera_metadata_corrected = pd.DataFrame(columns = cols)


        n_trials = len(pulse_indices_by_trial)  # this estimate is from neuropixels sync
        num_pulses_by_trial = [len(ii) for ii in pulse_indices_by_trial]

        # before reading the data structure fetch the number of pulses per trial to get the integrity of the each
        # video file
        n_pulses_behav = []
        for ii in tqdm(range(n_trials)):
            pulses_this_video = CareyUtils.getRisingLoweringEdges(
                camera_metadata[camera_metadata['trial']==ii+1]['syncpulse'].values)[0]
            if pulses_this_video is None:
                n_pulses_behav.append(0)
            else:
                n_pulses_behav.append(pulses_this_video.shape[0])

        # boolean array with the trials that should be kept.
        # Assumptions: modal value fo pulses is the correct one (there should be more uncorrupted than corrupted values)
        intact_trials = n_pulses_behav==scipy.stats.mode(n_pulses_behav)[0][0]
        if np.any(~intact_trials):
            raise ValueError('Non intact trials found')
        else:
            print('All trials have consistent number of pulses.')

        slopes      = []
        intercepts  = []

        # trial by trial
        print('Estimating temporal correction per trial')
        for ii in tqdm(range(n_trials)):
            # get te pulse signal for the camera in this trial
            this_trial = camera_metadata[camera_metadata['trial']==ii+1]
            cam_sync_this_trial = this_trial['syncpulse'].values
            # get rising edges of this pulse for the first trial
            rising_edges_this_trial,__ = CareyUtils.getRisingLoweringEdges(cam_sync_this_trial)

            # check if pulses match up
            if pulse_indices_by_trial[0].shape[0] == n_pulses_behav[ii]:
                # raise ValueError('Number of pulses does not match up!')

                # prepare values for time regression / alignment / correction
                timepoints_reference    = sessionwise_time_array[pulse_indices_by_trial[ii]]
                timepoints_camera       = this_trial['trialwise_time'].values[rising_edges_this_trial]
                # do the regression
                timing_model = LinearRegression()
                timing_model.fit(timepoints_camera[:,None], timepoints_reference)
                # apply the correction

                slopes.append(timing_model.coef_[0])
                intercepts.append(timing_model.intercept_)

                sessionwise_time_corrected = this_trial['trialwise_time'].values*timing_model.coef_[0] + timing_model.intercept_
                # the abve line needs to include the time of first...
                trialwise_time_corrected = sessionwise_time_corrected - sessionwise_time_corrected[0] # setting the first
                # time at zero index to zero (the definition of a trialwise time)

                # update the camera_metadata dataframe with the updated information
                first_pulse = rising_edges_this_trial[0]

                pd.options.mode.chained_assignment = None
                this_trial['sessionwise_time'] = sessionwise_time_corrected
                this_trial['trialwise_time'] = trialwise_time_corrected
                pd.options.mode.chained_assignment = 'warn'

                camera_metadata_corrected = camera_metadata_corrected.append(this_trial, ignore_index=True)

                # timing of the pulses in camera
                # print(this_trial['sessionwise_time'].values[rising_edges_this_trial][-1])
                # timing of the pulses in npx
                # print(timepoints_reference[-1])
                a=1

            else:
                warnings.warn('Number of pulses does not match up!')
                pd.options.mode.chained_assignment = None
                this_trial['sessionwise_time'] = np.NaN
                this_trial['trialwise_time'] = np.NaN
                camera_metadata_corrected = camera_metadata_corrected.append(this_trial, ignore_index=True)
                pd.options.mode.chained_assignment = 'warn'
                print('Inputed NaNs in time for trial ' + str(ii+1))

        # save the corrected file
        if outputfile == 'default':
            outputfile = os.path.join(self.processing_dir, prefix + 'behavior_metadata_corrected.csv')
        camera_metadata_corrected.to_csv(outputfile)

        return slopes, intercepts

    def segmentTrialsFromSyncSignal(syncpulse, thresh=2):
        """finds the beggining and enfding of each sync pulse by trial
        returns: start_indices:
        end_indices, start_indices_by_trial"""
        rising, lowering = CareyUtils.getRisingLoweringEdges(syncpulse)

        # get intervals between rising edges
        intervals = np.diff(rising)

        # a typical interval within pulses will be the mode of intervals
        inter_pulse_interval = scipy.stats.mode(intervals)[0][0]

        # where are there intervals higher than the model? these will be between trials
        inter_trial = np.where( intervals > (inter_pulse_interval * thresh) )[0]

        num_trials = inter_trial.shape[0] + 1
        print(f'Found {num_trials} trials on the synchronizer pulse in the neuropixels stream.')

        start_indices   = np.zeros((num_trials))
        end_indices     = np.zeros((num_trials)) # still rising edges!

        start_indices[0] = rising[0]
        for ii in range(inter_trial.shape[0]):
            start_indices[ii+1] = rising[inter_trial[ii]+1]
            end_indices[ii]     = rising[inter_trial[ii]]

        end_indices[-1] = rising[-1]

        start_indices_by_trial = [None] * num_trials # list of len num_trials, each element is an array of the indices of
        # pulses for ech trial

        for ii in range(num_trials):
            start_indices_by_trial[ii] = rising[np.where( np.logical_and( rising>=start_indices[ii], rising<=end_indices[ii] ))[0]]

        # TODO: return a list of pulses already organized by trial. a list where each element is an array of pulse
        # timings for that trial

        return start_indices, end_indices, start_indices_by_trial

    def compileSessionwiseMetadataBehavior(videosdir, save_to_file=0, fileext='.csv', csv_frame_idx=1,
                                           sync_ch=2, framerate=430, videoformat='.avi', verbose=True):
        # file={.csv, .avi}, avi not implemented
        # csv_frame_idx = 1, this is the frame index from column 1 (idx = 2-1=1 a la python)
        # sync_ch= 2, this is the index for the sync pulse (idx = 3-1=2 a la python)

        # list metadata files from videosdir
        file_list = []
        for file in os.listdir(videosdir):
            if file.endswith(fileext):
                # add file to list
                file_list.append(file)

        # ..........   sort files by trial   ......................
        trials_unsorted = []
        for fileidx, file in enumerate(file_list):
            file_props, __, __ = CareyFileProcessing.getFileProps(file, standard='NPXRIG')
            trial = file_props['trial']
            trials_unsorted.append(trial)

        trials_unsorted = np.asarray(trials_unsorted)
        sort_indices = np.argsort(trials_unsorted)
        #  list comprehesion
        file_list_unsorted = file_list
        file_list = [file_list_unsorted[i] for i in sort_indices]
        [file_list[i] for i in sort_indices]
        # .........................................................

        # iterate through files
        for fileidx, file in enumerate(tqdm(file_list)):
            # assuming the 3 column weird thing I had:
            if verbose:
                print(f'Compiling metadata behavior for file {fileidx+1} of {len(file_list)}... ', end="")
            if fileidx == 0:
                this_trial = pd.read_csv(os.path.join(videosdir, file),
                                         header=None)
                A = np.arange(this_trial.columns.shape[0]).tolist()
                column_names_csv = [str(x) for x in A]
                column_names_session = ['trial', 'filename_avi', 'filename_csv', 'frame_idx',
                                        'trialwise_idx', 'trialwise_time', 'sessionwise_time',
                                        'syncpulse']
                column_names_session = column_names_session + column_names_csv
                # frame_idx is different from trialwise_idx in that frame_idx is always continuous
                # when frames are lost, trialwise_idx will skip the number of lost frames such that
                # trialwise_idx will always be equal or higher than frame_idx. trialwise_idx will have
                # a matching time called trialwise_time

                metadata_session = pd.DataFrame(columns=column_names_session)

            this_trial = pd.read_csv(os.path.join(videosdir, file),
                                     header=None)
            this_trial_metadata = pd.DataFrame(columns=column_names_session)
            this_trial_metadata['trial'] = np.ones(this_trial.shape[0]) * (fileidx + 1)
            # indexing of trials starting at 1
            file_props, __, __ = CareyFileProcessing.getFileProps(file, standard='NPXRIG')
            this_filename_avi = CareyLib.NeuropixelsExperiment.getSingleFileFromFolder(
                videosdir, trial=file_props['trial'], method='trial', ext='.avi')
            this_trial_metadata = this_trial_metadata.assign(filename_avi=this_filename_avi)
            this_trial_metadata = this_trial_metadata.assign(filename_csv=file)
            this_trial_metadata['frame_idx'] = np.arange(this_trial.shape[0])
            this_trial_metadata['trialwise_idx'] = this_trial[1] - this_trial[1][0]
            this_trial_metadata['trialwise_time'] = this_trial_metadata['trialwise_idx'] * 1 / framerate
            this_trial_metadata = this_trial_metadata.assign(sessionwise_time=0)  # placeholder
            this_trial_metadata['syncpulse'] = this_trial[2]
            this_trial_metadata['0'] = this_trial[0]
            this_trial_metadata['1'] = this_trial[1]
            this_trial_metadata['2'] = this_trial[2]

            metadata_session = metadata_session.append(this_trial_metadata)
            if verbose:
                print('Done.')

        print(f'{fileidx+1} behavioral trial files found in this session')

        # save to file
        if isinstance(save_to_file, str):
            __, ext = os.path.splitext(save_to_file)
            if ext == '.pkl':
                metadata_session.to_pickle(save_to_file)
            elif ext == '.csv':
                metadata_session.to_csv(save_to_file)
            elif ext == '.hdf':
                metadata_session.to_hdf(save_to_file, key='data')

            print('saved sync pulses in {}'.format(save_to_file))

    def getSingleFileFromFolder(folder, match='', trial=None, ext='', method='expression_match'):

        if method == 'expression_match':
            file_list = []
            for file in os.listdir(folder):
                if file.__contains__(match) and file.endswith(ext):
                    # add file to list
                    file_list.append(file)

        if method == 'trial':
            file_list = []
            for file in os.listdir(folder):
                if file.endswith(ext):
                    file_props, __, __ = CareyFileProcessing.getFileProps(file, standard='NPXRIG')
                    if file_props['trial'] == trial:
                        # add file to list
                        file_list.append(file)

        if len(file_list) > 1:
            raise ValueError('CareyLib.NeuropixelsExperiment.getSingleFileFromFolder: More than one file found!')
        elif len(file_list) < 1:
            raise ValueError('CareyLib.NeuropixelsExperiment.getSingleFileFromFolder: No file found!')

        return file_list[0]

    def compileSessionwiseTracks(self, tracksdir, match='shuffle1', ext='h5', mouse_name=True, verbose=True,
                                 sortby='trial', dt=1/432, qthresh=0.005, tCov=1.0, obsCov=1.0):
        """
        compileSessionwiseTracks compiles DLC tracks into a file with all tracks for the whole session
        :param match:
        :param ext:
        :param mouse_name indicates whether theres an underscore in the mouse name or not
        :return:
        """

        track_mapping = {   'FRx': ('FR_bottom', 'x', 'y'),
                            'FRy': ('FR_bottom', 'y', 'x'),
                            'FRz': ('FR_side', 'y', '__'),
                            'FLx': ('FL_bottom', 'x', 'y'),
                            'FLy': ('FL_bottom', 'y', 'x'),
                            'FLz': ('FL_side', 'y', '__'),
                            'HLx': ('HL_bottom', 'x', 'y'),
                            'HLy': ('HL_bottom', 'y', 'x'),
                            'HLz': ('HL_side', 'y', '__'),
                            'Tail1x': ('tail_bottom_1', 'x', 'y'),
                            'Tail1y': ('tail_bottom_1', 'y', 'x'),
                            'Tail1z': ('tail_side_1', 'y', '__'),
                            'Tail2x': ('tail_bottom_2', 'x', 'y'),
                            'Tail2y': ('tail_bottom_2', 'y', 'x'),
                            'Tail2z': ('tail_side_2', 'y', '__')}

        file_list = []
        for file in os.listdir(tracksdir):
            if file.__contains__(match) and file.endswith(ext):
                # add file to list
                file_list.append(file)

        # sort these files by trial
        sorted_list, __, __ = CareyFileProcessing.sortFilesByTrial(file_list, standard='NPXRIG_DLC',
                                             name_contains_underscore=mouse_name)

        columns = ['trial', 'frame',
                   'FRx', 'FRy', 'FRz', 'HRx', 'HRy', 'HRz', 'FLx', 'FLy', 'FLz', 'HLx', 'HLy', 'HLz',
                   'Tail1x', 'Tail1y', 'Tail1z', 'Tail2x', 'Tail2y', 'Tail2z']

        df = pd.DataFrame(data=None, columns=columns)
        print('Compiling tracks from DLC files')
        for fileidx, file in enumerate(tqdm(sorted_list)):
            file_props, __, __ = CareyFileProcessing.getFileProps(
                                        os.path.join(tracksdir, file), standard='NPXRIG_DLC',
                                        name_contains_underscore=mouse_name)
            DLC_df = pd.read_hdf(os.path.join(tracksdir, file))
            these_tracks = DLC_df[DLC_df.columns[0][0]]
            n_frames = these_tracks.shape[0]
            this_trial = pd.DataFrame(data=None, columns=columns)
            this_trial['trial']     = np.ones(n_frames, int) * file_props['trial']
            this_trial['frame']     = np.arange(n_frames)

            for trial_key, (track_key, coord, alt_coord) in track_mapping.items():
                if coord == 'x':
                    this_trial[trial_key], this_trial[trial_key[:-1]+'y'] = \
                        CareyBehavior.kalman_smooth_low_confidence_tracks(these_tracks, track_key, dt=dt,
                                                                          confThresh=qthresh, tCov=tCov, obsCov=obsCov)
                if track_key.endswith('bottom'):
                    __, this_trial[trial_key] = CareyBehavior.kalman_smooth_low_confidence_tracks(
                        these_tracks, track_key, dt=dt, confThresh=qthresh, tCov=tCov, obsCov=obsCov)

            this_trial['tracksfile'] = file

            df = df.append(this_trial, ignore_index=True)

        csvfilename = file_props['mouse'] + '_' + file_props['session'] + '_DLC_tracks.csv'
        if verbose:
            print(f'Writing tracks to {csvfilename}')

        df.to_csv(os.path.join(self.processing_dir, csvfilename)) # todo: compile all the tracks and likelihood, then smooth sessionwise. otherwise kalman filter instantiation takes too long
        return df

    def runSwingStanceDetection(tracksdir, outputfolder, ext='h5', match='shuffle1', naming_convention='NPXRIG_DLC',
                                CleanArtifs=True, FiltCutOff=60, Acq_Freq=430, SwSt_Outlier_Rej=True, Belts_Dict=None,
                                Type_Experiment=0, graph=False, mouse_name=True, verbose=True, skip_if_exists=True,
                                file_list=None):
        """
        runSwingStanceDetection iteratively calls Jorge's SwSt_Det_SlopeThres.Pocket_SwSt_Det for each file in the
        given folder
        :param outputfolder:
        :param match:
        :param naming_convention:
        :param CleanArtifs:
        :param FiltCutOff:
        :param Acq_Freq:
        :param SwSt_Outlier_Rej:
        :param Type_Experiment:
        :param graph:
        :param verbose:
        :return:
        """

        if (file_list is None):
            file_list = []
            for file in os.listdir(tracksdir):
                if file.__contains__(match) and file.endswith(ext):
                    # add file to list
                    file_list.append(file)

        # sort these files by trial
        sorted_list, __, __ = CareyFileProcessing.sortFilesByTrial(file_list, standard=naming_convention,
                                                                   name_contains_underscore=mouse_name)

        for f in sorted_list:

            video_file = f[:f.find('DLC')]
            tmp_beltspeed_idx = Belts_Dict['filename'] == video_file
            this_Belts_Dict = {}
            for k in Belts_Dict.keys():
                this_Belts_Dict[k] = Belts_Dict[k][tmp_beltspeed_idx]

            # build the output name
            if naming_convention == 'NPXRIG_DLC':
                file_props, __, __ = CareyFileProcessing.getFileProps(f,
                                        standard=naming_convention, name_contains_underscore=mouse_name)
                strides_filename = file_props['mouse'] + '_'  + naming_convention + '_SwSt_strides' +\
                                   '_' + file_props['session'] + '_'+ str(file_props['trial'])

            if skip_if_exists and os.path.exists(os.path.join(outputfolder, (strides_filename + '.h5'))):
                continue
            else:
                # try:
                CareyBehavior.SwingAndStanceDetection(
                    os.path.join(tracksdir, f), os.path.join(outputfolder, strides_filename),
                    CleanArtifs=CleanArtifs, FiltCutOff=FiltCutOff,
                    Acq_Freq=Acq_Freq, SwSt_Outlier_Rej=SwSt_Outlier_Rej,
                    Type_Experiment=Type_Experiment,
                    Belts_Dict=this_Belts_Dict,
                    graph=graph, verbose=verbose)
                # block raising an exception
                # except:
                #    pass  # doing nothing on exception

    def addTracksToMetadata(metadata, tracks, verbose=1):
        pd.set_option('mode.chained_assignment', None)
        if isinstance(metadata, pd.DataFrame):
            df_metadata = metadata
            metadata = None
            del metadata
        else:
            df_metadata = pd.read_csv(metadata)
        if isinstance(tracks, pd.DataFrame):
            df_tracks = tracks
            tracks = None
            del tracks
        else:
            df_tracks   = pd.read_csv(tracks_file)
        '''
        df_metadata.columns
        Out[70]:
        Index(['Unnamed: 0', 'Unnamed: 0.1', 'trial', 'filename_avi', 'filename_csv',
               'frame_idx', 'trialwise_idx', 'trialwise_time', 'sessionwise_time',
               'syncpulse', '0', '1', '2'],
              dtype='object')
        df_tracks.columns
        Out[71]: 
        Index(['Unnamed: 0', 'trial', 'frame', 'FRx', 'FRy, FRz', 'HRx', 'HRy, HRz',
               'FLx', 'FLy, FLz', 'HLx', 'HLy, HLz', 'Tail1x', 'Tail1y', 'Tail1z',
               'Tail2x', 'Tail2y', 'Tail2z',
              dtype='object')
        '''
        columns = [ 'trial', 'filename_avi', 'filename_csv',
                    'frame_idx', 'trialwise_idx', 'trialwise_time', 'sessionwise_time',
                    'syncpulse', '0', '1', '2',
                    'FRx', 'FRy', 'FRz', 'HRx', 'HRy', 'HRz',
                    'FLx', 'FLy', 'FLz', 'HLx', 'HLy', 'HLz', 'Tail1x', 'Tail1y', 'Tail1z',
                    'Tail2x', 'Tail2y', 'Tail2z']

        df = pd.DataFrame(data=None, columns=columns)

        alltrials = np.asarray(np.unique(df_metadata['trial'].values), 'int')
        for trial in alltrials:

            if verbose:
                print('Appending tracks from trial ' + str(trial))

            trial_metadata  = df_metadata[df_metadata['trial']==trial]
            trial_tracks    = df_tracks[df_tracks['trial']==trial]

            mismatch = trial_metadata.shape[0] - trial_tracks.shape[0]
            n_trial_frames = trial_tracks.shape[0]

            if mismatch <= -1:
                trial_tracks.drop(trial_tracks.index[-1], inplace=True)
                warnings.warn('There is one more frame in the video than in metadata')
                mismatch = 0

            if mismatch >= 0 and mismatch < 2: # I'll allow one frame, let's not push it
                this_trial = pd.DataFrame(data=None, columns=columns)
                this_trial['trial']             = trial_metadata['trial'][:n_trial_frames].astype('int')
                this_trial['filename_avi']      = trial_metadata['filename_avi'][:n_trial_frames]
                this_trial['filename_csv']      = trial_metadata['filename_csv'][:n_trial_frames]
                this_trial['tracksfile']        = trial_tracks['tracksfile']
                this_trial['frame']             = trial_metadata['frame_idx'][:n_trial_frames]
                this_trial['trialwise_idx']     = trial_metadata['trialwise_idx'][:n_trial_frames]
                this_trial['trialwise_time']    = trial_metadata['trialwise_time'][:n_trial_frames]
                this_trial['sessionwise_time']  = trial_metadata['sessionwise_time'][:n_trial_frames]
                this_trial['syncpulse']         = trial_metadata['syncpulse'][:n_trial_frames]

                this_trial['FRx']       = trial_tracks['FRx']
                this_trial['FRy']       = trial_tracks['FRy']
                this_trial['FRz']       = trial_tracks['FRz']
                this_trial['HRx']       = trial_tracks['HRx']
                this_trial['HRy']       = trial_tracks['HRy']
                this_trial['HRz']       = trial_tracks['HRz']
                this_trial['FLx']       = trial_tracks['FLx']
                this_trial['FLy']       = trial_tracks['FLy']
                this_trial['FLz']       = trial_tracks['FLz']
                this_trial['HLx']       = trial_tracks['HLx']
                this_trial['HLy']       = trial_tracks['HLy']
                this_trial['HLz']       = trial_tracks['HLz']
                this_trial['Tail1x']    = trial_tracks['Tail1x']
                this_trial['Tail1y']    = trial_tracks['Tail1y']
                this_trial['Tail1z']    = trial_tracks['Tail1z']
                this_trial['Tail2x']    = trial_tracks['Tail2x']
                this_trial['Tail2y']    = trial_tracks['Tail2y']
                this_trial['Tail2z']    = trial_tracks['Tail2z']

                df = df.append(this_trial, ignore_index=True)

            else:
                raise ValueError(f'unhandled mismatch of frames in trial {trial}')
                # if mistmatch > 0:
                #     # which I think I understand
                #     print(f'Mistmatch of {mistmatch} frames on trial {trial}')
                #     # append from the beginning of the metadata
        pd.set_option('mode.chained_assignment', 'warn')
        return df

    def addSwingAndStanceEvents(dataframe, swing_and_stance_output_folder, pattern='SwSt_strides'):
        """
        :param swing_and_stance_output_folder:
        :return:
        """
        pd.set_option('mode.chained_assignment', None)
        if not isinstance(dataframe, pd.DataFrame):
            dataframe = pd.read_csv(dataframe)

        # 1) find stride segmentation / swing and stance files
        file_obj = CareyFileProcessing.FileOps(swing_and_stance_output_folder)
        file_obj.listdir().filterFilesFromList('.h5').filterFilesFromList(pattern)

        # 2) sort them by trials. some trials may not exist, so explicitly pull out a trial number from the file name
        sorted_list, __, __ = CareyFileProcessing.sortFilesByTrial(file_obj.file_list,
                                                                   standard='NPXRIG')

        vars = ['FR_SwOn', 'FR_StOn', 'HR_SwOn', 'HR_StOn', 'FL_SwOn', 'FL_StOn', 'HL_SwOn', 'HL_StOn']
        for var in vars:
            # do this by generating a variable from names. makes the code less repetitive and less error prone (typo)
            locals()[var] = None
            dataframe[var] = False

        # 3) iterate through each trial. match to the variable "frame"
        for strides_file in sorted_list:

            file_props, __, __ = CareyFileProcessing.getFileProps(strides_file, standard='NPXRIG')
            trial_number = file_props['trial']
            these_events = h5py.File(os.path.join(swing_and_stance_output_folder, strides_file), 'r')
            print(f'Adding events from trial {trial_number}')

            # let's stick to the basics for now and use only the essential events
            for var in vars:
                locals()[var] = None
                paw = var[:2]
                sw_or_st = 'Swing' if var[3:5]=='Sw' else 'Stance'
                field           = sw_or_st + ' Onset F val.'
                field_accepted  = sw_or_st + ' Onset Accepted'

                try:
                    locals()[var] = np.asarray(these_events[paw][field])[these_events[paw][field_accepted]]
                    idx = np.logical_and(   dataframe['trial'] == trial_number,
                                            CareyUtils.findAny(dataframe['frame'].astype('int'),  locals()[var]))
                    dataframe[var][idx] = True
                except:
                    print(f'Failed to retrieve data of {var} on trial {trial_number}')
                    pass
        pd.set_option('mode.chained_assignment', 'warn')
        print('Done!')
        return dataframe

    def addWheelDistanceAndSpeed(metadata, speed_data, distance_data, time_array):
        """
        interpolates and adds distances to metadata file. arguments of speed, distance and time are after the call:
        speed_data, distance_data, time_array = CareyLib.RotaryEncoder.getSpeedAndDistance_fromFile(nibinfile)
        :param speed_data:
        :param distance_data:
        :param time_array:
        :return:
        """
        # speed_data, distance_data, time_array = CareyLib.RotaryEncoder.getSpeedAndDistance_fromFile(nibinfile)
        metadata['wheel_distance']  = np.interp(metadata['sessionwise_time'], time_array, distance_data)
        metadata['wheel_speed']     = np.interp(metadata['sessionwise_time'], time_array, speed_data)

        return metadata

    def interpolateLowConfidenceTracks(ps_position, ps_likelihood, qthresh):
        badpoints = ps_likelihood < np.quantile(ps_likelihood, qthresh)
        ps_position.loc[badpoints] = np.nan
        ps_position = ps_position.interpolate(method='polynomial', order=3)
        return ps_position

    def compile_wheel_speed(self, nidq_corrected_time=None, outfile=None):

        if nidq_corrected_time is None:
            print('Reading NI DAQ time from default location')
            nidq_corrected_time = os.path.join(self.processing_dir, self.mouse + '_' + self.session + '_nidq_time_array.npy')
            time_array = np.load(nidq_corrected_time)
        else:
            if isinstance(nidq_corrected_time, str):
                print(f'Reading NI DAQ time from {nidq_corrected_time}')
                time_array = np.load(nidq_corrected_time)
            elif isinstance(time_array, np.ndarray):
                print(f'Parsing NI DAQ time from numerical input')
                time_array = nidq_corrected_time

        nimeta =    [x for x in os.listdir(os.path.join(self.npx_apmeta, '..', '..')) if x[-9:] == 'nidq.meta'][0]
        nibin =     [x for x in os.listdir(os.path.join(self.npx_apmeta, '..', '..')) if x[-8:] == 'nidq.bin'][0]
        nimeta = os.path.join(self.npx_apmeta, '..', '..', nimeta)
        nibin  = os.path.join(self.npx_apmeta, '..', '..', nibin)
        metadata = ibllib.io.spikeglx.read_meta_data(nimeta)

        DT = 1 / metadata['niSampRate']
        speed, dist, _ = CareyLib.RotaryEncoder.getSpeedAndDistance_fromFile(nibin)

        # downsample
        ds_rate = 1000  # Hz
        time_array_downsample = np.arange(0, time_array[-1], 1 / ds_rate)

        dist_downsample = np.interp(time_array_downsample, time_array, dist)
        speed_downsample= np.interp(time_array_downsample, time_array, speed)

        dist = dist_downsample
        tim = time_array_downsample

        df = pd.DataFrame(data=None, columns=['mouse', 'session', 'time', 'distance', 'speed'])
        df = df.assign(mouse=self.mouse)
        df = df.assign(session=self.session)
        df['time']     = time_array_downsample
        df['distance'] = dist_downsample
        df['speed']    = speed_downsample

        print(f'Saving wheel speed data to file {outfile}.')
        if outfile is not None:
            df.to_csv(outfile)

        return df

def periCShistogram(cs, ss, plotting=0):
    # plots the peri-ccomplex spike histogram of simple spikes
    # peri peri peri peri
    # assumes that cs and ss are spike times in seconds, preferably in double
    # format
    ms2s = 1000; # to convert milliseconds to s, multiply this
    interval = np.array([-100, 100]); # ms
    binsize = 2; # ms

    binlimits = np.arange(interval[0], interval[1]+binsize, binsize);
    hist_values = np.zeros(binlimits.shape[0]-1);

    numCS = cs.shape[0];

    # interate through all complex spikes and count simple spikes before and
    # after
    for ii in range(cs.shape[0]):
        # set the time limits for searching simple spikes specific to this
        # CS
        timelimits_this_cs = interval/ms2s+cs[ii]
        # bounds to protect going under zero or above unexisting times
        if timelimits_this_cs[0]<ss[0] or timelimits_this_cs[1] > ss[-1]:
            N = np.zeros(hist_values.shape)
        else:
            edges = binlimits/ms2s+cs[ii];
            N, edges = np.histogram(ss, edges)
        hist_values = hist_values + N

    if plotting:
        fig = plt.figure()
        plt.hist(hist_values, edges)

    return hist_values, binlimits

def logWriter(file_full_path, message_to_write):
    # check if folder exists
    if not os.path.exists(os.path.dirname(file_full_path)):
        # create such folder
        os.mkdir(os.path.dirname(file_full_path))

    file_handle = open(file_full_path, "a+")
    dated_message = str(datetime.datetime.now()) +  "\t" + message_to_write

    file_handle.write(dated_message)






























# qApp = QtWidgets.QApplication(sys.argv)

# aw = ApplicationWindow()
# aw.setWindowTitle("%s" % progname)
# aw.show()
# sys.exit(qApp.exec_())
# #qApp.exec_()
