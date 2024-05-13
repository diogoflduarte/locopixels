import os
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
import tkinter.filedialog
from natsort import natsorted, ns
import simpleaudio as sa
from matplotlib.widgets import Button

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