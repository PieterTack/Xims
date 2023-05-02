# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:08:34 2020

@author: prrta
"""
import h5py
import numpy as np
import sys
import os

import matplotlib
matplotlib.use('Qt5Agg') #Render to Pyside/PyQt canvas
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QApplication, QWidget, QDialog
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QCheckBox, QPushButton, QLabel, QScrollArea, \
        QLineEdit, QTabWidget, QFileDialog, QComboBox, QTreeWidget, QTreeWidgetItem, \
            QRadioButton


def gplot_rh5(h5file, channel='channel00'):
    f = h5py.File(h5file, 'r')
    spe = np.array(f['raw/'+channel+'/sumspec'])
    try:
        names = [n.decode('utf8') for n in f['fit/'+channel+'/names']]
        cfg = f['fit/'+channel+'/cfg'][()]
    except KeyError:
        names = None
        cfg = None

    if cfg is not None: # we're looking for the energy calibration values...
        from PyMca5.PyMca import ConfigDict
        config = ConfigDict.ConfigDict()
        config.read(cfg)
        cfg = [config['detector']['zero'], config['detector']['gain'], config['fit']['energy']] #zero, gain, E_Rayleigh

    f.close()

    return spe, names, cfg

def h5_plot(h5file, channel='channel00', label=None, xrange=None, normtochan=None, yrange=None, peak_id=True):
    # read the h5 file, formatted according to the XMI format
    #   If  fit was performed, also read in the PyMCA config file to figure out detector calibration
    h5file = np.array(h5file)
    plt.figure(figsize=(20,15))
    for j in range(0, h5file.size):
        if h5file.size == 1:
            savename = str(h5file).split('.')[0]+'_'+channel+'.png'
            h5 = str(h5file)
        else:
            savename = str(h5file[0]).split('.')[0]+'-'+str(h5file[-1]).split('.')[0].split('/')[-1]+'_'+channel+'.png'
            h5 = str(h5file[j])
        fileid = str(h5.split('.')[0].split('/')[-1])
        if channel == 'all':
            chnl = ['channel00', 'channel02']
            for i in range(0, 2):
                spe, names, cfg = gplot_rh5(h5, chnl[i])
                if normtochan is not None:
                    spe = spe[:]/spe[normtochan]
                if cfg is None:
                    zero = 0.
                    gain = 1.
                    xtitle = "Channel Number"
                else:
                    zero = cfg[0]
                    gain = cfg[1]
                    xtitle = "Energy [keV]"
                # plot the spectrum, Ylog, X-axis converted to Energy (keV) if PyMca cfg provided
                if label is None:
                    plt_lbl = fileid+'_'+str(chnl[i])
                else:
                    plt_lbl = label[i]
                plt.plot(np.linspace(0, spe.shape[0]-1, num=spe.shape[0])*gain+zero, spe, label=plt_lbl, linestyle='-')
        else:        
            spe, names, cfg = gplot_rh5(h5, channel)
            if normtochan is not None:
                spe = spe[:]/spe[normtochan]
            if cfg is None:
                zero = 0.
                gain = 1.
                xtitle = "Channel Number"
            else:
                zero = cfg[0]
                gain = cfg[1]
                xtitle = "Energy [keV]"
            # plot the spectrum, Ylog, X-axis converted to Energy (keV) if PyMca cfg provided
            if xrange is None:
                xrange = (zero, (spe.shape[0]-1)*gain+zero)
            if label is None:
                plt_lbl = fileid+'_'+str(channel)
            else:
                plt_lbl = label
            plt.plot(np.linspace(0, spe.shape[0]-1, num=spe.shape[0])*gain+zero, spe, label=plt_lbl, linestyle='-')

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    plt.yscale('log')
    plt.xlim(xrange)
    if yrange is not None:
        plt.ylim(yrange)
    plt.legend(handles, labels, loc='best', fontsize=16)
    ax.set_xlabel(xtitle, fontsize=16)
    ax.set_ylabel("Intensity [counts]", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='x', which='minor', bottom=True)
    
    # add peak annotation if names and cfg provided
    if peak_id is True and cfg is not None and names is not None:
        # x axis of plot is in Energy (keV)
        # determine peak energy values (Rayl energy = cfg[2])
        from PyMca5.PyMcaPhysics.xrf import Elements
        for n in names:
            if n != 'Rayl' and n != 'Compt':
                if n.split(" ")[1][0] == 'K':
                    energy = Elements.getxrayenergy(n.split(" ")[0],'KL3') #Ka1
                elif n.split(" ")[1][0] == 'L':
                    energy = Elements.getxrayenergy(n.split(" ")[0],'L3M5') #La1
                elif n.split(" ")[1][0] == 'M':
                    energy = Elements.getxrayenergy(n.split(" ")[0],'M5N7') #Ma1
            elif n == 'Rayl':
                if type(cfg[2]) is type(list()):
                    energy = cfg[2][0]
                else:
                    energy = cfg[2]
            else:
                energy = None
            # if energy not None, plot label at this energy value
            if energy is not None:
                idx = max(np.where(handles[0].get_xdata() <= energy)[-1])
                yval = 10**(np.log10(max([hand.get_ydata()[idx] for hand in handles]))*1.025)
                # plot the text label X% above this value
                plt.text(energy, yval, n, horizontalalignment='center', fontsize=16)
    
    #TODO: Increase fontsizes, add minor tickmarks on x-axis        
    #plt.show()

    plt.savefig(savename)#, bbox_inches='tight', pad_inches=0)
    plt.close() 

def plot(data, labels=None, cfg=None, xrange=None, yrange=None, normtochan=None, savename='plot.png', plotelnames=True):
    tickfontsize=22
    titlefontsize=28
    figsize= (21,7)#(20,15)

    data = np.array(data) #expected dimensions: [N, channels]
    if len(data.shape) == 1:
        data = np.reshape((1,data.shape[0]))
    elif len(data.shape) >= 3:
        print("Error: expected data dimensions: [N, channels]; ", data.shape)
        return
    
    names = []
    if cfg is not None: # we're looking for the energy calibration values...
        from PyMca5.PyMca import ConfigDict
        config = ConfigDict.ConfigDict()
        config.read(cfg)
        cfg = [config['detector']['zero'], config['detector']['gain'], config['fit']['energy'][0]] #zero, gain, E_Rayleigh
        peaks = list(config['peaks'].keys())
        for peak in peaks:
            for linetype in config['peaks'][peak]:
                names.append(peak+' '+str(linetype))    
    
    plt.figure(figsize=figsize)
    for j in range(0, data.shape[0]):
        spe = data[j,:]
        if labels is None:
            lbl = 'spectrum '+str(j)
        else:
            lbl = labels[j]
        if normtochan is not None:
            spe = spe[:]/spe[normtochan]
        if cfg is None:
            zero = 0.
            gain = 1.
            xtitle = "Channel Number"
        else:
            zero = cfg[0]
            gain = cfg[1]
            xtitle = "Energy (keV)"
        # plot the spectrum, Ylog, X-axis converted to Energy (keV) if PyMca cfg provided
        if xrange is None:
            xrange = (zero, (spe.shape[0]-1)*gain+zero)
        plt.plot(np.linspace(0, spe.shape[0]-1, num=spe.shape[0])*gain+zero, spe, label=lbl, linestyle='-')
    
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    plt.yscale('log')
    plt.xlim(xrange)
    if yrange is not None:
        plt.ylim(yrange)
    plt.legend(handles, labels, loc='best', fontsize=tickfontsize)
    ax.set_xlabel(xtitle, fontsize=titlefontsize)
    ax.set_ylabel("Intensity (counts)", fontsize=titlefontsize)
    ax.tick_params(axis='both', which='major', labelsize=tickfontsize)
    ax.tick_params(axis='x', which='minor', bottom=True)
    
    # add peak annotation if names and cfg provided
    if cfg is not None:
        # x axis of plot is in Energy (keV)
        # determine peak energy values (Rayl energy = cfg[2])
        #TODO account for occurences of both Ka and Kb...
        from PyMca5.PyMcaPhysics.xrf import Elements
        for n in names:
            if n != 'Rayl' and n != 'Compt':
                if n.split(" ")[1][0] == 'K':
                    energy = Elements.getxrayenergy(n.split(" ")[0],'KL3') #Ka1
                elif n.split(" ")[1][0] == 'L':
                    energy = Elements.getxrayenergy(n.split(" ")[0],+'L3M5') #La1
                elif n.split(" ")[1][0] == 'M':
                    energy = Elements.getxrayenergy(n.split(" ")[0],'M5N7') #Ma1
            elif n == 'Rayl':
                energy = cfg[2]
            else:
                energy = None
            # if energy not None, plot label at this energy value
            if energy is not None and plotelnames is True:
                idx = max(np.where(handles[0].get_xdata() <= energy)[-1])
                yval = 10**(np.log10(max([hand.get_ydata()[idx] for hand in handles]))*1.025)
                # plot the text label X% above this value
                plt.text(energy, yval, n, horizontalalignment='center', fontsize=tickfontsize)

    plt.show()
    plt.tight_layout()

    plt.savefig(savename)#, bbox_inches='tight', pad_inches=0)
    plt.close() 

class Poll_h5dir():
    def __init__(self, h5file, parent=None):
        self.paths=[]
        self.h5file = h5file
        # extract all Dataset paths from the H5 file
        f = h5py.File(self.h5file, 'r')
        self.paths = self.descend(f, paths=None)

        # in this case, we only want spectra
        self.paths = [path for path in self.paths if 'sumspec' in path or 'maxspec' in path]    
        self.specs = [np.array(f[path]) for path in self.paths]
        
        f.close()

    def spe(self, path):
        return self.specs[self.paths.index(path)]

    def dirs(self):
        return self.paths
        
    def descend(self, obj, paths=None):
        if paths is None:
            paths = []
            
        if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
            for key in obj.keys():
                self.descend(obj[key], paths=paths)
        elif type(obj)==h5py._hl.dataset.Dataset:
            paths.append(obj.name)
        return paths


class MatplotlibWidget(QWidget):
    
    def __init__(self, parent = None):
        
        QWidget.__init__(self, parent)
        
        self.fig = Figure(figsize=(7,3.5), dpi=100, tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(NavigationToolbar(self.canvas, self))
        vertical_layout.addWidget(self.canvas)
        
        # self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.axes = self.fig.add_subplot(111)
        
        self.setLayout(vertical_layout)

    
class Xplot_GUI(QWidget):
    
    def __init__(self, parent=None):
        super(Xplot_GUI, self).__init__(parent)

        self.new_window = None
        self.filename = ""


        # create widgets

        # create main layout for widgets
        layout_main = QVBoxLayout()
        layout_browseh5 = QHBoxLayout()
        layout_subdir = QHBoxLayout()
        layout_body = QHBoxLayout()
        layout_canvas = QVBoxLayout()

        # browse buttons
        self.file_lbl = QLabel("File:")
        self.filedir = QLineEdit("")
        self.browse = QPushButton("...")
        self.browse.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.browse.setMaximumWidth(25)
        layout_browseh5.addWidget(self.file_lbl)
        layout_browseh5.addWidget(self.filedir)
        layout_browseh5.addWidget(self.browse)
        layout_main.addLayout(layout_browseh5)

        # curves display
        self.mpl = MatplotlibWidget()
        self.mpl.axes.set_xlabel('Energy [keV]')
        self.mpl.axes.set_ylabel('Intensity [counts]')
        layout_canvas.addWidget(self.mpl)
    
        # dropdown box to select file subdir
        self.subdir_lbl = QLabel('     Sub directory:')
        self.subdir = QComboBox()
        self.subdir.addItems([''])
        self.subdir.setMinimumWidth(200)
        layout_subdir.addWidget(self.subdir_lbl)
        layout_subdir.addWidget(self.subdir)
        layout_subdir.addStretch()
        layout_main.addLayout(layout_subdir)
        
        layout_body.addLayout(layout_canvas)

        # plotting options
        # 3 tabs:
        #   Labels
        #   Layout
        #   Options
        self.menu_tabs = QTabWidget()
        
        self.tab_labels = QWidget() #Labels
        tab_labels_layout = QVBoxLayout()
        labels_graphtitle_layout = QHBoxLayout()
        graphtitle_lbl = QLabel("Title:")
        self.graphtitle = QLineEdit("")
        labels_graphtitle_layout.addWidget(graphtitle_lbl)
        labels_graphtitle_layout.addWidget(self.graphtitle)
        tab_labels_layout.addLayout(labels_graphtitle_layout)
        tab_labels_layout.insertSpacing(1, 10)
        axis_opts_layout = QGridLayout()
        axis_opts_layout.addWidget(QLabel(""),0,0)
        xtitle_lbl = QLabel("X-axis:")
        ytitle_lbl = QLabel("Y-axis:")
        axis_opts_layout.addWidget(xtitle_lbl,1,0)
        axis_opts_layout.addWidget(ytitle_lbl,2,0)
        axis_opts_layout.addWidget(QLabel("Title"),0,1)
        self.xtitle = QLineEdit("Energy [keV]")
        self.xtitle.setMaximumWidth(500)
        self.ytitle = QLineEdit("Intensity [Counts]")
        self.ytitle.setMaximumWidth(500)
        axis_opts_layout.addWidget(self.xtitle,1,1)
        axis_opts_layout.addWidget(self.ytitle,2,1)
        axis_opts_layout.addWidget(QLabel("Log"),0,2)
        self.xlinlog = QCheckBox("")
        self.xlinlog.setChecked(False)
        self.ylinlog = QCheckBox("")
        self.ylinlog.setChecked(True)
        axis_opts_layout.addWidget(self.xlinlog,1,2)
        axis_opts_layout.addWidget(self.ylinlog,2,2)
        axis_opts_layout.addWidget(QLabel("Multiplier"),0,3)
        self.xmult = QLineEdit("1")
        self.xmult.setValidator(QDoubleValidator(-1E9, 1E9, 0))
        self.xmult.setMaximumWidth(50)
        self.ymult = QLineEdit("1")
        self.ymult.setValidator(QDoubleValidator(-1E9, 1E9, 0))
        self.ymult.setMaximumWidth(50)
        axis_opts_layout.addWidget(self.xmult,1,3)
        axis_opts_layout.addWidget(self.ymult,2,3)
        tab_labels_layout.addLayout(axis_opts_layout)
        axbox_layout = QHBoxLayout()
        axbox_layout.addStrut(50)
        axbox_title = QLabel("Axes type:")
        axbox_layout.addWidget(axbox_title)
        self.axboxtype_single = QRadioButton("single")
        axbox_layout.addWidget(self.axboxtype_single)
        self.axboxtype_box = QRadioButton("box")
        self.axboxtype_box.setChecked(True)
        axbox_layout.addWidget(self.axboxtype_box)
        self.xkcd = QCheckBox("xkcd plottype") #with plt.xkcd():
        axbox_layout.addWidget(self.xkcd)
        axbox_layout.insertSpacing(4, 25)
        tab_labels_layout.addLayout(axbox_layout)
        tab_labels_layout.insertSpacing(3, 10)
        axis_minmax_layout = QGridLayout()
        axis_minmax_layout.addWidget(QLabel(""),0,0)
        axis_minmax_layout.addWidget(QLabel("X-axis:"),1,0)
        axis_minmax_layout.addWidget(QLabel("Y-axis:"),2,0)
        axis_minmax_layout.addWidget(QLabel("Min:"),0,1)
        axis_minmax_layout.addWidget(QLabel("Max:"),0,2)
        self.xmin = QLineEdit("")
        self.xmin.setMaximumWidth(500)
        self.ymin = QLineEdit("")
        self.ymin.setMaximumWidth(500)
        axis_minmax_layout.addWidget(self.xmin,1,1)
        axis_minmax_layout.addWidget(self.ymin,2,1)
        self.xmax = QLineEdit("")
        self.xmax.setMaximumWidth(500)
        self.ymax = QLineEdit("")
        self.ymax.setMaximumWidth(500)
        axis_minmax_layout.addWidget(self.xmax,1,2)
        axis_minmax_layout.addWidget(self.ymax,2,2)
        tab_labels_layout.addLayout(axis_minmax_layout)
        axis_Eplot_layout = QHBoxLayout()
        axis_Eplot_layout.addStrut(50)
        self.Eplot = QCheckBox("Display X-axis as Energy")
        axis_Eplot_layout.addWidget(self.Eplot)
        tab_labels_layout.addLayout(axis_Eplot_layout)
        tab_labels_layout.addStretch()
        self.tab_labels.setLayout(tab_labels_layout)
        self.menu_tabs.addTab(self.tab_labels, "Labels")
        
        self.tab_custom = QWidget() #Layout
        tab_custom_layout = QVBoxLayout()
        tab_custom_layout.addWidget(QLabel("Font sizes:"))
        custom_fontsizes_layout = QGridLayout()
        custom_fontsizes_layout.addWidget(QLabel('Main Title:'), 0,0)
        custom_fontsizes_layout.addWidget(QLabel('Axes Title:'), 1,0)
        custom_fontsizes_layout.addWidget(QLabel('Axes Labels:'), 2,0)
        self.fontsize_maintitle = QLineEdit("")
        self.fontsize_maintitle.setValidator(QDoubleValidator(1, 100, 0))
        self.fontsize_maintitle.setMaximumWidth(30)
        self.fontsize_axtitle = QLineEdit("")
        self.fontsize_axtitle.setValidator(QDoubleValidator(1, 100, 0))
        self.fontsize_axtitle.setMaximumWidth(30)
        self.fontsize_axlbl = QLineEdit("")
        self.fontsize_axlbl.setValidator(QDoubleValidator(1, 100, 0))
        self.fontsize_axlbl.setMaximumWidth(30)
        custom_fontsizes_layout.addWidget(self.fontsize_maintitle, 0,1)
        custom_fontsizes_layout.addWidget(self.fontsize_axtitle, 1,1)
        custom_fontsizes_layout.addWidget(self.fontsize_axlbl, 2,1)        
        custom_fontsizes_layout.addWidget(QLabel('Legend:'), 0,2)
        custom_fontsizes_layout.addWidget(QLabel('Annotations:'), 1,2)
        custom_fontsizes_layout.addWidget(QLabel('Curve Thick:'), 2,2)
        self.fontsize_legend = QLineEdit("")
        self.fontsize_legend.setValidator(QDoubleValidator(1, 1E2, 0))
        self.fontsize_legend.setMaximumWidth(30)
        self.fontsize_annot = QLineEdit("")
        self.fontsize_annot.setValidator(QDoubleValidator(1, 1E2, 0))
        self.fontsize_annot.setMaximumWidth(30)
        self.curve_thick = QLineEdit("")
        self.curve_thick.setValidator(QDoubleValidator(1, 1E2, 0))
        self.curve_thick.setMaximumWidth(30)
        custom_fontsizes_layout.addWidget(self.fontsize_legend, 0,3)
        custom_fontsizes_layout.addWidget(self.fontsize_annot, 1,3)
        custom_fontsizes_layout.addWidget(self.curve_thick, 2,3)        
        tab_custom_layout.addLayout(custom_fontsizes_layout)
        tab_custom_layout.addSpacing(10)
        custom_curve_layout = QHBoxLayout()
        self.curve_sequence = QPushButton("Curve Sequence/Label/Colour")
        self.curve_sequence.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.curve_sequence.setMaximumWidth(200)
        custom_curve_layout.addWidget(self.curve_sequence)
        custom_curve_layout.addStretch(10)
        custom_curve_layout.addWidget(QLabel("Vert. Offset:"))
        self.vert_offset = QLineEdit("0")
        self.vert_offset.setValidator(QDoubleValidator(0, 1E9, 3))
        self.vert_offset.setMaximumWidth(30)
        custom_curve_layout.addWidget(self.vert_offset)
        tab_custom_layout.addLayout(custom_curve_layout)
        tab_custom_layout.addSpacing(10)
        tab_custom_layout.addWidget(QLabel("Plot Positions:"))
        custom_positions_layout = QGridLayout()
        custom_positions_layout.addWidget(QLabel('Plot area:'), 0,0)
        custom_positions_layout.addWidget(QLabel('Legend position:'), 1,0)
        self.plotarea = QLineEdit("tight") #tight or other layout or xmin,ymin,xmax,ymax
        self.legendpos = QLineEdit("best") #best or other loc or xpos,ypos
        custom_positions_layout.addWidget(self.plotarea, 0,1)
        custom_positions_layout.addWidget(self.legendpos, 1,1)
        tab_custom_layout.addLayout(custom_positions_layout)
        tab_custom_layout.addStretch()
        self.tab_custom.setLayout(tab_custom_layout)
        self.menu_tabs.addTab(self.tab_custom, "Layout")
        
        self.tab_options = QWidget() #Options
        tab_options_layout = QVBoxLayout()
        options_smoothderiv_layout = QHBoxLayout()
        self.smooth = QCheckBox("Smooth data") #using from scipy.signal import savgol_filter; savgol_filter(y, 51, 3) # window size 51, polynomial order 3
        self.smooth.setMaximumWidth(100)
        self.savgol_window = QLineEdit("51")
        self.savgol_window.setValidator(QDoubleValidator(1, 1E2, 0))
        self.savgol_window.setMaximumWidth(30)
        self.savgol_poly = QLineEdit("3")
        self.savgol_poly.setValidator(QDoubleValidator(1, 1E2, 0))
        self.savgol_poly.setMaximumWidth(30)
        self.deriv = QCheckBox("Plot Deriv")
        options_smoothderiv_layout.addWidget(self.smooth)
        options_smoothderiv_layout.addWidget(self.savgol_window)
        options_smoothderiv_layout.addWidget(self.savgol_poly)
        options_smoothderiv_layout.addWidget(self.deriv)
        tab_options_layout.addLayout(options_smoothderiv_layout)     
        tab_options_layout.addSpacing(10)
        self.interpolate = QCheckBox("Interpolate data")
        tab_options_layout.addWidget(self.interpolate)
        tab_options_layout.addSpacing(10)
        options_errorbar_layout = QHBoxLayout()
        self.errorbar_flag = QCheckBox("Display Errorbars: ")
        options_errorbar_layout.addWidget(self.errorbar_flag)
        self.errorbar_bars = QRadioButton("Bars")
        self.errorbar_area = QRadioButton("Area") #plt.fill_between(x, low, up, alpha=0.3)
        self.errorbar_bars.setChecked(True)
        options_errorbar_layout.addWidget(self.errorbar_bars)
        options_errorbar_layout.addWidget(self.errorbar_area)
        options_errorbar_layout.addStretch()
        tab_options_layout.addLayout(options_errorbar_layout)
        tab_options_layout.addSpacing(10)
        options_peakid_layout = QHBoxLayout()
        options_peakid_layout.addWidget(QLabel("Peak ID:"))
        self.peakid_none = QRadioButton("None")
        self.peakid_main = QRadioButton("Main")
        self.peakid_all = QRadioButton("All")
        self.peakid_none.setChecked(True)
        options_peakid_layout.addWidget(self.peakid_none)
        options_peakid_layout.addWidget(self.peakid_main)
        options_peakid_layout.addWidget(self.peakid_all)
        options_peakid_layout.addSpacing(20)
        self.peakid_arrows = QCheckBox("Arrows")
        options_peakid_layout.addWidget(self.peakid_arrows)
        options_peakid_layout.addStretch()
        tab_options_layout.addLayout(options_peakid_layout)
        tab_options_layout.addStretch()
        self.tab_options.setLayout(tab_options_layout)
        self.menu_tabs.addTab(self.tab_options, "Options")
        layout_body.addWidget(self.menu_tabs)        
        self.menu_tabs.setCurrentWidget(self.tab_labels)

    
        # show window
        layout_main.addLayout(layout_body)
        self.setLayout(layout_main)
        self.setWindowTitle('Xplot GUI')
        self.show()

        # event handling
        self.browse.clicked.connect(self.browse_app) # browse button
        self.filedir.returnPressed.connect(self.browse_app)
        self.subdir.currentIndexChanged.connect(self.subdir_change) #different data directory selected
        
    def browse_app(self):
        self.filename = QFileDialog.getOpenFileName(self, caption="Open spectrum file", filter="H5 (*.h5);;SPE (*.spe);;CSV (*.csv)")[0]
        if len(self.filename) != 0:
            self.filedir.setText("'"+str(self.filename)+"'")
            # read in first ims file, to obtain data on elements and dimensions
            if(self.filename != "''"):
                extension = os.path.splitext(self.filename)[-1].lower()
                if extension == '.spe':
                    pass #TODO
                elif extension == '.csv':
                    pass #TODO
                elif extension == '.h5':
                    self.file = Poll_h5dir(self.filename)
                    self.subdirs = []
                    self.subdirs = self.file.dirs()
                    self.rawspe = self.file.spe([dirs for dirs in self.subdirs if 'raw' in dirs and 'sumspec' in dirs][0])
                    # change dropdown widget to display appropriate subdirs
                    self.subdir.clear()
                    self.subdir.addItems(self.subdirs)
                    self.subdir.setCurrentIndex(self.subdirs.index([dirs for dirs in self.subdirs if 'raw' in dirs and 'sumspec' in dirs][0]))
            # now adjust plot window (if new file or dir chosen, the fit results should clear and only self.rawspe is displayed)
            self.update_plot(update=False)
            
    def subdir_change(self, index):
        if self.subdirs != []:
            self.rawspe = self.file.spe(self.subdirs[index])
            self.update_plot(update=False)

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    xplot = Xplot_GUI()
    xplot.show()
    sys.exit(app.exec_())

