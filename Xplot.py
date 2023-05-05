# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:08:34 2020

@author: prrta
"""
import h5py
import numpy as np
import sys
import os
import itertools

import matplotlib
matplotlib.use('Qt5Agg') #Render to Pyside/PyQt canvas
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QCheckBox, QPushButton, QDialog, QLabel, \
        QLineEdit, QTabWidget, QFileDialog, QRadioButton, QTableWidgetItem, \
            QListWidget, QAbstractItemView, QSplitter, QTableWidget, QHeaderView


def Xplot_rh5(h5file, channel='raw/channel00/sumspec'):
    with h5py.File(h5file, 'r') as f:
        spe = np.array(f[channel])
        
        # see if names is in same channel directory, or in folder above
        if '/'.join(channel.split('/')[0:-1])+'/names' in f.keys():
            names = [n.decode('utf8') for n in f['/'.join(channel.split('/')[0:-1])+'/names']]
        elif '/'.join(channel.split('/')[0:-2])+'/names' in f.keys():
            names = [n.decode('utf8') for n in f['/'.join(channel.split('/')[0:-2])+'/names']]
        else:
            names = None

        cfgdir = 'fit/'+channel.split('/')[1]+'/cfg'
        if cfgdir in f.keys():
            cfg = f[cfgdir][()]
        else:
            cfg = None

    if cfg is not None:
        if os.path.exists(cfg): # we're looking for the energy calibration values...
            from PyMca5.PyMca import ConfigDict
            config = ConfigDict.ConfigDict()
            config.read(cfg)
            zgr = [config['detector']['zero'], config['detector']['gain'], config['fit']['energy']] #zero, gain, E_Rayleigh
        elif os.path.exists(os.path.dirname(h5file)+'/'+os.path.basename(cfg)): #also check if the cfg file happens to be in same folder as h5file...
            from PyMca5.PyMca import ConfigDict
            config = ConfigDict.ConfigDict()
            config.read(os.path.dirname(h5file)+'/'+os.path.basename(cfg))
            zgr = [config['detector']['zero'], config['detector']['gain'], config['fit']['energy']] #zero, gain, E_Rayleigh
        else:
            zgr = None
    else:
        zgr = None

    return spe, names, zgr

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
                spe, names, cfg = Xplot_rh5(h5, chnl[i])
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
            spe, names, cfg = Xplot_rh5(h5, channel)
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

class Poll_h5dir(QDialog):
    def __init__(self, h5file, parent=None):
        super(Poll_h5dir, self).__init__(parent)
        # extract all Dataset paths from the H5 file
        f = h5py.File(h5file, 'r')
        self.paths = self.descend(f, paths=None)
        f.close()
        
        # spawn screen allowing the user to select a given path, or multiple
        self.paths = [path for path in self.paths]        
        # build widgets
        layout = QVBoxLayout()
        self.task = QLabel('Select your H5 file directory of choice:')
        layout.addWidget(self.task)
        self.path_select = QListWidget()
        self.path_select.addItems(self.paths)
        self.path_select.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.path_select)
        self.read_but = QPushButton("Read")
        layout.addWidget(self.read_but)
        # show window
        self.setLayout(layout)
        self.show()
        # handle events
        self.read_but.clicked.connect(self.read_path)
        
    def descend(self, obj, paths=None):
        if paths is None:
            paths = []
            
        if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
            for key in obj.keys():
                self.descend(obj[key], paths=paths)
        elif type(obj)==h5py._hl.dataset.Dataset:
            paths.append(obj.name)
        return paths

    def read_path(self):
        self.h5dir = [item.text() for item in self.path_select.selectedItems()]
        # close spawned window and return selected elements...
        self.hide()
        super().accept()


class CurveSequence(QDialog):
    def __init__(self, mainobj, parent=None):
        super(CurveSequence, self).__init__(parent)

        layout_main = QVBoxLayout()
        
        # we need 5 columns: Sequence, filename (ineditable), datadir (ineditable), label, colour
        table_widget = QTableWidget(len(mainobj.datadic), 5)
        header = table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        table_widget.verticalHeader().setVisible(False)
        layout_main.addWidget(table_widget)
        # Set some Title cells
        table_widget.setHorizontalHeaderLabels(["Sequence", "File", "Directory", "Label", "Colour"])
        # Set all the subsequent datadic cells
        for index,data in enumerate(mainobj.datadic):
            table_widget.setItem(index, 0, QTableWidgetItem("{:0.0f}".format(index))) #sequence in the dictionary
            item = QTableWidgetItem(data["filename"])
            item.setFlags(item.flags() ^ Qt.ItemIsEditable) # filename
            table_widget.setItem(index, 1, item)
            item = QTableWidgetItem(data["h5dir"])
            item.setFlags(item.flags() ^ Qt.ItemIsEditable) # datadir in filename
            table_widget.setItem(index, 2, item)
            table_widget.setItem(index, 3,QTableWidgetItem(data["label"]))
            if data["colour"] is None:
                itemtext = "None"
            else:
                itemtext = data["colour"]
            table_widget.setItem(index, 4,QTableWidgetItem(itemtext))
        
        # Apply changes button
        self.set = QPushButton("Apply")
        self.set.setAutoDefault(False) # set False as otherwise this button is called on each return
        layout_main.addWidget(self.set)

        # show window
        self.setLayout(layout_main)
        self.setWindowTitle('Curve Sequence and Labeling')
        self.setWindowModality(Qt.ApplicationModal)
        self.setFocusPolicy(Qt.StrongFocus)
        self.resize(700,350)
        self.show()

        # event handling
        self.set.clicked.connect(self.go) # calculate button
        
    def go(self):
        # 
        # close spawned window and return selected elements...
        self.hide()
        super().accept()
        


class MatplotlibWidget(QWidget):
    
    def __init__(self, parent = None):
        
        QWidget.__init__(self, parent)
        
        self.fig = Figure(figsize=(10,5), dpi=100, tight_layout=True)
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
        self.datadic = []


        # create widgets

        # create main layout for widgets
        layout_main = QVBoxLayout()
        layout_browseh5 = QHBoxLayout()
        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)
        

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
        splitter.addWidget(self.mpl)
        
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
        tab_labels_layout.addSpacing(20)
        axis_Eplot_layout = QHBoxLayout()
        self.Eplot = QCheckBox("Display X-axis as Energy")
        axis_Eplot_layout.addWidget(self.Eplot)
        tab_labels_layout.addLayout(axis_Eplot_layout)
        axis_ZeroGain_layout = QHBoxLayout()
        axis_ZeroGain_layout.addSpacing(50)
        axis_ZeroGain_layout.addWidget(QLabel("Zero:"))
        self.Eplot_zero = QLineEdit("0")
        self.Eplot_zero.setValidator(QDoubleValidator(0, 1E6, 3))
        self.Eplot_zero.setMaximumWidth(50)
        self.Eplot_gain = QLineEdit("1")
        self.Eplot_gain.setValidator(QDoubleValidator(0, 1E6, 3))
        self.Eplot_gain.setMaximumWidth(50)
        axis_ZeroGain_layout.addWidget(self.Eplot_zero)
        axis_ZeroGain_layout.addWidget(QLabel("Gain:"))
        axis_ZeroGain_layout.addWidget(self.Eplot_gain)
        axis_ZeroGain_layout.addStretch()
        tab_labels_layout.addLayout(axis_ZeroGain_layout)
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
        self.fontsize_maintitle = QLineEdit("18")
        self.fontsize_maintitle.setValidator(QDoubleValidator(1, 100, 0))
        self.fontsize_maintitle.setMaximumWidth(30)
        self.fontsize_axtitle = QLineEdit("16")
        self.fontsize_axtitle.setValidator(QDoubleValidator(1, 100, 0))
        self.fontsize_axtitle.setMaximumWidth(30)
        self.fontsize_axlbl = QLineEdit("14")
        self.fontsize_axlbl.setValidator(QDoubleValidator(1, 100, 0))
        self.fontsize_axlbl.setMaximumWidth(30)
        custom_fontsizes_layout.addWidget(self.fontsize_maintitle, 0,1)
        custom_fontsizes_layout.addWidget(self.fontsize_axtitle, 1,1)
        custom_fontsizes_layout.addWidget(self.fontsize_axlbl, 2,1)        
        custom_fontsizes_layout.addWidget(QLabel('Legend:'), 0,2)
        custom_fontsizes_layout.addWidget(QLabel('Annotations:'), 1,2)
        custom_fontsizes_layout.addWidget(QLabel('Curve Thick:'), 2,2)
        self.fontsize_legend = QLineEdit("12")
        self.fontsize_legend.setValidator(QDoubleValidator(1, 1E2, 0))
        self.fontsize_legend.setMaximumWidth(30)
        self.fontsize_annot = QLineEdit("12")
        self.fontsize_annot.setValidator(QDoubleValidator(1, 1E2, 0))
        self.fontsize_annot.setMaximumWidth(30)
        self.curve_thick = QLineEdit("1.5")
        self.curve_thick.setValidator(QDoubleValidator(1, 1E2, 1))
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
        custom_positions_layout = QHBoxLayout()
        custom_positions_layout.addWidget(QLabel('Legend position:'))
        self.legendpos = QLineEdit("best") #best or other loc or xpos,ypos
        custom_positions_layout.addWidget(self.legendpos)
        custom_positions_layout.addStretch()
        tab_custom_layout.addLayout(custom_positions_layout)
        legend_bbox_layout = QHBoxLayout()
        legend_bbox_layout.addSpacing(50)
        self.legend_bbox = QCheckBox("Legend Bbox")
        self.legend_bbox.setChecked(True)
        legend_bbox_layout.addWidget(self.legend_bbox)
        tab_custom_layout.addLayout(legend_bbox_layout)
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
        splitter.addWidget(self.menu_tabs)        
        self.menu_tabs.setCurrentWidget(self.tab_labels)

    
        # show window
        layout_main.addWidget(splitter)
        self.setLayout(layout_main)
        self.setWindowTitle('Xplot GUI')
        self.show()

        # event handling
        self.browse.clicked.connect(self.browse_app) # browse button
        self.filedir.returnPressed.connect(self.read_files)
        self.graphtitle.returnPressed.connect(self.update_plot)
        self.xtitle.returnPressed.connect(self.update_plot)
        self.ytitle.returnPressed.connect(self.update_plot)
        self.xmin.returnPressed.connect(self.update_plot)
        self.xmax.returnPressed.connect(self.update_plot)
        self.ymin.returnPressed.connect(self.update_plot)
        self.ymax.returnPressed.connect(self.update_plot)
        self.xmult.returnPressed.connect(self.update_plot)
        self.ymult.returnPressed.connect(self.update_plot)
        self.xlinlog.stateChanged.connect(self.update_plot)
        self.ylinlog.stateChanged.connect(self.update_plot)
        self.axboxtype_single.toggled.connect(self.update_plot)
        self.axboxtype_box.toggled.connect(self.update_plot)
        self.xkcd.stateChanged.connect(self.update_plot)
        self.Eplot.stateChanged.connect(self.ctegain_update)
        self.Eplot_gain.returnPressed.connect(self.ctegain_update)
        self.Eplot_zero.returnPressed.connect(self.ctegain_update)
        self.fontsize_maintitle.returnPressed.connect(self.update_plot)  
        self.fontsize_maintitle.returnPressed.connect(self.update_plot)  
        self.fontsize_axtitle.returnPressed.connect(self.update_plot)  
        self.fontsize_axlbl.returnPressed.connect(self.update_plot)  
        self.fontsize_legend.returnPressed.connect(self.update_plot)  
        self.curve_thick.returnPressed.connect(self.update_plot)  
        self.legendpos.returnPressed.connect(self.update_plot)  
        self.vert_offset.returnPressed.connect(self.update_plot)
        self.legend_bbox.stateChanged.connect(self.update_plot)
        self.curve_sequence.clicked.connect(self.change_curve_seq)

    def update_plot(self):
        if self.datadic != []:
            if self.xkcd.isChecked() is True:
                plt.xkcd()
            else:
                plt.rcdefaults()
            self.mpl.axes.cla()
            self.mpl.fig.clear()
            self.mpl.axes = self.mpl.fig.add_subplot(111)
            self.mpl.axes.set_xlim((float(self.xmin.text()),float(self.xmax.text())))
            self.mpl.axes.set_ylim((float(self.ymin.text()),float(self.ymax.text())))
            if self.xlinlog.isChecked() is True:
                self.mpl.axes.set_xscale('log')
            if self.ylinlog.isChecked() is True:
                self.mpl.axes.set_yscale('log')
            for index, item in enumerate(self.datadic):
                # Apply vertical offset to all curves, taking into account a coordinate transform as y-axis could be log scaled
                xdata = item["xvals"]*float(self.xmult.text())
                ydata = item["data"]*float(self.ymult.text())
                if float(self.vert_offset.text()) != 0.0:
                    newTransform = self.mpl.axes.transScale + self.mpl.axes.transLimits
                    for i in range(len(ydata)):
                        axcoords = newTransform.transform([xdata[i], ydata[i]])
                        axcoords[1] += float(self.vert_offset.text())*index # add vertical offset in relative axes height
                        ydata[i] = newTransform.inverted().transform(axcoords)[1]
                self.mpl.axes.plot(xdata, ydata, label=item["label"], linewidth=float(self.curve_thick.text()), color=item["colour"])
            if self.axboxtype_single.isChecked() is True:
                self.mpl.axes.spines[['right', 'top']].set_visible(False)
            if self.graphtitle.text() != "":
                self.mpl.axes.set_title(self.graphtitle.text(), fontsize=np.around(float(self.fontsize_maintitle.text())).astype(int))
            self.mpl.axes.set_xlabel(self.xtitle.text(), fontsize=np.around(float(self.fontsize_axtitle.text())).astype(int))
            self.mpl.axes.xaxis.set_tick_params(labelsize=np.around(float(self.fontsize_axlbl.text())).astype(int))
            self.mpl.axes.set_ylabel(self.ytitle.text(), fontsize=np.around(float(self.fontsize_axtitle.text())).astype(int))
            self.mpl.axes.yaxis.set_tick_params(labelsize=np.around(float(self.fontsize_axlbl.text())).astype(int))

            handles, labels = self.mpl.axes.get_legend_handles_labels()
            if ',' in self.legendpos.text():
                legend_pos = tuple(map(float,self.legendpos.text().split(',')))
            elif self.legendpos.text() in ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center']:
                legend_pos = self.legendpos.text()
            else:
                legend_pos = None
                print("Warning: did you mean one of the following: best, center right, lower center?")
            if legend_pos is not None:
                self.mpl.axes.legend(handles, labels, loc=legend_pos, fontsize=np.around(float(self.fontsize_legend.text())).astype(int), frameon=self.legend_bbox.isChecked())



            self.mpl.canvas.draw()
            # to make sure we don't screw up matplotlib for other processes, undo the xkcd style
            plt.rcdefaults()
        
    def browse_app(self):
        self.filenames = QFileDialog.getOpenFileNames(self, caption="Open spectrum file(s)", filter="H5 (*.h5);;SPE (*.spe);;CSV (*.csv)")[0]
        if len(self.filenames) != 0:
            # read in first file, to obtain data on elements and dimensions
            if(self.filenames != []):
                extension = os.path.splitext(self.filenames[0])[-1].lower()
                if extension == '.spe':
                    pass #TODO
                elif extension == '.csv':
                    pass #TODO
                elif extension == '.h5':
                    self.new_window = Poll_h5dir(self.filenames[0])
                    if self.new_window.exec_() == QDialog.Accepted:
                        self.subdirs = self.new_window.h5dir
                    self.filedir.setText('"'+'","'.join([pair for pair in map(':'.join, list(itertools.product(self.filenames, self.subdirs)))])+'"')
            self.read_files()

    def read_files(self):
            # read the data from all files/directories
            files = self.filedir.text()[1:-1].split('","')
            self.datadic = []
            for file in files:
                h5file = ':'.join(file.split(':')[0:-1]) #silly win folders may have colons in the directory...
                h5dir = file.split(':')[-1]
                # check datatype based on h5dir
                if 'spe' in h5dir:
                    datatype = 'spe'
                elif 'quant' in h5dir:
                    datatype = 'scatter'
                elif 'elyield' in h5dir:
                    datatype = 'scatter'
                elif 'detlim' in h5dir:
                    datatype = 'scatter'
                else:
                    datatype = 'spe'
                data, lines, config = Xplot_rh5(h5file, channel=h5dir)  #Todo: in principle we could also have this function look for a unit attribute to data

                if datatype == 'spe' and config is not None:
                    xvals = np.arange(len(data))*config[1]+config[0]
                else:
                    xvals = np.arange(len(data))
                self.datadic.append({'filename' : h5file,
                                     'h5dir' : h5dir,
                                     'label' : os.path.basename(h5file)+':'+h5dir,
                                     'colour' : None,
                                     'data' : data,
                                     'xvals' : xvals,
                                     'datatype' : datatype,
                                     'lines' : lines,
                                     'cfg' : config
                                     })

            # set GUI fields to the appropriate values
            self.xmin.setText("{:0.0f}".format(np.around(np.min([item["xvals"] for item in self.datadic]))))
            self.xmax.setText("{:0.0f}".format(np.around(np.max([item["xvals"] for item in self.datadic]))))
            self.ymin.setText("{:.3}".format(0.5*np.min([item["data"] for item in self.datadic])))
            self.ymax.setText("{:.3}".format(2.*np.max([item["data"] for item in self.datadic])))
            if self.datadic[0]['datatype'] == 'spe':
                self.ytitle.setText("Intensity [Counts]")
                if self.datadic[0]['cfg'] is None:
                    self.xtitle.setText("Detector Channel Number")
                else:
                    self.Eplot.setChecked(True)
                    self.Eplot_zero.setText("{:.3}".format(self.datadic[0]["cfg"][0]))
                    self.Eplot_gain.setText("{:.3}".format(self.datadic[0]["cfg"][1]))
                    self.xtitle.setText("Energy [keV]")
            elif 'quant' in self.datadic[0]['h5dir']:
                self.xtitle.setText("Atomic Number [Z]")
                self.ytitle.setText("Concentration [ppm]")
            elif 'elyield' in self.datadic[0]['h5dir']:
                self.xtitle.setText("Atomic Number [Z]")
                self.ytitle.setText("Elemental yield [(ct/s)/(ug/cm²)]")
            elif 'detlim' in self.datadic[0]['h5dir']:
                self.xtitle.setText("Atomic Number [Z]")
                self.ytitle.setText("Detection Limit [ppm]")
            

            # now adjust plot window (if new file or dir chosen, the fit results should clear and only self.rawspe is displayed)
            self.update_plot()
            
    def ctegain_invert(self):
        if self.datadic != []:
            if self.Eplot.isChecked() is True:
                self.xtitle.setText("Energy [keV]")
                if self.datadic[0]["cfg"] is not None:
                    self.xmin.setText("{:0.0f}".format(np.around(float(self.xmin.text())*self.datadic[0]["cfg"][1]+self.datadic[0]["cfg"][0])))
                    self.xmax.setText("{:0.0f}".format(np.around(float(self.xmax.text())*self.datadic[0]["cfg"][1]+self.datadic[0]["cfg"][0])))
                else:
                    self.xmin.setText("{:0.0f}".format(np.around(float(self.xmin.text())*float(self.Eplot_gain.text())+float(self.Eplot_zero.text()))))
                    self.xmax.setText("{:0.0f}".format(np.around(float(self.xmax.text())*float(self.Eplot_gain.text())+float(self.Eplot_zero.text()))))
                for item in self.datadic:
                    if item["cfg"] is not None:
                        item["xvals"] = item["xvals"] = np.arange(len(item["data"]))*float(item["cfg"][1])+float(item["cfg"][0])
                    else:
                        item["xvals"] = np.arange(len(item["data"]))*float(self.Eplot_gain.text())+float(self.Eplot_zero.text())
                self.update_plot()
            else:
                self.ctegain_update()
        
    def ctegain_update(self):
        if self.datadic != []:
            if self.Eplot.isChecked() is True:
                self.xtitle.setText("Energy [keV]")
                if self.datadic[0]["cfg"] is not None:
                    self.xmin.setText("{:0.0f}".format(np.around(float(self.xmin.text())*self.datadic[0]["cfg"][1]+self.datadic[0]["cfg"][0])))
                    self.xmax.setText("{:0.0f}".format(np.around(float(self.xmax.text())*self.datadic[0]["cfg"][1]+self.datadic[0]["cfg"][0])))
                else:
                    self.xmin.setText("{:0.0f}".format(np.around(float(self.xmin.text())*float(self.Eplot_gain.text())+float(self.Eplot_zero.text()))))
                    self.xmax.setText("{:0.0f}".format(np.around(float(self.xmax.text())*float(self.Eplot_gain.text())+float(self.Eplot_zero.text()))))
                for item in self.datadic:
                    if item["cfg"] is not None:
                        item["xvals"] = item["xvals"] = np.arange(len(item["data"]))*float(item["cfg"][1])+float(item["cfg"][0])
                    else:
                        item["xvals"] = np.arange(len(item["data"]))*float(self.Eplot_gain.text())+float(self.Eplot_zero.text())
            else:
                self.xtitle.setText("Detector Channel Number")
                if self.datadic[0]["cfg"] is not None:
                    self.xmin.setText("{:0.0f}".format(np.around((float(self.xmin.text())-self.datadic[0]["cfg"][0])/self.datadic[0]["cfg"][1])))
                    self.xmax.setText("{:0.0f}".format(np.around((float(self.xmax.text())-self.datadic[0]["cfg"][0])/self.datadic[0]["cfg"][1])))
                else:
                    self.xmin.setText("{:0.0f}".format(np.around((float(self.xmin.text())-float(self.Eplot_zero.text()))/float(self.Eplot_gain.text()))))
                    self.xmax.setText("{:0.0f}".format(np.around((float(self.xmax.text())-float(self.Eplot_zero.text()))/float(self.Eplot_gain.text()))))
                self.Eplot_zero.setText("0")
                self.Eplot_gain.setText("1")
                for item in self.datadic:
                    item["xvals"] = np.arange(len(item["data"]))*float(self.Eplot_gain.text())+float(self.Eplot_zero.text())
                
            self.update_plot()

    def change_curve_seq(self):
        if self.datadic != []:
            self.new_window = CurveSequence(self)
            self.new_window.setFocus()
            if self.new_window.exec_() == QDialog.Accepted:
                pass #TODO: copy the new info to the old dict self.datadic
            self.new_window.close()
            self.new_window = None


    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    xplot = Xplot_GUI()
    xplot.show()
    sys.exit(app.exec_())

