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
from PyQt5.QtWidgets import QCheckBox, QPushButton, QDialog, QLabel, QButtonGroup, \
        QLineEdit, QTabWidget, QFileDialog, QRadioButton, QTableWidgetItem, QFrame, \
            QListWidget, QAbstractItemView, QSplitter, QTableWidget, QHeaderView


def Xplot_rh5(h5file, channel='raw/channel00/sumspec'):
    with h5py.File(h5file, 'r') as f:
        spe = np.array(f[channel])
        spe[~np.isfinite(spe)] = 0. # remove NaN and Inf values
        
        # find detector channel
        detchnl = [detchnl for detchnl in channel.split('/') if 'channel' in detchnl]
        
        # see if names is in same channel directory, or in folder above
        if '/'.join(channel.split('/')[0:-1])+'/names' in f.keys():
            names = [n.decode('utf8') for n in f['/'.join(channel.split('/')[0:-1])+'/names']]
        elif '/'.join(channel.split('/')[0:-2])+'/names' in f.keys():
            names = [n.decode('utf8') for n in f['/'.join(channel.split('/')[0:-2])+'/names']]
        elif 'fit' in f.keys():
            if 'fit/'+detchnl[0]+'/names' in f.keys():
                names = [n.decode('utf8') for n in f['fit/'+detchnl[0]+'/names']]
            else:
                names = None
        else:
            names = None

        cfgdir = 'fit/'+detchnl[0]+'/cfg'
        if cfgdir in f.keys():
            cfg = f[cfgdir][()].decode('utf8')
        else:
            cfg = None

        # look for error values, else return None
        if channel+'_stddev' in f.keys():
            error = np.asarray(f[channel+'_stddev'])
        elif '/'.join(channel.split('/')[0:-1])+'/stddev' in f.keys():
            error = np.asarray(f['/'.join(channel.split('/')[0:-1])+'/stddev'])
        else:
            error = None
        if error is not None:
            error[~np.isfinite(error)] = 0. # remove NaN and Inf values

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

    return spe, names, zgr, error

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
                spe, names, cfg, _ = Xplot_rh5(h5, chnl[i])
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
            spe, names, cfg, _ = Xplot_rh5(h5, channel)
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
    if cfg is not None and plotelnames is True:
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
            if energy is not None:
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
        self.mainobj = mainobj
        
        # we need 5 columns: Sequence, filename (ineditable), datadir (ineditable), label, colour
        self.table_widget = QTableWidget(len(self.mainobj.datadic), 7)
        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.table_widget.verticalHeader().setVisible(False)
        layout_main.addWidget(self.table_widget)
        # Set some Title cells
        self.table_widget.setHorizontalHeaderLabels(["Sequence", "File", "Directory", "Label", "Colour", "Linetype", "Marker"])
        # Set all the subsequent datadic cells
        for index,data in enumerate(self.mainobj.datadic):
            self.table_widget.setItem(index, 0, QTableWidgetItem("{:0.0f}".format(index))) #sequence in the dictionary
            item = QTableWidgetItem(data["filename"])
            item.setFlags(item.flags() ^ Qt.ItemIsEditable) # filename
            self.table_widget.setItem(index, 1, item)
            item = QTableWidgetItem(data["h5dir"])
            item.setFlags(item.flags() ^ Qt.ItemIsEditable) # datadir in filename
            self.table_widget.setItem(index, 2, item)
            self.table_widget.setItem(index, 3,QTableWidgetItem(data["label"]))
            if data["colour"] is None:
                itemtext = "None"
            else:
                itemtext = data["colour"]
            self.table_widget.setItem(index, 4,QTableWidgetItem(itemtext))
            if data["plotline"] is None:
                itemtext = "None"
            else:
                itemtext = data["plotline"]
            self.table_widget.setItem(index, 5,QTableWidgetItem(itemtext))
            if data["plotmark"] is None:
                itemtext = "None"
            else:
                itemtext = data["plotmark"]
            self.table_widget.setItem(index, 6,QTableWidgetItem(itemtext))
        
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
        # read the table info
        nrows, ncols = self.table_widget.rowCount(), self.table_widget.columnCount()
        table = []
        for i in range(nrows):
            for j in range(ncols):
                table.append(self.table_widget.item(i,j).text())
        table = np.asarray(table).reshape(nrows, ncols)
        # Check if the sequence makes sense, i.e. if each integer appears only once etc.
        sequence = [int(i) for i in table[:,0]] #sequence is the first column
        if np.sum(abs(np.sort(sequence) - np.arange(nrows))) == 0:
            # sort datadic and table according to new sequence
            self.mainobj.datadic = np.asarray(self.mainobj.datadic)[sequence]
            for i in range(ncols):
                table[:,i] = table[sequence,i]
            for i in range(nrows):
                self.mainobj.datadic[i]["label"] = table[i,3]
                if table[i,4].lower() != 'none':
                    self.mainobj.datadic[i]["colour"] = table[i,4]
                else:
                    self.mainobj.datadic[i]["colour"] = None
                if table[i,5].lower() != 'none':
                    self.mainobj.datadic[i]["plotline"] = table[i,5]
                else:
                    self.mainobj.datadic[i]["plotline"] = None
                if table[i,6].lower() != 'none':
                    self.mainobj.datadic[i]["plotmark"] = table[i,6]
                else:
                    self.mainobj.datadic[i]["plotmark"] = None
            self.mainobj.datadic = [dic for dic in self.mainobj.datadic]
            # close spawned window and return selected elements...
            self.hide()
            super().accept()
        else:
            print("WARNING: the curve sequence column must contain unique integer values from 0 (first curve to plot) to %s (last curve to plot)" % nrows-1)


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

def ReturnLabels(labels, quantifier):
    # if quantifier is "main" only return the Ka, La and Ma energies,
    #   if quantifier is "all" return Ka, Kb, La, Lb, Lg and Ma, Mb and Mg
    from PyMca5.PyMcaPhysics.xrf import Elements

    energies = []
    labeltext = []
    for label in labels:
        ele = label.split(" ")[0]
        line = label.split(" ")[1]
        linedict = Elements._getUnfilteredElementDict(ele, None)
        if 'K' in line:
            # add Ka1 line in any case, can add others if requested
            energies.append(linedict['KL3']['energy'])
            labeltext.append(ele+r' K$\alpha$')
            if line == 'Kb' or quantifier == 'all': #There is the chance that this occurs multiple time, so will have to take unique items at the end still
                energies.append(linedict['KM3']['energy'])
                labeltext.append(ele+r' K$\beta$')

        if 'L' in line:
            # add La1 line in any case, can add others if requested
            energies.append(linedict['L3M5']['energy'])
            labeltext.append(ele+r' L$\alpha$')
            if quantifier == 'all':
                #Lb1
                energies.append(linedict['L2M4']['energy'])
                labeltext.append(ele+r' L$\beta$')
                #Lg1
                energies.append(linedict['L2N4']['energy'])
                labeltext.append(ele+r' L$\gamma$')
        if 'M' in line:
            # add Ma1 line in any case
            energies.append(linedict['M5N7']['energy'])
            labeltext.append(ele+r' M$\alpha$')
            if quantifier == 'all':
                #Lb1
                energies.append(linedict['M4N6']['energy'])
                labeltext.append(ele+r' M$\beta$')
                #Lg1
                energies.append(linedict['M3N5']['energy'])
                labeltext.append(ele+r' M$\gamma$')
    
    # sort energies and obtain unique labeltexts
    unique_labels, unique_id = np.unique(labeltext, return_index=True)
    unique_energies = [energies[i] for i in unique_id]
    sort_id = np.argsort(unique_energies)
    unique_labels = [unique_labels[i] for i in sort_id]
    unique_energies = [unique_energies[i] for i in sort_id]
    
    return unique_energies, unique_labels
    
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
        self.xmin.setValidator(QDoubleValidator(0, 1E9, 3))
        self.ymin = QLineEdit("")
        self.ymin.setMaximumWidth(500)
        axis_minmax_layout.addWidget(self.xmin,1,1)
        axis_minmax_layout.addWidget(self.ymin,2,1)
        self.xmax = QLineEdit("")
        self.xmax.setMaximumWidth(500)
        self.xmax.setValidator(QDoubleValidator(0, 1E9, 3))
        self.ymax = QLineEdit("")
        self.ymax.setMaximumWidth(500)
        axis_minmax_layout.addWidget(self.xmax,1,2)
        axis_minmax_layout.addWidget(self.ymax,2,2)
        tab_labels_layout.addLayout(axis_minmax_layout)
        tab_labels_layout.addSpacing(20)
        axis_Eplot_layout = QHBoxLayout()
        self.Eplot = QCheckBox("Convert X-axis to Energy or AtomicSymbol")
        self.Eplot.setToolTip ("Display SPE data with Energy X-axis or display scatter plot data with Atomic Symbol names.\n If zero is not 0 the X-axis is replaced by LabelNames, if gain is not 0 the top X-axis is replaced by LabelNames.")
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
        set_Xticks_layout = QHBoxLayout()
        self.setXticks = QCheckBox("Set X-ticks:")
        self.setXticks.setToolTip("Define Xticks as [[positions], [labels], rotation].\n E.g. [[57,58,59], ['La', 'Ce', 'Pr'], 90]")
        set_Xticks_layout.addWidget(self.setXticks)
        self.setXticks_values = QLineEdit("None")
        self.setXticks_values.setMaximumWidth(250)
        self.setXticks_values.setToolTip("Define Xticks as [[positions], [labels], rotation].\n E.g. [[57,58,59], ['La', 'Ce', 'Pr'], 90]")
        set_Xticks_layout.addWidget(self.setXticks_values)
        set_Xticks_layout.addStretch()
        tab_custom_layout.addLayout(set_Xticks_layout)
        tab_custom_layout.addSpacing(10)
        set_Yticks_layout = QHBoxLayout()
        self.setYticks = QCheckBox("Set Y-ticks:")
        self.setYticks.setToolTip("Define Yticks as [[positions], [labels], rotation].\n E.g. [[57,58,59], ['La', 'Ce', 'Pr'], 90]")
        set_Yticks_layout.addWidget(self.setYticks)
        self.setYticks_values = QLineEdit("None")
        self.setYticks_values.setMaximumWidth(250)
        self.setYticks_values.setToolTip("Define Yticks as [[positions], [labels], rotation].\n E.g. [[57,58,59], ['La', 'Ce', 'Pr'], 90]")
        set_Yticks_layout.addWidget(self.setYticks_values)
        set_Yticks_layout.addStretch()
        tab_custom_layout.addLayout(set_Yticks_layout)
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
        self.savgol_window = QLineEdit("5")
        self.savgol_window.setValidator(QDoubleValidator(1, 1E2, 0))
        self.savgol_window.setMaximumWidth(30)
        self.savgol_poly = QLineEdit("1")
        self.savgol_poly.setValidator(QDoubleValidator(1, 1E2, 0))
        self.savgol_poly.setMaximumWidth(30)
        self.deriv = QCheckBox("Plot Deriv")
        options_smoothderiv_layout.addWidget(self.smooth)
        options_smoothderiv_layout.addWidget(self.savgol_window)
        options_smoothderiv_layout.addWidget(self.savgol_poly)
        options_smoothderiv_layout.addWidget(self.deriv)
        tab_options_layout.addLayout(options_smoothderiv_layout)     
        tab_options_layout.addSpacing(10)
        interpolate_layout = QHBoxLayout()
        self.interpolate = QCheckBox("Interpolate data")
        interpolate_layout.addWidget(self.interpolate)
        interpolate_layout.addSpacing(5)
        interpolate_layout.addWidget(QLabel("Order:"))
        self.interpolate_order = QLineEdit("2")
        self.interpolate_order.setValidator(QDoubleValidator(1, 1E2, 0))
        self.interpolate_order.setMaximumWidth(30)
        interpolate_layout.addWidget(self.interpolate_order)
        interpolate_layout.addStretch()
        tab_options_layout.addLayout(interpolate_layout)        
        tab_options_layout.addSpacing(10)
        options_errorbar_layout = QHBoxLayout()
        self.errorbar_flag = QCheckBox("Display Errorbars: ")
        options_errorbar_layout.addWidget(self.errorbar_flag)
        self.errorbar_bars = QRadioButton("Bars")
        self.errorbar_area = QRadioButton("Area") 
        self.errorbar_bars.setChecked(True)
        options_errorbar_layout.addWidget(self.errorbar_bars)
        options_errorbar_layout.addWidget(self.errorbar_area)
        options_errorbar_layout.addStretch()
        tab_options_layout.addLayout(options_errorbar_layout)
        errorbar_nsigma_layout = QHBoxLayout()
        errorbar_nsigma_layout.addSpacing(50)
        errorbar_nsigma_layout.addWidget(QLabel("# strd dev:"))
        self.errorbar_nsigma = QLineEdit("3")
        self.errorbar_nsigma.setValidator(QDoubleValidator(1, 1E2, 0))
        self.errorbar_nsigma.setMaximumWidth(30)
        errorbar_nsigma_layout.addWidget(self.errorbar_nsigma)
        errorbar_nsigma_layout.addStretch()
        tab_options_layout.addLayout(errorbar_nsigma_layout)
        tab_options_layout.addSpacing(10)
        normtochan_layout = QHBoxLayout()
        self.normtochan = QCheckBox("Normalise to X-Value:")
        normtochan_layout.addWidget(self.normtochan)
        self.normtochan_channel = QLineEdit("")
        self.normtochan_channel.setValidator(QDoubleValidator(1, 1E9, 3))
        self.normtochan_channel.setMaximumWidth(30)
        normtochan_layout.addWidget(self.normtochan_channel)
        normtochan_layout.addStretch()
        tab_options_layout.addLayout(normtochan_layout)        
        tab_options_layout.addSpacing(10)
        omitXrange_layout = QHBoxLayout()
        self.omitXrange = QCheckBox("Omit X-range:")
        self.omitXrange.setToolTip ("Set a range for which X values should be removed from plotting. E.g. 0-20;28;56-70")
        omitXrange_layout.addWidget(self.omitXrange)
        self.omitXrange_range = QLineEdit("")
        self.omitXrange_range.setToolTip ("Set a range for which X values should be removed from plotting. E.g. 0-20;28;56-70")
        self.omitXrange_range.setMaximumWidth(80)
        omitXrange_layout.addWidget(self.omitXrange_range)
        omitXrange_layout.addStretch()
        tab_options_layout.addLayout(omitXrange_layout)
        tab_options_layout.addSpacing(10)
        options_peakid_layout = QHBoxLayout()
        options_peakid_layout.addWidget(QLabel("Peak ID:"))
        self.button_group = QButtonGroup()
        self.peakid_none = QRadioButton("None")
        self.peakid_main = QRadioButton("Main")
        self.peakid_all = QRadioButton("All")
        self.button_group.addButton(self.peakid_none)
        self.button_group.addButton(self.peakid_main)
        self.button_group.addButton(self.peakid_all)        
        self.peakid_none.setChecked(True)
        options_peakid_layout.addWidget(self.peakid_none)
        options_peakid_layout.addWidget(self.peakid_main)
        options_peakid_layout.addWidget(self.peakid_all)
        options_peakid_layout.addSpacing(20)
        self.peakid_arrows = QCheckBox("Arrows")
        options_peakid_layout.addWidget(self.peakid_arrows)
        options_peakid_layout.addStretch()
        tab_options_layout.addLayout(options_peakid_layout)
        self.scatterframe = QFrame()
        KLline_layout = QHBoxLayout()
        KLline_lbl = QLabel("Display ")
        self.show_Klines = QCheckBox("K-")
        self.show_Llines = QCheckBox("L-line data")
        self.show_Klines.setChecked(True)
        self.show_Llines.setChecked(True)
        KLline_layout.addWidget(KLline_lbl)
        KLline_layout.addWidget(self.show_Klines)
        KLline_layout.addWidget(self.show_Llines)
        KLline_layout.addStretch()
        self.scatterframe.setLayout(KLline_layout)
        tab_options_layout.addWidget(self.scatterframe)
        tab_options_layout.addStretch()
        self.tab_options.setLayout(tab_options_layout)
        self.menu_tabs.addTab(self.tab_options, "Options")
        splitter.addWidget(self.menu_tabs)        
        self.menu_tabs.setCurrentWidget(self.tab_labels)
        layout_main.addWidget(splitter)

        button_layout = QHBoxLayout()
        self.refresh = QPushButton("Refresh")
        self.refresh.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.refresh.setMaximumWidth(200)
        button_layout.addWidget(self.refresh)
        self.load_settings = QPushButton("Load Settings")
        self.load_settings.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.load_settings.setMaximumWidth(200)
        button_layout.addWidget(self.load_settings)
        self.save_settings = QPushButton("Save Settings")
        self.save_settings.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.save_settings.setMaximumWidth(200)
        button_layout.addWidget(self.save_settings)
        self.savepng = QPushButton("Save Image")
        self.savepng.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.savepng.setMaximumWidth(200)
        button_layout.addWidget(self.savepng)

        layout_main.addLayout(button_layout)
    
        # show window
        self.setLayout(layout_main)
        self.setWindowTitle('Xplot GUI')
        self.show()
        self.scatterframe.hide()

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
        self.fontsize_annot.returnPressed.connect(self.update_plot)  
        self.fontsize_axtitle.returnPressed.connect(self.update_plot)  
        self.fontsize_axlbl.returnPressed.connect(self.update_plot)  
        self.fontsize_legend.returnPressed.connect(self.update_plot)  
        self.curve_thick.returnPressed.connect(self.update_plot)  
        self.legendpos.returnPressed.connect(self.update_plot)  
        self.vert_offset.returnPressed.connect(self.update_plot)
        self.legend_bbox.stateChanged.connect(self.update_plot)
        self.curve_sequence.clicked.connect(self.change_curve_seq)
        self.smooth.stateChanged.connect(self.update_plot)
        self.savgol_window.returnPressed.connect(self.update_plot)
        self.savgol_poly.returnPressed.connect(self.update_plot)
        self.deriv.stateChanged.connect(self.update_plot) #TODO: should change ylim as well, but no idea how to do this reversible...
        self.errorbar_flag.stateChanged.connect(self.update_plot)
        self.errorbar_area.toggled.connect(self.update_plot)
        self.errorbar_bars.toggled.connect(self.update_plot)
        self.interpolate.stateChanged.connect(self.update_plot)
        self.interpolate_order.returnPressed.connect(self.update_plot)
        self.normtochan.stateChanged.connect(self.update_plot)
        self.normtochan_channel.returnPressed.connect(self.update_plot)
        self.button_group.buttonClicked.connect(self.update_plot)
        self.peakid_arrows.stateChanged.connect(self.update_plot)
        self.refresh.clicked.connect(self.update_plot)
        self.savepng.clicked.connect(self.save_png)
        self.show_Klines.stateChanged.connect(self.update_plot)
        self.show_Llines.stateChanged.connect(self.update_plot)
        self.omitXrange.stateChanged.connect(self.update_plot)
        self.omitXrange_range.returnPressed.connect(self.update_plot)
        self.setXticks.stateChanged.connect(self.update_plot)
        self.setXticks_values.returnPressed.connect(self.update_plot)
        self.setYticks.stateChanged.connect(self.update_plot)
        self.setYticks_values.returnPressed.connect(self.update_plot)


    def update_plot(self):
        if self.datadic:
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
            normfactor = []
            if self.normtochan.isChecked() is True:
                xval2norm = float(self.normtochan_channel.text())
                xvals = self.datadic[0]["xvals"]*float(self.xmult.text())
                x_id = np.max(np.where(xvals <= xval2norm))
                if x_id:
                    yval2norm = (self.datadic[0]["data"]*float(self.ymult.text()))[x_id]
                    for index, item in enumerate(self.datadic):
                         xvals = item["xvals"]*float(self.xmult.text())
                         x_id = np.max(np.where(xvals <= xval2norm))
                         if x_id:
                             normfactor.append((item["data"]*float(self.ymult.text()))[x_id]/yval2norm)
                         else:
                            normfactor.append(1)
                            print("Warning: X value to normalise to is not in range of curve %i" %index)
                else:
                    print("Warning: X value to normalise to is not in range of curve 0")
            if self.Eplot.isChecked() is True and self.datadic[0]["datatype"] == "scatter":
                from PyMca5.PyMcaPhysics.xrf import Elements
                all_lines = []
                all_ydata = []
                for index, item in enumerate(self.datadic):
                    lines = np.asarray(item["lines"])
                    ydata = item["data"]*float(self.ymult.text())
                    if self.show_Klines.isChecked() is False:
                        # remove data points where item["lines"] represents a K line
                        nonK_id = [i for i,k in enumerate(lines) if "K" not in k.split(" ")[1]]
                        lines = lines[nonK_id]
                        ydata = ydata[nonK_id]
                    if self.show_Llines.isChecked() is False:
                        # remove data points where item["lines"] represents a K line
                        nonL_id = [i for i,k in enumerate(lines) if "L" not in k.split(" ")[1]]
                        lines = lines[nonL_id]
                        ydata = ydata[nonL_id]
                    if normfactor:
                        ydata = ydata/normfactor[index]
                    for line in lines:
                        all_lines.append(line)
                    for y in ydata:
                        all_ydata.append(y)
                unique_lines = np.unique(np.asarray(all_lines, dtype='str'))
                all_lines = np.asarray(all_lines, dtype='str')
                all_ydata = np.asarray(all_ydata)
                ydata_av = np.asarray([np.average(all_ydata[np.where(all_lines == tag)]) for tag in unique_lines])
                lines_z = np.asarray([Elements.getz(name.split(" ")[0]) for name in unique_lines])
                unique_z = np.unique(lines_z)
                if unique_z.size != lines_z.size:
                    new_z = unique_z
                    new_labels = []
                    for i in range(0, unique_z.size):
                        z_indices = np.where(lines_z == unique_z[i])[0].astype(int)
                        y_order = np.flip(np.argsort(ydata_av[z_indices]))
                        new_labels.append("\n".join(unique_lines[z_indices][y_order]))
                    new_labels = np.asarray(new_labels)
                else:
                    new_z = np.asarray(lines_z)
                    new_labels = np.asarray(unique_lines)
                new_labels = new_labels[np.argsort(new_z)]
                new_z = new_z[np.argsort(new_z)]
            curves = []
            for index, item in enumerate(self.datadic):
                # Apply vertical offset to all curves, taking into account a coordinate transform as y-axis could be log scaled
                lines = np.asarray([line for line in item["lines"]])
                xdata = np.asarray(item["xvals"]*float(self.xmult.text()))
                ydata = np.asarray(item["data"]*float(self.ymult.text()))
                if normfactor:
                    ydata = ydata/normfactor[index]
                if item["error"] is None:
                    yerr = None
                else:
                    yerr = np.asarray(item["error"]*0.)
                    np.divide(item["error"],item["data"], out=yerr, where=item["data"]!=0) #relative error, will convert to absolute again during plotting
                if self.omitXrange.isChecked() is True:
                    Xomission = self.omitXrange_range.text()
                    if ';' in Xomission: #several sections provided, so loop through them
                        for omit in Xomission.split(';'):
                            if '-' in omit: #a range provided, so omit values within the range
                                keep_id = np.where([(xdata < float(omit.split('-')[0])) | (xdata > float(omit.split('-')[1]))])[1]
                                ydata = ydata[keep_id]
                                xdata = xdata[keep_id]
                                if self.datadic[0]["datatype"] == "scatter":
                                    lines = lines[keep_id]
                                if yerr is not None:
                                    yerr = yerr[keep_id]
                            else: #only a single value provided, only omit this one
                                keep_id = np.where(xdata != float(omit))
                                ydata = ydata[keep_id]
                                xdata = xdata[keep_id]
                                if self.datadic[0]["datatype"] == "scatter":
                                    lines = lines[keep_id]
                                if yerr is not None:
                                    yerr = yerr[keep_id]
                    elif Xomission != '':
                        if '-' in Xomission: #a range provided, so omit values within the range
                            keep_id = np.where([(xdata < float(Xomission.split('-')[0])) | (xdata > float(Xomission.split('-')[1]))])[1]
                            ydata = ydata[keep_id]
                            xdata = xdata[keep_id]
                            if self.datadic[0]["datatype"] == "scatter":
                                lines = lines[keep_id]
                            if yerr is not None:
                                yerr = yerr[keep_id]
                        else: #only a single value provided, only omit this one
                            keep_id = np.where(xdata != float(Xomission))
                            ydata = ydata[keep_id]
                            xdata = xdata[keep_id]
                            if self.datadic[0]["datatype"] == "scatter":
                                lines = lines[keep_id]
                            if yerr is not None:
                                yerr = yerr[keep_id]
                if self.datadic[0]["datatype"] == "scatter":
                    if self.show_Klines.isChecked() is False:
                        # remove data points where item["lines"] represents a K line
                        nonK_id = [i for i,k in enumerate(lines) if "K" not in k.split(" ")[1]]
                        lines = lines[nonK_id]
                        xdata = xdata[nonK_id]
                        ydata = ydata[nonK_id]
                        if yerr is not None:
                            yerr = yerr[nonK_id]
                    if self.show_Llines.isChecked() is False:
                        # remove data points where item["lines"] represents a L line
                        nonL_id = [i for i,k in enumerate(lines) if "L" not in k.split(" ")[1]]
                        lines = lines[nonL_id]
                        xdata = xdata[nonL_id]
                        ydata = ydata[nonL_id]
                        if yerr is not None:
                            yerr = yerr[nonL_id]
                if self.smooth.isChecked() is True:
                    if 'savgol_filter' not in dir():
                        from scipy.signal import savgol_filter
                    ydata = savgol_filter(ydata, int(self.savgol_window.text()), int(self.savgol_poly.text()))
                if self.deriv.isChecked() is True:
                    ydata = np.gradient(ydata, xdata)
                if float(self.vert_offset.text()) != 0.0:
                    newTransform = self.mpl.axes.transScale + self.mpl.axes.transLimits
                    for i in range(len(ydata)):
                        axcoords = newTransform.transform([xdata[i], ydata[i]])
                        axcoords[1] += float(self.vert_offset.text())*index # add vertical offset in relative axes height
                        ydata[i] = newTransform.inverted().transform(axcoords)[1]
                curves.append(self.mpl.axes.plot(xdata, ydata, label=item["label"], linewidth=float(self.curve_thick.text()), 
                                                 linestyle=item["plotline"], marker=item["plotmark"], color=item["colour"]))
                if index == 0:
                    self.mpl.axes.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
                    if self.setXticks.isChecked() is True:
                        if self.setXticks_values.text() == "" or self.setXticks_values.text().lower() == "none":
                            self.mpl.axes.set_xticks([])
                        else:
                            xtick_list = eval(self.setXticks_values.text())
                            if len(xtick_list) >= 3:
                                rotation = xtick_list[2]
                            else:
                                rotation = 0
                            self.mpl.axes.set_xticks(xtick_list[0], labels=xtick_list[1], fontsize=np.around(float(self.fontsize_annot.text())).astype(int), rotation=rotation)
                    if self.setYticks.isChecked() is True:
                        if self.setYticks_values.text() == "" or self.setYticks_values.text().lower() == "none":
                            self.mpl.axes.set_yticks([])
                            self.mpl.axes.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                        else:
                            ytick_list = eval(self.setYticks_values.text())
                            if len(ytick_list) >= 3:
                                rotation = ytick_list[2]
                            else:
                                rotation = 0
                            self.mpl.axes.set_yticks(ytick_list[0], labels=ytick_list[1], fontsize=np.around(float(self.fontsize_annot.text())).astype(int), rotation=rotation)
                    if self.Eplot.isChecked() is True and self.datadic[0]["datatype"] == "scatter":
                        self.mpl.axes.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
                        if float(self.Eplot_gain.text()) == 1:
                            secaxx = self.mpl.axes.secondary_xaxis('top')
                            secaxx.set_xticks(new_z, labels=new_labels, fontsize=np.around(float(self.fontsize_annot.text())).astype(int))
                        if float(self.Eplot_zero.text()) == 1:
                            self.mpl.axes.set_xticks(new_z, labels=new_labels, fontsize=np.around(float(self.fontsize_annot.text())).astype(int))
                # display error values
                if self.errorbar_flag.isChecked() is True and yerr is not None:
                    if self.errorbar_bars.isChecked() is True:
                        eb = self.mpl.axes.errorbar(xdata, ydata, yerr*ydata*float(self.errorbar_nsigma.text()), ecolor=curves[-1][0].get_color(), 
                                                    fmt='none', capsize=float(self.curve_thick.text())*2, elinewidth=float(self.curve_thick.text()))
                        eb[-1][0].set_linestyle('solid')
                    elif self.errorbar_area.isChecked() is True:
                        self.mpl.axes.fill_between(xdata, ydata-(yerr*ydata*float(self.errorbar_nsigma.text())), 
                                                        ydata+(yerr*ydata*float(self.errorbar_nsigma.text())), alpha=0.3, color=curves[-1][0].get_color())
                # fit curve through points and plot as dashed line in same color
                if self.interpolate.isChecked():
                    polyfactor = np.around(float(self.interpolate_order.text())).astype(int)
                    try:
                        if self.ylinlog.isChecked() is True:
                            fit_par = np.polyfit(xdata, np.log10(ydata), polyfactor) #TODO: need to prune 0 values! RuntimeWarning: divide by zero encountered in log10
                        else:
                            fit_par = np.polyfit(xdata, ydata, polyfactor)
                        func = np.poly1d(fit_par)
                        fit_x = np.linspace(np.min(xdata), np.max(xdata), num=np.around(np.max(xdata)-np.min(xdata)).astype(int)*2)
                        if self.ylinlog.isChecked() is True:
                            fit_y = 10.**(func(fit_x))
                        else:
                            fit_y = func(fit_x)
                        self.mpl.axes.plot(fit_x, fit_y, linestyle='--', color=curves[-1][0].get_color())
                    except Exception:
                        print("Error: polynomial interpolation did not work out. Perhaps something is wrong with the data?")

            if self.axboxtype_single.isChecked() is True:
                self.mpl.axes.spines[['right', 'top']].set_visible(False)
            if self.graphtitle.text() != "":
                self.mpl.axes.set_title(self.graphtitle.text(), fontsize=np.around(float(self.fontsize_maintitle.text())).astype(int))
            self.mpl.axes.set_xlabel(self.xtitle.text(), fontsize=np.around(float(self.fontsize_axtitle.text())).astype(int))
            self.mpl.axes.xaxis.set_tick_params(labelsize=np.around(float(self.fontsize_axlbl.text())).astype(int))
            self.mpl.axes.set_ylabel(self.ytitle.text(), fontsize=np.around(float(self.fontsize_axtitle.text())).astype(int))
            self.mpl.axes.yaxis.set_tick_params(labelsize=np.around(float(self.fontsize_axlbl.text())).astype(int))
            
            # add peak ID labels to curve
            # make list of all labels used across all spectra
            if self.peakid_none.isChecked() is not True:
                labels = []
                for item in self.datadic:
                    if item["lines"] is not None:
                        for n in item["lines"]:
                            if n == 'Rayl' and item['cfg'] is not None:
                                if type(item["cfg"][2]) is type(list()):
                                    if item["cfg"][2][0] is not None and item["cfg"][2][0] != "None":
                                        labels.append(n+':'+"{:.3}".format(item["cfg"][2][0]))
                                elif item["cfg"][2] is not None and item["cfg"][2] != "None":
                                    labels.append(n+':'+"{:.3}".format(item["cfg"][2]))
                            elif n != 'Compt' and n != 'Rayl':
                                labels.append(n)
                if labels:
                    labelpeaks = ''
                    if self.peakid_main.isChecked() is True:
                        labelpeaks = 'main'
                        NiterMax = 99
                    elif self.peakid_all.isChecked() is True:
                        labelpeaks = 'all'
                        NiterMax = 299
                    labels = np.unique(np.asarray(labels))
                    # remove Rayl from this list, and treat them separately
                    linelabels = [lbl for lbl in labels if 'Rayl' not in lbl]
                    # obtain the corresponding energies for each label
                    labelenergies, labeltext = ReturnLabels(linelabels, labelpeaks)
                    labels = [lbl for lbl in labels if 'Rayl' in lbl]
                    labels = np.unique(labels)
                    for lbl in labels:
                        if 'Rayl' in lbl:
                            labelenergies.append(float(lbl.split(":")[1]))
                            labeltext.append("Rayleigh")
                    # sort energies
                    sort_id = np.argsort(labelenergies)
                    labelenergies = [labelenergies[i] for i in sort_id]
                    labeltext = [labeltext[i] for i in sort_id]
                    handles, labels = self.mpl.axes.get_legend_handles_labels()
                    texts = []
                    yval = []
                    render = self.mpl.canvas.renderer
                    newTransform = self.mpl.axes.transScale + self.mpl.axes.transLimits
                    for index, text in enumerate(labeltext):
                        idx = max(np.where(handles[0].get_xdata() <= labelenergies[index])[-1])
                        yval.append(max([hand.get_ydata()[idx] for hand in handles]))
                        # plot the text label X% above this value
                        temp = newTransform.transform([labelenergies[index],yval[-1]])
                        temp[1] += 0.025
                        xyann = newTransform.inverted().transform(temp)
                        texts.append(self.mpl.axes.annotate(text, (labelenergies[index], yval[-1]), xytext=xyann, 
                                                            xycoords='data', horizontalalignment='center', verticalalignment='bottom', rotation=0,
                                                            fontsize=np.around(float(self.fontsize_annot.text())).astype(int)))

                    # obtain maxima of all curves in axes coordinates so we can check whether there is overlap with the labels
                    curve_x = np.arange(1001)/1000 #array from 0 to 1, representing curve axis values
                    curve_y = np.arange(len(curve_x)).astype(float)*0.
                    for curve in curves:
                        xdata = [newTransform.transform(x)[0] for x in zip(curve[0].get_xdata(),curve[0].get_ydata())]
                        ydata = [newTransform.transform(x)[1] for x in zip(curve[0].get_xdata(),curve[0].get_ydata())]
                        for i in range(1,len(curve_x)):
                            ymax = [ydata[nr] for nr in np.where(xdata <= curve_x[i])[0] if xdata[nr] > curve_x[i-1]]
                            if ymax:
                                if np.max(ymax) > curve_y[i]:
                                    curve_y[i] = np.max(ymax)
                    # iterate over the labels and adjust positions where necessary
                    changed = 1
                    N_iter = 0
                    while changed != 0 and N_iter <= NiterMax:
                        changed = 0
                        N_iter+=1
                        txt_clim_ax = []
                        for text in texts:
                            bbox = text.get_tightbbox(render)  #get_window_extent(render)
                            # these are the coordinates (axis) of the text bounding boxes, leftbot to righttop
                            txt_clim_ax.append(np.vstack( (self.mpl.axes.transAxes.inverted().transform([bbox.xmin, bbox.ymin]),
                                                self.mpl.axes.transAxes.inverted().transform([bbox.xmax, bbox.ymax])) ))
                        # now go through the coordinates and make sure they are not overlapping
                        for i, coord in enumerate(txt_clim_ax):
                            # see if this label overlaps with curve or is below it. If so, translate it to above the curve
                            ymax_curve = [curve_y[nr] for nr in np.where(curve_x >= coord[0,0])[0] if curve_x[nr] < coord[1,0] ]
                            if ymax_curve:
                                if np.max(ymax_curve)*1.025 > coord[0,1]:
                                    new_ymin = np.max(ymax_curve)*1.025
                                    txt_clim_ax[i][1,1] = new_ymin+(txt_clim_ax[i][1,1]-txt_clim_ax[i][0,1])
                                    txt_clim_ax[i][0,1] = new_ymin
                                    texts[i].set(position=(labelenergies[i], newTransform.inverted().transform((coord[0,0], new_ymin))[1]))
                                    changed +=1
                            # see if there is overlap in x between this label and all previous ones. If not: all's fine.
                            previous_lbls_xoverlap = [previous for previous in txt_clim_ax[:i] if coord[0,0] < previous[1,0] ]
                            if previous_lbls_xoverlap:
                                # if there is an x overlap, see if there is a y overlap with those labels that overlap in x
                                # first check if we can fit the label in between the curve and the other labels
                                if (np.min([previous[0,1] for previous in previous_lbls_xoverlap])-np.max(ymax_curve)*1.025) > (txt_clim_ax[i][1,1]-txt_clim_ax[i][0,1]):
                                    new_ymin = np.max(ymax_curve)*1.025
                                    txt_clim_ax[i][1,1] = new_ymin+(txt_clim_ax[i][1,1]-txt_clim_ax[i][0,1])
                                    txt_clim_ax[i][0,1] = new_ymin
                                    texts[i].set(position=(labelenergies[i], newTransform.inverted().transform((coord[0,0], new_ymin))[1]))
                                    changed +=1
                                previous_lbls_yoverlap = [previous for previous in previous_lbls_xoverlap if (previous[0,1] <= coord[0,1] and coord[0,1] < previous[1,1]) or (previous[0,1] < coord[1,1] and coord[1,1] <= previous[1,1])]
                                                          #ymin tussen previous y-range of ymax tussen previous y-range
                                if previous_lbls_yoverlap:
                                    # if we cannot put it between the curve and others, let's just put it above them
                                    new_ymin = np.max([previous[1,1] for previous in previous_lbls_yoverlap])
                                    txt_clim_ax[i][1,1] = new_ymin+(txt_clim_ax[i][1,1]-txt_clim_ax[i][0,1])
                                    txt_clim_ax[i][0,1] = new_ymin
                                    texts[i].set(position=(labelenergies[i], newTransform.inverted().transform((coord[0,0], new_ymin))[1]))
                                    changed +=1
                    print("Label positioning finished in %i iterations." % N_iter)
                    if self.peakid_arrows.isChecked() is True:
                        for text in texts:
                            self.mpl.axes.annotate("", xy=text.xy, xycoords='data', xytext=text.xyann, textcoords='data',
                                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", alpha=0.5))

                    
                    
            handles, labels = self.mpl.axes.get_legend_handles_labels()
            if ',' in self.legendpos.text():
                legend_pos = tuple(map(float,self.legendpos.text().split(',')))
            elif self.legendpos.text() in ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 
                                           'center left', 'center right', 'lower center', 'upper center', 'center']:
                legend_pos = self.legendpos.text()
            else:
                legend_pos = None
                print("Warning: did you mean one of the following: best, center right, lower center?")
            if legend_pos is not None:
                self.mpl.axes.legend(handles, labels, loc=legend_pos, fontsize=np.around(float(self.fontsize_legend.text())).astype(int), 
                                     frameon=self.legend_bbox.isChecked())



            self.mpl.canvas.draw()
            # to make sure we don't screw up matplotlib for other processes, undo the xkcd style
            plt.rcdefaults()

    def save_png(self):
        imagename = QFileDialog.getSaveFileName(self, caption="Save PNG in:", filter="PNG (*.png)")[0]
        if len(imagename) != 0:
            self.mpl.canvas.print_figure(imagename, dpi=300)
        
    def browse_app(self):
        self.filenames = QFileDialog.getOpenFileNames(self, caption="Open spectrum file(s)", filter="H5 (*.h5);;SPE (*.spe);;CSV (*.csv);;NXS (*.nxs)")[0]
        if len(self.filenames) != 0:
            # read in first file, to obtain data on elements and dimensions
            if(self.filenames != []):
                extension = os.path.splitext(self.filenames[0])[-1].lower()
                if extension == '.spe':
                    pass #TODO
                elif extension == '.csv':
                    pass #TODO
                elif extension == '.h5' or extension == '.nxs':
                    self.new_window = Poll_h5dir(self.filenames[0])
                    if self.new_window.exec_() == QDialog.Accepted:
                        self.subdirs = self.new_window.h5dir
                    self.filedir.setText('"'+'","'.join([pair for pair in map(':'.join, list(itertools.product(self.filenames, self.subdirs)))])+'"')
            self.read_files()

    def read_files(self):
            # read the data from all files/directories
            files = self.filedir.text()[1:-1].split('","')
            self.datadic = []
            from matplotlib.lines import Line2D
            marks = [mark for mark in Line2D.markers.keys()][2:18]
            for index, file in enumerate(files):
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
                data, lines, config, error = Xplot_rh5(h5file, channel=h5dir)  #Todo: in principle we could also have this function look for a unit attribute to data

                linetype = '-'
                marker = ''
                if datatype == 'spe' and config is not None:
                    xvals = np.arange(len(data)).astype(float)*config[1]+config[0]
                elif datatype == 'spe' and config is None:
                    xvals = np.arange(len(data)).astype(float)
                elif datatype == 'scatter': # all scatter types need x-axis as Z based on lines
                    from PyMca5.PyMcaPhysics.xrf import Elements
                    xvals = np.asarray([Elements.getz(label.split(" ")[0]) for label in lines])
                    linetype = ''
                    marker = marks[index%len(marks)]
                else:
                    xvals = np.arange(len(data)).astype(float)
                 
                if error is None and datatype == 'spe':
                    error = np.sqrt(data)
                self.datadic.append({'filename' : h5file,
                                     'h5dir' : h5dir,
                                     'label' : os.path.basename(h5file)+':'+h5dir,
                                     'colour' : None,
                                     'plotline' : linetype,
                                     'plotmark' : marker,
                                     'data' : data,
                                     'error' : error,
                                     'xvals' : xvals,
                                     'datatype' : datatype,
                                     'lines' : lines,
                                     'cfg' : config
                                     })

            # set GUI fields to the appropriate values
            self.ymax.setText("{:.3}".format(1.5*np.max([item["data"] for item in self.datadic])))
            self.ymin.setText("{:.3}".format(0.5*np.min([np.min(item["data"][np.where(item["data"]!=0)]) for item in self.datadic])))
            if self.datadic[0]['datatype'] == 'spe':
                self.scatterframe.hide()
                self.ymin.setText("{:.3}".format(0.5*np.min([item["data"] for item in self.datadic])))
                self.ytitle.setText("Intensity [Counts]")
                if self.datadic[0]['cfg'] is None:
                    self.xtitle.setText("Detector Channel Number")
                else:
                    self.Eplot.setChecked(True)
                    self.Eplot_zero.setText("{:.3}".format(self.datadic[0]["cfg"][0]))
                    self.Eplot_gain.setText("{:.3}".format(self.datadic[0]["cfg"][1]))
                    self.xtitle.setText("Energy [keV]")
            elif 'quant' in self.datadic[0]['h5dir']:
                self.scatterframe.show()
                self.xtitle.setText("Atomic Number [Z]")
                self.ytitle.setText("Concentration [ppm]")
            elif 'elyield' in self.datadic[0]['h5dir']:
                self.scatterframe.show()
                self.xtitle.setText("Atomic Number [Z]")
                self.ytitle.setText("Elemental yield [(ct/s)/(ug/cm²)]")
            elif 'detlim' in self.datadic[0]['h5dir']:
                self.scatterframe.show()
                self.xtitle.setText("Atomic Number [Z]")
                self.ytitle.setText("Detection Limit [ppm]")
            self.xmin.setText("{:.3}".format(np.min([np.min(item["xvals"].astype(float)) for item in self.datadic])))
            self.xmax.setText("{:.3}".format(np.max([np.max(item["xvals"].astype(float)) for item in self.datadic])))
        

            # now adjust plot window (if new file or dir chosen, the fit results should clear and only self.rawspe is displayed)
            self.update_plot()
            
    def ctegain_invert(self):
        if self.datadic:
            if self.Eplot.isChecked() is True and self.datadic[0]["datatype"] == 'spe':
                self.xtitle.setText("Energy [keV]")
                if self.datadic[0]["cfg"] is not None:
                    self.xmin.setText("{:.3}".format(np.around(float(self.xmin.text())*float(self.datadic[0]["cfg"][1])+float(self.datadic[0]["cfg"][0]))))
                    self.xmax.setText("{:.3}".format(np.around(float(self.xmax.text())*float(self.datadic[0]["cfg"][1])+float(self.datadic[0]["cfg"][0]))))
                else:
                    self.xmin.setText("{:.3}".format(np.around(float(self.xmin.text())*float(self.Eplot_gain.text())+float(self.Eplot_zero.text()))))
                    self.xmax.setText("{:.3}".format(np.around(float(self.xmax.text())*float(self.Eplot_gain.text())+float(self.Eplot_zero.text()))))
                for item in self.datadic:
                    if item["cfg"] is not None:
                        item["xvals"] = item["xvals"] = np.arange(len(item["data"]))*float(item["cfg"][1])+float(item["cfg"][0])
                    else:
                        item["xvals"] = np.arange(len(item["data"]))*float(self.Eplot_gain.text())+float(self.Eplot_zero.text())
                self.update_plot()
            else:
                self.ctegain_update()
        
    def ctegain_update(self):
        if self.datadic:
            old_xmin = self.xmin.text()
            if old_xmin == '': old_xmin = 0
            old_xmax = self.xmax.text()
            if old_xmax == '': old_xmax = 0
            if self.datadic[0]["datatype"] == 'spe':
                if self.Eplot.isChecked() is True:
                    self.xtitle.setText("Energy [keV]")
                    if self.datadic[0]["cfg"] is not None:
                        self.xmin.setText("{:.3}".format(np.around(float(old_xmin)*float(self.datadic[0]["cfg"][1])+float(self.datadic[0]["cfg"][0]))))
                        self.xmax.setText("{:.3}".format(np.around(float(old_xmax)*float(self.datadic[0]["cfg"][1])+float(self.datadic[0]["cfg"][0]))))
                    else:
                        self.xmin.setText("{:.3}".format(np.around(float(old_xmin)*float(self.Eplot_gain.text())+float(self.Eplot_zero.text()))))
                        self.xmax.setText("{:.3}".format(np.around(float(old_xmax)*float(self.Eplot_gain.text())+float(self.Eplot_zero.text()))))
                    for item in self.datadic:
                        if item["cfg"] is not None:
                            item["xvals"] = item["xvals"] = np.arange(len(item["data"]))*float(item["cfg"][1])+float(item["cfg"][0])
                        else:
                            item["xvals"] = np.arange(len(item["data"]))*float(self.Eplot_gain.text())+float(self.Eplot_zero.text())
                else:
                    self.xtitle.setText("Detector Channel Number")
                    if self.datadic[0]["cfg"] is not None:
                        self.xmin.setText("{:.3}".format(np.around((float(old_xmin)-self.datadic[0]["cfg"][0])/self.datadic[0]["cfg"][1])))
                        self.xmax.setText("{:.3}".format(np.around((float(old_xmax)-self.datadic[0]["cfg"][0])/self.datadic[0]["cfg"][1])))
                    else:
                        self.xmin.setText("{:.3}".format(np.around((float(old_xmin)-float(self.Eplot_zero.text()))/float(self.Eplot_gain.text()))))
                        self.xmax.setText("{:.3}".format(np.around((float(old_xmax)-float(self.Eplot_zero.text()))/float(self.Eplot_gain.text()))))
                    self.Eplot_zero.setText("0")
                    self.Eplot_gain.setText("1")
                    for item in self.datadic:
                        item["xvals"] = np.arange(len(item["data"]))*float(self.Eplot_gain.text())+float(self.Eplot_zero.text())
                
            self.update_plot()

    def change_curve_seq(self):
        if self.datadic:
            self.new_window = CurveSequence(self)
            self.new_window.setFocus()
            if self.new_window.exec_() == QDialog.Accepted:
                self.update_plot()
            self.new_window.close()
            self.new_window = None


    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    xplot = Xplot_GUI()
    xplot.show()
    sys.exit(app.exec_())

