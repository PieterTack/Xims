# -*- coding: utf-8 -*-
"""
Plotims_gui

Images mutliple ims/h5 arrays and allows for the addition of
    scale bars, color bars etc using GUI interface
Based on plotims_gui from IDL
"""
import sys
sys.path.insert(1, 'D:/School/PhD/python_pro/plotims')

import plotims as IMS
import matplotlib
matplotlib.use('Qt5Agg') #Render to Pyside/PyQt canvas
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import h5py


from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator
from PyQt5.QtWidgets import QApplication, QDialog, QWidget, QHBoxLayout, \
    QGridLayout, QVBoxLayout, QCheckBox, QGroupBox, QPushButton, QLabel, \
    QRadioButton, QLineEdit, QTabWidget, QFileDialog, QComboBox
    


class Poll_h5dir(QDialog):
    def __init__(self, h5file, parent=None):
        super(Poll_h5dir, self).__init__(parent)
        # extract all Dataset paths from the H5 file
        f = h5py.File(h5file, 'r')
        self.paths = self.descend(f, paths=None)
        f.close()
        
        # spawn screen allowing the user to select a given path
        #   in this case, due to plotims only accepting ims directories, prune the list
        self.paths = [path for path in self.paths if 'ims' in path]        
        # build widgets
        layout = QVBoxLayout()
        self.task = QLabel('Select your H5 file directory of choice:')
        layout.addWidget(self.task)
        self.path_select = QComboBox()
        self.path_select.addItems(self.paths)
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
        self.h5dir = self.paths[self.path_select.currentIndex()]
        # close spawned window and return selected elements...
        self.hide()
        super().accept()
        


class El_sel_window(QDialog):
    
    def __init__(self, el_array, el_selection, parent=None):
        super(El_sel_window, self).__init__(parent)
        self.parent = parent
        self.el_array = el_array
        self.el_selection = el_selection
        # build widgets
        layout = QVBoxLayout()
        self.opt_layout = QGridLayout()
        self.buts = QGroupBox("Select Elements:")
        self.buts.setLayout(self.opt_layout)
        layout.addWidget(self.buts)
        self.add_checkbox()
        but_layout = QHBoxLayout()
        self.all_none = QGroupBox()
        self.all_none.setFlat(True)
        self.all_none.setStyleSheet("border:0")
        self.all = QRadioButton("All")
        but_layout.addWidget(self.all)
        self.none = QRadioButton("None")
        but_layout.addWidget(self.none)
        self.select = QPushButton("Select")
        but_layout.addWidget(self.all_none)
        but_layout.addWidget(self.select)
        layout.addLayout(but_layout)        
        # show window
        self.setLayout(layout)
        self.show()
        # handle events
        self.select.clicked.connect(self.read_boxes)
        self.all.clicked.connect(self.select_all)
        self.none.clicked.connect(self.select_none)
        
    def add_checkbox(self):
        row = 0
        col = 0
        for i in range(0, len(self.el_array)): 
            self.checkbox = QCheckBox(self.el_array[i])
            self.opt_layout.addWidget(self.checkbox, row, col)
            if(self.el_array[i] in self.el_selection):
                self.checkbox.setChecked(True)
            col += 1
            if(col == 3):
                row += 1
                col = 0
           
    def read_boxes(self):
        self.el_selection = list("")
        for i in range(self.opt_layout.count()):
            self.current_box = self.opt_layout.itemAt(i)
            if self.current_box.widget().isChecked() :
                self.el_selection.append(self.el_array[i])
        # close spawned window and return selected elements...
        self.hide()
        super().accept()
        
    def select_all(self):
        if self.all.isChecked():
            for i in range(self.opt_layout.count()):
                self.current_box = self.opt_layout.itemAt(i)
                self.current_box.widget().setChecked(True)
            self.all.setChecked(False)
    
    def select_none(self):
        if self.none.isChecked():
            for i in range(self.opt_layout.count()):
                self.current_box = self.opt_layout.itemAt(i)
                self.current_box.widget().setChecked(False)
            self.none.setChecked(False)
                

class Plotims(QDialog):
    
    def __init__(self, parent=None):
        super(Plotims, self).__init__(parent)

        # some variables to store data
        self.el_selection = ""
        self.element_array = [""]
        self.ims_data = IMS.ims()
        self.units = ["Å","nm", "µm", "mm", "cm"]
        self.rot_angle = 0
        
        # create widgets

        # create main layout for widgets
        layout_main = QVBoxLayout()
        layout_browse = QHBoxLayout()
        layout_info = QHBoxLayout()
        layout_pixsize = QHBoxLayout()
        layout_option0 = QHBoxLayout()
        layout_option1 = QHBoxLayout()
        layout_tabs = QHBoxLayout()
        layout_element_select = QHBoxLayout()
        layout_cb_title = QHBoxLayout()
        layout_colortable = QHBoxLayout()
        layout_cb_dir = QHBoxLayout()
        layout_filetype = QHBoxLayout()
        layout_exe_buts = QHBoxLayout()
        
        # main widgets
        self.label_select = QLabel("Select image files:")
        layout_main.addWidget(self.label_select)
        # browse widgets
        self.filedir = QLineEdit("")
        self.browse = QPushButton("...")
        self.browse.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.browse.setMaximumWidth(25)
        layout_browse.addWidget(self.filedir)
        layout_browse.addWidget(self.browse)
        layout_main.addLayout(layout_browse)
        # info widgets
        layout_info.addSpacing(50)
        layout_info.addWidget(QLabel("ims info:   #X"))
        self.npix_x = QLabel("---")
        self.npix_x.setMaximumWidth(25)
        layout_info.addWidget(self.npix_x)
        layout_info.addWidget(QLabel(" #Y"))
        self.npix_y = QLabel("---")
        self.npix_y.setMaximumWidth(25)
        layout_info.addWidget(self.npix_y)
        layout_info.addWidget(QLabel(" #elem"))
        self.nelem = QLabel("---")
        self.nelem.setMaximumWidth(25)
        layout_info.addWidget(self.nelem)
        self.pix_scale = QCheckBox("Scale to pixel size")
        layout_info.addWidget(self.pix_scale)
        layout_info.addStretch()
        layout_main.addLayout(layout_info)
        # pixelsize widgets
        layout_pixsize.addSpacing(50)
        self.pix_label_x = QLabel("Pixelsize (micrometer):  X:")
        layout_pixsize.addWidget(self.pix_label_x)
        self.pix_label_x.setEnabled(False)
        self.pix_x = QLineEdit("---")
        self.pix_x.setMaximumWidth(25)
        self.pix_x.setValidator(QDoubleValidator(-1E6, 1E6,3))
        layout_pixsize.addWidget(self.pix_x)
        self.pix_x.setEnabled(False)
        self.pix_label_y = QLabel(" Y:")
        layout_pixsize.addWidget(self.pix_label_y)
        self.pix_label_y.setEnabled(False)
        self.pix_y = QLineEdit("---")
        self.pix_y.setMaximumWidth(25)
        self.pix_y.setValidator(QDoubleValidator(-1E6, 1E6,3))
        layout_pixsize.addWidget(self.pix_y)
        self.pix_y.setEnabled(False)
        layout_pixsize.addStretch()
        layout_main.addLayout(layout_pixsize)
        # data option button widgets
        data_opts_layout = QHBoxLayout()
        self.data_opts = QGroupBox("Data options:")
        self.data_opts_bin = QCheckBox("binning")
        data_opts_layout.addWidget(self.data_opts_bin)
        self.data_opts_ratio = QCheckBox("ratio")
        data_opts_layout.addWidget(self.data_opts_ratio)
        self.data_opts_resize = QCheckBox("resize")
        data_opts_layout.addWidget(self.data_opts_resize)
        self.data_opts_sqrt = QCheckBox("sqrt")
        data_opts_layout.addWidget(self.data_opts_sqrt)
        self.data_opts_log = QCheckBox("log")
        data_opts_layout.addWidget(self.data_opts_log)
        self.data_opts.setLayout(data_opts_layout)
        layout_option0.addWidget(self.data_opts)
        layout_main.addLayout(layout_option0)
        # plot option button widgets
        plot_opts_layout = QGridLayout()
        self.plot_opts = QGroupBox("Plot options:")
        self.plot_opts_normplot = QCheckBox("normal plot")
        plot_opts_layout.addWidget(self.plot_opts_normplot, 0, 0)
        self.plot_opts_colim = QCheckBox("collated image")
        plot_opts_layout.addWidget(self.plot_opts_colim, 0, 1)
        self.plot_opts_colim.setChecked(True)
        self.plot_opts_scale = QCheckBox("scaling")
        plot_opts_layout.addWidget(self.plot_opts_scale, 0, 2)
        self.plot_opts_hide = QCheckBox("hide plotting")
        plot_opts_layout.addWidget(self.plot_opts_hide, 0, 3)
        self.plot_opts_interpol = QCheckBox("interpolate bicubic")
        plot_opts_layout.addWidget(self.plot_opts_interpol, 1, 0)
        self.plot_opts_cbcut = QCheckBox("colorbar cutoff")
        plot_opts_layout.addWidget(self.plot_opts_cbcut, 1, 1)
        self.plot_opts_rotate = QCheckBox("rotate")
        plot_opts_layout.addWidget(self.plot_opts_rotate, 1, 2)
        self.plot_opts_rgb = QCheckBox("RGB")
        plot_opts_layout.addWidget(self.plot_opts_rgb, 1, 3)
        self.plot_opts_frame = QCheckBox("frame image")
        self.plot_opts_frame.setChecked(True)
        plot_opts_layout.addWidget(self.plot_opts_frame, 2, 0)
        self.plot_opts_fontsize = QCheckBox("adjust font sizes")
        plot_opts_layout.addWidget(self.plot_opts_fontsize, 2, 1)
        self.plot_opts_sb = QCheckBox("scale bar")
        plot_opts_layout.addWidget(self.plot_opts_sb, 2, 2)
        self.plot_opts_neg2zero = QCheckBox("Neg2Zero")
        self.plot_opts_neg2zero.setChecked(True)
        plot_opts_layout.addWidget(self.plot_opts_neg2zero, 2, 3)
        self.plot_opts.setLayout(plot_opts_layout)
        layout_option1.addWidget(self.plot_opts)
        layout_main.addLayout(layout_option1)
        # tabs widgets
        self.opt_tabs = QTabWidget()
        self.tab_bin = QWidget()
        tab_bin_layout = QVBoxLayout()
        bin_line0_layout = QHBoxLayout()
        bin_line0_layout.addWidget(QLabel("Binning: "))
        self.tab_bin_binsize = QRadioButton("Define Binning Size")
        self.tab_bin_binsize.setChecked(True)
        bin_line0_layout.addWidget(self.tab_bin_binsize)
        self.tab_bin_resmod = QRadioButton("Modify Resolution")
        bin_line0_layout.addWidget(self.tab_bin_resmod)
        bin_line0_layout.addStretch()
        tab_bin_layout.addLayout(bin_line0_layout)
            # widgets on binning size
        self.tab_binsize = QWidget()
        bin_line1_layout = QHBoxLayout()
        bin_line1_layout.addWidget(QLabel("Binning size X: "))
        self.tab_bin_xbin = QLineEdit("")
        self.tab_bin_xbin.setMaximumWidth(25)
        self.tab_bin_xbin.setValidator(QDoubleValidator(-1E6, 1E6,0))
        bin_line1_layout.addWidget(self.tab_bin_xbin)
        bin_line1_layout.addWidget(QLabel(" Y: "))
        self.tab_bin_ybin = QLineEdit("")
        self.tab_bin_ybin.setMaximumWidth(25)
        self.tab_bin_ybin.setValidator(QDoubleValidator(-1E6, 1E6,0))
        bin_line1_layout.addWidget(self.tab_bin_ybin)
        bin_line1_layout.addStretch()
        self.tab_binsize.setLayout(bin_line1_layout)
        tab_bin_layout.addWidget(self.tab_binsize)
            #widgets on modify resolution
        self.tab_resmod = QWidget()
        bin_line2_layout = QHBoxLayout()
        bin_line2_layout.addWidget(QLabel("Resolution modifier: "))
        self.tab_bin_mod = QLineEdit("")
        self.tab_bin_mod.setMaximumWidth(25)
        self.tab_bin_mod.setValidator(QDoubleValidator(-1E6, 1E6,0))
        bin_line2_layout.addWidget(self.tab_bin_mod)
        bin_line2_layout.addStretch()
        self.tab_resmod.setLayout(bin_line2_layout)
        tab_bin_layout.addWidget(self.tab_resmod)
        self.tab_resmod.setVisible(False)
            #continue tab_bin widgets
        bin_line3_layout = QHBoxLayout()
        bin_line3_layout.addWidget(QLabel("New *.ims info #X: "))
        self.tab_bin_newx = QLabel("---")
        self.tab_bin_newx.setMaximumWidth(25)
        bin_line3_layout.addWidget(self.tab_bin_newx)
        bin_line3_layout.addWidget(QLabel(" #Y: "))
        self.tab_bin_newy = QLabel("---")
        self.tab_bin_newy.setMaximumWidth(25)
        bin_line3_layout.addWidget(self.tab_bin_newy)
        self.tab_bin_save = QCheckBox("Save as separate ims file")
        bin_line3_layout.addWidget(self.tab_bin_save)
        bin_line3_layout.addStretch()
        tab_bin_layout.addLayout(bin_line3_layout)
        tab_bin_layout.addStretch()
        self.tab_bin.setLayout(tab_bin_layout)
        self.opt_tabs.addTab(self.tab_bin, "Binning")

        self.tab_resize = QWidget()
        tab_resize_layout = QVBoxLayout()
        resize_line0_layout = QHBoxLayout()
        resize_line1_layout = QHBoxLayout()
        resize_line2_layout = QHBoxLayout()
        resize_line0_layout.addWidget(QLabel("Resized Map Dimensions: "))
        resize_line0_layout.addStretch()
        resize_line1_layout.addWidget(QLabel("X start:"))
        self.resize_xstart = QLineEdit("")
        self.resize_xstart.setMaximumWidth(25)
        self.resize_xstart.setValidator(QDoubleValidator(-1E6, 1E6,0))
        resize_line1_layout.addWidget(self.resize_xstart)
        resize_line1_layout.addWidget(QLabel("X end:"))
        self.resize_xend = QLineEdit("")
        self.resize_xend.setMaximumWidth(25)
        self.resize_xend.setValidator(QDoubleValidator(-1E6, 1E6,0))
        resize_line1_layout.addWidget(self.resize_xend)
        resize_line1_layout.addWidget(QLabel("Y start:"))
        self.resize_ystart = QLineEdit("")
        self.resize_ystart.setMaximumWidth(25)
        self.resize_ystart.setValidator(QDoubleValidator(-1E6, 1E6,0))
        resize_line1_layout.addWidget(self.resize_ystart)
        resize_line1_layout.addWidget(QLabel("Y end:"))
        self.resize_yend = QLineEdit("")
        self.resize_yend.setMaximumWidth(25)
        self.resize_yend.setValidator(QDoubleValidator(-1E6, 1E6,0))
        resize_line1_layout.addWidget(self.resize_yend)     
        resize_line1_layout.addStretch()
        self.tab_resize_save = QCheckBox("Save as separate ims file")
        resize_line2_layout.addWidget(self.tab_resize_save)
        resize_line2_layout.addStretch()
        tab_resize_layout.addLayout(resize_line0_layout)
        tab_resize_layout.addLayout(resize_line1_layout)
        tab_resize_layout.addLayout(resize_line2_layout)
        tab_resize_layout.addStretch()
        self.tab_resize.setLayout(tab_resize_layout)
        self.opt_tabs.addTab(self.tab_resize, "Resize")        

        self.tab_cbcut = QWidget()
        tab_cbcut_layout = QVBoxLayout()
        cbcut_line0_layout = QHBoxLayout()
        cbcut_line1_layout = QHBoxLayout()
        cbcut_line2_layout = QHBoxLayout()
        cbcut_line0_layout.addWidget(QLabel("Colorbar Cutoff:"))
        cbcut_line1_layout.addWidget(QLabel("Min. Int:"))
        self.cbcut_min = QLineEdit("")
        self.cbcut_min.setMaximumWidth(50)
        self.cbcut_min.setValidator(QDoubleValidator(-1E6, 1E6,3))
        cbcut_line1_layout.addWidget(self.cbcut_min)
        cbcut_line1_layout.addWidget(QLabel("Max. Int:"))
        self.cbcut_max = QLineEdit("")
        self.cbcut_max.setMaximumWidth(50)
        self.cbcut_max.setValidator(QDoubleValidator(-1E6, 1E6,3))
        cbcut_line1_layout.addWidget(self.cbcut_max)
        cbcut_line1_layout.addWidget(QLabel("EoI:"))
        self.cbcut_eoi = QComboBox()
        self.cbcut_eoi.addItems(self.element_array)
        cbcut_line1_layout.addWidget(self.cbcut_eoi)
        cbcut_line1_layout.addStretch()
        self.cbcut_canvas = QLabel("")
        cbcut_line2_layout.addWidget(self.cbcut_canvas)
        cbcut_line2_layout.addStretch()
        tab_cbcut_layout.addLayout(cbcut_line0_layout)
        tab_cbcut_layout.addLayout(cbcut_line1_layout)
        tab_cbcut_layout.addLayout(cbcut_line2_layout)
        tab_cbcut_layout.addStretch()
        self.tab_cbcut.setLayout(tab_cbcut_layout)
        self.opt_tabs.addTab(self.tab_cbcut, "CB Cutoff")        
        
        self.tab_scale = QWidget()
        tab_scale_layout = QVBoxLayout()
        scale_line0_layout = QHBoxLayout()
        scale_line1_layout = QHBoxLayout()
        scale_line2_layout = QHBoxLayout()
        self.scale = QGroupBox()
        self.scale.setFlat(True)
        self.scale.setStyleSheet("border:0")
        scale_line0_layout.addWidget(QLabel("Scale Bar: "))
        self.scale_xscale = QCheckBox("X Scale")
        scale_line0_layout.addWidget(self.scale_xscale)
        self.scale_yscale = QCheckBox("Y Scale")
        scale_line0_layout.addWidget(self.scale_yscale)
        scale_line0_layout.addWidget(self.scale)
        scale_line0_layout.addStretch()
        scale_line1_layout.addWidget(QLabel("Pixel Size:"))
        scale_line1_layout.addSpacing(3)
        validator = QDoubleValidator(-1E6, 1E6,3)
        validator.setLocale(QtCore.QLocale("en_US")
        self.scale_xpixsize = QLineEdit("")
        self.scale_xpixsize.setMaximumWidth(25)
        self.scale_xpixsize.setValidator(validator)
        scale_line1_layout.addWidget(self.scale_xpixsize)
        scale_line1_layout.addSpacing(40)
        self.scale_ypixsize = QLineEdit("")
        self.scale_ypixsize.setMaximumWidth(25)
        self.scale_ypixsize.setValidator(validator)
        scale_line1_layout.addWidget(self.scale_ypixsize)
        self.scale_pixunit = QComboBox()
        self.scale_pixunit.addItems(self.units)
        self.scale_pixunit.setCurrentIndex(2)
        scale_line1_layout.addWidget(self.scale_pixunit)
        scale_line1_layout.addStretch()
        scale_line2_layout.addWidget(QLabel("Scale Size:"))
        self.scale_xsclsize = QLineEdit("")
        self.scale_xsclsize.setMaximumWidth(25)
        self.scale_xsclsize.setValidator(validator)
        scale_line2_layout.addWidget(self.scale_xsclsize)
        scale_line2_layout.addSpacing(40)
        self.scale_ysclsize = QLineEdit("")
        self.scale_ysclsize.setMaximumWidth(25)
        self.scale_ysclsize.setValidator(validator)
        scale_line2_layout.addWidget(self.scale_ysclsize)
        self.scale_sclunit = QComboBox()
        self.scale_sclunit.addItems(self.units)
        self.scale_sclunit.setCurrentIndex(2)
        scale_line2_layout.addWidget(self.scale_sclunit)
        scale_line2_layout.addStretch()
        tab_scale_layout.addLayout(scale_line0_layout)
        tab_scale_layout.addLayout(scale_line1_layout)
        tab_scale_layout.addLayout(scale_line2_layout)
        tab_scale_layout.addStretch()
        self.tab_scale.setLayout(tab_scale_layout)
        self.opt_tabs.addTab(self.tab_scale, "Scale Bar")        

        self.tab_rgb = QWidget()
        tab_rgb_layout = QHBoxLayout()
        rgb_line0_layout = QVBoxLayout()
        rgb_line1_layout = QVBoxLayout()
        rgb_line2_layout = QVBoxLayout()
        rgb_line3_layout = QVBoxLayout()
        rgb_line4_layout = QVBoxLayout()
        rgb_line0_layout.addWidget(QLabel("RGB imaging: "))
        self.rgb_preview = QPushButton("Show Preview")
        rgb_line0_layout.addWidget(self.rgb_preview)
        rgb_line0_layout.addSpacing(3)
        rgb_line0_layout.addWidget(QLabel("EoI: "))
        rgb_line0_layout.addSpacing(4)
        rgb_line0_layout.addWidget(QLabel("Min. Int.: "))
        rgb_line0_layout.addSpacing(3)
        rgb_line0_layout.addWidget(QLabel("Max. Int.: "))
        rgb_line0_layout.addSpacing(3)
        rgb_line0_layout.addWidget(QLabel("Cutoff Min.: "))
        rgb_line0_layout.addSpacing(4)
        rgb_line0_layout.addWidget(QLabel("Cutoff Max.: "))
        rgb_line0_layout.addStretch()
        rgb_line1_layout.addSpacing(30)
        rgb_line1_layout.addWidget(QLabel("Red"))
        self.rgb_red = QComboBox()
        self.rgb_red.addItems(self.element_array)
        rgb_line1_layout.addWidget(self.rgb_red)
        self.rgb_red_minint = QLabel("---")
        rgb_line1_layout.addWidget(self.rgb_red_minint)
        self.rgb_red_maxint = QLabel("---")
        rgb_line1_layout.addWidget(self.rgb_red_maxint)
        self.rgb_red_mincut = QLineEdit("")
        self.rgb_red_mincut.setMaximumWidth(60)
        self.rgb_red_mincut.setValidator(QDoubleValidator(-1E6, 1E6,3))
        rgb_line1_layout.addWidget(self.rgb_red_mincut)
        self.rgb_red_maxcut = QLineEdit("")
        self.rgb_red_maxcut.setMaximumWidth(60)
        self.rgb_red_maxcut.setValidator(QDoubleValidator(-1E6, 1E6,3))
        rgb_line1_layout.addWidget(self.rgb_red_maxcut)
        rgb_line1_layout.addStretch()
        rgb_line2_layout.addSpacing(30)
        rgb_line2_layout.addWidget(QLabel("Green"))
        self.rgb_green = QComboBox()
        self.rgb_green.addItems(self.element_array)
        rgb_line2_layout.addWidget(self.rgb_green)
        self.rgb_green_minint = QLabel("---")
        rgb_line2_layout.addWidget(self.rgb_green_minint)
        self.rgb_green_maxint = QLabel("---")
        rgb_line2_layout.addWidget(self.rgb_green_maxint)
        self.rgb_green_mincut = QLineEdit("")
        self.rgb_green_mincut.setMaximumWidth(60)
        self.rgb_green_mincut.setValidator(QDoubleValidator(-1E6, 1E6,3))
        rgb_line2_layout.addWidget(self.rgb_green_mincut)
        self.rgb_green_maxcut = QLineEdit("")
        self.rgb_green_maxcut.setMaximumWidth(60)
        self.rgb_green_maxcut.setValidator(QDoubleValidator(-1E6, 1E6,3))
        rgb_line2_layout.addWidget(self.rgb_green_maxcut)
        rgb_line2_layout.addStretch()
        rgb_line3_layout.addSpacing(30)
        rgb_line3_layout.addWidget(QLabel("Blue"))
        self.rgb_blue = QComboBox()
        self.rgb_blue.addItems(self.element_array)
        rgb_line3_layout.addWidget(self.rgb_blue)
        self.rgb_blue_minint = QLabel("---")
        rgb_line3_layout.addWidget(self.rgb_blue_minint)
        self.rgb_blue_maxint = QLabel("---")
        rgb_line3_layout.addWidget(self.rgb_blue_maxint)
        self.rgb_blue_mincut = QLineEdit("")
        self.rgb_blue_mincut.setMaximumWidth(60)
        self.rgb_blue_mincut.setValidator(QDoubleValidator(-1E6, 1E6,3))
        rgb_line3_layout.addWidget(self.rgb_blue_mincut)
        self.rgb_blue_maxcut = QLineEdit("")
        self.rgb_blue_maxcut.setMaximumWidth(60)
        self.rgb_blue_maxcut.setValidator(QDoubleValidator(-1E6, 1E6,3))
        rgb_line3_layout.addWidget(self.rgb_blue_maxcut)
        rgb_line3_layout.addStretch()
        self.rgb_canvas = QLabel("")
        rgb_line4_layout.addWidget(self.rgb_canvas)
        rgb_line4_layout.addStretch()
        tab_rgb_layout.addLayout(rgb_line0_layout)
        tab_rgb_layout.addLayout(rgb_line1_layout)
        tab_rgb_layout.addLayout(rgb_line2_layout)
        tab_rgb_layout.addLayout(rgb_line3_layout)
        tab_rgb_layout.addLayout(rgb_line4_layout)
        tab_rgb_layout.addStretch()
        self.tab_rgb.setLayout(tab_rgb_layout)
        self.opt_tabs.addTab(self.tab_rgb, "RGB")        
        
        self.tab_colim = QWidget()
        tab_colim_layout = QVBoxLayout()
        colim_line0_layout = QHBoxLayout()
        colim_line1_layout = QHBoxLayout()
        colim_line0_layout.addWidget(QLabel("Collated Imaging:"))
        colim_line0_layout.addStretch()
        colim_line1_layout.addWidget(QLabel("# of columns:"))
        self.colim_ncol = QLineEdit("")
        self.colim_ncol.setMaximumWidth(25)
        self.colim_ncol.setValidator(QDoubleValidator(-1E6, 1E6,3))
        colim_line1_layout.addWidget(self.colim_ncol)
        self.colim_plotcb = QCheckBox("Plot Colorbar")
        self.colim_plotcb.setChecked(True)
        colim_line1_layout.addWidget(self.colim_plotcb)
        colim_line1_layout.addStretch()
        tab_colim_layout.addLayout(colim_line0_layout)
        tab_colim_layout.addLayout(colim_line1_layout)
        tab_colim_layout.addStretch()
        self.tab_colim.setLayout(tab_colim_layout)
        self.opt_tabs.addTab(self.tab_colim, "Collated Image")        
        
        self.tab_fontsize = QWidget()
        tab_fontsize_layout = QVBoxLayout()
        fontsize_line0_layout = QHBoxLayout()
        fontsize_line1_layout = QHBoxLayout()
        fontsize_line2_layout = QHBoxLayout()
        fontsize_line0_layout.addWidget(QLabel("Adjust Font Sizes:"))
        fontsize_line0_layout.addStretch()
        fontsize_line1_layout.addWidget(QLabel("Image Title"))
        fontsize_line1_layout.addSpacing(20)
        fontsize_line1_layout.addWidget(QLabel("Colorbar Numbers"))
        fontsize_line1_layout.addSpacing(20)
        fontsize_line1_layout.addWidget(QLabel("Colorbar Title"))
        fontsize_line1_layout.addSpacing(20)
        fontsize_line1_layout.addWidget(QLabel("Scale Bar"))
        fontsize_line1_layout.addStretch()
        fontsize_line2_layout.addSpacing(15)
        self.fontsize_im_tit = QLineEdit("20")
        self.fontsize_im_tit.setMaximumWidth(25)
        self.fontsize_im_tit.setValidator(QDoubleValidator(-1E6, 1E6,3))
        fontsize_line2_layout.addWidget(self.fontsize_im_tit)
        fontsize_line2_layout.addSpacing(65)
        self.fontsize_cb_num = QLineEdit("14")
        self.fontsize_cb_num.setMaximumWidth(25)
        self.fontsize_cb_num.setValidator(QDoubleValidator(-1E6, 1E6,3))
        fontsize_line2_layout.addWidget(self.fontsize_cb_num)
        fontsize_line2_layout.addSpacing(65)
        self.fontsize_cb_tit = QLineEdit("14")
        self.fontsize_cb_tit.setMaximumWidth(25)
        self.fontsize_cb_tit.setValidator(QDoubleValidator(-1E6, 1E6,3))
        fontsize_line2_layout.addWidget(self.fontsize_cb_tit)
        fontsize_line2_layout.addSpacing(52)
        self.fontsize_scale = QLineEdit("16")
        self.fontsize_scale.setMaximumWidth(25)
        self.fontsize_scale.setValidator(QDoubleValidator(-1E6, 1E6,3))
        fontsize_line2_layout.addWidget(self.fontsize_scale)
        fontsize_line2_layout.addStretch()
        tab_fontsize_layout.addLayout(fontsize_line0_layout)
        tab_fontsize_layout.addLayout(fontsize_line1_layout)
        tab_fontsize_layout.addLayout(fontsize_line2_layout)
        tab_fontsize_layout.addStretch()
        self.tab_fontsize.setLayout(tab_fontsize_layout)
        self.opt_tabs.addTab(self.tab_fontsize, "Font Size")        
        
        self.tab_ratio = QWidget()
        tab_ratio_layout = QVBoxLayout()
        ratio_line0_layout = QHBoxLayout()
        ratio_line1_layout = QHBoxLayout()
        ratio_line2_layout = QHBoxLayout()
        ratio_line3_layout = QHBoxLayout()
        ratio_line0_layout.addSpacing(75)
        ratio_line0_layout.addWidget(QLabel("Nominator"))
        ratio_line0_layout.addSpacing(50)
        ratio_line0_layout.addWidget(QLabel("Denominator"))
        ratio_line0_layout.addStretch()
        ratio_line1_layout.addWidget(QLabel("EoI:"))
        ratio_line1_layout.addSpacing(35)
        self.ratio_nom = QComboBox()
        self.ratio_nom.addItems(self.element_array)
        ratio_line1_layout.addWidget(self.ratio_nom)
        ratio_line1_layout.addSpacing(12)
        ratio_line1_layout.addWidget(QLabel("/"))
        ratio_line1_layout.addSpacing(12)
        self.ratio_den = QComboBox()
        self.ratio_den.addItems(self.element_array)
        ratio_line1_layout.addWidget(self.ratio_den)
        ratio_line1_layout.addStretch()
        ratio_line2_layout.addWidget(QLabel("Ratio Min:"))
        self.ratio_min = QLineEdit("")
        self.ratio_min.setMaximumWidth(50)
        self.ratio_min.setValidator(QDoubleValidator(-1E6, 1E6,3))
        ratio_line2_layout.addWidget(self.ratio_min)
        ratio_line2_layout.addSpacing(25)
        ratio_line2_layout.addWidget(QLabel("Ratio Max:"))
        self.ratio_max = QLineEdit("")
        self.ratio_max.setMaximumWidth(50)
        self.ratio_max.setValidator(QDoubleValidator(-1E6, 1E6,3))
        ratio_line2_layout.addWidget(self.ratio_max)
        ratio_line2_layout.addStretch()
        self.ratio_canvas = QLabel("")
        ratio_line3_layout.addWidget(self.ratio_canvas)
        ratio_line3_layout.addStretch()
        tab_ratio_layout.addLayout(ratio_line0_layout)
        tab_ratio_layout.addLayout(ratio_line1_layout)
        tab_ratio_layout.addLayout(ratio_line2_layout)
        tab_ratio_layout.addLayout(ratio_line3_layout)
        tab_ratio_layout.addStretch()
        self.tab_ratio.setLayout(tab_ratio_layout)
        self.opt_tabs.addTab(self.tab_ratio, "Ratio")        
        
        self.tab_rotate = QWidget()
        tab_rotate_layout = QVBoxLayout()
        rotate_line0_layout = QHBoxLayout()
        rotate_line1_layout = QHBoxLayout()
        rotate_line1_sub0 = QVBoxLayout()
        rotate_line1_sub1 = QVBoxLayout()
        rotate_line1_sub2 = QVBoxLayout()
        rotate_line1_sub3 = QVBoxLayout()
        rotate_line2_layout = QHBoxLayout()
        rotate_line0_layout.addWidget(QLabel("Rotation Angle:"))
        rotate_line0_layout.addStretch()
        rotate_line1_sub0.addSpacing(40)
        self.rotate_90 = QPushButton("90")
        self.rotate_90.setMaximumWidth(25)
        rotate_line1_sub0.addWidget(self.rotate_90)
        rotate_line1_sub0.addStretch()
        self.rotate_0 = QPushButton("0")
        self.rotate_0.setMaximumWidth(25)
        rotate_line1_sub1.addWidget(self.rotate_0)
        rotate_line1_sub1.addSpacing(50)
        self.rotate_180 = QPushButton("180")
        self.rotate_180.setMaximumWidth(25)
        rotate_line1_sub1.addWidget(self.rotate_180)
        rotate_line1_sub1.addStretch()
        rotate_line1_sub2.addSpacing(40)
        self.rotate_270 = QPushButton("270")
        self.rotate_270.setMaximumWidth(25)
        rotate_line1_sub2.addWidget(self.rotate_270)
        rotate_line1_sub2.addStretch()
        rotate_line1_sub3.addSpacing(33)
        self.rotate_flipv = QCheckBox("Flip Vertical Axis")
        rotate_line1_sub3.addWidget(self.rotate_flipv)
        self.rotate_fliph = QCheckBox("Flip Horizontal Axis")
        rotate_line1_sub3.addWidget(self.rotate_fliph)
        rotate_line1_sub3.addStretch()
        rotate_line1_layout.addLayout(rotate_line1_sub0)
        rotate_line1_layout.addSpacing(10)
        rotate_line1_layout.addLayout(rotate_line1_sub1)
        rotate_line1_layout.addSpacing(10)
        rotate_line1_layout.addLayout(rotate_line1_sub2)
        rotate_line1_layout.addSpacing(10)
        rotate_line1_layout.addLayout(rotate_line1_sub3)
        rotate_line1_layout.addStretch()
        rotate_line2_layout.addWidget(QLabel("New #X:"))
        self.rotate_newx = QLabel("---")
        rotate_line2_layout.addWidget(self.rotate_newx)
        rotate_line2_layout.addWidget(QLabel("#Y:"))
        self.rotate_newy = QLabel("---")
        rotate_line2_layout.addWidget(self.rotate_newy)
        rotate_line2_layout.addStretch()
        tab_rotate_layout.addLayout(rotate_line0_layout)
        tab_rotate_layout.addLayout(rotate_line1_layout)
        tab_rotate_layout.addLayout(rotate_line2_layout)
        self.tab_rotate.setLayout(tab_rotate_layout)
        self.opt_tabs.addTab(self.tab_rotate, "Rotate")
        
        layout_tabs.addWidget(self.opt_tabs)
        layout_main.addLayout(layout_tabs)
        # elements to plot widgets
        el2plot = QLabel("Elements to plot: ")
        el2plot.setMaximumWidth(80)
        layout_element_select.addWidget(el2plot)
        el_sel_layout = QHBoxLayout()
        self.el_sel = QGroupBox()
        self.el_sel.setFlat(True)
        self.el_sel.setStyleSheet("border:0")
        self.el_sel.setMaximumWidth(150)
        self.el_sel_all = QRadioButton("All")
        self.el_sel_all.setChecked(True)
        el_sel_layout.addWidget(self.el_sel_all)
        self.el_sel_some = QRadioButton("Selection")
        el_sel_layout.addWidget(self.el_sel_some)
        self.el_sel.setLayout(el_sel_layout)
        layout_element_select.addWidget(self.el_sel)
        layout_element_select.addStretch()
        layout_main.addLayout(layout_element_select)
        # colorbar title widgets
        cb_title = QLabel("Colorbar Title: ")
        cb_title.setMaximumWidth(80)
        layout_cb_title.addWidget(cb_title)
        cb_title_layout = QHBoxLayout()
        self.cb_title = QGroupBox()
        self.cb_title.setFlat(True)
        self.cb_title.setStyleSheet("border:0")
        self.cb_title.setMaximumWidth(300)
        self.cb_title_none = QRadioButton("None")
        self.cb_title_none.setChecked(True)
        cb_title_layout.addWidget(self.cb_title_none)
        self.cb_title_int = QRadioButton("Int.;[cts]")
        cb_title_layout.addWidget(self.cb_title_int)
        self.cb_title_conc = QRadioButton("Conc.;[µg/cm$^2$]")
        cb_title_layout.addWidget(self.cb_title_conc)
        self.cb_title_rand = QRadioButton()
        cb_title_layout.addWidget(self.cb_title_rand)
        self.cb_title.setLayout(cb_title_layout)
        layout_cb_title.addWidget(self.cb_title)
        self.cb_title_rand_lbl = QLineEdit("Title;[unit]")
        self.cb_title_rand_lbl.setMaximumWidth(75)
        layout_cb_title.addWidget(self.cb_title_rand_lbl)
        layout_cb_title.addStretch()
        layout_main.addLayout(layout_cb_title)
        # color table widgets
        ct_title = QLabel("Image Color Table:")
        ct_title.setMaximumWidth(90)
        layout_colortable.addWidget(ct_title)
        colortable_layout = QHBoxLayout()
        self.colortable = QGroupBox()
        self.colortable.setFlat(True)
        self.colortable.setStyleSheet("border:0")
        self.colortable.setMaximumWidth(300)
        self.colortable_red = QRadioButton("Temperature Red")
        self.colortable_red.setChecked(True)
        colortable_layout.addWidget(self.colortable_red)
        self.colortable_wb = QRadioButton("White-Black")
        colortable_layout.addWidget(self.colortable_wb)
        self.colortable_rain = QRadioButton("Rainbow")
        colortable_layout.addWidget(self.colortable_rain)
        self.colortable_rand = QRadioButton()
        colortable_layout.addWidget(self.colortable_rand)
        self.colortable.setLayout(colortable_layout)
        layout_colortable.addWidget(self.colortable)
        self.colortable_rand_lbl = QLineEdit("plasma")
        self.colortable_rand_lbl.setMaximumWidth(70)
        layout_colortable.addWidget(self.colortable_rand_lbl)
        layout_colortable.addStretch()
        layout_main.addLayout(layout_colortable)
        # colorbar direction widgets
        cb_dir_title = QLabel("Colorbar Orientation:")
        cb_dir_title.setMaximumWidth(100)
        layout_cb_dir.addWidget(cb_dir_title)
        cb_dir_layout = QHBoxLayout()
        self.cb_dir = QGroupBox()
        self.cb_dir.setFlat(True)
        self.cb_dir.setStyleSheet("border:0")
        self.cb_dir.setMaximumWidth(200)
        self.cb_dir_hor = QRadioButton("Horizontal")
        cb_dir_layout.addWidget(self.cb_dir_hor)
        self.cb_dir_ver = QRadioButton("Vertical")
        self.cb_dir_ver.setChecked(True)
        cb_dir_layout.addWidget(self.cb_dir_ver)
        self.cb_dir.setLayout(cb_dir_layout)
        layout_cb_dir.addWidget(self.cb_dir)
        self.cb_discr = QCheckBox("Discrete")
        layout_cb_dir.addWidget(self.cb_discr)
        layout_cb_dir.addStretch()
        layout_main.addLayout(layout_cb_dir)
        # filetype save widgets
        filetype_title = QLabel("Save as:")
        filetype_title.setMaximumWidth(80)
        layout_filetype.addWidget(filetype_title)
        filetype_layout = QHBoxLayout()
        self.filetype = QGroupBox()
        self.filetype.setFlat(True)
        self.filetype.setStyleSheet("border:0")
        self.filetype.setMaximumWidth(200)
        self.filetype_png = QRadioButton("*.png")
        self.filetype_png.setChecked(True)
        filetype_layout.addWidget(self.filetype_png)
        self.filetype_eps = QRadioButton("*.eps")
        filetype_layout.addWidget(self.filetype_eps)
        self.filetype_bmp = QRadioButton("*.bmp")
        filetype_layout.addWidget(self.filetype_bmp)
        self.filetype.setLayout(filetype_layout)
        layout_filetype.addWidget(self.filetype)
        layout_filetype.addStretch()
        layout_main.addLayout(layout_filetype)
        # execute button widgets
        self.plot = QPushButton("Plot")
        self.plot.setMaximumWidth(50)
        layout_exe_buts.addWidget(self.plot)
        self.close = QPushButton("Close")
        self.close.setMaximumWidth(50)
        layout_exe_buts.addWidget(self.close)
        layout_exe_buts.addStretch()
        layout_main.addLayout(layout_exe_buts)

        # set dialog layout
        self.setLayout(layout_main)
        self.opt_tabs.setCurrentWidget(self.tab_colim)

        # add button signal to greeting slot
        self.setWindowTitle("Plotims GUI")
        self.browse.clicked.connect(self.browse_app) # browse button
        self.filedir.returnPressed.connect(self.browse_app)
        self.close.clicked.connect(self.quit_app) # close button
        self.plot.clicked.connect(self.plotims) # plot button
        self.data_opts_sqrt.stateChanged.connect(self.toggle_sqrt_log) # toggle log if sqrt on
        self.data_opts_log.stateChanged.connect(self.toggle_log_sqrt) # toggle sqrt if log on
        self.data_opts_bin.stateChanged.connect(self.switch_tab)  # activate tab window when corresponding option selected
        self.data_opts_ratio.stateChanged.connect(self.switch_tab)
        self.data_opts_resize.stateChanged.connect(self.switch_tab)
        self.plot_opts_cbcut.stateChanged.connect(self.switch_tab)
        self.plot_opts_sb.stateChanged.connect(self.switch_tab)
        self.plot_opts_rgb.stateChanged.connect(self.switch_tab)
        self.plot_opts_fontsize.stateChanged.connect(self.switch_tab)
        self.plot_opts_rotate.stateChanged.connect(self.switch_tab)
        self.plot_opts_colim.stateChanged.connect(self.switch_tab)
        self.pix_scale.stateChanged.connect(self.pix_scale_toggle) # enable or disable pixel scale options
        self.el_sel_some.toggled.connect(self.select_elements) # spawn window to select elements
        self.tab_bin_resmod.toggled.connect(self.toggle_binmode) # toggle bin mode widgets within tab window
        self.rgb_preview.clicked.connect(self.rgb_view) # preview rgb image
        self.rgb_red.currentIndexChanged.connect(self.rgb_eoi_red) # RGB red EoI changed
        self.rgb_green.currentIndexChanged.connect(self.rgb_eoi_green) # RGB green EoI changed
        self.rgb_blue.currentIndexChanged.connect(self.rgb_eoi_blue) # RGB blue EoI changed
        self.cbcut_eoi.currentIndexChanged.connect(self.cbcut_eoi_change) # colorbar cutoff EoI changed
        self.rotate_0.clicked.connect(self.set_rot_angle) # rotation angle buttons and store angle in self.rot_angle
        self.rotate_90.clicked.connect(self.set_rot_angle) # rotation angle buttons and store angle in self.rot_angle
        self.rotate_180.clicked.connect(self.set_rot_angle) # rotation angle buttons and store angle in self.rot_angle
        self.rotate_270.clicked.connect(self.set_rot_angle) # rotation angle buttons and store angle in self.rot_angle        
        self.tab_bin_xbin.returnPressed.connect(self.adj_imsdim) # update new ims dimension
        self.tab_bin_ybin.returnPressed.connect(self.adj_imsdim) # update new ims dimension
        self.tab_bin_mod.returnPressed.connect(self.adj_imsdim) # update new ims dimension
        self.ratio_nom.currentIndexChanged.connect(self.ratio_eoi_change) # update min and max values
        self.ratio_den.currentIndexChanged.connect(self.ratio_eoi_change)
        self.cbcut_min.returnPressed.connect(self.cbcut_preview) # this and following to make preview image
        self.cbcut_max.returnPressed.connect(self.cbcut_preview)
        self.ratio_min.returnPressed.connect(self.ratio_preview) # this and following to make preview image
        self.ratio_max.returnPressed.connect(self.ratio_preview)


    def quit_app(self):
        print('CLEAN EXIT')
        app.quit()
        
    def browse_app(self):
        self.filenames = QFileDialog.getOpenFileNames(self, caption="Open IMS file", filter="H5 file (*.h5);;IMS file (*.ims)")
        self.filedir.setText("'"+"','".join([str(file) for file in self.filenames[0]])+"'")
        # read in first ims file, to obtain data on elements and dimensions
        if(self.filenames[0][0] != "''"):
            if self.filenames[0][0].split('.')[-1] == 'ims':
                self.ims_data = IMS.read_ims(self.filenames[0][0])
            elif self.filenames[0][0].split('.')[-1] == 'h5':
                self.h5dir = []
                self.new_window = Poll_h5dir(self.filenames[0][0])
                if self.new_window.exec_() == QDialog.Accepted:
                    self.h5dir = self.new_window.h5dir
                self.ims_data = IMS.read_h5(self.filenames[0][0], self.h5dir)
            ims_dim = self.ims_data.data.shape
            self.npix_x.setText(str(ims_dim[1]))
            self.npix_y.setText(str(ims_dim[0]))
            self.nelem.setText(str(ims_dim[2]))
            self.element_array = self.ims_data.names
            self.el_selection = self.ims_data.names
            # by default select all elements to plot at start
            self.el_sel_all.setChecked(True)
            # update elements of interest list in CB cutoff tab
            self.cbcut_eoi.clear()
            self.cbcut_eoi.addItems(self.element_array)
            # update elements list and intensities in RGB tab
            self.rgb_red.clear()
            self.rgb_green.clear()
            self.rgb_blue.clear()
            self.rgb_red.addItems(self.element_array)
            self.rgb_green.addItems(self.element_array)
            self.rgb_blue.addItems(self.element_array)
            self.rgb_red_minint.setText(str(self.ims_data.data[:,:,0].min()))
            self.rgb_green_minint.setText(str(self.ims_data.data[:,:,0].min()))
            self.rgb_blue_minint.setText(str(self.ims_data.data[:,:,0].min()))
            self.rgb_red_maxint.setText(str(self.ims_data.data[:,:,0].max()))
            self.rgb_green_maxint.setText(str(self.ims_data.data[:,:,0].max()))
            self.rgb_blue_maxint.setText(str(self.ims_data.data[:,:,0].max()))
            self.tab_bin_newx.setText(str(self.ims_data.data.shape[1]))
            self.tab_bin_newy.setText(str(self.ims_data.data.shape[0]))
            self.rotate_newx.setText(str(self.ims_data.data.shape[1]))
            self.rotate_newy.setText(str(self.ims_data.data.shape[0]))
            # update elements list in ratio tab
            self.ratio_nom.clear()
            self.ratio_den.clear()
            self.ratio_nom.addItems(self.element_array)
            self.ratio_den.addItems(self.element_array)

    def toggle_sqrt_log(self):
        if(self.data_opts_sqrt.isChecked()):
            self.data_opts_log.setChecked(False)
            self.cb_title_int.setText("Int.;[$\sqrt{cts}$]")
            self.cb_title_conc.setText("Conc.;[$\sqrt{µg/cm^2}$]")
        else:
            self.cb_title_int.setText("Int.;[cts]")
            self.cb_title_conc.setText("Conc.;[µg/cm$^2$]")

            
    def toggle_log_sqrt(self):
        if(self.data_opts_log.isChecked()):
            self.data_opts_sqrt.setChecked(False)
            self.cb_title_int.setText("Int.;[log(cts)]")
            self.cb_title_conc.setText("Conc.;[log(µg/cm$^2$)]")  
        else:
            self.cb_title_int.setText("Int.;[cts]")
            self.cb_title_conc.setText("Conc.;[µg/cm$^2$]")
       
    def switch_tab(self):
        if(self.sender() == self.data_opts_bin):
                if(self.data_opts_bin.isChecked()):
                    self.opt_tabs.setCurrentWidget(self.tab_bin)
        elif(self.sender() == self.data_opts_ratio):
                if(self.data_opts_ratio.isChecked()):
                    self.opt_tabs.setCurrentWidget(self.tab_ratio)
        elif(self.sender() == self.data_opts_resize):
                if(self.data_opts_resize.isChecked()):
                    self.opt_tabs.setCurrentWidget(self.tab_resize)
        elif(self.sender() == self.plot_opts_cbcut):
                if(self.plot_opts_cbcut.isChecked()):
                    self.opt_tabs.setCurrentWidget(self.tab_cbcut)
        elif(self.sender() == self.plot_opts_sb):
                if(self.plot_opts_sb.isChecked()):
                    self.opt_tabs.setCurrentWidget(self.tab_scale)
        elif(self.sender() == self.plot_opts_rgb):
                if(self.plot_opts_rgb.isChecked()):
                    self.opt_tabs.setCurrentWidget(self.tab_rgb)
        elif(self.sender() == self.plot_opts_fontsize):
                if(self.plot_opts_fontsize.isChecked()):
                    self.opt_tabs.setCurrentWidget(self.tab_fontsize)
        elif(self.sender() == self.plot_opts_rotate):
                if(self.plot_opts_rotate.isChecked()):
                    self.opt_tabs.setCurrentWidget(self.tab_rotate)
        elif(self.sender() == self.plot_opts_colim):    
                if(self.plot_opts_colim.isChecked()):
                    self.opt_tabs.setCurrentWidget(self.tab_colim)
        else:
            None

    def pix_scale_toggle(self):
        if(self.pix_scale.isChecked()):
            self.pix_label_x.setEnabled(True)
            self.pix_x.setEnabled(True)
            self.pix_label_y.setEnabled(True)
            self.pix_y.setEnabled(True)
        else:
            self.pix_label_x.setEnabled(False)
            self.pix_x.setEnabled(False)
            self.pix_label_y.setEnabled(False)
            self.pix_y.setEnabled(False)
            
    def select_elements(self):
        if(self.el_sel_some.isChecked()):
            #spawn window showing elements in first ims file, providing selection
            self.new_window = El_sel_window(self.element_array, self.el_selection)
            if self.new_window.exec_() == QDialog.Accepted:
                self.el_selection = self.new_window.el_selection                
        else:
            self.el_selection = self.element_array
            
    def toggle_binmode(self, state):
        self.tab_binsize.setVisible(not state)
        self.tab_resmod.setVisible(state)

    def rgb_eoi_red(self):
        # extract data and perform various operations
        imsdata = self.ims_data.data.copy()
        self.set_opts()
        imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)
        eoi_min = imsdata[:,:,self.rgb_red.currentIndex()].min()
        eoi_max = imsdata[:,:,self.rgb_red.currentIndex()].max()
        self.rgb_red_minint.setText(str(eoi_min))
        self.rgb_red_maxint.setText(str(eoi_max))

    def rgb_eoi_green(self):
        # extract data and perform various operations
        imsdata = self.ims_data.data.copy()
        self.set_opts()
        imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)
        eoi_min = imsdata[:,:,self.rgb_green.currentIndex()].min()
        eoi_max = imsdata[:,:,self.rgb_green.currentIndex()].max()
        self.rgb_green_minint.setText(str(eoi_min))
        self.rgb_green_maxint.setText(str(eoi_max))
        
    def rgb_eoi_blue(self):
        # extract data and perform various operations
        imsdata = self.ims_data.data.copy()
        self.set_opts()
        imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)
        eoi_min = imsdata[:,:,self.rgb_blue.currentIndex()].min()
        eoi_max = imsdata[:,:,self.rgb_blue.currentIndex()].max()
        self.rgb_blue_minint.setText(str(eoi_min))
        self.rgb_blue_maxint.setText(str(eoi_max))
        
    def cbcut_eoi_change(self):
        # extract data and perform various operations
        imsdata = self.ims_data.data.copy()
        self.set_opts()
        imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)
        eoi_min = imsdata[:,:,self.cbcut_eoi.currentIndex()].min()
        eoi_max = imsdata[:,:,self.cbcut_eoi.currentIndex()].max()
        self.cbcut_min.setText(str(eoi_min))
        self.cbcut_max.setText(str(eoi_max))
        self.cbcut_preview()
    
    def ratio_eoi_change(self):
        # extract data and perform various operations
        imsdata = self.ims_data.data.copy()
        self.set_opts()
        imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)
        eoi_nom = self.ratio_nom.currentIndex()
        eoi_den = self.ratio_den.currentIndex()
        ratio = np.zeros((imsdata.shape[0], imsdata.shape[1]))
        nom_data = imsdata[:,:,eoi_nom]
        den_data = imsdata[:,:,eoi_den]
        ratio[den_data != 0 ] = nom_data[den_data != 0 ] / den_data[den_data != 0 ]
        if self.ratio_min.text() == '':
            eoi_min = ratio.min()
        else:
            eoi_min = float(self.ratio_min.text())
        if self.ratio_max.text() == '':
            eoi_max = ratio.max()
        else:
            eoi_max = float(self.ratio_max.text())
        self.ratio_min.setText(str(eoi_min))
        self.ratio_max.setText(str(eoi_max))
        self.ratio_preview()
    
    def set_rot_angle(self):
        new_angle = float(self.sender().text())
        self.rot_angle = new_angle
        self.adj_imsdim()

    def adj_imsdim(self):
        # extract data and perform various operations
        imsdata = self.ims_data.data.copy()
        self.set_opts()
        imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)
        # update the appropriate fields
        self.tab_bin_newx.setText(str(imsdata.shape[1]))
        self.tab_bin_newy.setText(str(imsdata.shape[0]))
        self.rotate_newx.setText(str(imsdata.shape[1]))
        self.rotate_newy.setText(str(imsdata.shape[0]))

    def cbcut_preview(self):
        # extract data and perform various operations
        imsdata = self.ims_data.data.copy()
        self.set_opts()
        imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)
        eoi = self.cbcut_eoi.currentIndex()
        if self.cbcut_min.text() == '':
            eoi_min = imsdata[:,:,eoi].min()
        else:
             eoi_min = float(self.cbcut_min.text())
        if self.cbcut_max.text() == '':
             eoi_max = imsdata[:,:,eoi].max()
        else:
             eoi_max = float(self.cbcut_max.text())
        rgb_im = IMS.prepare_rgb_data(imsdata, eoi, eoi, eoi,  eoi_min,  eoi_max,  eoi_min,  eoi_max,  eoi_min,  eoi_max)
        rgb_im_trp = rgb_im.astype(np.uint8).copy() # need to copy data for QImage
        qim = QImage(rgb_im_trp, rgb_im_trp.shape[1], rgb_im_trp.shape[0], 3*rgb_im_trp.shape[1], QImage.Format_RGB888)
        pixmap = QPixmap(qim)
        pixmap = pixmap.scaled(175, 175, Qt.KeepAspectRatio)
        self.cbcut_canvas.setPixmap(pixmap)
        
    def ratio_preview(self):
        # extract data and perform various operations
        imsdata = self.ims_data.data.copy()
        self.set_opts()
        imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)
        eoi_nom = self.ratio_nom.currentIndex()
        eoi_den = self.ratio_den.currentIndex()
        ratio = np.zeros((imsdata.shape[0], imsdata.shape[1]))
        nom_data = imsdata[:,:,eoi_nom]
        den_data = imsdata[:,:,eoi_den]
        ratio[den_data != 0 ] = nom_data[den_data != 0 ] / den_data[den_data != 0 ]
        if self.ratio_min.text() == '':
            eoi_min = ratio.min()
        else:
            eoi_min = float(self.ratio_min.text())
        if self.ratio_max.text() == '':
            eoi_max = ratio.max()
        else:
            eoi_max = float(self.ratio_max.text())
        rgb_im = IMS.prepare_rgb_data(ratio, None, None, None,  eoi_min,  eoi_max,  eoi_min,  eoi_max,  eoi_min,  eoi_max)
        rgb_im_trp = rgb_im.astype(np.uint8).copy() # need to copy data for QImage
        qim = QImage(rgb_im_trp, rgb_im_trp.shape[1], rgb_im_trp.shape[0], 3*rgb_im_trp.shape[1], QImage.Format_RGB888)
        pixmap = QPixmap(qim)
        pixmap = pixmap.scaled(175, 175, Qt.KeepAspectRatio)
        self.ratio_canvas.setPixmap(pixmap)     

    def rgb_view(self):
        # perform data manipulations
        imsdata = self.ims_data.data.copy()
        # prepare rgb array to plot (each colour channel must be scaled between 0 and 255)
        self.set_opts()
        imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)
        # select rgb channels
        eoi_red = self.rgb_red.currentIndex()
        eoi_green = self.rgb_green.currentIndex()
        eoi_blue = self.rgb_blue.currentIndex()
        # extract min and max cutoff values
        if(self.rgb_red_mincut.text() == ""):
            rmin = imsdata[:,:,eoi_red].min()
        else:
            rmin = float(self.rgb_red_mincut.text())
        if(self.rgb_green_mincut.text() == ""):
            gmin = imsdata[:,:,eoi_green].min()
        else:
            gmin = float(self.rgb_green_mincut.text())
        if(self.rgb_blue_mincut.text() == ""):
            bmin = imsdata[:,:,eoi_blue].min()
        else:
            bmin = float(self.rgb_blue_mincut.text())
        if(self.rgb_red_maxcut.text() == ""):
            rmax = imsdata[:,:,eoi_red].max()
        else:
            rmax = float(self.rgb_red_maxcut.text())
        if(self.rgb_green_maxcut.text() == ""):
            gmax = imsdata[:,:,eoi_green].max()
        else:
            gmax = float(self.rgb_green_maxcut.text())
        if(self.rgb_blue_maxcut.text() == ""):
            bmax = imsdata[:,:,eoi_blue].max()
        else:
            bmax = float(self.rgb_blue_maxcut.text())
        if(self.plot_opts_neg2zero.isChecked()):
            if(rmin < 0):
                rmin = 0
            if(gmin < 0):
                gmin = 0
            if(bmin < 0):
                bmin = 0
        # check input values: max must be higher than min
        if rmax <= rmin or gmax <= gmin or bmax <= bmin:
            print("Error: rgb_view: minimum cutoff values must be strictly smaller than maximum cutoff values.")
        else:
            rgb_im = IMS.prepare_rgb_data(imsdata, eoi_red, eoi_green, eoi_blue, rmin, rmax, gmin, gmax, bmin, bmax)
            rgb_im_trp = rgb_im.astype(np.uint8).copy() # need to copy data for QImage
            qim = QImage(rgb_im_trp, rgb_im_trp.shape[1], rgb_im_trp.shape[0], 3*rgb_im_trp.shape[1], QImage.Format_RGB888)
            pixmap = QPixmap(qim)
            pixmap = pixmap.scaled(175, 175, Qt.KeepAspectRatio)
            self.rgb_canvas.setPixmap(pixmap)
   
    def set_opts(self):
        # obtain interpolation. By default no or 'nearest' interpolation
        if(self.plot_opts_interpol.isChecked()):
            interpol = 'bicubic'
        else:
            interpol = 'nearest'

        # extract font sizes
        fs_im_tit = int(self.fontsize_im_tit.text())
        fs_cb_num = int(self.fontsize_cb_num.text())
        fs_cb_tit = int(self.fontsize_cb_tit.text())
        fs_scale = int(self.fontsize_scale.text())
            
        # set colorbar options
        if(self.cb_dir_hor.isChecked()):
            cb_dir = 'horizontal'
        elif(self.cb_dir_ver.isChecked()):
            cb_dir = 'vertical'
        else:
            cb_dir = ''
        if(self.cb_discr.isChecked()):
            cb_discr = True
        else:
            cb_discr = False
        if(self.cb_title_none.isChecked()):
            cb_title = ''
        elif(self.cb_title_int.isChecked()):
            cb_title = self.cb_title_int.text().split(";")
            cb_title = "\n".join(cb_title)
        elif(self.cb_title_conc.isChecked()):
            cb_title = self.cb_title_conc.text().split(";")
            cb_title = "\n".join(cb_title)
        elif(self.cb_title_rand.isChecked()):
            cb_title = self.cb_title_rand_lbl.text().split(";")
            cb_title = "\n".join(cb_title)
        if(self.colortable_red.isChecked()):
            colortable = 'OrRd'
        elif(self.colortable_wb.isChecked()):
            colortable = 'Greys'
        elif(self.colortable_rain.isChecked()):
            colortable = 'Spectral'
        elif(self.colortable_rand.isChecked()):
            colortable = self.colortable_rand_lbl.text()
        
        # retreive binning parameters
        if(self.data_opts_bin.isChecked()):
            self.bin_opts = IMS.Binning()
            if(self.tab_bin_binsize.isChecked()): # join every X and Y pixels to one. Add them to preserve total flux.
                if(self.tab_bin_xbin.text() == ""):
                    binx = 1
                else:
                    binx = int(self.tab_bin_xbin.text())
                if(self.tab_bin_ybin.text() == ""):
                    biny = 1
                else:
                    biny = int(self.tab_bin_ybin.text())
            elif(self.tab_bin_resmod.isChecked()):
                if(self.tab_bin_mod.text() == ""):
                    mod = 1
                else:
                    mod = int(self.tab_bin_mod.text())
                # in resolution modifier, we simply divide each dimension by the resolution modifier
                binx = mod
                biny = mod                
            self.bin_opts.binx = binx
            self.bin_opts.biny = biny
        else:
            self.bin_opts = None
            binx = 1
            biny = 1        

        # neg2zero
        if(self.plot_opts_neg2zero.isChecked()):
            neg2zero = True
        else:
            neg2zero = False
        
        # sqrt or log
        if(self.data_opts_sqrt.isChecked()):
            self.math_opt = 'sqrt'
        elif(self.data_opts_log.isChecked()):
            self.math_opt = 'log'
        else:
            self.math_opt = ''
            
        # rotate
        if(self.plot_opts_rotate.isChecked()):
            self.rot_opts = IMS.Rotate()
            self.rot_opts.angle = self.rot_angle
            if(self.rotate_fliph.isChecked()):
                self.rot_opts.fliph = True
            if(self.rotate_flipv.isChecked()):
                self.rot_opts.flipv = True
        else: self.rot_opts = None

        # scale bar pixel size
        if(self.plot_opts_sb.isChecked()):
            if(self.scale_xscale.isChecked()):
                pix_size_x = self.scale_xpixsize.text()
                if(pix_size_x == ""):
                    print("Error: scale bar: pixel size cannot be None")
                else:
                    pix_size_x = float(pix_size_x)
                scl_size_x = self.scale_xsclsize.text()
                if(scl_size_x == ""):
                    print("Error: scale bar: scale bar size cannot be None")
                else:
                    scl_size_x = float(scl_size_x)
                # convert scl_size to same unit as pix_size.
                #   first set scale to m
                if(self.scale_sclunit.currentIndex() == 0): #Angstrom
                    scl_size_x = scl_size_x*1E-10
                elif(self.scale_sclunit.currentIndex() == 1): #nm
                    scl_size_x = scl_size_x*1E-9
                elif(self.scale_sclunit.currentIndex() == 2): #um
                    scl_size_x = scl_size_x*1E-6
                elif(self.scale_sclunit.currentIndex() == 3): #mm
                    scl_size_x = scl_size_x*1E-3
                elif(self.scale_sclunit.currentIndex() == 4): #cm
                    scl_size_x = scl_size_x*1E-2
                # then convert back to pix_size unit    
                if(self.scale_pixunit.currentIndex() == 0): #Angstrom
                    scl_size_x = scl_size_x*1E10
                elif(self.scale_pixunit.currentIndex() == 1): #nm
                    scl_size_x = scl_size_x*1E9
                elif(self.scale_pixunit.currentIndex() == 2): #um
                    scl_size_x = scl_size_x*1E6
                elif(self.scale_pixunit.currentIndex() == 3): #mm
                    scl_size_x = scl_size_x*1E3
                elif(self.scale_pixunit.currentIndex() == 4): #cm
                    scl_size_x = scl_size_x*1E2
                # adjust pix_size if scale to pixel size was selected
                #   binning groups integer amount of pixels, just multiply pix_size by this value
                pix_size_x = pix_size_x * binx

            if(self.scale_yscale.isChecked()):
                pix_size_y = self.scale_ypixsize.text()
                if(pix_size_y == ""):
                    print("Error: scale bar: pixel size cannot be None")
                else:
                    pix_size_y = float(pix_size_y)
                scl_size_y = self.scale_ysclsize.text()
                if(scl_size_y == ""):
                    print("Error: scale bar: scale bar size cannot be None")
                else:
                    scl_size_y = float(scl_size_y)
                # convert scl_size to same unit as pix_size.
                #   first set scale to m
                if(self.scale_sclunit.currentIndex() == 0): #Angstrom
                    scl_size_y = scl_size_y*1E-10
                elif(self.scale_sclunit.currentIndex() == 1): #nm
                    scl_size_y = scl_size_y*1E-9
                elif(self.scale_sclunit.currentIndex() == 2): #um
                    scl_size_y = scl_size_y*1E-6
                elif(self.scale_sclunit.currentIndex() == 3): #mm
                    scl_size_y = scl_size_y*1E-3
                elif(self.scale_sclunit.currentIndex() == 4): #cm
                    scl_size_y = scl_size_y*1E-2
                # then convert back to pix_size unit    
                if(self.scale_pixunit.currentIndex() == 0): #Angstrom
                    scl_size_y = scl_size_y*1E10
                elif(self.scale_pixunit.currentIndex() == 1): #nm
                    scl_size_y = scl_size_y*1E9
                elif(self.scale_pixunit.currentIndex() == 2): #um
                    scl_size_y = scl_size_y*1E6
                elif(self.scale_pixunit.currentIndex() == 3): #mm
                    scl_size_y = scl_size_y*1E3
                elif(self.scale_pixunit.currentIndex() == 4): #cm
                    scl_size_y = scl_size_y*1E2
                # adjust pix_size if scale to pixel size was selected
                #   binning groups integer amount of pixels, just multiply pix_size by this value
                pix_size_y = pix_size_y * biny
    
        # scale to pixel size
        #   actual scaling will occur by setting aspect ratio of image in imshow
        if(self.pix_scale.isChecked()):
            if(self.pix_x.text() == "" or self.pix_y.text() == ""):
                xpix = 1
                ypix = 1
            else:
                xpix = float(self.pix_x.text())
                ypix = float(self.pix_y.text())
        else:
            xpix = 1
            ypix = 1
            
        #resize
        if(self.data_opts_resize.isChecked()):
            self.resize_opts = IMS.Resize()
            if(self.resize_xstart.text() == ""):
                self.resize_opts.xstart = 0
            else:
                self.resize_opts.xstart = int(self.resize_xstart.text())
            if(self.resize_xend.text() == ""):
                self.resize_opts.xend = None
            else:
                self.resize_opts.xend = int(self.resize_xend.text())
            if(self.resize_ystart.text() == ""):
                self.resize_opts.ystart = 0
            else:
                self.resize_opts.ystart = int(self.resize_ystart.text())
            if(self.resize_yend.text() == ""):
                self.resize_opts.yend = None
            else:
                self.resize_opts.yend = int(self.resize_yend.text())
        else: self.resize_opts = None

        # set scaling min and max. Note that these values have to be adjusted for mathop, binning etc...
        if self.plot_opts_scale.isChecked():
            # read in all ims files, note outer min and max for each element of the complete dataset
            #   as this has to take into account mathop, binning, resize, etc, easiest is to actually perform these operations
            clim = np.zeros((2,len(self.el_selection)))
            for i in range(len(self.filenames[0][:])):
                if i == 0:
                    ims = self.ims_data
                    imsdata = self.ims_data.data.copy()
                else:
                    if self.filenames[0][i].split('.')[-1] == 'h5':
                        ims = IMS.read_h5(self.filenames[0][i], self.h5dir)
                    elif self.filenames[0][i].split('.')[-1] == 'ims':
                        ims = IMS.read_ims(self.filenames[0][i])
                    imsdata = ims.data.copy()
                # perform data operations
                imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)
                # obtain min and max for each element, and check if higher than previously registered value
                for j in range(len(self.el_selection)):
                    # find index of selected element
                    if self.el_selection[j] in ims.names:
                        for k in range(len(ims.names)):
                            if ims.names[k] == self.el_selection[j]:
                                if i == 0:
                                    clim[1,j] = imsdata[:,:,k].max()
                                    clim[0,j] = imsdata[:,:,k].min()
                                else:
                                    if imsdata[:,:,k].max() > clim[1,j]:
                                        clim[1,j] = imsdata[:,:,k].max()
                                    if imsdata[:,:,k].min() < clim[0,j]:
                                        clim[0,j] = imsdata[:,:,k].min()
        else: 
           clim=None

        # collated image options
        if self.plot_opts_colim.isChecked():
            if self.colim_ncol.text() == '':
                colim_ncol = int(np.floor(np.sqrt(len(self.el_selection))))
            else:
                colim_ncol = int(self.colim_ncol.text())
            colim_nrow = int(np.ceil(len(self.el_selection)/colim_ncol))
            colim_cb = self.colim_plotcb.isChecked()
            self.colim_opts = IMS.Collated_image_opts(ncol=colim_ncol, nrow=colim_nrow, cb=colim_cb)
        else:
            self.colim_opts = None

        # set plotting options
        self.plt_opts = IMS.Plot_opts(aspect=xpix/ypix, interpol=interpol, title_fontsize=fs_im_tit, clim=clim, ct=colortable, n2z=neg2zero, frame=self.plot_opts_frame.isChecked())
        if(self.plot_opts_sb.isChecked()):
            self.sb_opts = IMS.Scale_opts()
            self.sb_opts.fontsize = fs_scale
            if(self.scale_xscale.isChecked()):
                self.sb_opts.xscale = True
                self.sb_opts.x_pix_size = pix_size_x
                self.sb_opts.x_scl_size = scl_size_x
                self.sb_opts.x_scl_text = self.scale_xsclsize.text()+' '+self.scale_pixunit.currentText()
            if(self.scale_yscale.isChecked()):
                self.sb_opts.yscale = True
                self.sb_opts.y_pix_size = pix_size_y
                self.sb_opts.y_scl_size = scl_size_y
                self.sb_opts.y_scl_text = self.scale_ysclsize.text()+' '+self.scale_pixunit.currentText()
        else: self.sb_opts = None
        if(cb_dir == 'vertical' or cb_dir == 'horizontal'):
            self.cb_opts = IMS.Colorbar_opt(discr=cb_discr, dir=cb_dir, fs_num=fs_cb_num, fs_title=fs_cb_tit, title=cb_title)
        else: 
            self.cb_opts = None


    def plotims(self):
        # determine output file extension
        if(self.filetype_png.isChecked()):
            out_ext = '.png'
        elif(self.filetype_eps.isChecked()):
            out_ext = '.eps'
        elif(self.filetype_bmp.isChecked()):
            out_ext = '.bmp'
        else:
            out_ext = '.png'
        filename_base = self.filenames[0][0].split(".")
        filename_base = filename_base[0]
        if self.plot_opts_scale.isChecked():
            addendum = '_scaled' # filename addendum
        else: 
           addendum = ''
        self.set_opts() #set all plotting options etc, extracted from gui

        # perform RGB plotting
        #   RGB only images first ims file, for convenience reasons.
        #       perform all data manipulations within this if clause
        if(self.plot_opts_rgb.isChecked()):
            # prepare ims data
            imsdata = self.ims_data.data.copy()
            # perform data operations
            imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)

            # continue plotting
            eoi_red = self.rgb_red.currentIndex()
            eoi_green = self.rgb_green.currentIndex()
            eoi_blue = self.rgb_blue.currentIndex()
            # extract min and max cutoff values
            if(self.rgb_red_mincut.text() == ""):
                rmin = imsdata[:,:,eoi_red].min()
            else:
                rmin = float(self.rgb_red_mincut.text())
            if(self.rgb_green_mincut.text() == ""):
                gmin = imsdata[:,:,eoi_green].min()
            else:
                gmin = float(self.rgb_green_mincut.text())
            if(self.rgb_blue_mincut.text() == ""):
                bmin = imsdata[:,:,eoi_blue].min()
            else:
                bmin = float(self.rgb_blue_mincut.text())
            if(self.rgb_red_maxcut.text() == ""):
                rmax = imsdata[:,:,eoi_red].max()
            else:
                rmax = float(self.rgb_red_maxcut.text())
            if(self.rgb_green_maxcut.text() == ""):
                gmax = imsdata[:,:,eoi_green].max()
            else:
                gmax = float(self.rgb_green_maxcut.text())
            if(self.rgb_blue_maxcut.text() == ""):
                bmax = imsdata[:,:,eoi_blue].max()
            else:
                bmax = float(self.rgb_blue_maxcut.text())
            # check input values: max must be higher than min
            if rmax <= rmin or gmax <= gmin or bmax <= bmin:
                print("Error: rgb_view: minimum cutoff values must be strictly smaller than maximum cutoff values.")
            else:
                # prepare rgb array to plot (each colour channel must be scaled between 0 and 255)
                rgb_im = IMS.prepare_rgb_data(imsdata.copy(), eoi_red, eoi_green, eoi_blue, rmin, rmax, gmin, gmax, bmin, bmax)
                # matplotlib plotting, to save image
                fig = plt.figure(figsize=(10,10))
                gs = gridspec.GridSpec(2,1, height_ratios=[0.2,1])
                rgb0 = plt.subplot(gs[0], anchor='SW')
                rgb_triangle = IMS.make_rgb_triangle()
                rgb0.imshow(rgb_triangle.astype(np.uint8))
                rgb0.axis('off')
                # add labels on rgb triangle
                green_lbl = self.ims_data.names[eoi_green].split("-")
                green_lbl = green_lbl[0]
                red_lbl = self.ims_data.names[eoi_red].split("-")
                red_lbl = red_lbl[0]
                blue_lbl = self.ims_data.names[eoi_blue].split("-")
                blue_lbl = blue_lbl[0]
                rgb0.text(0.5, 0.56, green_lbl, size=16, ha="center", transform=rgb0.transAxes) #green label
                rgb0.text(0.25, 0.1, red_lbl, size=16, color='w', ha="center", transform=rgb0.transAxes) #red label
                rgb0.text(0.75, 0.1, blue_lbl, size=16, color='w', ha="center", transform=rgb0.transAxes) #blue label
                # add text with cutoff values
                red_cutoff = self.ims_data.names[eoi_red]+" cutoff value min: "+str(rmin)+" max:"+str(rmax)
                green_cutoff = self.ims_data.names[eoi_green]+" cutoff value min: "+str(gmin)+" max:"+str(gmax)
                blue_cutoff = self.ims_data.names[eoi_blue]+" cutoff value min: "+str(bmin)+" max:"+str(bmax)
                rgb0.text(1.2, 0.75, red_cutoff, size=12, ha="left", transform=rgb0.transAxes)
                rgb0.text(1.2, 0.625, green_cutoff, size=12, ha="left", transform=rgb0.transAxes)
                rgb0.text(1.2, 0.5, blue_cutoff, size=12, ha="left", transform=rgb0.transAxes)
                # plot actual rgb image
                rgb1 = plt.subplot(gs[1])
                rgb1.imshow(rgb_im.astype(np.uint8), aspect=1./self.plt_opts.aspect, interpolation=self.plt_opts.interpol)
                rgb1.axis('off')
                plt.tight_layout()
                # add scale_bar if requested
                if(self.plot_opts_sb.isChecked() and self.scale_xscale.isChecked()):
                    IMS.add_scalebar(rgb1, self.sb_opts.x_pix_size, self.sb_opts.x_scl_size, self.sb_opts.x_scl_text, scale_fontsize=self.sb_opts.fontsize, dir='h')
                if(self.plot_opts_sb.isChecked() and self.scale_yscale.isChecked()):
                    IMS.add_scalebar(rgb1, self.sb_opts.y_pix_size, self.sb_opts.y_scl_size, self.sb_opts.y_scl_text, scale_fontsize=self.sb_opts.fontsize, dir='v')
                filename = "_rgb_"+red_lbl+green_lbl+blue_lbl+out_ext
                filename = filename_base+filename.replace(" ","") #remove all white spaces
                fig.savefig(filename, dpi=420)
                plt.close()        
        

        # loop over all selected ims files
        for i in range(len(self.filenames[0][:])):
            # read ims file
            if i == 0:
                ims = self.ims_data
                imsdata = self.ims_data.data
            else:
                print(self.filenames[0][i])
                if self.filenames[0][i].split('.')[-1] == 'h5':
                        ims = IMS.read_h5(self.filenames[0][i], self.h5dir)
                elif self.filenames[0][i].split('.')[-1] == 'ims':
                    ims = IMS.read_ims(self.filenames[0][i])
                imsdata = ims.data
            filename_base = self.filenames[0][i].split(".")
            filename_base = filename_base[0]

            # perform data manipulations (resize, sqrt, log, rotation, binning, ...)
            imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)            

            # save new ims files if requested, and adjust filename addendum
            if self.data_opts_bin.isChecked():
                addendum = addendum + '_binx'+str(self.bin_opts.binx)+'biny'+str(self.bin_opts.biny)
            if self.data_opts_resize.isChecked():
                addendum = addendum + 'resizex'+str(self.resize_opts.xstart)+'-'+str(self.resize_opts.xend)+'y'+str(self.resize_opts.ystart)+'-'+str(self.resize_opts.yend)
            if (self.tab_bin_save.isChecked() or self.tab_resize_save.isChecked() ):
                IMS.write_ims(imsdata, ims.names, filename_base+addendum+'.ims')

            # perform collated imaging
            if self.plot_opts_colim.isChecked():
                filename = "_"+self.plt_opts.ct+"_overview"+addendum+out_ext
                filename = filename_base+filename.replace(" ","") #remove all white spaces
                ims.data = imsdata
                IMS.plot_colim(ims, self.el_selection, self.plt_opts.ct, plt_opts=self.plt_opts, sb_opts=self.sb_opts, cb_opts=self.cb_opts, colim_opts=self.colim_opts, save=filename)

            # perform individual image plotting
            if self.plot_opts_normplot.isChecked():
                for j in range(len(self.el_selection)):
                    # find index of selected element
                    if self.el_selection[j] in ims.names:
                        eoi = -1
                        for k in range(len(ims.names)):
                            if ims.names[k] == self.el_selection[j]:
                                eoi = k
                        if self.plt_opts.clim:
                            clim = self.plt_opts.clim[:,eoi]
                        else: clim = None
                        # perform plotting
                        filename = "_"+self.plt_opts.ct+"_"+ims.names[eoi]+addendum+out_ext
                        filename = filename_base+filename.replace(" ","") #remove all white spaces
                        IMS.plot_image(imsdata[:,:,eoi], ims.names[eoi], self.plt_opts.ct, plt_opts=self.plt_opts, sb_opts=self.sb_opts, cb_opts=self.cb_opts, clim=clim, save=filename)

        # colorbar cutoff image
        if self.plot_opts_cbcut.isChecked():
            ims = self.ims_data
            imsdata = self.ims_data.data.copy()
            # perform data manipulations (resize, sqrt, log, rotation, binning, ...)
            imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)            
            # plot cutoff image with selected limits
            eoi = self.cbcut_eoi.currentIndex()
            clim = np.zeros(2)
            if self.cbcut_min.text() == '':
                clim[0] = imsdata[:,:,eoi].min()
            else:
                clim[0] = float(self.cbcut_min.text())
            if self.cbcut_max.text() == '':
                clim[1] = imsdata[:,:,eoi].max()
            else:
                clim[1] = float(self.cbcut_max.text())
            filename = "_"+self.plt_opts.ct+"_"+ims.names[eoi]+"_cbcut"+out_ext
            filename = filename_base+filename.replace(" ","") #remove all white spaces
            IMS.plot_image(imsdata[:,:,eoi], ims.names[eoi], self.plt_opts.ct, plt_opts=self.plt_opts, sb_opts=self.sb_opts, cb_opts=self.cb_opts, clim=clim, save=filename)
            
        # ratio image
        if self.data_opts_ratio.isChecked():
            ims = self.ims_data
            imsdata = self.ims_data.data.copy()
            # perform data manipulations (resize, sqrt, log, rotation, binning, ...)
            imsdata = IMS.ims_data_manip(imsdata, resize=self.resize_opts, binning=self.bin_opts, neg2zero=self.plt_opts.n2z, mathop=self.math_opt, rotate=self.rot_opts)            
            eoi_nom = self.ratio_nom.currentIndex()
            eoi_den = self.ratio_den.currentIndex()
            ratio = np.zeros((imsdata.shape[0], imsdata.shape[1]))
            nom_data = imsdata[:,:,eoi_nom]
            den_data = imsdata[:,:,eoi_den]
            ratio[den_data != 0 ] = nom_data[den_data != 0 ] / den_data[den_data != 0 ]
            clim = np.zeros(2)
            if self.cbcut_min.text() == '':
                clim[0] = ratio.min()
            else:
                clim[0] = float(self.ratio_min.text())
            if self.ratio_max.text() == '':
                clim[1] = ratio.max()
            else:
                clim[1] = float(self.ratio_max.text())
            filename = "_"+self.plt_opts.ct+"_"+ims.names[eoi_nom]+'_'+ims.names[eoi_den]+"_ratio"+out_ext
            filename = filename_base+filename.replace(" ","") #remove all white spaces
            IMS.plot_image(ratio, ims.names[eoi_nom]+'/'+ims.names[eoi_den], self.plt_opts.ct, plt_opts=self.plt_opts, sb_opts=self.sb_opts, cb_opts=self.cb_opts, clim=clim, save=filename) 

    def empty(self):
        None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    plotims = Plotims()
    plotims.show()
    sys.exit(app.exec_())

