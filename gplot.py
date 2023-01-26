# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:08:34 2020

@author: prrta
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np


def gplot_rh5(h5file, channel='channel00'):
    f = h5py.File(h5file, 'r')
    spe = np.array(f['raw/'+channel+'/sumspec'])
    try:
        names = [n.decode('utf8') for n in f['fit/'+channel+'/names']]
        cfg = f['fit/'+channel+'/cfg'][()]
        # if type(names[0]) is type(bytes()):
        #     names = [n.decode('utf8') for n in f['fit/'+channel+'/names']]
        # if type(cfg) is type(bytes()):
        #     cfg = cfg.decode('utf8')
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

def h5_plot(h5file, channel='channel00', label=None, xrange=None, normtochan=None, yrange=None):
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
    if cfg is not None and names is not None:
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
    
    if cfg is not None: # we're looking for the energy calibration values...
        from PyMca5.PyMca import ConfigDict
        config = ConfigDict.ConfigDict()
        config.read(cfg)
        cfg = [config['detector']['zero'], config['detector']['gain'], config['fit']['energy'][0]] #zero, gain, E_Rayleigh
        names = []
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
    

