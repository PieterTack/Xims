# -*- coding: utf-8 -*-
"""
Plotims

Images mutliple ims/h5 arrays and allows for the addition of
    scale bars, color bars etc.
Based on plotims_gui from IDL
"""

import numpy as np
from scipy import stats
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec


class ims():    
    def __init__(self):
        self.data = np.zeros((2,2))
        self.names = ""
    
class Resize():
    def __init__(self):
        self.xstart = 0
        self.ystart = 0
        self.xend = 0
        self.yend = 0

class Binning():
    def __init__(self):
        self.binx = 1
        self.biny = 1
    
class Rotate():
    def __init__(self):
        self.angle = 0
        self.fliph = False
        self.flipv = False
    
class Colorbar_opt():
    def __init__(self, discr=False, dir='vertical', fs_num=14, fs_title=14, title=''):
        if discr:
            self.discr = discr
        else: self.discr = False
        if dir:
            self.dir = dir
        else: self.dir = 'vertical'
        if fs_num:
            self.fs_num = fs_num
        else: self.fs_num = 14
        if fs_title:
            self.fs_title = fs_title
        else: self.fs_title = 14
        if title:
            self.title = title
        else: self.title = ''
    
class Plot_opts():
    def __init__(self, aspect=1, interpol='nearest', title_fontsize=20, clim=None, ct='OrRd', n2z=None, frame=True):
        if aspect:
            self.aspect = aspect
        else: self.aspect = 1
        if interpol:
            self.interpol = interpol
        else: self.interpol = 'nearest'
        if title_fontsize:
            self.title_fontsize = title_fontsize
        else: self.title_fontsize = 20
        self.clim = clim
        if ct:
            self.ct = ct
        else: self.ct = 'OrRd'
        if n2z:
            self.n2z = True
        else: self.n2z = False
        if frame:
            self.frame = frame
        else: self.frame = True
    
class Scale_opts():
    def __init__(self, xscale=False, x_pix_size=0, x_scl_size=0, x_scl_text='', yscale=False, y_pix_size=0, y_scl_size=0, y_scl_text='', fontsize=16):
        if xscale:
            self.xscale = xscale
        else: self.xscale = False
        if yscale:
            self.yscale = yscale
        else: self.yscale = False
        if x_pix_size:
            self.x_pix_size = x_pix_size
        else: self.x_pix_size = 0
        if x_scl_size:
            self.x_scl_size = x_scl_size
        else: self.x_scl_size = 0
        if y_pix_size:
            self.y_pix_size = y_pix_size
        else: self.y_pix_size = 0
        if y_scl_size:
            self.y_scl_size = y_scl_size
        else: self.y_scl_size = 0
        if x_scl_text:
            self.x_scl_text = x_scl_text
        else: self.x_scl_text = ''
        if y_scl_text:
            self.y_scl_text = y_scl_text
        else: self.y_scl_text = ''
        if fontsize:
            self.fontsize = fontsize
        else: self.fontsize = 16
        
class Collated_image_opts():
    def __init__(self, ncol=False, nrow=False, cb=False):
        if ncol:
            self.ncol = ncol
        else: self.ncol = 3
        if nrow:
            self.nrow = nrow
        else: self.nrow = 3
        if cb:
            self.cb = cb
        else: self.cb = False

def read_ims(imsfile):
    element_array = ""
    line = ""
    with open(imsfile, "r") as f:
        f.readline()
        dim = [int(i) for i in f.readline().split(" ") if i.strip()] #should contain 3 or 4 elements, depending on ims dimensions (2 or 3D)
        if(len(dim) != 3):
            print("TODO") #TODO
        imsf = ims()
        imsf.data = np.zeros(dim)
        for j in range(0, dim[2]):
            for k in range(0,dim[1]):
                dim0count = 0
                while dim0count < dim[0]:
                    line = [float(i) for i in f.readline().split(" ") if i.strip()]
                    imsf.data[dim0count:dim0count+len(line),k,j] = line
                    dim0count += len(line)
        element_array = f.readlines()
    for i in range(0,dim[2]):
        element_array[i] = element_array[i].strip()
    print("Succesfully read "+imsfile)
    imsf.names = element_array
    return imsf

def save_as_tif(h5file, h5channel, el2plot, savefile_prefix):
    """
    Convert a H5 file data directory to separate *.TIF files, one for each provided element (el2plot keyword).

    Parameters
    ----------
    h5file : string
        File directory path to the H5 file containing the data to be converted.
    h5channel : string
        H5 data directory containing the (ims) data to be converted.
    el2plot : string (list)
        To convert all elements within the h5channel, set to 'All'. Alternatively, provide a list of strings with the element name identifiers to be converted.
    savefile_prefix : string
        A prefix for the file name the generated TIF files should contain.

    Returns
    -------
    bool
        returs True if conversion succeeded, False if unsuccesful.

    """
    import tifffile
    imsdata = read_h5(h5file, h5channel)
    if el2plot.lower() == 'all':
        el2plot = np.asarray(imsdata.names)
    else:
        el2plot = np.array(el2plot)
    
    if el2plot.size == 1:
        if el2plot not in imsdata.names:
            print('ERROR: save_as_tif: '+el2plot+' not in imsdata.names')
            return False
        idx = imsdata.names.index(el2plot)
        data = imsdata.data[:,:,idx]
        data[np.isnan(data)] = 0.
        tifffile.imwrite(savefile_prefix+'_'+''.join(imsdata.names[idx].split(" "))+'.tif', data.astype('float32'), photometric='minisblack')
        print('Saved '+savefile_prefix+'_'+''.join(imsdata.names[idx].split(" "))+'.tif')
    else:
        for i in range(0, el2plot.size):
            if el2plot[i] not in imsdata.names:
                print('ERROR: save_as_tif: '+el2plot+' not in imsdata.names')
                return False
            idx = imsdata.names.index(el2plot[i])
            data = imsdata.data[:,:,idx]
            data[np.isnan(data)] = 0.
            tifffile.imwrite(savefile_prefix+'_'+''.join(imsdata.names[idx].split(" "))+'.tif', data.astype('float32'), photometric='minisblack')
            print('Saved '+savefile_prefix+'_'+''.join(imsdata.names[idx].split(" "))+'.tif')
    return True
            
def read_h5(h5file, datadir):
    if type(datadir) is not type(list()):
        datadir = [datadir]
    with h5py.File(h5file, 'r') as file:
        data = []
        names = []
        for path in datadir:
            try:
                imsdat = np.array(file[path])
                if imsdat.ndim == 2:
                    imsdat = imsdat.reshape((1,imsdat.shape[0], imsdat.shape[1]))
                try:
                    nms = file['/'.join(path.split('/')[0:-1])+'/names']
                except Exception:
                    nms = [''.encode('utf8')]*imsdat.shape[0]
                for nm in nms: 
                    names.append(nm)
                data.append(imsdat)
            except Exception:
                print("Error: unknown data directory: "+path+" in "+h5file)
                return None
    
    imsdat = np.concatenate(data, axis=0)
    
    # rearrange ims array to match what plotims expects
    imsdata = np.moveaxis(imsdat, 0, -1)
    imsdata[np.isnan(imsdata)] = 0.
    
    rv = ims()
    rv.data = np.asarray(imsdata)
    rv.names = [n.decode('utf8') for n in names[:]]
    return rv

def write_ims(imsdata, names, filename):
    with open(filename, "w") as f:
        f.write(str(len(imsdata.shape)-1)+'\n')
        line = list('')
        for i in range(0, len(imsdata.shape)):
            line.append(str(imsdata.shape[i]))
        f.write(" ".join(line)+'\n')
        for i in range(0, len(names)):
            for j in range(0, imsdata.shape[1]):
                f.write(str(" ".join(str(k) for k in imsdata[:,j,i]))+'\n')
        for i in range(0, len(names)):
            f.write(names[i]+'\n')

def prepare_rgb_data(data, r_eoi, g_eoi, b_eoi, rmin, rmax, gmin, gmax, bmin, bmax):
    rgb_im = np.zeros([data.shape[0], data.shape[1], 3])
    r_im = data[:, :, r_eoi]
    g_im = data[:, :, g_eoi]
    b_im = data[:, :, b_eoi]
    # set lowest and highest value to selected min and max
    r_im[r_im < rmin] = rmin
    r_im[r_im > rmax] = rmax
    g_im[g_im < gmin] = gmin
    g_im[g_im > gmax] = gmax
    b_im[b_im < bmin] = bmin
    b_im[b_im > bmax] = bmax
    # fill rgb array
    rgb_im[:, :, 0] = np.squeeze(r_im)
    rgb_im[:, :, 1] = np.squeeze(g_im)
    rgb_im[:, :, 2] = np.squeeze(b_im)
    # normalise values between 0 and 255 for each RGB channel
    for i in range(0,3):
        if rgb_im[:, :, i].max() != rgb_im[:, :, i].min():
            rgb_im[:, :, i] = 255 * (rgb_im[:, :, i]-rgb_im[:, :, i].min())/(rgb_im[:, :, i].max()-rgb_im[:, :, i].min())
        else:
            rgb_im[:, :, i] = 0
    return rgb_im.astype(np.uint8)

def make_rgb_triangle():
    width = 296
    height = 296
    triangle = np.zeros([width,height,4])
    green = np.zeros([width,height])
    red = np.zeros([width,height])
    blue = np.zeros([width,height])
    alpha = np.zeros([width,height])
    alpha[:,:] = 255
    # make gradients
    for i in range(0,width):
        for j in range(0,height):
            green[i,j] = j
            if (green[i,j] > 255):
                green[i,j] = 255
            red[i,j] = 0.5*(255 - np.sqrt(3)*(i-147) - j)
            if(red[i,j] > 255):
                red[i,j] = 255
            if(red[i,j] < 0):
                red[i,j] = 0
            blue[i,j] = 0.5*(255 + np.sqrt(3)*(i-147) - j)
            if(blue[i,j] > 255):
                blue[i,j] = 255
            if(blue[i,j] < 0):
                blue[i,j] = 0
    # cut away corners to a triangle
    for i in range(0,width):
        for j in range(0,height):
            if(blue[i,j] == 0):
                red[i,j] = 0
            if(red[i,j] == 0):
                blue[i,j] = 0
            if(red[i,j] == 0 and blue[i,j] == 0):
                green[i,j] = 0
                alpha[i,j] = 0 # set background transparent
    triangle[:,:,0] = red
    triangle[:,:,1] = green
    triangle[:,:,2] = blue
    triangle[:,:,3] = alpha
    # scale max intensity to 255
    triangle2 = triangle
    for i in range(0, width):
        for j in range(0, height):
            if(triangle[i,j,0:3].max() > 0):
                triangle[i,j,0:3] = 255*triangle2[i,j,0:3]/triangle[i,j,0:3].max()
    rotate_triangle = np.zeros([height,width,4])
    for i in range(0, width):
        for j in range(0, height):
            rotate_triangle[height-j-1,i,:] = triangle[i,j,:]
    return rotate_triangle

def plot_rgb(imsdata, eoi_red, eoi_green, eoi_blue, filename, rmin=None, rmax=None, gmin=None, gmax=None, bmin=None, bmax=None, pix_size=None, scl_size=None, scl_unit=None, dpi=420):
    # we expect type of eoi_red/green/blue to be integer indices. If they are string, let's look in imsdata.names for their indices
    if type(eoi_red) is str:
        eoi_red = imsdata.names.index(eoi_red)
    if type(eoi_green) is str:
        eoi_green = imsdata.names.index(eoi_green)
    if type(eoi_blue) is str:
        eoi_blue = imsdata.names.index(eoi_blue)
    
    if rmin is None:
        rmin = np.min(imsdata.data[:,:,eoi_red])
    if rmax is None:
        rmax = np.max(imsdata.data[:,:,eoi_red])
    if gmin is None:
        gmin = np.min(imsdata.data[:,:,eoi_green])
    if gmax is None:
        gmax = np.max(imsdata.data[:,:,eoi_green])
    if bmin is None:
        bmin = np.min(imsdata.data[:,:,eoi_blue])
    if bmax is None:
        bmax = np.max(imsdata.data[:,:,eoi_blue])    

    plt_opts = Plot_opts(aspect=1.)
    
    # prepare rgb array to plot (each colour channel must be scaled between 0 and 255)
    rgb_im = prepare_rgb_data(imsdata.data.copy(), eoi_red, eoi_green, eoi_blue, rmin, rmax, gmin, gmax, bmin, bmax)
    # matplotlib plotting, to save image
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(2,1, height_ratios=[0.2,1])
    rgb0 = plt.subplot(gs[0], anchor='SW')
    rgb_triangle = make_rgb_triangle()
    rgb0.imshow(rgb_triangle.astype(np.uint8))
    rgb0.axis('off')
    # add labels on rgb triangle
    green_lbl = imsdata.names[eoi_green].split("-")
    green_lbl = green_lbl[0]
    red_lbl = imsdata.names[eoi_red].split("-")
    red_lbl = red_lbl[0]
    blue_lbl = imsdata.names[eoi_blue].split("-")
    blue_lbl = blue_lbl[0]
    rgb0.text(0.5, 0.56, green_lbl, size=16, ha="center", transform=rgb0.transAxes) #green label
    rgb0.text(0.25, 0.1, red_lbl, size=16, color='w', ha="center", transform=rgb0.transAxes) #red label
    rgb0.text(0.75, 0.1, blue_lbl, size=16, color='w', ha="center", transform=rgb0.transAxes) #blue label
    
    # add text with cutoff values
    red_cutoff = imsdata.names[eoi_red]+" cutoff value min: "+str(rmin)+" max:"+str(rmax)
    green_cutoff = imsdata.names[eoi_green]+" cutoff value min: "+str(gmin)+" max:"+str(gmax)
    blue_cutoff = imsdata.names[eoi_blue]+" cutoff value min: "+str(bmin)+" max:"+str(bmax)
    rgb0.text(1.2, 0.75, red_cutoff, size=12, ha="left", transform=rgb0.transAxes)
    rgb0.text(1.2, 0.625, green_cutoff, size=12, ha="left", transform=rgb0.transAxes)
    rgb0.text(1.2, 0.5, blue_cutoff, size=12, ha="left", transform=rgb0.transAxes)
    
    # plot actual rgb image
    rgb1 = plt.subplot(gs[1])
    rgb1.imshow(rgb_im.astype(np.uint8), aspect=1./plt_opts.aspect, interpolation=plt_opts.interpol)
    rgb1.axis('off')
    plt.tight_layout()
    # add scale_bar if requested
    if pix_size is not None and scl_size is not None:
        add_scalebar(rgb1, pix_size, scl_size, str(scl_size)+' '+scl_unit, dir='h')
    fig.savefig(filename, dpi=dpi)
    plt.close()  

def ims_data_manip(imsdata, resize=None, binning=None, neg2zero=None, mathop=None, rotate=None):
    # resize
    if resize:
        imsdata = imsdata[resize.ystart:resize.yend, resize.xstart:resize.xend, :]
    # complete rest of binning
    if binning:
        binx = binning.binx
        biny = binning.biny
        newims = np.zeros( (int(np.floor(imsdata.shape[0]/biny)), int(np.floor(imsdata.shape[1]/binx)), imsdata.shape[2]) )
        xshift = int(round((imsdata.shape[1]/binx-newims.shape[1])*binx/2))
        yshift = int(round((imsdata.shape[0]/biny-newims.shape[0])*biny/2))
        for i in range(0, newims.shape[0]):
            for j in range(0, newims.shape[1]):
                for k in range(0, newims.shape[2]):
                    newims[i, j, k] = np.sum(imsdata[i*biny+yshift:(i+1)*biny+yshift, j*binx+xshift:(j+1)*binx+xshift, k])
        imsdata = newims
    # neg2zero
    if neg2zero:
        imsdata[imsdata < 0] = 0
    # sqrt or log
    if mathop:
        if(mathop == 'sqrt'):
            imsdata[imsdata < 0] = 0
            imsdata = np.sqrt(imsdata)
        elif(mathop == 'log'):
            imsdata[imsdata < 1] = 1
            imsdata = np.log10(imsdata)
    # rotate
    if rotate:
        # first rotate over chosen angle
        imsdata = np.rot90(imsdata, int(rotate.angle/90), (0,1))
        # then flip/invert axis
        if(rotate.fliph == True):
            imsdata = np.flip(imsdata, 1)
        if(rotate.flipv == True):
            imsdata = np.flip(imsdata, 0)
    return imsdata

def plot_correl(imsdata, imsnames, el_id=None, save=None, dpi=420):
    """
    Display correlation plots.

    Parameters
    ----------
    imsdata : float array
        imsdata is a N*M*Y float array containing the signal intensities of N*M datapoints for Y elements.
    imsnames : string
        imsnames is a string array of Y elements, containing the names of the corresponding elements.
    el_id : integer list, optional
        el_id should be a integer list containing the indices of the elements to include in the plot. The default is None.
    save : string, optional
        File path as which the image should be saved. The default is None.

    Returns
    -------
    None.

    """
    imsdata = np.array(imsdata)
    imsnames = np.array(imsnames)
    data =  imsdata.reshape(imsdata.shape[0]*imsdata.shape[1], imsdata.shape[2])
    if el_id is not None:
        data = data[:, el_id]
        imsnames = imsnames[el_id]
    
    plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(imsnames.shape[0], imsnames.shape[0])
    num_bins = 5
    scatterplot = []
    for i in range(imsnames.shape[0]):
        for j in range(imsnames.shape[0]):
            scatterplot.append(plt.subplot(gs[i,j]))
            if j>i:
                scatterplot[-1].plot(data[:,j], data[:,i], linewidth=0, marker='.', alpha=0.5)  # correlation plots
                scatterplot[-1].margins(0.05)
                # fit and plot regression line
                z = np.polyfit(data[:,j], data[:,i], 1)
                p = np.poly1d(z)
                y_model = p(data[:,j])
                n = data[:,i].size                                           # number of observations
                m = z.size                                                 # number of parameters
                dof = n - m                                                # degrees of freedom
                t = stats.t.ppf(0.975, n - m)                              # used for CI and PI bands 0.975=95%CI
                resid = data[:,i] - y_model                           
                # chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
                # chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
                s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error
                scatterplot[-1].plot(data[:,j], y_model, "--", color="black", linewidth=1.0, alpha=0.5, label="Fit")
                x2 = np.linspace(np.min(data[:,j]), np.max(data[:,j]), 100)
                y2 = p(x2)
                ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(data[:,j]))**2 / np.sum((data[:,j] - np.mean(data[:,j]))**2))
                scatterplot[-1].fill_between(x2, y2 + ci, y2 - ci, color="gray", alpha=0.8)
                # calculate R² value
                rsq = np.sum((y_model - np.mean(data[:,i]))**2) / np.sum((data[:,i] - np.mean(data[:,i]))**2)
                scatterplot[-1].annotate('R² = {:.3f} '.format(rsq), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=8, color='gray')

            elif j==i:
                ymin = np.min(data[:,j]) - (np.max(data[:,j])-np.min(data[:,j]))*0.05
                ymax = np.max(data[:,j]) + (np.max(data[:,j])-np.min(data[:,j]))*0.05
                hist, edges = np.histogram(data[:,j], bins=num_bins*2)
                hist = np.min(data[:,j]) + ((hist-np.min(hist))/(np.max(hist)-np.min(hist))) * (np.max(data[:,j])-np.min(data[:,j]))
                xhist = np.zeros(hist.shape[0]*2)
                yhist = np.zeros(hist.shape[0]*2)
                for k in range(hist.shape[0]):
                    if k == 0:
                        xhist[k] = np.min(data[:,j])
                        xhist[-1] = np.max(data[:,j])
                    else:
                        xhist[2*k-1:2*k+1] = edges[k]
                    yhist[2*k:2*k+2] = hist[k]
                scatterplot[-1].fill_between(xhist, yhist, np.zeros(hist.shape[0]*2)+ymin, color='tab:Blue', alpha=0.5)
                scatterplot[-1].set_ylim(ymin, ymax)
                # scatterplot[-1].hist(data[:,j], bins=num_bins*2, density=True, alpha=0.5) # density plots
                # scatterplot[-1].margins(0.05)
            else:
                # kernel density estimate plots
                kernel = stats.gaussian_kde(np.vstack([data[:,j], data[:,i]]))
                xmin = np.min(data[:,j]) - (np.max(data[:,j])-np.min(data[:,j]))*0.05
                xmax = np.max(data[:,j]) + (np.max(data[:,j])-np.min(data[:,j]))*0.05
                ymin = np.min(data[:,i]) - (np.max(data[:,i])-np.min(data[:,i]))*0.05
                ymax = np.max(data[:,i]) + (np.max(data[:,i])-np.min(data[:,i]))*0.05
                X, Y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(kernel(positions).T, X.shape)
                scatterplot[-1].contour(X, Y, np.sqrt(Z), num_bins, cmap='Blues')
                # calculate Pearson correlation and give idea of confidence interval
                r, p = stats.pearsonr(data[:,j], data[:,i])
                p_stars = ''
                if p <= 0.05:  
                    p_stars = '*'
                if p <= 0.01:  
                    p_stars = '**'
                if p <= 0.001:  
                    p_stars = '***'
                scatterplot[-1].annotate('r = {:.2f} '.format(r) + p_stars, xy=(0.05, 0.85), xycoords='axes fraction', fontsize=8, color='gray')

            if i == imsnames.shape[0]-1:
                scatterplot[-1].set_xlabel(imsnames[j], fontsize=12)
                scatterplot[-1].xaxis.set_major_locator(plt.MaxNLocator(3))
            if i != imsnames.shape[0]-1:
                scatterplot[-1].set_xticklabels('')
                scatterplot[-1].set_xticks([])
                
            if j == 0:
                scatterplot[-1].set_ylabel(imsnames[i], fontsize=12)
                scatterplot[-1].yaxis.set_major_locator(plt.MaxNLocator(3))
            if j != 0:
                scatterplot[-1].set_yticklabels('')
                scatterplot[-1].set_yticks([])
             
    plt.tight_layout()
    plt.show()
    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()        



def add_scalebar(target, pix_size, scl_size, scale_text,scale_fontsize=16, dir=''):
    if(dir == 'horizontal' or dir == 'h'):
        target.plot([0,scl_size/pix_size], [-2,-2], lw=2, color='black', clip_on=False)
        target.text(scl_size/(2.*pix_size), -3, scale_text, ha='center', va='bottom', size=scale_fontsize, clip_on=False)
    if(dir == 'vertical' or dir == 'v' ):
        target.plot([-2,-2], [0,scl_size/pix_size], lw=2, color='black', clip_on=False)
        target.text(-3, scl_size/(2.*pix_size), scale_text, ha='right', va='center', size=scale_fontsize, rotation=90, clip_on=False)

def plot_image(imsdata, imsname, ctable, plt_opts=None, sb_opts=None, cb_opts=None, clim=None, save=None, subplot=None, dpi=420):
    # set option for discrete colorbar, only if 10 or less values are plotted
    if(cb_opts and cb_opts.discr and imsdata.max()-imsdata.min() <= 10):
        ctable = plt.cm.get_cmap(ctable, np.around(imsdata.max()-imsdata.min()+1).astype(int))
        clim = [imsdata.min()-0.5, imsdata.max()+0.5]
    if plt_opts:
        aspect = plt_opts.aspect
        interpol = plt_opts.interpol
        fs_im_tit = plt_opts.title_fontsize
        frame = plt_opts.frame
    else:
        aspect = 1
        interpol = 'nearest'
        fs_im_tit = 20
        frame = True
    # if this plot is part of a subplot, provide subplot axes. Otherwise extract plt axes
    if type(subplot) != type(None):
        ncols = subplot[0]
        nrows = subplot[1]
        n_el = imsdata.shape[2]
    else:
        ncols, nrows, n_el = 1, 1, 1

    dx, dy = imsdata.shape[1], imsdata.shape[0]
    img_high = 6 # height in inches of all image rows stacked, omitting whitespaces
    pad = 0.05 # padding between cb or title and image in inches
    
    ws_ver, ws_hor = 0.35, 0.01
    if cb_opts:
        if cb_opts.dir == 'vertical': #---vertical colorbar---#
            ws_ver = 0.35 # vertical whitespace between images [inch]
            ws_hor = 0.6 # horizontal whitespace between images [inch]
        elif cb_opts.dir == 'horizontal': #---horizontal colorbar---#
            ws_ver = 0.6 # vertical whitespace between images [inch]
            if cb_opts.title == '':
                ws_hor = 0.01 # horizontal whitespace between images [inch]
            elif cb_opts.title == "Int.;[$\sqrt{cts}$]" or cb_opts.title == "Int.;[cts]" or cb_opts.title == "Int.;[log(cts)]":
                ws_hor = 0.3
            else:
                ws_hor = 0.55
        cb_opts.title = '\n'.join(cb_opts.title.split(';'))

    height = img_high + (nrows+1)*(ws_ver+pad)  # width and height are in inches
    width = (ws_hor+pad)*(ncols+1) + ncols*(img_high/nrows)*(dx/dy)*aspect # each image has width dependent on aspect ratio and dx/dy ratio
    fig = plt.figure(figsize=(width, height))

    # now we know each image is ((img_high/nrows)*(dx/dy)*aspect, (img_high/nrows)) inches wide,high. Convert this to relative ratio of image and whitespace:
    im_rel_wide, im_rel_high = ((img_high/nrows)*(dx/dy)*aspect)/width, (img_high/nrows)/height
    ws_rel_wide, ws_rel_high = ws_hor/width, ws_ver/height
    col_id, row_id = 0, 0
    for j in range(n_el):
        if n_el == 1:
            data = imsdata[:,:]
            name = imsname
        else:
            data = imsdata[:,:,j]
            name = imsname[j]
        if type(clim) != type(None):
            # clim = clim
            if(clim[0] > np.min(data) and clim[1] < np.max(data)):
                extend = 'both'
            elif clim[0] > np.min(data):
                extend = 'min'
            elif clim[1] < np.max(data):
                extend = 'max'
            else:
                extend = 'neither'
            limit = clim
        else:
            limit = None
            extend = 'neither'
        im_xstart = ws_rel_wide+(im_rel_wide+ws_rel_wide+pad/width)*col_id
        im_ystart = 1.-(im_rel_high+ws_rel_high)*(row_id+1)
        ax_im = fig.add_axes([im_xstart, im_ystart, im_rel_wide, im_rel_high])
        img = ax_im.imshow(data, interpolation=interpol, cmap=ctable, aspect='auto', clim=limit)
        plt.text(dx/2, -1*(pad/(im_rel_high*height))*dy, name, ha='center', va='bottom', size=fs_im_tit, clip_on=False)
        if frame: # frame image or not
            ax_im.set(xticks=[], yticks=[])
        else:
            ax_im.axis('off')
        # add scale_bar if requested
        if(sb_opts and sb_opts.xscale):
            sb_ax = fig.add_axes([im_xstart, im_ystart+im_rel_high+pad/height, im_rel_wide, 0], sharex=ax_im)
            sb_ax.axis('off')
            add_scalebar(sb_ax, sb_opts.x_pix_size, sb_opts.x_scl_size, sb_opts.x_scl_text, scale_fontsize=sb_opts.fontsize, dir='h')
        if(sb_opts and sb_opts.yscale):
            sb_ax = fig.add_axes([im_xstart-pad/width, im_ystart, 0, im_rel_high], sharey=ax_im)
            sb_ax.axis('off')
            add_scalebar(sb_ax, sb_opts.y_pix_size, sb_opts.y_scl_size, sb_opts.y_scl_text, scale_fontsize=sb_opts.fontsize, dir='v')
        # draw colorbar on its own axis, within image dimensions
        if cb_opts:
            if cb_opts.dir == 'vertical':
                ax_cb = fig.add_axes([im_xstart+im_rel_wide+pad/width, im_ystart, 0.1/width, im_rel_high])
                ax_cb.text(2, 1+(pad/(im_rel_high*height)), cb_opts.title, ha='center', va='bottom', size=cb_opts.fs_title, clip_on=False, transform=ax_cb.transAxes)
                cbar = fig.colorbar(img, orientation=cb_opts.dir, extend=extend, cax=ax_cb)
            if cb_opts.dir == 'horizontal':
                ax_cb = fig.add_axes([im_xstart, im_ystart-(pad+0.1)/height, im_rel_wide, 0.1/height])
                ax_cb.text(1+(pad+ws_hor)/(2*im_rel_wide*width), 0, cb_opts.title, ha='center', va='center', size=cb_opts.fs_title, clip_on=False, transform=ax_cb.transAxes)
                cbar = fig.colorbar(img, orientation=cb_opts.dir, extend=extend, cax=ax_cb)
            cbar.ax.tick_params(labelsize=cb_opts.fs_num)
            
        col_id = col_id+1
        if col_id >= ncols:
            col_id = 0
            row_id = row_id+1
    # save the image (and close it)
    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()        

def plot_colim(imsdata, el_selection, colortable, plt_opts=None, sb_opts=None, cb_opts=None, colim_opts=None, save=None):
    if colim_opts:
        ncols = colim_opts.ncol
        nrows = colim_opts.nrow
        if colim_opts.cb == False:
            cb_opts = None # dont't draw colorbar if chosen (default is to draw)
    else:
        ncols = int(np.floor(np.sqrt(len(el_selection))))
        nrows = int(np.ceil(len(el_selection)/ncols))
    if plt_opts:
        if type(plt_opts.clim) != type(None):
            clim = plt_opts.clim
        else: clim = None
        plt_opts.title_fontsize = plt_opts.title_fontsize/2
    else:
        clim = None
    if sb_opts:
        sb_opts.fontsize = sb_opts.fontsize/2 #divide fontsize by 2 for collated imaging
    if cb_opts:
        cb_opts.fs_title = cb_opts.fs_title/2
        cb_opts.fs_num = cb_opts.fs_num/2
    
    # Make data array containing only the elements to plot.
    #   First let's check how many elements are common between el_selection and imsdata.names
    cnt = 0
    for i in range(len(el_selection)):
        if el_selection[i] in np.arange(len(imsdata.names)):
            for k in range(len(imsdata.names)):
                if k == el_selection[i]:
                    cnt = cnt+1                
    #    Now make empty array, and repeat loop above to fill array appropiately
    datacube = np.zeros((imsdata.data.shape[0], imsdata.data.shape[1], cnt))
    names = list('')
    if clim:
        cb_lim = np.zeros((cnt,2))
    else:
        cb_lim = None
    cnt = 0
    for i in range(len(el_selection)):
        if el_selection[i] in np.arange(len(imsdata.names)):
            for k in range(len(imsdata.names)):
                if k == el_selection[i]:
                    datacube[:,:,cnt] = imsdata.data[:,:,k]
                    names.append(imsdata.names[k])
                    if clim:
                        cb_lim[cnt,0] = np.min(imsdata.data[:,:,k])+clim[0]*(np.max(imsdata.data[:,:,k])-np.min(imsdata.data[:,:,k]))
                        cb_lim[cnt,1] = np.min(imsdata.data[:,:,k])+clim[1]*(np.max(imsdata.data[:,:,k])-np.min(imsdata.data[:,:,k]))
                    cnt = cnt+1

    # Perform plotting. plot_image will know how to handle collated images based on amount of rows and columns.
    plot_image(datacube, names, colortable, plt_opts=plt_opts, sb_opts=sb_opts, cb_opts=cb_opts, clim=cb_lim, save=save, subplot=(ncols, nrows))
    # set fonts back to normal value for further imaging; probably useless as sb_opts will be local variable, but meh
    if sb_opts:
        sb_opts.fontsize = sb_opts.fontsize*2 
    if plt_opts:
        plt_opts.title_fontsize = plt_opts.title_fontsize*2
    if cb_opts:
        cb_opts.fs_title = cb_opts.fs_title*2
        cb_opts.fs_num = cb_opts.fs_num*2
