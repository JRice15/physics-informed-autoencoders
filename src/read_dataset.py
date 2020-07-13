import numpy as np
from scipy.io import loadmat
import re
import abc

import cmocean
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# do 'conda install basemap' to get this module for some reason
from mpl_toolkits.basemap import Basemap



def data_from_name(name, flat):
    """
    convert multiple forms of dataset names to one canonical short name
    """
    # get rid of dashes and underscores to match easier
    filtered_name = re.sub(r"[-_]", "", name.lower().strip())
    if filtered_name in ("cylinder", "flowcylinder"):
        return FlowCylinder(flat)
    if filtered_name in ("cylinderfull", "flowcylinderfull"):
        return FlowCylinder(flat, full=True)
    if filtered_name in ("sst", "seasurfacetemp", "seasurfacetemperature"):
        return SST(flat)
    raise ValueError("Unknown dataset " + name)



class CustomDataset(abc.ABC):

    def __init__(self, name, X, Xtest, imshape, input_shape, write_index):
        self.dataname = name
        self.imshape = imshape
        self.input_shape = input_shape
        self.write_index = write_index
        self.X = X
        self.Xtest = Xtest
        self.Y = None
        self.Ytest = None

    def set_Y(self, Y, Ytest):
        self.Y = Y
        self.Ytest = Ytest

    @abc.abstractmethod
    def write_im(self, img, title, filename, directory="train_results", 
            subtitle="", show=False, outline=False):
        """
        write image to dir/filename
        """
        ...


class FlowCylinder(CustomDataset):

    def __init__(self, flat, full=False):
        self.full = full
        self.flat = flat

        X = np.load('data/flow_cylinder.npy')
        # remove leading edge and halve horizontal resolution
        if not self.full:
            X = X[:,65::2,:]
        imshape = X[0].shape

        # Split into train and test set
        Xsmall = X[0:100, :, :]
        Xsmall_test = X[100:151, :, :]
        if flat:
            Xsmall = Xsmall.reshape(100, -1)
            Xsmall_test = Xsmall_test.reshape(51, -1)
        print("Flow cylinder X shape:", Xsmall.shape, "Xtest shape:", Xsmall_test.shape)

        name = "cyl-full" if full else "cylndr"
        super().__init__(name=name, X=Xsmall, Xtest=Xsmall_test, 
            imshape=imshape, input_shape=Xsmall[0].shape, write_index=7)


    def write_im(self, img, title, filename, directory="train_results", 
            subtitle="", show=False, outline=False):
        """
        if show=True, filename and directory can be None
        """
        shape = self.imshape
        img = img.reshape(shape)
        if not self.full:
            img = np.repeat(img, 2, axis=0)
        img = img.T

        shape0 = shape[0] if self.full else 2*shape[0]
        x2 = np.arange(0, shape[1], 1)
        y2 = np.arange(0, shape0, 1)
        mX, mY = np.meshgrid(x2, y2)
        minmax = np.max(np.abs(img)) * 0.65

        plt.figure(facecolor="white",  edgecolor='k', figsize=(7.9,4.7))
        # light contour (looks blurry otherwise)
        plt.contourf(mY.T, mX.T, img, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
        # heavy contour
        if outline:
            plt.contour(mY.T, mX.T, img, 40, colors='black', alpha=0.5, vmin=-minmax, vmax=minmax)
        im = plt.imshow(img, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)

        if self.full:
            wedge = mpatches.Wedge((0,99), 33, 270, 90, ec="#636363", color='#636363',lw = 5, zorder=200)
            im.axes.add_patch(p=wedge)
        plt.tight_layout()
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.xlabel(subtitle)
        plt.title(title.title())

        if show:
            plt.show()
        else:
            if filename[-1] == ".":
                ext = "png"
            else:
                ext = ".png"
            filename = directory + "/" + filename + ext
            plt.savefig(filename)
        plt.close()


class SST(CustomDataset):

    def __init__(self, flat):
        self.flat = flat

        X = np.load('data/sstday.npy')
        lats = np.load('data/lats.npy')
        lons = np.load('data/lons.npy')

        # make 64x64 crop
        ybottom = 6
        xleft = 28
        xright = -58
        X = X[:,ybottom:,xleft:xright]
        lats = lats[ybottom:,xleft:xright]
        lons = lons[ybottom:,xleft:xright]
        imshape = X[0].shape

        indices = range(3600)
        #training_idx, test_idx = indices[:730], indices[730:1000] 
        training_idx, test_idx = indices[220:1315], indices[1315:2557] # 6 years
        #training_idx, test_idx = indices[230:2420], indices[2420:2557] # 6 years
        #training_idx, test_idx = indices[0:1825], indices[1825:2557] # 5 years    
        #training_idx, test_idx = indices[230:1325], indices[1325:2000] # 3 years
        
        # scale
        m, n = imshape
        X = X.reshape(-1,m*n)
        X -= X.mean(axis=0)    
        X = X.reshape(-1,m*n)
        X = 2 * (X - np.min(X)) / np.ptp(X) - 1
        if flat:
            X = X.reshape(-1,m*n) 
        else:
            X = X.reshape(-1,m,n) 
        
        # split into train and test set
        X_train = X[training_idx]  
        X_test = X[test_idx]

        print("SST X shape:", X_train.shape, "Xtest shape:", X_test.shape)

        super().__init__(name="sst", X=X_train, Xtest=X_test, imshape=imshape,
            input_shape=X_train[0].shape, write_index=140)
        self.lats = lats
        self.lons = lons


    def write_im(self, img, title, filename, directory="train_results", 
            subtitle="", show=False, outline=False):
        img = img.reshape(self.imshape)
        fig, ax = plt.subplots(1, 1, facecolor="white",  edgecolor='k', figsize=(7,4))
        # ax = ax.ravel()
        plt.title(title)
        mintemp = np.min(img)
        maxtemp = np.max(img)
        minmax = np.maximum(mintemp,maxtemp)

        m = Basemap(projection='mill',
                    lon_0 = 180,
                    llcrnrlat = 16.6,
                    llcrnrlon = 261.5,
                    urcrnrlat = 32,
                    urcrnrlon = 277.9,
                    #resolution='l',
                    ax=ax)
        m.pcolormesh(self.lons, self.lats, img, cmap=cmocean.cm.balance, 
            latlon=True, alpha=1.0, shading='gouraud', vmin = -minmax, vmax=minmax)
        m.fillcontinents(color='lightgray', lake_color='aqua')
        m.drawmapboundary(fill_color='lightgray')
        m.drawcoastlines(3)
        plt.tight_layout()

        plt.xlabel(subtitle)
        plt.title(title.title())
        if show:
            plt.show()
        else:
            if filename[-1] == ".":
                ext = "png"
            else:
                ext = ".png"
            filename = directory + "/" + filename + ext
            plt.savefig(filename)
        plt.close()



