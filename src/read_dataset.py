import numpy as np
from scipy.io import loadmat
import re
import abc

import cmocean
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib


def data_from_name(name, flat, **kwargs):
    """
    convert multiple forms of dataset names to one canonical short name
    """
    # get rid of dashes and underscores to match easier
    filtered_name = re.sub(r"[-_]", "", name.lower().strip())
    if filtered_name in ("cylinder", "flowcylinder"):
        return FlowCylinder(flat, **kwargs)
    if filtered_name in ("cylinderfull", "flowcylinderfull"):
        return FlowCylinder(flat, full=True, **kwargs)
    if filtered_name in ("sst", "seasurfacetemp", "seasurfacetemperature"):
        return SST(flat, full=False, **kwargs)
    if filtered_name in ("sstfull", "fullsst"):
        return SST(flat, full=True, **kwargs)
    raise ValueError("Unknown dataset '" + str(name) + "'")



class CustomDataset(abc.ABC):

    def __init__(self, name, X, Xtest, imshape, input_shape, write_index, 
            test_write_inds=None, mask=None):
        self.dataname = name
        self.imshape = imshape
        self.input_shape = input_shape
        self.write_index = write_index
        self.test_write_inds = [] if test_write_inds is None else test_write_inds
        self.X = X
        self.Xtest = Xtest
        self.Y = None
        self.Ytest = None
        self.mask = mask

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

    def __init__(self, flat, full=False, **kwargs):
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
            plt.savefig(filename, dpi=300)
        plt.close()


class SST(CustomDataset):

    def __init__(self, flat, full=False, full_test=False, no_basemap=False, 
            do_mask=False, no_continents=False, **kwargs):
        self.flat = flat
        self.full = full
        self.no_basemap = no_basemap
        self.no_continents = no_continents

        X = np.load('data/sstday.npy')
        lats = np.load('data/lats.npy')
        lons = np.load('data/lons.npy')

        if not full:
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
        # 3 years, 1 years
        training_idx = indices[220:1315]
        if full_test:
            test_idx = indices[1315:2590] # 3 years + 180 day buffer, for testing
            # test_idx = indices[1315:1315+30+180]
        else:
            test_idx = indices[1315:1680] # 1 year, for training
        #training_idx, test_idx = indices[230:2420], indices[2420:2557] # 6 years
        #training_idx, test_idx = indices[0:1825], indices[1825:2557] # 5 years    
        #training_idx, test_idx = indices[230:1325], indices[1325:2000] # 3 years
        
        # mask: True where sea, False where land
        mask = np.logical_not(np.any(X==0, axis=0))
        if flat:
            mask = mask.flatten()

        def mask_func(x):
            landmask = np.logical_not(mask)
            x[:,landmask] = 0.0
            return x

        m, n = imshape

        # nonland = mask_func(X)
        self.descaled_Xmin = 12
        self.descaled_Xmax = 33
        # unq = np.unique(nonland, return_counts=True)
        # print([(unq[0][i], unq[1][i]) for i in range(len(unq[0]))])
        # print(self.descaled_Xmin, self.descaled_Xmax)

        # scale
        X = X.reshape(-1,m*n)
        mean = X.mean(axis=0)
        X = X - mean
        X = X.reshape(-1,m*n)
        minm = np.min(X)
        ptp = np.ptp(X)
        X = 2 * (X - minm) / ptp - 1
        if flat:
            X = X.reshape(-1,m*n) 
        else:
            X = X.reshape(-1,m,n) 
    
        def de_scale(x, unflatten=False):
            """
            convert scaled -> celcius
            Returns:
                de-scaled units (np array), new units name (str)
            """
            x = x.reshape(-1,m*n)
            x = (x + 1) * ptp / 2 + minm
            x = x + mean
            if unflatten:
                x = x.reshape(-1,m,n)
            if do_mask:
                x = mask_func(x)
            return x.squeeze()

        self.de_scale = de_scale
        self.de_scale_units = "Celcius MAE"

        # split into train and test set
        X_train = X[training_idx]
        X_test = X[test_idx]

        if do_mask:
            print("doing masking")
            X_train = mask_func(X_train)
            X_test = mask_func(X_test)

        print("SST X shape:", X_train.shape, "Xtest shape:", X_test.shape)

        name="sst-full" if full else "sst"
        super().__init__(name=name, X=X_train, Xtest=X_test, imshape=imshape,
            input_shape=X_train[0].shape, write_index=140, test_write_inds=[380],
            mask=mask)
        self.lats = lats
        self.lons = lons
        # shift scaling to make slightly more vibrant, generally warmer
        self.Xmin = 0.8 * min(np.min(self.X), np.min(self.Xtest))
        self.Xmax = 0.65 * max(np.max(self.X), np.max(self.Xtest))

    def write_im(self, img, title, filename, directory="train_results", 
            subtitle="", show=False, outline=False, descale=True):

        # sometimes its just too hard to get this basemap package to work.
        # Easier to train in my gpu env and bring it back to my laptop to test 
        # and generate ims
        if self.no_basemap:
            if show:
                raise ValueError("Cannot make and show plots in '--no-basemap' mode")
            return
        # do 'conda install basemap' to get this module, for some reason
        from mpl_toolkits.basemap import Basemap

        img = img.reshape(self.imshape)
        if descale:
            img = self.de_scale(img, unflatten=True)

        fig, ax = plt.subplots(1, 1, facecolor="white",  edgecolor='k', figsize=(6,3))
        # ax = ax.ravel()
        # mintemp = np.min(img)
        # maxtemp = np.max(img)
        # minmax = np.maximum(mintemp,maxtemp)

        if self.full:
            m = Basemap(projection='mill',
                        lon_0 = 160,
                        llcrnrlat = 15.1,
                        llcrnrlon = 255.1,
                        urcrnrlat = 32.4,
                        urcrnrlon = 292.4,
                        #resolution='l',
                        ax=ax)
        else:
            m = Basemap(projection='mill',
                        lon_0 = 180,
                        llcrnrlat = 16.6,
                        llcrnrlon = 261.5,
                        urcrnrlat = 32,
                        urcrnrlon = 277.9,
                        #resolution='l',
                        ax=ax)
        if descale:
            vmin, vmax = self.descaled_Xmin, self.descaled_Xmax
        else:
            vmin, vmax = self.Xmin, self.Xmax
        m.pcolormesh(self.lons, self.lats, img, cmap=cmocean.cm.balance, 
            latlon=True, alpha=1.0, shading='gouraud', vmin=vmin, vmax=vmax)
        if not self.no_continents:
            m.fillcontinents(color='lightgray', lake_color='aqua')
            m.drawmapboundary(fill_color='lightgray')
            m.drawcoastlines(3)

        if descale:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmocean.cm.balance),
                fraction=0.0256, pad=0.04)

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
            plt.savefig(filename, dpi=300)
        plt.close()



