import numpy as np
from scipy.io import loadmat
import re
import abc

"""
from github.com/erichson/ShallowDecoder.git
"""

class CustomDataset(abc.ABC):

    def __init__(self, name, X, Xtest):
        self.dataname = name
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

    def __init__(self, full=False):
        self.full = full

        X = np.load('data/flow_cylinder.npy')
        # remove leading edge and halve horizontal resolution
        if not self.full:
            X = X[:,65::2,:]

        # Split into train and test set
        Xsmall = X[0:100, :, :]
        t, m, n = Xsmall.shape
        
        Xsmall = Xsmall.reshape(100, -1)
        Xsmall_test = X[100:151, :, :].reshape(51, -1)
        print("Flow cylinder X shape:", Xsmall.shape, "Xtest shape:", Xsmall_test.shape)

        super().__init__("cylndr")

    def write_im(self, img, title, filename, directory="train_results", 
            subtitle="", show=False, outline=False):


def data_from_name(name):
    """
    convert multiple forms of dataset names to one canonical short name
    """
    # get rid of dashes and underscores to match easier
    name = re.sub(r"[-_]", "", name)
    if name in ("cylinder", "flowcylinder"):
        return FlowCylinder()
    if name in ("cylinderfull", "flowcylinderfull"):
        return "cyl-full"
    if name in ("sst", "seasurfacetemp", "seasurfacetemperature"):
        return "sst"
    raise ValueError("Unknown dataset " + name)


def rescale(Xsmall, Xsmall_test):
    """
    rescale data to between 0 and 1
    """
    Xmin = Xsmall.min()
    Xmax = Xsmall.max()
    
    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin)) 
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin)) 

    return Xsmall, Xsmall_test



def flow_cylinder(full=False):
    X = np.load('data/flow_cylinder.npy')
    
    # remove leading edge and halve horizontal resolution
    if not full:
        X = X[:,65::2,:]

    # Split into train and test set
    Xsmall = X[0:100, :, :]
    t, m, n = Xsmall.shape
    
    Xsmall = Xsmall.reshape(100, -1)

    Xsmall_test = X[100:151, :, :].reshape(51, -1)

    print("Flow cylinder X shape:", Xsmall.shape, "Xtest shape:", Xsmall_test.shape)
    # Xsmall, Xsmall_test = rescale(Xsmall, Xsmall_test)

    def formatter(x):
        x = x.reshape((m,n))
        if not full:
            x = np.repeat(x,2,axis=0)
        x = np.rot90(x)
        return x

    return Xsmall, Xsmall_test, formatter, (m,n)



def sst():

    X = np.load('data/sstday.npy')
    #******************************************************************************
    # Preprocess data
    #******************************************************************************
    t, m, n = X.shape


    #******************************************************************************
    # Slect train data
    #******************************************************************************
    #indices = np.random.permutation(1400)
    indices = range(3600)
    #training_idx, test_idx = indices[:730], indices[730:1000] 
    training_idx, test_idx = indices[220:1315], indices[1315:2557] # 6 years
    #training_idx, test_idx = indices[230:2420], indices[2420:2557] # 6 years
    #training_idx, test_idx = indices[0:1825], indices[1825:2557] # 5 years    
    #training_idx, test_idx = indices[230:1325], indices[1325:2000] # 3 years
    
    
    # mean subtract
    X = X.reshape(-1,m*n)
    X -= X.mean(axis=0)    
    
    # scale 
    X = X.reshape(-1,m*n)
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    X = X.reshape(-1,m,n) 
    
    # split into train and test set
    
    X_train = X[training_idx]  
    X_test = X[test_idx]

 
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    print("SST X shape:", X_train.shape, "Xtest shape:", X_test.shape)
    # return X_train, X_test, X_train, X_test, m, n

    def formatter(x):
        return x.reshape((m,n))

    return X_train, X_test, formatter, (m,n)

