import numpy as np
from scipy.io import loadmat

"""
from github.com/erichson/ShallowDecoder.git
"""

def data_from_name(name):
    dataset_map = {
        "flow_cylinder": flow_cylinder,
        "sst": sst,
    }
    try:
        dataset_func = dataset_map[name]
    except KeyError:
        raise ValueError('dataset {} not recognized, choose one of: {}'.format(
            name, list(dataset_map.keys())))

    return dataset_func()


def rescale(Xsmall, Xsmall_test):
    """
    rescale data to between 0 and 1
    """
    Xmin = Xsmall.min()
    Xmax = Xsmall.max()
    
    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin)) 
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin)) 

    return Xsmall, Xsmall_test



def flow_cylinder():
    X = np.load('data/flow_cylinder.npy')
    
    # remove leading edge and halve horizontal resolution
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
        x = np.repeat(x,2,axis=0)
        x = np.rot90(x)
        return x

    return Xsmall, Xsmall_test, formatter



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

    return X_train, X_test, formatter

