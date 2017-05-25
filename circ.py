#peter weir's code

import numpy as np
#from scipy.stats import chi2
#from scipy.stats import norm
from scipy import stats
import warnings
import pdb

def circmean(alpha,axis=None, **kwargs):
    
    if 'weights' in kwargs:
        
        wts=kwargs['weights']
        mean_angle = np.arctan2(stats.nanmean(wts*np.sin(alpha),axis),stats.nanmean(wts*np.cos(alpha),axis))
    else:
        mean_angle = np.arctan2(stats.nanmean(np.sin(alpha),axis),stats.nanmean(np.cos(alpha),axis))
    return mean_angle
    
def circvar(alpha,axis=None,**kwargs):
    if np.ma.isMaskedArray(alpha) and alpha.mask.shape!=():
        N = np.sum(~alpha.mask,axis)
    else:
        if axis is None:
            N = alpha.size
        else:
            N = alpha.shape[axis]
    
    if 'weights' in kwargs:
        
        wts=kwargs['weights']
        R = np.sqrt(np.sum(wts*np.sin(alpha),axis)**2 + np.sum(wts*np.cos(alpha),axis)**2)/N
    
    else:
        R = np.sqrt(np.sum(np.sin(alpha),axis)**2 + np.sum(np.cos(alpha),axis)**2)/N
    
    V = 1-R
    return V

def circdiff(alpha,beta):
    D = np.arctan2(np.sin(alpha-beta),np.cos(alpha-beta))
    return D



