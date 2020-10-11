#peter weir's code

import numpy as np
#from scipy.stats import chi2
#from scipy.stats import norm
from scipy import stats
import warnings
import pdb
from scipy.signal import butter, lfilter, freqz

def circmean(alpha,axis=None, **kwargs):
    
    if 'weights' in kwargs:
        
        vals_list=get_vec_list(alpha,**kwargs)
        
        N=len(vals_list)
       
        mean_angle = np.arctan2(np.nanmean(np.sin(vals_list),axis),np.nanmean(np.cos(vals_list),axis))
    else:
        mean_angle = np.arctan2(np.nanmean(np.sin(alpha),axis),np.nanmean(np.cos(alpha),axis))
    
    return mean_angle
    

def circfilter(alpha):
    order = 6
    fs = 40.0       # sample rate, Hz
    cutoff = 0.08  
    sinvls=np.sin(alpha)
    cosvls=np.cos(alpha)
    filt={}
    filt['sin']=butter_lowpass_filter(sinvls,cutoff,fs,order)
    filt['cos']=butter_lowpass_filter(cosvls,cutoff,fs,order)
    ret_angle=np.arctan2(filt['sin'], filt['cos'])
    return ret_angle
def vector_coherence_with_zero_deg(alpha, **kwargs):
    axis=None
    vals_list=get_vec_list(alpha,**kwargs)
    N=len(vals_list)
    R=np.sum(np.cos(alpha))/N
    V=1-R
    return V

def circvar(alpha,axis=None,**kwargs):
    if np.ma.isMaskedArray(alpha) and alpha.mask.shape!=():
        N = np.sum(~alpha.mask,axis)
    else:
        if axis is None:
            N = alpha.size
        else:
            N = alpha.shape[axis]
    
    #this is accomplished in an inelegant manner,
    #where I asemble 10,000 values proportioned according to the weights, and then compute.
    if 'weights' in kwargs:
        

        vals_list=get_vec_list(alpha,**kwargs)
        
        N=len(vals_list)
        R = np.sqrt(np.sum(np.sin(vals_list),axis)**2 + np.sum(np.cos(vals_list),axis)**2)/N
    
    else:
        R = np.sqrt(np.sum(np.sin(alpha),axis)**2 + np.sum(np.cos(alpha),axis)**2)/N
    
    V = 1-R
    return V

def get_vec_list(alpha,**kwargs):
    nvls=1e4
    vals_list=[]
    for crind, cr_alpha in enumerate(alpha):
        num_vls_to_add=np.round(kwargs['weights'][crind]*nvls)
            
        vals_list=np.append(vals_list,cr_alpha*np.ones(int(num_vls_to_add)))
    return vals_list

def circdiff(alpha,beta):
    D = np.arctan2(np.sin(alpha-beta),np.cos(alpha-beta))
    return D

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

