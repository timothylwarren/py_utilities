#!/usr/bin/python
import pdb
import pylab
import numpy as np
import datetime
import numpy.random as npr
import scipy.stats as st
import scipy.signal as sig
import circ as circ
import cmath as cmath
import scipy
#import tw_plot_library3 as plt
import csv

#pylab.ion()
#from numpy import sin, linspace, pi
#from pylab import plot, show, title, xlabel, ylabel, subplot
#from scipy import fft, arange


def permute_test_for_mean_diff_between_two_groups(dat1, dat2, num_permutations=10000):
    len_dat1=len(dat1)
    len_dat2=len(dat2)
    diff_list=[]
    for cr_permutation in np.arange(num_permutations):
        tmptst=dat1+dat2
        npr.shuffle(tmptst)
        diff_list.append(np.mean(tmptst[0:len_dat1])-np.mean(tmptst[len_dat1:]))
    
    return diff_list

def calc_offset_hist(indt,offset_value):
    mot_inds=np.arange(0,len(indt['time_in_min']))
    
    tmp_mot_rad=indt['mot_rad'][mot_inds]

    mot_rad=tmp_mot_rad-offset_value
    crdeg=rad_to_deg(standardize_angle(mot_rad,2*np.pi,force_positive=1))
    deg_per_bin=10
    degbins = np.arange(0,370,deg_per_bin)           
                
    hstout_dt=make_hist_calculations(crdeg,degbins)
    return hstout_dt
def heat_map_relative_weights(heatmap,rvalues):
    #find indices between -90,90
    #90,270
    supra_270=np.where(rvalues>=3*np.pi/2)
    sub_90=np.where(rvalues<=np.pi/2)
    
    
    middle=np.intersect1d(np.where(rvalues>=np.pi/2)[0],np.where(rvalues<3*np.pi/2)[0])
    ratio=[]
    for crht in heatmap:
        reverse_sum=np.sum(np.sum(crht[middle,:],axis=0))
        sum_180=np.sum(np.sum(crht[supra_270,:],axis=0))+np.sum(np.sum(crht[sub_90,:],axis=0))
        ratio.append(sum_180/reverse_sum)

    return ratio
#peter weir's function
def count_ommatidia(dataFileName):
    numOmmatidia = 1398
    ox, oy, oz = np.ones(numOmmatidia), np.ones(numOmmatidia), np.ones(numOmmatidia)

    ommatidiumInd = 0
    with open(dataFileName, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ox[ommatidiumInd], oy[ommatidiumInd], oz[ommatidiumInd] = float(row[0]), float(row[1]), float(row[2])
            ommatidiumInd += 1

    # convert to spherical coordinates:

    r = np.sqrt(ox**2 + oy**2)

    ommatidiaElevations = np.arctan2(oz,r)
    ommatidiaAzimuths = np.arctan2(oy,ox)

    # arena geometry:
    arenaMirrorCurvatureRadius_mm = 50.
    arenaMirrorFocalDistance_mm = 25.
    arenaOuterRadius_mm = 22.
    arenaInnerRadius_mm = 5.

    yOuter = arenaMirrorFocalDistance_mm + np.sqrt(arenaMirrorCurvatureRadius_mm**2 - arenaOuterRadius_mm**2) - arenaMirrorCurvatureRadius_mm
    yInner = arenaMirrorFocalDistance_mm + np.sqrt(arenaMirrorCurvatureRadius_mm**2 - arenaInnerRadius_mm**2) - arenaMirrorCurvatureRadius_mm

    arenaOuterElevation = np.pi/2 - np.arctan(arenaOuterRadius_mm/yOuter)
    arenaInnerElevation = np.pi/2 - np.arctan(arenaInnerRadius_mm/yInner)

    print 'mirror spans an annulus from elevation', arenaOuterElevation*180/np.pi, 'to elevation', arenaInnerElevation*180/np.pi
    print 'assuming DRA spans entire northern hemisphere and samples evenly across it, this means our stimulus covers', (arenaInnerElevation-arenaOuterElevation)*100./np.pi, 'percent of the DRA'

    ommatidiaViewingMirror = (ommatidiaElevations<arenaInnerElevation) & (ommatidiaElevations>arenaOuterElevation)

    print 'number of ommatidia viewing mirror =', np.sum(ommatidiaViewingMirror.astype('int'))
    print 'out of', numOmmatidia, 'total ommatidia'
    print 'so', np.sum(ommatidiaViewingMirror.astype('int'))*100./float(numOmmatidia), 'percent of ommatidia view the mirror'


    arena_stats={}
    arena_stats['ommatidiaAzimuths']=ommatidiaAzimuths
    arena_stats['ommatidiaElevations']=ommatidiaElevations
    arena_stats['arenaOuterElevation']=arenaOuterElevation
    arena_stats['arenaInnerElevation']=arenaInnerElevation
    arena_stats['ommatidiaViewingMirror']=ommatidiaViewingMirror

    return arena_stats







def decimate(crdt,decimate_ratio):
    #remove first N values as they appear to be erroneous
    chop_values=10
    tst=sig.decimate(crdt,decimate_ratio)
    return tst[chop_values:]
def make_histogram_confidence_intervals(indata,bnds,num_bins):
    #calculate histogram n times
    sumhist=[]
    for crind in np.arange(1000):
        axh=[]
        
        dat=np.reshape(indata,len(indata))
        crdt=np.random.choice(dat,len(dat),replace=True)
        
        bins=np.linspace(bnds[0],bnds[1],num_bins+1)
        inarray=indata[~np.isnan(indata)]
        weights = np.ones_like(inarray)/len(inarray)

    
        hist, bins = np.histogram(crdt[~np.isnan(crdt)], bins=bins, weights=weights)



        
        sumhist.append(hist)
    #calculate hist confidence intervals
    return bins,np.percentile(sumhist,2.5,axis=0),np.percentile(sumhist,97.5,axis=0)

def get_clim_max(arrays):
    max=np.max(arrays)


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if type(ndarray) is list:
        out_vl=[]
        for cr_array in ndarray:
            
            out_vl.append(compute_bins(cr_array,new_shape,operation))

    else:

        out_vl=compute_bins(ndarray,new_shape,operation)
    return out_vl
def compute_bins(ndarray,new_shape,operation):
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray



def nanmean_matrix(indata,**kwargs):
    try:
        ax=kwargs['axis']
    except:
        ax=0
   
    mdat=np.ma.masked_array(indata,np.isnan(indata))
    mm=np.mean(mdat,axis=ax)
    return mm

def nanmedian_matrix(indata,**kwargs):
    try:
        ax=kwargs['axis']
    except:
        ax=0
   
    mdat=np.ma.masked_array(indata,np.isnan(indata))
    
    mm=np.median(mdat,axis=ax)
    return mm

def steppify(arr,isX=False,interval=0):
    
    if isX and interval==0:
        newlist=[]
        for crind in np.arange(len(arr)):
            for k in [0,1]:
                newlist.append(arr[crind])
        
        return np.array(newlist[0:-1])
    else:
        newlist=[]
        for crind in np.arange(len(arr)):
            for k in [0,1]:
                newlist.append(arr[crind])
        return np.array(newlist[1:])


def nanstd_matrix(indata,**kwargs):
    try:
        ax=kwargs['axis']
    except:
        ax=0
   
    mdat=np.ma.masked_array(indata,np.isnan(indata))
    mm=np.std(mdat,axis=ax)
    return mm
    
def intersection(first, *others):
    return set(first).intersection(*others)

def downsample(iny,ratio,**kwargs):
    #if 'ratio' in kwargs:
    #    ratio=kwargs['ratio']
    #else:
    #    ratio=len(newx)/len(inx)
    try:
        CIRC_FLAG=kwargs['circ']
    except:
        CIRC_FLAG=False


    if 'method' in kwargs:
        if kwargs['method'] is 'chunk':
            outvls=[]
            
            if ratio>1:
                inds=np.arange(0,len(iny),ratio)
                for i in np.arange(len(inds)-1):
                    
                    crvls=np.array(iny[inds[i]:inds[i+1]])
                    nonaninds=np.where(np.isnan(crvls)==0)
                    try:
                        outvls.append(np.mean(crvls[nonaninds[0]]))
                    except:
                        pdb.set_trace()
                return outvls
            else:
                return
    else:
        pad_size=np.ceil(float(iny.size)/ratio)*ratio-iny.size
        iny_padded=np.append(iny,np.zeros(pad_size)*np.NaN)
        if CIRC_FLAG:

            return circ.circmean(iny_padded.reshape(-1,ratio),axis=1)
        else:
            return scipy.nanmean(iny_padded.reshape(-1,ratio),axis=1)
        
def get_dynamic_vec_strength(mot_dt,time_dt,calculation_type,vec_calc_type,filter_len_in_sec,**kwargs):
    if time_dt.size==0:
        return
    
    
    total_time=time_dt[-1]*60.
    
    datalen=len(time_dt)
    sample_rate=datalen/total_time
    filter_len_in_pts=filter_len_in_sec*sample_rate
    gaussian_wts=make_gaussian_weights(filter_len_in_pts,normalize_type='sum')
    
    
    if calculation_type is 'convolution':
            #for variance, first perform convolution
            #of sin and cos with gaussian
        
            #allvls=np.convolve(crdt['mot_rad'],gaussian_wts,mode='valid')
        if vec_calc_type=='360':
            sinvls=np.convolve(np.sin(mot_dt),gaussian_wts,mode='valid')
            cosvls=np.convolve(np.cos(mot_dt),gaussian_wts,mode='valid')
        elif vec_calc_type=='180':
            sinvls=np.convolve(np.sin(2.0*mot_dt),gaussian_wts,mode='valid')
            cosvls=np.convolve(np.cos(2.0*mot_dt),gaussian_wts,mode='valid')
        vec_time_list=np.convolve(time_dt,gaussian_wts,mode='valid')
            
        R = np.sqrt(sinvls**2 + cosvls**2)
        lenlst=R
            
        mn_lst=np.arctan2(sinvls,cosvls)
        
        return lenlst,mn_lst,vec_time_list
            
            #mn_dot_product=np.mean(np.abs(np.cos(allvls-crdt['mnrad'])))
        
        
    elif calculation_type is 'sliding_window':
        end_data_vl=datalen-filter_len_in_pts
        STEP_LEN_IN_SEC=0.1
        STEP_LEN_IN_PTS=STEP_LEN_IN_SEC*sample_rate
        vector_vls=np.arange(0,end_data_vl,STEP_LEN_IN_PTS)
        mn_lst=[]
        lenlst=[]
        vec_time_list=[]
        for crvl in vector_vls:
            
            heading_data=crdt['mot_rad'][crvl:crvl+filter_len_in_pts]
            
            vec_time_list.append(np.mean(crdt['time_in_min'][crvl:crvl+filter_len_in_pts]))
            doubled_data=np.mod(2*heading_data,2*np.pi)
            gaussian_weights=make_gaussian_weights(len(heading_data),normalize_type='mean')
                
            mn_lst.append(circ.circmean(heading_data,weights=gaussian_weights))
                #modified to use heading_data, not doubled_data for vector strength
            lenlst.append(1-circ.circvar(heading_data,weights=gaussian_weights))
                
            #mn_dot_product.append(np.mean(np.abs(np.cos(heading_data-crdt['mnrad']))))
        return lenlst,mn_lst,vec_time_list
#set to have a sum of 1

#idea here is to calculate confidence interval 
#simple way to do this is resample with replacement
#1000 times to get smooth distribution
def resample_for_threshold(indata,**kwargs):
    prctile=kwargs['prctile']
    thresh_val=np.percentile(np.random.choice(indata,1000),95)
    return thresh_val



def make_gaussian_weights(numpts,**kwargs):
        #6 is chosen to get ends to 0
    tmp_weights=sig.get_window(('gaussian',numpts/6.),numpts)
    #sum of weights is 1
    if kwargs['normalize_type']=='sum':
        return tmp_weights/np.sum(tmp_weights)        
    elif kwargs['normalize_type']=='mean':
        return tmp_weights/np.mean(tmp_weights)
        
        
def bootstrap(x,**kwargs):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    
    # dat=np.array(data)
    # n_vl = len(dat)
    # idx = npr.randint(0, n_vl, (num_samples, n_vl))
    # samples = dat[idx]
    
    # stat = np.sort(np.max(samples, 1))
    # return (stat[int((alpha/2.0)*num_samples)],stat[int((1-alpha/2.0)*num_samples)])
    try:
        conf_intervals=kwargs['conf_intervals']
    except:
        conf_intervals=[2.5, 97.5]


    n = len(x)
    reps = 10000
    xb = np.random.choice(x, (n, reps))
    mb = xb.mean(axis=0)
    #mb = np.percentile(xb, 10, axis=0)
    #mb.sort()

    [lower, upper]= np.percentile(mb, conf_intervals)
    return lower, upper




#inheadings(radians)
def make_forward_trajectory(input_headings, **kwargs):
    #generate a series of r,theta headings in polar coordinates
    cr_position={}
    if kwargs['calc_type'] is 'doubled':
    	input_headings=2.*input_headings
    elif kwargs['calc_type'] is 'mod180':
    	input_headings=np.mod(input_headings,np.pi)
    
    cr_position['x']=np.cumsum(np.cos(input_headings))
    cr_position['y']=np.cumsum(np.sin(input_headings))
    
    return cr_position

def vector_addition(pos1,pos2,**kwargs):
    if kwargs['mod90']==1:
    	pos1[1]=radians_to_quadrant(pos1[1])
    	pos2[1]=radians_to_quadrant(pos2[1])
    new_xvl=pos1[0]*np.cos(pos1[1])+pos2[0]*np.cos(pos2[1])
    new_yvl=pos1[0]*np.sin(pos1[1])+pos2[0]*np.sin(pos2[1])
    return linear_to_polar([new_xvl,new_yvl])

def rotate_xy_by_angle(inx,iny,shift_heading):
    invl={}
    new_polar_vl={}
    invl['x']=inx
    invl['y']=iny
    polar_vl=linear_to_polar(invl)
    new_polar_vl=polar_vl
    new_polar_vl['theta']=polar_vl['theta']+shift_heading
    output_vl=polar_to_linear(new_polar_vl)

    return output_vl

def linear_to_polar(input_pos):
    output_pos={}
    
    if type(input_pos['x']) is list:
        for key in ['theta','len']:
            output_pos[key]=[]
        for crind in np.arange(len(input_pos['x'])):
            output_pos['theta'].append(np.arctan2(input_pos['y'][crind],input_pos['x'][crind]))
            output_pos['len'].append(np.sqrt(input_pos['x'][crind]**2+input_pos['y'][crind]**2))
    else:    
        
        output_pos['theta']=np.arctan2(input_pos['y'],input_pos['x'])
        output_pos['len']=np.sqrt(input_pos['x']**2+input_pos['y']**2)
    return output_pos
    
def polar_to_linear(input_pos):
    output_pos={}
    output_pos['x']=input_pos['len']*np.cos(input_pos['theta'])
    output_pos['y']=input_pos['len']*np.sin(input_pos['theta'])
    return output_pos
    
def deg_to_rad(degree_vl):
    return np.array(degree_vl)*(2*np.pi)/360

def rad_to_deg(rad_vl):
    return (np.array(rad_vl)*360)/(2*np.pi)

def radians_to_quadrant(input_radian):
	mod_180_radian=np.mod(input_radian,np.pi)
	if mod_180_radian>np.pi/2:
		mod_180_radian=np.pi-mod_180_radian
	return mod_180_radian

def entropy(normhst):
    entropyvl=0
    for crbinvl in normhst:
        entropyvl=entropyvl-(crbinvl*np.log(crbinvl))
    return entropyvl

    #assumes list of lists or list of arrays as input

#assumes input in radians
def force_angle_to_range(input_angle,**kwargs):
    try:
        force_range=kwargs['force_range']
    except:
        force_range='0_pi'
    if force_range is '0_pi':
        mod_angle=input_angle
        posinds=np.where(input_angle>0)[0]
        mod_angle[posinds]=np.mod(input_angle,np.pi)
        neginds=np.where(mod_angle<0)[0]
        mod_angle[neginds]=np.mod(input_angle,-np.pi)
        mod_angle=abs(mod_angle)
        

    return mod_angle

def get_2darray(input_dt):
    numrow=len(input_dt)
    numcol=len(input_dt[0])
    
    output_array=np.zeros((numrow,numcol))
    for ind,cr_row in enumerate(input_dt):
        output_array[ind]=cr_row
    return output_array
        

    
def make_hist_calculations(crdeg,degbins):
        out_dt={}
        comb_raddata=deg_to_rad(crdeg)
        histout=np.histogram(crdeg[~np.isnan(crdeg)],degbins)
        out_dt['normhst']=histout[0]/float(sum(histout[0]))
        rad_data=comb_raddata[~pylab.isnan(comb_raddata)]
        doubled_rad_data=np.mod(2*rad_data,2*np.pi)
        
        tmpmean=circ.circmean(doubled_rad_data)/2
        
        out_dt['circvar']=(circ.circvar(2*rad_data))
        out_dt['circvar_mod360']=circ.circvar(rad_data)
       
        tmp360mean=circ.circmean(rad_data)
        
        if tmpmean<0:
            out_dt['mnrad']=np.pi+tmpmean
        else:
            out_dt['mnrad']=tmpmean
        if tmp360mean<0:
            out_dt['mnrad_360']=tmp360mean+2*np.pi
        else:
            out_dt['mnrad_360']=tmp360mean
            
        if np.abs(out_dt['mnrad']-out_dt['mnrad_360'])>(np.pi/2):
            out_dt['mnrad_max180']=out_dt['mnrad']+np.pi
        else:
            out_dt['mnrad_max180']=out_dt['mnrad']
        if out_dt['mnrad_360']<0:
            pdb.set_trace()
        
        return out_dt
#refactor by making above function call this one
def circ_mean(vls,**kwargs):
    comb_raddata=vls
    if 'anal_180' in kwargs:
        if kwargs['anal_180']:
            anal_180_flag=True
        else:
            anal_180_flag=False
    else:
        anal_180_flag=False
    if anal_180_flag:    
        rad_data=comb_raddata[~pylab.isnan(comb_raddata)]
        doubled_rad_data=np.mod(2*rad_data,2*np.pi)
        tmpmean=circ.circmean(doubled_rad_data)/2
        if tmpmean<0:
            mnout=np.pi+tmpmean
        else:
            mnout=tmpmean
        tmpvar=circ.circvar(2*rad_data)
    
    else:
        rad_data=comb_raddata[~pylab.isnan(comb_raddata)]
        mnout=circ.circmean(rad_data)
        tmpvar=circ.circvar(rad_data)

    return rad_to_deg(mnout),tmpvar






#lost original weighted mean on Thursday 4/6...recover version prior to that.
def weighted_mean(mnvls,edges,**kwargs):
    
    try:
        mean_type=kwargs['mn_type']
    except:
        mean_type='circ'

    try:
        anal_180_flag=kwargs['anal_180']
    except:
        anal_180_flag=False
    pdb.set_trace()
    bin_middles = (edges[:-1] + edges[1:]) / 2

    if len(bin_middles)<len(mnvls):
        bin_middles=mnvls

    veclist=[]
    
    try:
        for ind,crmnvl in enumerate(mnvls):
            try:
                veclist.append(bin_middles[ind]*crmnvl)
            except:
                pdb.set_trace()
    except:
        pdb.set_trace()
    
    if mean_type is 'circ':
        if kwargs['anal_180']:
            total_mean,total_var=circ_mean(np.array(veclist),anal_180=1)
        else:
            total_mean,total_var=circ_mean(np.array(veclist),anal_180=0)
        #needs to be fixed
        #pdb.set_trace()
        return total_mean,total_var
        pdb.set_trace()
    else:
        
        return sum(veclist)
        
    

#takes a list of 1xn vectors,
#returns the mean of those values as a 1*n and the standard deviation 
#excludes nan
def mnstd(input_list):
    dat=np.array(input_list)
    mdat = np.ma.masked_array(dat,np.isnan(dat))
    return np.mean(mdat,axis=0), np.std(mdat,axis=0)

def cos_fit(input_xvls,input_yvls):
    from scipy.optimize import curve_fit
    try:
        fitpars, covmat = curve_fit(cos_twofunc,input_xvls,input_yvls)
    except:
        pdb.set_trace()
    return fitpars,covmat

def asy_fit(xvls,yvls,asy_vl):
    from scipy import optimize
    fitfunc = lambda p, x:p[0]+((2*(asy_vl-p[0]))/(1+np.exp(p[1]*x))-(asy_vl-p[0]))

    
    errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function    
    p0 = [0.4,-0.07] # Initial guess for the parameters
    p1, success = optimize.leastsq(errfunc, p0[:], args=(xvls, yvls))
    
    
            
       
    return p1
    
    
    
    
    
    try:
        fitpars, covmat = curve_fit(asy_func,input_xvls,input_yvls)
    except:
        pdb.set_trace()
    return fitpars,covmat    
    
    
def smooth_minimum_fit(xvls,yvls,**kwargs):
    neg_xvls=-xvls
    neg_yvls=yvls
    pdb.set_trace()
    xinds=np.argsort(xvls)
    sorted_xvls=xvls[xinds]
    neg_sorted_
    sorted_yvls=yvls[xinds]
    smoothed_vls=smooth(sorted_yvls,16)
    pdb.set_trace()
    
    
    return minvl

def smooth(x,beta):
        """ kaiser window smoothing """
        window_len=11
        # extending the data at beginning and at the end
        # to apply the window at the borders
        s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        pdb.set_trace()
        w = np.kaiser(window_len,beta)
        y = np.convolve(w/w.sum(),s,mode='valid')
        return y[5:len(y)-5]    
    
    
def cos_ls_fit(xvls,yvls,**kwargs): 
    from scipy import optimize
    if 'num_cycles' in kwargs:
        num_cycles=kwargs['num_cycles']
    else:
        num_cycles=0.5
    fitfunc = lambda p, x: p[0]*np.cos(num_cycles*(x-p[1]))  # Target function
    
    errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function    
    p0 = [80.0, 0.] # Initial guess for the parameters
    p1, success = optimize.leastsq(errfunc, p0[:], args=(xvls, yvls))
    []
    if 'plot_flag' in kwargs:
        if kwargs['plot_flag']:
            pylab.figure()
            pylab.plot(xvls,yvls)
            pylab.plot(xvls,p1[0]*np.cos(num_cycles*(xvls-p1[1])),'r')
            
       
    return p1

def vonmises_fit_new(xdata,ydata):
    # define a von-Mises fitting function: B + A exp(k cos(2(theta - phi)) -1)
    # p[0] = B
    # p[1] = A
    # p[2] = k (1/D)
    # p[3] = phi
    
    #fitfunc = lambda p, x: p[0] * np.exp( p[2]*( np.cos((x - p[1]))) )
    errfunc = lambda p, x, y: fitfunc(p,x)-y
    
    # initial guess
    p0 = [0.1,0.7,5]
    # fit
    p1, success = scipy.optimize.leastsq(errfunc, p0[:],args=(xdata,ydata))
    
    #tw = (90./np.pi) * np.arccos(abs(1 + np.log((1.+np.exp(-2.*p1[2]))/2)/p1[2]) )
    
    return p1, fitfunc(p1,xdata),success

def fitfunc(p,x):
    if p[2]>0.5:
        return p[0] * np.exp( p[2]*( np.cos((x - p[1]))) )
    else:
        return 1e10



def vonmises_fit(xvls,yvls,**kwargs): 
    from scipy import optimize
    from scipy.stats import vonmises
    
    #if 'num_cycles' in kwargs:
     #   num_cycles=kwargs['num_cycles']
    #else:
     #   num_cycles=0.5
    #fitfunc = lambda p, x: p[0]*vonmises.pdf(x-p[1],8)  # Target function
    
    #errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function    
    #p0 = [10.0, 0.] # Initial guess for the parameters
    
    popt,pcov = optimize.curve_fit(vonmises_fitfunc, xvls,yvls)
    
    #if 'plot_flag' in kwargs:
     #   if kwargs['plot_flag']:
      #      pylab.figure()
       #     pylab.plot(xvls,yvls)
        #    pylab.plot(xvls,p1[0]*vonmises.pdf(xvls-p1[1],8),'r')
            
       
    return popt
def errfunc_pos(p,x,y):
    
    return vonmises_fitfunc(p,x)-y
    
def vonmises_fitfunc(x,a,b):
    from scipy.stats import vonmises
    if a>1:
        return a*vonmises.pdf(x-b,8)
    else:
        return 1e10

def rectified_cos_ls_fit(xvls,yvls,**kwargs): 
    from scipy import optimize
    if 'num_cycles' in kwargs:
        num_cycles=kwargs['num_cycles']
    else:
        num_cycles=0.5
    #fitfunc = lambda p, x: p[0]*np.cos(num_cycles*(x-p[1]))  # Target function
    
    #errfunc = lambda p, x, y: rectified_fit_func(p, x) - y # Distance to the target function    
    #p0 = [20, 0.] # Initial guess for the parameters
    #p1, success = optimize.leastsq(errfunc, p0[:], args=(xvls, yvls))
    #pdb.set_trace()
    popt,pcov=optimize.curve_fit(rectified_fit_func,xvls,yvls)
    
    if 'plot_flag' in kwargs:
        if kwargs['plot_flag']:
            pylab.figure()
            pylab.plot(xvls,yvls)
            pylab.plot(xvls,rectified_fit_func(xvls,popt[0],popt[1]),'r')
            
       
    return popt[1]

def rectified_fit_func(x,a,b):
    num_cycles=1
    #zeroinds=np.where(a*np.cos(num_cycles*(x-b)) <0)
    try:
        outvl=a*np.cos(num_cycles*(x-b))
    except:
        pdb.set_trace()
    try:
        zeroinds=np.where(outvl<0)
        returnvl=outvl
        returnvl[zeroinds]=0
    except:
        pdb.set_trace()
        returnvl=outvl 
    #outvl[zeroinds[0]]=0
    #pdb.set_trace()
    return returnvl
def linear_ls_fit(xvls,yvls):
    from scipy import optimize
    fitfunc = lambda p, x: p[0]*x  # Target function
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    try:
        p0=np.max(yvls)
        p1, success = optimize.leastsq(errfunc, p0, args=(xvls, yvls))
    
        return p1
    except:
        p1=[]
        return p1

def cos_twofunc(x,a,b):
    #t=abs(a)
    return a*np.cos(2*(x-b))

#vls=[yr,mo,dy]
def date_to_num(vls):
    #pdb.set_trace()
    return pylab.date2num(datetime.date(vls[0],vls[1],vls[2]))

#returns total elapsed time in day in minutes
def get_time(timestr,**kwargs):
    #pdb.set_trace()
    if 'add_time' in kwargs:
        add_time=kwargs['add_time']
    else:
        add_time=0
    #pdb.set_trace()
    try:
        return 60*int(timestr[0:2])+int(timestr[2:4])+add_time
    except:
        pdb.set_trace()
def convert_str_time_to_min(timestr):
    
    return 60*int(timestr[0:2])+int(timestr[2:4])

#returns two columns with start and stop x values of continuous values
def find_consecutive_values(in_array):
    start_vls=[]
    end_vls=[]
    arrays=np.array_split(in_array,np.where(np.diff(in_array)!=1)[0]+1)
    try:
        for cr_array in arrays:
            start_vls.append(cr_array[0])
            end_vls.append(cr_array[-1])
    except:
        pdb.set_trace()
    return start_vls,end_vls    

def standardize_angle(input_angle,mod_angle,**kwargs):
    if 'test_flag' in kwargs:
        tst=1
    if isinstance(input_angle,np.ndarray):
        array_flag=True
    else:
        array_flag=False
    if array_flag:
        output_angle=np.zeros(len(input_angle))
        posinds=np.where(input_angle>=0)
        neginds=np.where(input_angle<0)
        output_angle[posinds[0]]=np.mod(input_angle[posinds[0]],mod_angle)
        neg=np.mod(input_angle[neginds[0]],-mod_angle)
        output_angle[neginds[0]]=mod_angle+neg

        if 'force_positive' in kwargs:
            output_neg_inds=np.where(output_angle<0)
            output_angle[output_neg_inds[0]]=output_angle[output_neg_inds[0]]+180.0

    else:
        
        if input_angle>=0:
            output_angle=np.mod(input_angle,mod_angle)
        elif input_angle<0:
            neg=np.mod(input_angle,-mod_angle)
            output_angle=mod_angle+neg
        if 'force_positive' in kwargs:
            if output_angle<0:
                output_angle=output_angle+180.0
    
    return output_angle
    
def calculate_offset(mot,light,sensor_offset):
           
    xvls=2*np.pi*np.mod(mot,stepsPerRotation)/stepsPerRotation
    yvls=self.light_level-np.mean(light)
    srtind=np.argsort(xvls)
    fitpars = cos_ls_fit(xvls[srtind],yvls[srtind])
        
    pdb.set_trace()    
    if fitpars[0]>0:
        minimum_pol_vl=np.pi/2+fitpars[1]
        
    else:
        minimum_pol_vl=np.pi+fitpars[1]
    correct_polarizer_minimum_in_radians=np.pi/2-sensor_offset_radians_CW
    offset_in_radians=correct_polarizer_minimum_in_radians-minimum_pol_vl
    self.indt['adjusted_raw_motor_position']=self.indt['raw_motor_position']+(offset_in_radians/(2*np.pi))*self.calculated_motor_period
    self.indt['motor_position_radians']=2*np.pi*((np.mod(self.indt['adjusted_raw_motor_position'],self.calculated_motor_period))/self.calculated_motor_period)



#y is data, Fs is sampling frequency
def calc_spectrum(y,Fs):

    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = scipy.fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]
    return frq,abs(Y)
   

def find_period(new_xvls,xvls_in,yvls_in):
    import peakdetect as peakdetect

    sorted_indices=np.argsort(xvls_in)
    #pdb.set_trace()
    interp_yvls=np.interp(new_xvls,np.array(xvls_in)[sorted_indices],np.array(yvls_in)[sorted_indices])
    norm_yvls=interp_yvls-np.mean(interp_yvls)
    #pdb.set_trace()
    output_vls=np.correlate(norm_yvls,norm_yvls,'same')
    peak_data=peakdetect.peakdetect(output_vls)
    numpeaks=np.shape(peak_data[0])[0]
    pdb.set_trace()
    half_distance=(peak_data[0][-1][0]-peak_data[0][0][0])/float(numpeaks-1)
    period=2.0*half_distance
    return output_vls, period,peak_data
def realign_radial_data(rad_data,realign_vl):
    
    offset_data=rad_data-realign_vl
    #this is a hack to keep values positive
    offset_data=offset_data+100*np.pi
    realigned_data=np.mod(offset_data,2*np.pi)
    return realigned_data

def make_polarizer_fit_calculation(params,xvls,yvls):
    if 'filter_type' not in params:
        params['filter_type']='linear'
    if 'filter_type' in params:
        
        if params['filter_type']=='linear':
            try:
                fit0,fit1=cos_ls_fit(xvls,yvls,num_cycles=2,plot_flag=1)
            except:
                fit0=0
                fit1=0
            offset_value_if_negative=np.pi/2
            polarizer_correction_value=np.pi/2
        elif self.params['filter_type']=='circular':
            fit0,fit1=calc.cos_ls_fit(xvls,yvls,num_cycles=1)
            offset_value_if_negative=np.pi
            polarizer_correction_value=0
        else:
            fit0,fit1=calc.cos_ls_fit(xvls,yvls,num_cycles=2,plot_flag=1)
            offset_value_if_negative=np.pi/2
            polarizer_correction_value=np.pi/2
        if fit0<0:
            horiz_offset=fit1+offset_value_if_negative
        else:
            horiz_offset=fit1
            
        if 'intensity_cue' in params.keys():
            if params['intensity_cue']:
                new_horiz_offset=polarizer_correction_value+self.adjust_motor_for_intensity_cues(xvls,yvls,horiz_offset)
            else:
                new_horiz_offset=polarizer_correction_value+horiz_offset
        else:
            new_horiz_offset=polarizer_correction_value+horiz_offset
    
    return new_horiz_offset   
