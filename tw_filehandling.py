#!/usr/bin/python
#this is twplot_library.py

import pickle
#import pdb
import pylab
import datetime
import numpy as np
import os
import shutil
import time
import sys
#from py_utilities import tw_calc_library as calc

#requires complete name
def print_vert(indict):
    for x in indict:
        print(x)

def parse_date(fname):
    split_fn=fname.split('/')
    date=split_fn[-4]
    fly=split_fn[-3]
    time=split_fn[-1].split('_')[-1][0:4]
    return date, fly, time

def check_file_size(infile):
    
    return os.stat(infile)[6]/1e6
def make_reduced_txtfile(fname,target_sample_rate=200):
    #preserve nan values
   
    original_dt=np.array(np.genfromtxt(fname))
    numcols=len(original_dt[0,:])
    timecol=original_dt[:,0]
    min_time=np.nanmin(timecol)
    max_time=np.nanmax(timecol)
    out_time=np.arange(min_time,max_time,1./target_sample_rate)
    
    new_time=np.interp(out_time,timecol,timecol)
    #where were nanvls in original data
    naninds=np.where(np.isnan(original_dt[:,0]))
    nantime=[]
    
    for crind in naninds[0]:
        nantime.append(original_dt[crind-1,0]) 

    for crnantime in nantime:
        tmdiff=new_time-crnantime
        closest_ind=np.nanargmin(abs(tmdiff))
        current_nan_inds=np.where(np.isnan(new_time))
        if len(current_nan_inds[0]):
            if np.min(abs(current_nan_inds-closest_ind))>10:
                new_time[closest_ind]=np.nan
        else:
            new_time[closest_ind]=np.nan
    

    #initialize_array
    out_array=np.zeros((len(new_time),numcols))
    out_array[:,0]=new_time
    for crcol in np.arange(1,numcols):
        new_col=np.interp(out_time,timecol,original_dt[:,crcol])
        out_array[:,crcol]=new_col
    redfilename=fname[:-3]+'red.txt'
    np.savetxt(redfilename,out_array)
    
    return redfilename
def change_pickle(pickle_fname,param_name,new_param_value):
    params=open_pickle(pickle_fname)
    
    params[param_name]=new_param_value
    

    save_to_pickle(pickle_fname,params)


def open_pickle(pickle_fname, **kwargs):
    try:
        encoding=kwargs['encoding']
        encoding_flag=True
    except:
        encoding_flag=False
    print('pickle_name is %s'%pickle_fname)
    
    try:
       
        pickle_handle = open(pickle_fname,"rb")
        if not encoding_flag:
            params=pickle.load(pickle_handle)
        else:
            params=pickle.load(pickle_handle,encoding=encoding)
        pickle_handle.close()
    except:
        print("no pickle file")
        return
    return params

def save_to_pickle(pickle_fname,data):

    paramfile = open(pickle_fname,"wb")
    pickle.dump(data,paramfile,protocol=2)
    paramfile.close()    
#crday is a datetime object, returns string
def make_data_path(input_data_path,crday):
    
    strday=str(crday)
    strday=strday.replace('-','').split(' ')[0]
    datapath=input_data_path+strday+'/'
    
    return(datapath)    
    
#called by get_data_by_day
def get_directories(datapath):
    
    try:
        
        dirs= [name for name in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, name))]
        return(sorted(dirs))
    except:
        return 
        
def read_days(input_file):
    date_array=np.array([])
    reader=open(input_file,'r')
    contents=reader.readlines()
    date=contents[0]
    
    if(len(contents)>1):
        exp_type=contents[1].strip('\n')
    else:
        exp_type='closed'
    if(len(contents)>2):
        anal_type=contents[2].strip('\n')
    else:
        anal_type='pair'
    
    crdate=convert_days_to_array(date_array,date)
   
    reader.close()
    
    if type(crdate) is float:
        
        crdate=[crdate]
    return crdate,exp_type,anal_type

def get_day_from_fname(fname):
    #this line reduces
    #'/home/asl/Dropbox/flystuff/currentdata/20130411/fly4/cloop/fly4_20130411_140953.txt'
    # to '20130411'
    if 'currentdata' in fname:
        fname_parts=fname.split('currentdata')
    elif 'archived_data' in fname:
        fname_parts=fname.split('archived_data')
    tst=fname.split('/')[-1].split('_')[-2]
    
    return fname_parts, pylab.date2num(datetime.datetime(int(tst[0:4]),int(tst[4:6]),int(tst[6:8])))
   
    
def convert_days_to_array(date_array,date):
    
    vls=date.strip('\n').split(' ')
    for crvls in vls:
        added_dates=get_dates(crvls)
        
        if len(added_dates)>0:
            
            date_array=np.append(date_array,added_dates)
   
    #two sets of dates
    
    
    return date_array
def get_dates(invls):
#only one set of dates
    vls=invls.split('.')
    #one date
    if len(vls)==3:
        int_date=pylab.date2num(datetime.date(int(vls[0]),int(vls[1]),int(vls[2])))
        crdate=np.linspace(int_date,int_date,1)
    #set of days, denoted by hyphen, 2013.02.01-2013.02.05
    
    elif len(vls)==5:
        splitvls=vls[2].split('-')
        mindate=pylab.date2num(datetime.date(int(vls[0]), int(vls[1]), int(splitvls[0])))
        maxdate=pylab.date2num(datetime.date(int(splitvls[1]), int(vls[3]), int(vls[4])))
        crdate=np.linspace(mindate,maxdate,maxdate-mindate+1)
    else:
        crdate=[]
    return crdate

def save_to_file(pckfname,sumstats,**kwargs):
    endbit=pckfname.split('/')[-1]
    base_str=pckfname.strip(endbit)
    if 'save_executable' in kwargs:
        executable_file=kwargs['save_executable']
        executable_flag=1
    else:
        executable_flag=0
    
    if 'fig' in kwargs:
        fig=kwargs['fig']
        figname=kwargs['figname']
    if 'fig_traj' in kwargs:
        fig_traj=kwargs['fig_traj']
        fig_trajname=kwargs['fig_trajname']    
        pdf_str=base_str+fig_trajname+'.pdf'
        fig_traj.savefig(pdf_str)
    
    paramfile = open(pckfname,"wb")
    
    pickle.dump(sumstats,paramfile)
    if executable_flag:
        shutil.copyfile(executable_file,base_str+executable_file.split('/')[-1])
    
    try:
        eps_str=base_str+figname+'.eps'
        pdf_str=base_str+figname+'.pdf'
        if(not type(fig) is list or len(fig)<2):
            if type(fig) is list:
                #fig[0].savefig(eps_str)
                print ('saving png')
                fig[0].savefig(pdf_str)
            else:
                #fig.savefig(eps_str)
                fig.savefig(pdf_str)
        else:
            for crnum,crfig in enumerate(fig):
                crfig.savefig(pdf_str.strip('.png')+'_fig%d'%crnum+'.pdf')
    except:
        #pdb.set_trace()
        print ('no figure save')
   
        
    
    paramfile.close()   
