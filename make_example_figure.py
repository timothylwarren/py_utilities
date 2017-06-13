
#this script just makes an individual example figure
#assumes that dt has been opened elsewhere
import numpy as np 
import os
import pylab 
import pdb

import  fly_plot_basics as fpb



def make_example_figure(self, crdt, axmotor,axpol_hist,**kwargs):
	try:
		displacement_ax=kwargs['displacement_ax']
		displacement_flag=True
	except:
		displacement_flag=False
       
           
                
    fpb.plot_motor(crdt,axmotor,plot_vector=False,plot_split=1,plot_start_angle=0,xlim=[0,5.5],plot_mean=mnvl_in_rad)
                
    fpb.plot_mot_hist(indt,axpol_hist)

    if displacement_flag:
    	fpb.plot_displacement(self,ax, indt,**kwargs):

               