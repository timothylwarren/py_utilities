
#!/usr/bin/python
import pdb
import pylab
import numpy as np
from matplotlib.lines import Line2D  
import tw_plot_library3 as plt
import fp_library as fpl
import tw_calc_library as calc
import tw_filehandling as fh
#from matplotlib.patches import Ellipse

import matplotlib.cm as cm
AXISPAD=2

#pylab.ion()

def make_example_figure(crdt, axmotor,axpol_hist,**kwargs):
    try:
        displacement_ax=kwargs['displacement_ax']
        displacement_flag=True
    except:
        displacement_flag=False
    try:
        text_ax=kwargs['text_ax']
        text_flag=True
    except:
        text_flag=False

           
             
    plot_motor(crdt,axmotor,plot_vector=False,plot_split=1,plot_start_angle=0,xlim=[0,5.5],plot_mean=crdt['mnrad_360'])        
    plot_mot_hist(crdt,axpol_hist)

    if displacement_flag:
        plot_displacement(displacement_ax, crdt,add_net_displacement=True)
    if text_flag:
        add_text_name(text_ax,crdt)

def line_vec_strength(indt,ax):
        #if 'type' in kwargs:
         #   inds=np.intersect1d(np.where(all_time_list>self.crtimeinds[0])[0],np.where(all_time_list<self.crtimeinds[1])[0])
        #else:
        
    all_time_list,all_vec_list=pad_vector_lists(indt)
    inds=np.arange(0,len(all_time_list))
        
    plt_veclst=all_vec_list[inds]
    plt_timelst=all_time_list[inds]
        
    ax.plot(plt_timelst,plt_veclst)
    fpl.adjust_spines(ax,['left', 'bottom'])
    ax.set_ylim([0,1])
    ax.set_xlim([0,15])


def add_text_name(ax,crdt):
    fname=crdt['fname']
    cutname=fname.split('/cloop')
        #tst=['', 'home', 'timothy', 'data', '20170505', 'fly1-rnach']
    tst=cutname[0].split('/')
        #printstr='20170505/fly1-rnach'
    printstr=tst[-2]+'/'+tst[-1]
    ax.text(0,0,printstr,fontsize=5)
    ax.axis('off')


def plot_traj_by_vec_strength(ax,mn_drxn,vec_strength,**kwargs):

    
    ax.plot(mn_drxn,vec_strength)
    ax.get_yaxis().set_ticks([])
    ax.title.set_visible(False)
    ax.get_xaxis().set_ticklabels([])
    ax.spines['polar'].set_color('none')

def plot_displacement(ax, indt,**kwargs):
    try:
        add_net_flag=kwargs['add_net_displacement']
    except:
        add_net_flag=False

    plot_positions={}
    for key in ['x','y']:
            plot_positions[key]=[]

    

    for key in ['x','y']:
            
            
        plot_positions[key].append(indt['displacement_traj']['raw'][key])
        


                            
           

    polar_positions=calc.linear_to_polar(plot_positions)
    col='b' 
        
    indvls=np.arange(len(polar_positions['theta']))

        
            

    for ind in indvls:
        ax.plot(polar_positions['theta'][ind]+np.pi/2,polar_positions['len'][ind],color=col)
        ax.plot(polar_positions['theta'][ind][-1]+np.pi/2,polar_positions['len'][ind][-1],'o',color='k')

        ax.get_xaxis().set_ticks([0,np.pi/2.,np.pi,3.*(np.pi/2.)])
        ax.get_xaxis().set_ticklabels(['0','90','180','270'],fontsize=8)
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([],fontsize=8)   
        #pdb.set_trace()
        
        ax.set_ylim([0,10000])

        if add_net_flag:
            ax.plot([indt['mnrad_360']+np.pi/2,indt['mnrad_360']+np.pi/2],[0,polar_positions['len'][ind][-1]],'k--')



def pad_vector_lists(indt):
       
        #vectimevls=self.crdt['vec_time_lst']-self.crdt['time_in_min'][0]    
    time_space=indt['vec_time_lst'][1]-indt['vec_time_lst'][0]
        
    init_time_vls=np.arange(indt['time_in_min'][0],np.min(indt['vec_time_lst']),time_space)
    end_time_vls=np.arange(np.max(indt['vec_time_lst']),indt['time_in_min'][-1],time_space)
        
    timelst=np.concatenate([init_time_vls,np.array(indt['vec_time_lst']),end_time_vls])-indt['time_in_min'][0]
        #pdb.set_trace()
    veclst=np.concatenate([np.zeros(len(init_time_vls)),np.array(indt['len_vector_lst']),np.zeros(len(end_time_vls))])
    return timelst,veclst


def plot_motor(indt,ax,**kwargs):
    try: 
        save_dt_flag=kwargs['save_dt_flag']
    except:
        save_dt_flag=False
    VERTVL=370
    if 'zoom_times' in kwargs:
        type='zoom'
        
        mot_inds=self.calc_zoom_inds(kwargs['zoom_times'])
    else:
       
        mot_inds=np.arange(0,len(indt['time_in_min']))
        


        type='nozoom'
    plot_split_flag=0
    
    try:
        mnvl=kwargs['plot_mean']
    except:
        mnvl=[]


    try:
        flag_360=kwargs['flag_360']
    except:
        flag_360=0
    try:
        offset_flag=kwargs['offset_value']
    except:
        offset_flag=False

    if 'plot_vertical' in kwargs:
        plot_vert_flag=1
    else:
        plot_vert_flag=0
    
    if 'xticks' in kwargs:
        xtickflag=1
    else:
        xtickflag=0
    
    if 'plot_split' in kwargs:
        if kwargs['plot_split']:
            plot_split_flag=1
    
    if 'boundary_times' in kwargs:
        time_flag=1
    else:
        time_flag=0
    if 'xlim' in kwargs:
        xlim=kwargs['xlim']
    else:
        xlim=[0,15]
    
    if 'plot_vector' in kwargs:
        if kwargs['plot_vector']:
            if 'vector_threshold' in kwargs:
                vec_threshold=kwargs['vector_threshold']
            else:
                vec_threshold=self.vec_threshold

            try:
                if len(self.crdt['vec_time_lst'])>1:
                    if kwargs['plot_vector']:
                        all_time_list,all_vec_list=self.pad_vector_lists()
                        if 'type' in kwargs:
                            inds=np.intersect1d(np.where(all_time_list>self.crtimeinds[0])[0],np.where(all_time_list<self.crtimeinds[1])[0])
                        else:
                            inds=np.arange(0,len(all_time_list))
                        
                        inds_thresh=np.where(all_vec_list[inds]>vec_threshold)
                        #2 columns with x values to plot.
                        if len(inds[inds_thresh]):
                            [startinds,stopinds]=calc.find_consecutive_values(inds[inds_thresh])
                            plt.plot_horizontal_lines(ax,all_time_list[startinds],all_time_list[stopinds],VERTVL)
            except:
                tst=1
            
    if 'plot_left_axis' in kwargs:
        plot_left_axis=kwargs['plot_left_axis']
    else:
        plot_left_axis=1
        
    
    mot_tmp=indt['mot_deg'][mot_inds]
    time=indt['time_in_min'][mot_inds]-indt['time_in_min'][0]

    
    
    
    
    if offset_flag:
        mot_rad=calc.deg_to_rad(mot_tmp)-kwargs['offset_value']
        mot=calc.rad_to_deg(calc.standardize_angle(mot_rad,2*np.pi,force_positive=1))
    else:
        mot=mot_tmp
    
    sub_plot_motor(ax,time,mot,save_dt_flag=save_dt_flag)
    
   
    if mnvl:
       
        deg_mn=calc.rad_to_deg(mnvl)
        ax.plot([0,15],[deg_mn, deg_mn],linestyle='--', dashes=(2, 1),color='r')
        ax.plot([0,15],[deg_mn+180, deg_mn+180],linestyle='--', dashes=(2, 1),color='r')
            #ax.plot([0, 1], [0, 1], linestyle='--', dashes=(5, 1)) #length of 5, space of 1


    if 'plot_start_angle' in kwargs:
        if kwargs['plot_start_angle']:
            
            #ax.plot(time[0]+0.1,calc.standardize_angle(self.params['adjusted_start_angle']),'c>')
         
            ax.plot(time[0]-0.5,calc.standardize_angle(calc.rad_to_deg(self.params['calculated_start_angle']),180.0),'c>',markersize=7)
    if 'marker_time' in kwargs:
        #need to find the value of the motor at that time_duration
        #first need to find the closest index
        
        crind=np.argmin(np.abs(time-kwargs['marker_time']))
        ax.plot(time[crind],mot[crind],'co')
    
    
    try:
       
        if plot_left_axis:
            fpl.adjust_spines(ax,['left','bottom'])
        
        else:
        
            fpl.adjust_spines(ax,['bottom'])
        
        #elif (plot_left_axis) and (self.last_row_flag==0):
         #   ax.axis('off')
            #fpl.adjust_spines(ax,['left'])
    except:
        fpl.adjust_spines(ax,['left','bottom'])
    if plot_left_axis:
        ax.get_yaxis().set_ticks([0,90,180,270,360])
        ax.get_yaxis().set_ticklabels(['0$^\circ$','90$^\circ$','180$^\circ$','270$^\circ$','360$^\circ$'],fontsize=6)
        ax.set_ylabel('polarizer', fontsize=6)
    
    if xtickflag:
        xticks=kwargs['xticks']
        xticklabels=kwargs['xticklabels']
    else:
        xticks=np.linspace(0,25,6)
        xticklabels=['0','5','10','15','20','25']
       
    if plot_vert_flag:
        ax.plot([kwargs['plot_vertical'],kwargs['plot_vertical']],[-20,380],'r')
    try:
        if self.last_row_flag:
        #ax.set_ylabel('polarizer heading', fontsize=9)
            ax.get_xaxis().set_ticks(xticks)
            ax.get_xaxis().set_ticklabels(xticklabels,fontsize=6)
            ax.set_xlabel('minutes', fontsize=6)
    except:
        ax.get_xaxis().set_ticks(xticks)
        ax.get_xaxis().set_ticklabels(xticklabels,fontsize=6)
        ax.set_xlabel('minutes', fontsize=6)
    #pdb.set_trace()
    ax.set_xlim(xlim)
    #ax.set_aspect(0.005)
    ax.xaxis.labelpad = AXISPAD
    ax.yaxis.labelpad= AXISPAD
    ax.set_ylim([-20,380])

##
#This function plots position values in a manner that removes artefactual lines from data wrapping around
#inputs are 
#ax, handle to axis
#time- list of timevalues
#mot - list of degrees between 0 and 360
def sub_plot_motor(ax,time,mot,**kwargs):
    try:
        save_dt_flag=kwargs['save_dt_flag']
    except:
        save_dt_flag=False
    SAVE_EX_DT=True
    MAXDEGPLOT=359
    MINDEGPLOT=1
    plotinds1=np.where(mot<MAXDEGPLOT)
    plotinds2=np.where(mot>MINDEGPLOT)
    
    allinds=np.intersect1d(np.array(plotinds1[0]),np.array(plotinds2[0]))
    splitinds=np.array_split(allinds,np.array(np.where(np.diff(allinds)!=1))[0]+1) 
    #added for instructive purposes, generally will be False, so ignore.
    if save_dt_flag:   
        ex_dt={}
        #ex_dt['ax']=ax
        ex_dt['time']=time
        ex_dt['mot']=mot
        ex_dt['splitinds']=splitinds
        pdb.set_trace()
        fh.save_to_file('/users/tim/motor_dt.pck', ex_dt)

    for crsplitinds in splitinds:
        if np.size(crsplitinds):
            try:
                ax.plot(time[crsplitinds],mot[crsplitinds],'b')
            except:
                pdb.set_trace()



def plot_mot_hist(indt,crax,**kwargs):
    
    crax.step(indt['normhst'],indt['xrad'][0:-1]+indt['rad_per_bin']/2,'k',linewidth=1)
    fpl.adjust_spines(crax,[])
    crax.set_ylim([-calc.deg_to_rad(20.),calc.deg_to_rad(380.0)])
    crax.plot(0.22,indt['mnrad_360'],'r<')
    crax.set_xlim([0,0.24])

def make_heat_map(heatdt,**kwargs):
    POWER_VALUE=5
   
    ax=kwargs['ax']
    plt_type=kwargs['plt_type']
    try:
        transect_ax=kwargs['transect_ax']
        plot_transect_flag=True
        try:
            ax_schematic=kwargs['ax_schematic']
        except:
            ax_schematic=[]
    except:
        plot_transect_flag=False

    try:
        sub_flag=kwargs['sub_heat_map_flag']
    except:
        sub_flag=False

    try:
        paired_flag=kwargs['paired_flag']
    except:
        paired_flag=False


    try:
        aligned_flag=kwargs['aligned']
    except:
        aligned_flag=False
    try:
        colorbar_ax=kwargs['colorbar_ax']
        fig_flag=kwargs['fig_flag']
        plot_colorbar_flag=True
    except:
        plot_colorbar_flag=False


    try:
        renorm_flag=kwargs['renorm']
    except:    
        renorm_flag=False
    if sub_flag:
        
        cr_heatmap_data=heatdt['sub_heat_map'][plt_type]
        if renorm_flag:
            
            cr_heatmap_data['norm_heat_map_vls']=cr_heatmap_data['norm_heat_map_vls']/sum(sum(cr_heatmap_data['norm_heat_map_vls']))

        if plot_colorbar_flag:
            plt.polar_heat_map(ax,cr_heatmap_data,shift_vertical_flag=True,aligned=aligned_flag,sub_flag=sub_flag,colorbar_ax=colorbar_ax,fig_flag=kwargs['fig_flag'])
        else:
            plt.polar_heat_map(ax,cr_heatmap_data,shift_vertical_flag=True,aligned=aligned_flag,sub_flag=sub_flag)

    elif paired_flag:
        
        heatmap_list=heatdt['full_heat_map'][plt_type]
        if not 'redges' in heatmap_list.keys():
            heatmap_list['redges']=heatmap_list['r'][:,0]
            heatmap_list['thetaedges']=heatmap_list['theta'][0,:]
            
        cr_heatmap_data=heatmap_list
        if plot_colorbar_flag:
            plt.polar_heat_map(ax,heatmap_list,shift_vertical_flag=True,aligned=aligned_flag,sub_flag=sub_flag,paired_flag=True,sep_max_flag=True,colorbar_ax=colorbar_ax,fig_flag=kwargs['fig_flag'])
        else:
            plt.polar_heat_map(ax,heatmap_list,shift_vertical_flag=True,aligned=aligned_flag,sub_flag=sub_flag,paired_flag=True,sep_max_flag=True)

    else:
       
        cr_heatmap_data=heatdt['full_heat_map'][plt_type]
        
        if renorm_flag:
            cr_heatmap_data['norm_heat_map_vls']=cr_heatmap_data['norm_heat_map_vls']/sum(sum(cr_heatmap_data['norm_heat_map_vls']))

        if plot_colorbar_flag:
            plt.polar_heat_map(ax,cr_heatmap_data,shift_vertical_flag=True,aligned=aligned_flag,sub_flag=sub_flag,colorbar_ax=colorbar_ax,fig_flag=kwargs['fig_flag'])
        else:
            plt.polar_heat_map(ax,cr_heatmap_data,shift_vertical_flag=True,aligned=aligned_flag,sub_flag=sub_flag)
    if plot_transect_flag:
        base_bnds=np.array([-np.pi/9, np.pi/9])
        bnd_sectors=[base_bnds, base_bnds+np.pi/2, base_bnds+2*np.pi/2, base_bnds+3*np.pi/2]
        plt.plot_transects(transect_ax,cr_heatmap_data,aligned=True,ax_schematic=ax_schematic,bnds=bnd_sectors,paired_flag=paired_flag)



def arbitary_transect_from_heat_map(ax,heatdt,**kwargs):
    try:
        vecmin=kwargs['vecmin']
    except:
        vecmin=0
    if vecmin:
        num_inds_to_use=len(np.where(heatdt['thetaedges']>0.9)[0])
    sub_array=sub_array=heatdt['norm_heat_map_vls'][:,-2:]
    sumvls=np.sum(sub_array,axis=1)
    norm_sumvls=sumvls/np.sum(sumvls)
    
    ax.step(heatdt['redges'][:-1],norm_sumvls,color='k',drawstyle='steps-post')
    


def plot_wings(indt,ax,**kwargs):
        colvls=['r','c']
        if 'xlim' in kwargs:
            xlim=kwargs['xlim']
        else:
            xlim=[0,25]
        
        if 'type' in kwargs:
            inds=self.cr_zoominds
            type='zoom'
        else:
            inds=np.arange(0,len(indt['time_in_min']))
            type='nozoom'
        plot_time=indt['time_in_min'][inds]-indt['time_in_min'][0]
        lftwng=indt['lftwng'][inds]
        rtwng=indt['rtwng'][inds]
        
        ax.plot(plot_time,np.array(lftwng),colvls[0],linewidth=0.7)
        ax.plot(plot_time,np.array(rtwng),colvls[1],linewidth=0.7)
        ax.get_yaxis().set_ticks([35,50,65,80])
        fpl.adjust_spines(ax,['left'])
        ax.get_yaxis().set_ticklabels(['35','50','65','80'],fontsize=9)
        ax.set_ylabel('wing angle', fontsize=9)
        ax.set_ylim([25,80])
        ax.text(0,70,'left wing',fontsize=9,color=colvls[0])
        ax.text(5,70,'right wing',fontsize=9,color=colvls[1])
        if type=='zoom':
            ax.set_xlim([plot_time[0],plot_time[-1]])
        else:
            ax.set_xlim(xlim)    