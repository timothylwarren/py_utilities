
#!/usr/bin/python
import pdb
import pylab
import numpy as np
from matplotlib.lines import Line2D  
import tw_plot_library3 as twplt
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
        
    ax.plot(plt_timelst,plt_veclst,linewidth=0.5)
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

def make_raw_plot(crdt,axmotor, axhist):

        #COLNUM=-1

        
    TIME_GAP=5            

    for cr_fltnum in crdt.keys():

        if crdt[cr_fltnum]:

            mnvl_in_rad=crdt[cr_fltnum]['mnrad_360']
            halt_flag=False
                    
            offset_time=0
            if cr_fltnum==1:
                offset_time=crdt[cr_fltnum-1]['time_in_min'][-1]
            elif cr_fltnum>1:
                offset_time=crdt[cr_fltnum-1]['time_in_min'][-1]-TIME_GAP
                    
            plot_motor(crdt[cr_fltnum],axmotor,plot_vector=False,plot_split=1,plot_start_angle=0,subtract_zero_time=True,offset_time=offset_time,plot_vert_line_at_end=True, halt_flag=halt_flag)
                    
            axmotor.set_xlim([0,15.5])

                    #if COLNUM:

                     #   axmotor[crkey][flyindnum][COLNUM].axis('off')
                      #  axhist[crkey][flyindnum][COLNUM].axis('off')

            try:
                crax=axhist[cr_fltnum]
            except:
                pdb.set_trace()
            
            crax.step(crdt[cr_fltnum]['normhst'],crdt[cr_fltnum]['xrad'][0:-1]+crdt[cr_fltnum]['rad_per_bin']/2,'k',linewidth=1)
                    #self.col_num[crkey]=self.col_num[crkey]+1
            fpl.adjust_spines(crax,[])
            crax.set_ylim([-calc.deg_to_rad(0.),calc.deg_to_rad(360.0)])
            crax.plot(0.21,mnvl_in_rad,'r<')
            crax.set_xlim([0,0.24])

   





def plot_motor(indt,ax,withhold_bottom_axis=False,one_line_label=False,xlabelpad=-3,**kwargs):
    
    VERTVL=370
    if 'zoom_times' in kwargs:
        type='zoom'
        
        mot_inds=self.calc_zoom_inds(kwargs['zoom_times'])
    else:
        
        mot_inds=np.arange(0,len(indt['time_in_min']))
        


        type='nozoom'
    plot_split_flag=0
    
    try:
        subtract_zero_time_flag=kwargs['subtract_zero_time']
    except:
        subtract_zero_time_flag=True
    try:
        plot_vert_line_at_end=kwargs['plot_vert_line_at_end']
    except:
        plot_vert_line_at_end=False

    try:
        mnvl=kwargs['plot_mean']
    except:
        mnvl=[]
    try:
        center_on_zero_flag=kwargs['center_on_zero_flag']
    except:
        center_on_zero_flag=False

    try:
        flag_360=kwargs['flag_360']
    except:
        flag_360=0
    try:
        offset_to_subtract=kwargs['offset_value_to_subtract']
    except:
        offset_to_subtract=0
    try:
        halt_flag=kwargs['halt_flag']
    except:
        halt_flag=False

    try:
        offset_time=kwargs['offset_time']
    except:
        offset_time=0
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
        xlim=[0,20]
    
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
                            twplt.plot_horizontal_lines(ax,all_time_list[startinds],all_time_list[stopinds],VERTVL)
            except:
                pdb.set_trace()
                tst=1
            
    if 'plot_left_axis' in kwargs:
        plot_left_axis=kwargs['plot_left_axis']
    else:
        plot_left_axis=True
        
    
    mot_tmp=indt['mot_deg'][mot_inds]
    
    if subtract_zero_time_flag:
        time=indt['time_in_min'][mot_inds]-indt['time_in_min'][0]
    else:
        time=indt['time_in_min'][mot_inds]

    if offset_time:
        time=time+offset_time
    
    
    if halt_flag:
        pdb.set_trace()
    
    mot_rad=calc.deg_to_rad(mot_tmp)-offset_to_subtract
    mot_tmp=calc.rad_to_deg(calc.standardize_angle(mot_rad,2*np.pi,force_positive=1))
    if center_on_zero_flag:
        mot=calc.center_deg_on_zero(mot_tmp)
        
    else:
        mot=mot_tmp        
    sub_plot_motor(ax,time,mot, **kwargs)
    if plot_vert_line_at_end:
        
        ax.plot([time[-1],time[-1]],[0,360],'b',linewidth=0.5)
   
    if mnvl:
       
        deg_mn=calc.rad_to_deg(mnvl)
        #ax.plot([0,15],[deg_mn, deg_mn],linestyle='--', dashes=(2, 1),color='r')
        #ax.plot([0,15],[deg_mn+180, deg_mn+180],linestyle='--', dashes=(2, 1),color='r')
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
    
    
    
       
    if plot_left_axis:
        fpl.adjust_spines(ax,['left','bottom'])
    
    elif withhold_bottom_axis:
        fpl.adjust_spines(ax,['left'])
        ax.get_xaxis().set_ticklabels([],fontsize=6)

    else:

    
        fpl.adjust_spines(ax,['bottom'])
        
        #elif (plot_left_axis) and (self.last_row_flag==0):
         #   ax.axis('off')
            #fpl.adjust_spines(ax,['left'])
    
    if plot_left_axis:
        if center_on_zero_flag:
            ax.get_yaxis().set_ticks([-180,0,180])
            ax.get_yaxis().set_ticklabels(['-180','0','180'],fontsize=6)
            ax.set_ylim([-180,180])
        else:
            ax.get_yaxis().set_ticks([0,90,180,270,360])
            ax.get_yaxis().set_ticklabels(['0','90','180','270','360'],fontsize=6)
            ax.set_ylim([0,360])
        if one_line_label:
            ylab='polarizer ($^\circ$)'
        else:
            ylab='polarizer\n($^\circ$)'
        ax.set_ylabel(ylab, fontsize=6)
    
    if xtickflag:
        xticks=kwargs['xticks']
        xticklabels=kwargs['xticklabels']
    else:
        xticks=[0,15]
        xticklabels=['0','15']
       
    if plot_vert_flag:
        ax.plot([kwargs['plot_vertical'],kwargs['plot_vertical']],[-20,380],'r')
    if not withhold_bottom_axis:
        
        try:
            if self.last_row_flag:
            #ax.set_ylabel('polarizer heading', fontsize=9)
                ax.get_xaxis().set_ticks(xticks)
                ax.get_xaxis().set_ticklabels(xticklabels,fontsize=6)
                ax.set_xlabel('time (min.)', fontsize=6)
        except:
            ax.get_xaxis().set_ticks(xticks)
            ax.get_xaxis().set_ticklabels(xticklabels,fontsize=6)
            ax.set_xlabel('time (min.)', fontsize=6)
    #pdb.set_trace()
    ax.set_xlim(xlim)
    #ax.set_aspect(0.005)
    ax.xaxis.labelpad = xlabelpad
    ax.yaxis.labelpad= 1
    

##
#This function plots position values in a manner that removes artefactual lines from data wrapping around
#inputs are 
#ax, handle to axis
#time- list of timevalues
#mot - list of degrees between 0 and 360

def sub_plot_motor(ax,time,mot,linewidth=0.5,**kwargs):
    
    try:
        max_allowed_difference=kwargs['max_allowed_difference']
    except:
        max_allowed_difference=50
    try:
        plot_flag=kwargs['plot_flag']
    except:
        plot_flag=True

    try:
        col=kwargs['color']
    except:
        col='k'
    
    absolute_diff_vls=abs(np.diff(mot))
    #these are indices to split the incoming array because the difference between neighboring
    #values exceeds threshold
    breakinds=np.where(absolute_diff_vls>max_allowed_difference)[0]
    
    #breakinds+1 is to get correct index
    #this outputs an array of arrays, which will be plotted
    mot_split_array=np.array_split(mot,breakinds+1)
    time_split_array=np.array_split(time,breakinds+1)
   
    #loops through the arrays to plot each value
    if plot_flag:
        for crind,crmot_splitinds in enumerate(mot_split_array):
            if np.size(crmot_splitinds):
                if len(crmot_splitinds>3):
                    ax.plot(time_split_array[crind],crmot_splitinds,color=col,linewidth=linewidth)
                else:
                    pdb.set_trace()
    return time_split_array, mot_split_array






def plot_mot_hist(indt,crax,**kwargs):
    
    crax.step(indt['normhst'],indt['xrad'][0:-1]+indt['rad_per_bin']/2,'k',linewidth=1)
    fpl.adjust_spines(crax,[])
    crax.set_ylim([-calc.deg_to_rad(20.),calc.deg_to_rad(380.0)])
    crax.plot(0.22,indt['mnrad_360'],'r<')
    crax.set_xlim([0,0.24])

def make_heat_map(ax,heatdt,**kwargs):
    POWER_VALUE=5
   
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
        paired_flagvl=kwargs['paired_flag']
    except:
        paired_flagvl=False


    # try:
    #     aligned_flag=kwargs['aligned']
    # except:
    #     aligned_flag=False
    #try:
     #   colorbar_ax=kwargs['colorbar_ax']
      #  fig_flag=kwargs['fig_flag']
       # plot_colorbar_flag=True
    #except:
     #   plot_colorbar_flag=False


    try:
        renorm_flag=kwargs['renorm']
    except:    
        renorm_flag=False
    if sub_flag:
        
        cr_heatmap_data=heatdt['sub_heat_map'][plt_type]
        if renorm_flag:
            
            cr_heatmap_data['norm_heat_map_vls']=cr_heatmap_data['norm_heat_map_vls']/sum(sum(cr_heatmap_data['norm_heat_map_vls']))

        
        twplt.polar_heat_map(cr_heatmap_data,ax=ax,shift_vertical_flag=True,sub_flag=sub_flag,**kwargs)
        

    elif paired_flagvl:
        try:
            heatmap_list=heatdt['full_heat_map'][plt_type]
        except:
            pdb.set_trace()
        if not 'redges' in heatmap_list.keys():
            heatmap_list['redges']=heatmap_list['r'][:,0]
            heatmap_list['thetaedges']=heatmap_list['theta'][0,:]
            
        cr_heatmap_data=heatmap_list
       

        twplt.polar_heat_map(heatmap_list,ax=ax,shift_vertical_flag=True,sub_flag=sub_flag,sep_max_flag=True,**kwargs)
      
            


    else:
       
        cr_heatmap_data=heatdt['full_heat_map'][plt_type]
        base_bnds=np.array([-np.pi/9, np.pi/9])
        #bnd_sectors=[base_bnds, base_bnds+np.pi/2, base_bnds+2*np.pi/2, base_bnds+3*np.pi/2]
        #arc_colvls=['r', 'k' ,'c' ,'b','m','g']

        arc_bnd_sectors=[base_bnds, base_bnds+np.pi/2, base_bnds+2*np.pi/2, base_bnds+3*np.pi/2]
        arc_colors=['lime','k','darkgreen',(0.5,0.5,0.5)]
        trans_bnd_sectors=[base_bnds, base_bnds+np.pi/2, base_bnds+2*np.pi/2, base_bnds+3*np.pi/2]
        transect_colors=['k','lime',(0.5,0.5,0.5),'darkgreen']

        if renorm_flag:
            cr_heatmap_data['norm_heat_map_vls']=cr_heatmap_data['norm_heat_map_vls']/sum(sum(cr_heatmap_data['norm_heat_map_vls']))

        #if plot_colorbar_flag:
        twplt.polar_heat_map(cr_heatmap_data,ax=ax,shift_vertical_flag=True,sub_flag=sub_flag,**kwargs)
        #else:
         #   twplt.polar_heat_map(ax,cr_heatmap_data,shift_vertical_flag=True,aligned=aligned_flag,sub_flag=sub_flag)
    if plot_transect_flag:
        base_bnds=np.array([-np.pi/9, np.pi/9])
        bnd_sectors=[base_bnds, base_bnds+np.pi/2, base_bnds+2*np.pi/2, base_bnds+3*np.pi/2]
        if transect_ax:
            twplt.plot_transects(transect_ax,cr_heatmap_data,ax_schematic=ax_schematic,bnds=bnd_sectors,**kwargs)



def arbitary_transect_from_heat_map(ax,heatdt,color='k',plot_mean=False,vecminvls=[0.9],withhold_plot=False,**kwargs):
    
    histvl={}
    
    #    num_inds_to_use=len(np.where(heatdt['thetaedges']>0.9)[0])
    for ind,crvecminvl in enumerate(vecminvls):
        
        if crvecminvl==0.9:
            startvl=2
        elif crvecminvl==0.8:
            startvl=4

        sub_array=heatdt['norm_heat_map_vls'][:,-startvl:]
        sumvls=np.sum(sub_array,axis=1)
        norm_sumvls=sumvls/np.sum(sumvls)
        histvl[crvecminvl]=norm_sumvls
        if not withhold_plot:
            crmean=calc.weighted_mean(norm_sumvls,heatdt['redges'],mn_type='norm')
            ax.step(heatdt['redges'][:-1],norm_sumvls,color=color,drawstyle='steps-post',linewidth=0.5)
            if plot_mean:
            
                ax.plot(crmean,kwargs['mnht'],'v',color=color,markersize=2,clip_on=False)
    return histvl


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
        
        ax.plot(plot_time,np.array(lftwng),colvls[0],linewidth=0.3)
        ax.plot(plot_time,np.array(rtwng),colvls[1],linewidth=0.3)
        ax.get_yaxis().set_ticks([25,80])
        fpl.adjust_spines(ax,['left','bottom'])
        ax.get_yaxis().set_ticklabels(['25','80'],fontsize=6)
        ax.set_ylabel('wing angle', fontsize=6)
        ax.set_ylim([25,80])
        ax.text(0,70,'left wing',fontsize=4,color=colvls[0])
        ax.text(5,70,'right wing',fontsize=4,color=colvls[1])
        if type=='zoom':
            ax.set_xlim([plot_time[0],plot_time[-1]])
        else:
            ax.set_xlim(xlim)    