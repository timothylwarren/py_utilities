#!/usr/bin/python
#this is twplot_library.py
import pdb
import pylab
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D  
import matplotlib.pyplot as plt
import fp_library as fpl
import tw_calc_library as calc
from matplotlib.patches import Ellipse
import matplotlib.colorbar as mpl_cbar

import matplotlib.cm as cm



from matplotlib.ticker import FuncFormatter

AXISPAD=-5
pylab.ion()


def my_formatter(x, pos):
    pdb.set_trace()
    if x.is_integer():
        return str(int(x))
    else:
        return str(x)





def plot_legend(crax,positions, colors, strvls):
    

    if type(positions[0]) is list:
        for crind, crposition in enumerate(positions):
            crax.text(crposition[0],crposition[1],strvls[crind],fontsize=5,color=colors[crind])
    else:
        crax.text(positions[0],positions[1],strvls,fontsize=5,color=colors)

def fill_between_steps(x, y1, y2=0, h_align='mid', ax=None, **kwargs):
    #this function written by someone else who posted it.
    #source is https://gist.github.com/thriveth/8352565

    ''' Fills a hole in matplotlib: fill_between for step plots.
    Parameters :
    ------------
    x : array-like
        Array/vector of index values. These are assumed to be equally-spaced.
        If not, the result will probably look weird...
    y1 : array-like
        Array/vector of values to be filled under.
    y2 : array-Like
        Array/vector or bottom values for filled area. Default is 0.
    **kwargs will be passed to the matplotlib fill_between() function.
    '''
    # If no Axes opject given, grab the current one:
    if ax is None:
        ax = plt.gca()
    # First, duplicate the x values
    xx = x.repeat(2)[1:]
    # Now: the average x binwidth
    xstep = np.repeat((x[1:] - x[:-1]), 2)
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    # Now: add one step at end of row.
    xx = np.append(xx, xx.max() + xstep[-1])

    # Make it possible to chenge step alignment.
    if h_align == 'mid':
        xx -= xstep / 2.
    elif h_align == 'right':
        xx -= xstep

    # Also, duplicate each y coordinate in both arrays
    y1 = y1.repeat(2)#[:-1]
    if type(y2) == np.ndarray:
        y2 = y2.repeat(2)#[:-1]

    # now to the plotting part:
    
    ax.fill_between(xx, y1, y2=y2, edgecolor='none',**kwargs)

    return ax

def plot_wing(ax,timevls,mot):

    MAXDEGPLOT=359
    MINDEGPLOT=1
    plotinds1=np.where(mot<MAXDEGPLOT)
    plotinds2=np.where(mot>MINDEGPLOT)
    allinds=np.intersect1d(np.array(plotinds1[0]),np.array(plotinds2[0]))
    splitinds=np.array_split(allinds,np.array(np.where(np.diff(allinds)!=1))[0]+1) 
    for rotnum,crsplitinds in enumerate(splitinds):
        if np.size(crsplitinds):
            try:
                ax.plot(timevls[crsplitinds],mot[crsplitinds],'b')
            except:
                pdb.set_trace()
def scatterplot(ax,xvl,yvl,sizefactor=0.3,**kwargs):
    try:
        ellipse_flag=kwargs['ellipse_flag']
    except:
        ellipse_flag=False

    try:
        plot_error_bar=kwargs['plot_error_bar']
    except:
        plot_error_bar=False
    try:
        set_axes=kwargs['set_axes']
    except:
        set_axes=False

    try:
        error_scale_factor=kwargs['error_scale_factor']
    except:
        error_scale_factor=1


    try:

        long_ax=kwargs['long_ax']
    except:
        long_ax=False
    #sizefactor=10.
    facecolor_vl=0.4
    colorvals=cm.get_cmap('gray')
    try:
        dynamic_sizes=True
        dyn_sizes=kwargs['dynamic_sizes']
    except:
        dynamic_sizes=False
    try:
        color_map=kwargs['color_map']
    except:
        color_map=False
    try:
        double_horizontal_ax=kwargs['double_horizontal_ax']
    except:
        double_horizontal_ax=False
    try:
        title=kwargs['title']
    except:
        title=False
    

    if dynamic_sizes:
       
        xdyn=np.array(dyn_sizes)[:,0]
        ydyn=np.array(dyn_sizes)[:,1]
    else:
        
        s=7
   
   
    if ellipse_flag:
        for i,crx in enumerate(xdyn):
            
            ells = Ellipse(xy= (np.array(xvl)[i],np.array(yvl)[i]), width=sizefactor*(1-xdyn[i]), height=sizefactor*(1-ydyn[i]) )
        
           
            mnvl=np.mean([xdyn[i],ydyn[i]])

            modify_ells(ax,ells,mnvl)
            if double_horizontal_ax:
                ells = Ellipse(xy= (np.array(xvl)[i]+180,np.array(yvl)[i]), width=sizefactor*(1-xdyn[i]), height=sizefactor*(1-ydyn[i]) )
                mnvl=np.mean([xdyn[i],ydyn[i]])

                modify_ells(ax,ells,mnvl)

    else:
        #ax.scatter(arena_stats['ommatidiaAzimuths']*180/np.pi,arena_stats['ommatidiaElevations']*180/np.pi,color='k',s=0.3)
       
        ax.scatter(np.array(xvl), np.array(yvl),  edgecolor='k',facecolor='k',s=sizefactor,alpha=0.5,clip_on=False)
        #pdb.set_trace()
        if double_horizontal_ax:
            ax.scatter(np.array(xvl)+180, np.array(yvl), edgecolor='k',facecolor='k', s=sizefactor,alpha=0.5,clip_on=False)
        
        if plot_error_bar:
            ax.errorbar(x=np.array(xvl),y=np.array(yvl),yerr=ydyn*error_scale_factor,xerr=xdyn*error_scale_factor,fmt=None, ecolor=[0.6,0.6,0.6], alpha=0.5,capsize=0, zorder=5)

            ax.errorbar(x=np.array(xvl)+180,y=np.array(yvl),yerr=ydyn*error_scale_factor,xerr=xdyn*error_scale_factor,fmt=None, alpha=0.5,ecolor=[.6, .6, .6], capsize=0, zorder=5)
    if set_axes:
        if long_ax:
            ax.set_xlim(6, 21)
            ax.set_ylim(0, 180)
        else:
            #ax.set_aspect('equal')
            if double_horizontal_ax:
                ax.set_xlim(0, 360)
            else:
                ax.set_xlim(0, 180)
            ax.set_ylim(0, 180)

    if title:
        ax.set_title(title)

def modify_ells(ax,ells,mnvl):
    ax.add_artist(ells)
    #ells.set_clip_box(ax.bbox)

    
       
        #ells.set_facecolor(colorvals(mnvl))
    
    colvl=1-3.*mnvl
    if colvl<0:
        colvl=0
    
    ells.set_alpha(1-colvl)    
    ells.set_facecolor([colvl, colvl ,colvl])
    ells.set_edgecolor('none')
    #ells.set_clim([0,1])
   
    #cbar.ax.set_yticklabels(['< -1', '0', '> 1'])

def lolli_plot(ax,head_dt,vec_dt,**kwargs):

    try:
        hist_flag=kwargs['make_hists']
    except:
        hist_flag=False

    try:
        shift_vertical=kwargs['shift_vertical']
    except:
        shift_vertical=True

    try:
        plot_thresh_vl=kwargs['thresh_vl']
    except:
        plot_thresh_vl=False
    try:
        thresh_inds=kwargs['thresh_inds']
        thresh_ind_flag=True
    except:
        thresh_ind_flag=False

    try:
        title=kwargs['title']
    except:
        title=False

    try:
        invert_flag=kwargs['invert']
    except:
        invert_flag=False
    try:
        plot_points=kwargs['plot_points']
    except:
        plot_points=True

    if invert_flag:
        head_dt=-head_dt

    
    #plot points
    if shift_vertical:
       
        head_dt=head_dt+np.pi/2

    
    for crind,crhead in enumerate(head_dt):
        #if shift_vertical:
         #   crhead=crhead+np.pi/2
        #ax[0].plot(np.hstack([head_dt, head_dt]),np.hstack((np.zeros(len(vec_dt)), vec_dt)))
        if 'plot_both_vec' in kwargs:
            
            ax.plot([crhead,crhead],[0,vec_dt[crind,0]/2.],'k',linewidth=0.2)
            ax.plot([crhead,crhead],[vec_dt[crind,0]/2.,vec_dt[crind,0]/2+vec_dt[crind,1]/2.],'r',linewidth=0.2)
        else:
            ax.plot([crhead,crhead],[0,vec_dt[crind]],'k',linewidth=0.2)
            if thresh_ind_flag:
                if np.in1d(crind,thresh_inds)[0]:
                
                    ax.plot([crhead,crhead],[0,vec_dt[crind]],'r')



    if plot_points:
        
        
        if thresh_ind_flag:
                
            try:   
                ax.scatter(head_dt,vec_dt,c='r',s=3,alpha=0.5,edgecolor='none')
            except:
                pdb.set_trace()
            try:
                ax.scatter(head_dt[thresh_inds],vec_dt[thresh_inds],c='c',s=3,alpha=0.5,edgecolor='none')
            except:
                pdb.set_trace()
        else:
            if 'plot_both_vec' in kwargs:

                ax.scatter(head_dt,np.mean(vec_dt,axis=1),c='c',s=3,alpha=0.5,edgecolor='none')
            else:
                ax.scatter(head_dt,vec_dt,c='c',s=3,alpha=0.5,edgecolor='none')
    else:
        if 'plot_both_vec' in kwargs:
            ax.scatter(head_dt, np.mean(vec_dt,axis=1),c='c',s=3,alpha=0.5,edgecolor='none')
        else:
            ax.scatter(head_dt,vec_dt,c='c',s=3,alpha=0.5,edgecolor='none')
    #plot_vec
   

    
    #for tick in ax.xaxis.get_major_ticks():
     #       tick.label1.set_fontsize(8)
    #for tick in ax.yaxis.get_major_ticks():
     #   tick.label1.set_fontsize(8)
        #axh.fill(crdt['xrad'][0:-1]+crdt['rad_per_bin']/2,crdt['normhst'],colvls[i],edgecolor=colvls[i],alpha=0.5,linewidth=1)
    #ax.get_yaxis().set_ticks([0.25,0.5])
    #if shift_vertical:
     #   ax.get_xaxis().set_ticks([0,np.pi/2,np.pi,3*np.pi/2])
      #  ax.get_xaxis().set_ticklabels(['270','0','90','180'])
    #else:
     #   ax.get_xaxis().set_ticks([0,np.pi/2,np.pi,3*np.pi/2])
      #  ax.get_xaxis().set_ticklabels(['0$^\circ$','90$^\circ$','180$^\circ$', '270$^\circ$'])
    #h.set_ylim([0,MAXYVL])
   
    ax.text(0,1.5,'n=%d'%(len(head_dt)),fontsize=6)
    #ax.text(0,1.5*mxvl,sum_str,fontsize=6,color=colvls[crptind])
    if title:
        
        ax.set_title(title)
    if 'thresh_vl' in kwargs:
        
        ax.plot([0,2*np.pi],[kwargs['thresh_vl'],kwargs['thresh_vl']],'r--')
    if thresh_ind_flag:
        
        ax.text(0,1.0,'n=%d'%(len(thresh_inds)),fontsize=6)
    #if thresh_inds.size:
     #       ax[0].plot([crhead[thresh_inds],crhead[thresh_inds]],[0,vec_dt[crind]],'color','c')

def polar_plot(axh,sumstats,**kwargs):
    
    indt=[]
    MAXYVL=0.12
    if 'fontsize' in kwargs:
        fontsize=kwargs['fontsize']
    else:
        fontsize=7
    if 'gain' in kwargs:
        crgain=sumstats['crgain']
    if 'title' in kwargs:
        title=kwargs['title']
    else:
        title=''    
    if 'mean' in kwargs:
        mnvl=kwargs['mean']
    else:
        mnvl=0
    if 'points' in kwargs:
        try:
            pts=kwargs['points']
        except:
            pts=False
    else:
        pts=False
    
    colvls=['k','c']
    
    #this is a hack to allow for multiple inputs
    #if sumstats is dict, turns it into a list
    if type(sumstats) is dict:
        
        indt.append(sumstats)
    else:
        indt=sumstats
    
    for i,crdt in enumerate(indt):
            
        if 'use_360_mean' in kwargs:
            if kwargs['use_360_mean']:
                plt_mnvl=crdt['mnrad_360']
            else:
                plt_mnvl=crdt['mnrad']
        else:
            plt_mnvl=crdt['mnrad']
        
        mxvl=np.max(crdt['normhst'])
        if plt_mnvl:
            axh.plot([plt_mnvl,plt_mnvl+np.pi],[1.2*mxvl, 1.2*mxvl],'r--',linewidth=1)
        else:
            axh.plot([plt_mnvl,plt_mnvl+np.pi],[1.2*mxvl, 1.2*mxvl],'r--',linewidth=1)
        if 'add_mean' in kwargs:
        
           
            axh.plot([kwargs['add_mean'][0],kwargs['add_mean'][0]+np.pi],[1.2*mxvl, 1.2*mxvl],'c',linewidth=1)
        if pts:
            for crptind,crpt in enumerate(pts):
                axh.plot([0,crpt],[0,1.5*mxvl],colvls[crptind])
                sum_str='part '+str(crptind)
                axh.text(crpt,1.5*mxvl,sum_str,fontsize=6,color=colvls[crptind])
        for tick in axh.xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        for tick in axh.yaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        axh.fill(crdt['xrad'][0:-1]+crdt['rad_per_bin']/2,crdt['normhst'],colvls[i],edgecolor=colvls[i],alpha=0.5,linewidth=1)
        axh.get_yaxis().set_ticks([])
        axh.get_xaxis().set_ticks([0,np.pi/2,np.pi,3*np.pi/2])
        axh.set_ylim([0,MAXYVL])
        if 'add_text' in kwargs:
            if kwargs['add_text']:
                titlestrbase=('M%.1f ' + 'V%.2f\n')% (crdt['mean_deg'],1-crdt['circvar']) 
                if 'add_360' in kwargs:
                    
                    titlestrbase=titlestrbase+('M%.4f ' + 'V%.2f\n')% (calc.rad_to_deg(crdt['mnrad_360']),1-crdt['circvar_mod360'])
                if 'add_str' in kwargs:
                    
                    titlestr=titlestrbase + '\n'+str(kwargs['add_str'])
                else:
                    titlestr=titlestrbase 
                axh.set_title(titlestr,fontsize=8)
            #pdb.set_trace()
            
    
    if 'plot_bare' in kwargs:
        if kwargs['plot_bare']==1:
        #axh.patch.set_visible(False)
        #axh.get_xaxis().set_ticks([])
            axh.get_yaxis().set_ticks([])
            axh.title.set_visible(False)
            axh.get_xaxis().set_ticklabels([])
            axh.spines['polar'].set_color('none')

#input is a dict of sumstats

def polar_heat_map(heat_data,ax=[],shift_vertical_flag=False,plot_colorbar_flag=False,split_flag=False,**kwargs):
    cbar_shrink=0.1
    cbar_pad=.1
    cbar_aspect=10
    paired_mesh=[]
    try:
        paired_flag=kwargs['paired_flag']
    except:
        paired_flag=False
    
    try:

        arc_positions=kwargs['arc_positions']
        arc_colors=kwargs['arc_colors']
        plot_arc_flag=True
    except:
        
        plot_arc_flag=False
    try:
        sub_flag=kwargs['sub_flag']
        
    except:
        sub_flag=False
    try:
        colorbar_ax=kwargs['colorbar_ax']
        #colorbar_flag=True
        fig_flag=kwargs['fig_flag']
    except:
        plot_colorbar_flag=False

    try:
        thetaedges=kwargs['thetaedges']
        redges=kwargs['redges']
    except:
        thetaedges=heat_data['thetaedges']
        redges=heat_data['redges']

    if 'clim' in kwargs:
        clim=np.array([0,kwargs['clim']])
        calc_clim_flag=False
    else:
        calc_clim_flag=True
    if 'aligned' in kwargs:
        
        if kwargs['aligned']:
            dat_type='realigned_norm_heat_map_vls'
        else:
            dat_type='norm_heat_map_vls'
    else:
       	dat_type='norm_heat_map_vls'
       	
    if 'ind' in kwargs:
        indflag=1
        ind=kwargs['ind']
    else:
        indflag=0

    
    try:
        sep_max_flag=kwargs['sep_max_flag']
    except:
        sep_max_flag=False


    if 'plot_power_scale' in kwargs:
        if kwargs['plot_power_scale']:
            plot_power_value=kwargs['plot_power_scale']
        else:
            plot_power_value=False
    else:
        plot_power_value=False
   
    try:
        offset_vl=kwargs['offset_vl']
    except:
        offset_vl=np.pi/2
  
    
    rmod, thetamod=np.meshgrid( thetaedges,redges)
    
    if plot_power_value:
        rplot=np.power(plot_power_value,rmod)
    else:
        rplot=rmod
    
    
    
    if shift_vertical_flag:
        thetamod=thetamod+offset_vl
        
        #kwargs['arc_positions']=kwargs['arc_positions']
    
    if calc_clim_flag:
        if not split_flag:
            clim=[0,np.max(heat_data[dat_type])]
        else:
            clim=[0,np.max(heat_data['norm_heat_map_vls'][1])]


    #check if sum >1



    if paired_flag:  
        crmax=[]
        for inds in [0,1]:
            if sep_max_flag:
                crmax.append(np.max(heat_data[dat_type][inds]))
        
        clim=[0,0.55*np.max(crmax)]
        for inds in [0,1]:
            mesh=ax[inds].pcolormesh(thetamod,rplot,heat_data[dat_type][inds],cmap='hot',vmin=clim[0],vmax=clim[1])
            paired_mesh.append(mesh)
    else:
        if not split_flag:
            
            mesh=ax.pcolormesh(thetamod,rplot,heat_data[dat_type],cmap='hot',vmin=clim[0],vmax=clim[1])
        else:
            #if kwargs['ind_to_plot']:
             #   pdb.set_trace()
            mesh=ax.pcolormesh(thetamod,rplot,heat_data[dat_type][kwargs['ind_to_plot']],cmap='hot',vmin=clim[0],vmax=clim[1])

    
    
    if paired_flag:
        for inds in [0,1]:
            crax=ax[inds]
            add_arc_fxn(crax,plot_arc_flag,**kwargs)
    else:
        
        add_arc_fxn(ax, plot_arc_flag,**kwargs)
        
    
    if plot_colorbar_flag:
       
        cmap=pylab.get_cmap('hot')
        #cbar=mpl_cbar.ColorbarBase(colorbar_ax,cmap=cmap,boundaries=[clim[0],clim[1]])
        
        make_colorbar(fig_flag,mesh,cax=colorbar_ax,ticks=clim)


    if type(ax) is list:
        for inds in [0,1]:
            adjust_polar_ax(ax[inds],plot_power_value,sub_flag,**kwargs)
    else:
        if plot_arc_flag:
            adjust_polar_ax(ax,plot_power_value,**kwargs) 
        else:
            adjust_polar_ax(ax,plot_power_value,**kwargs)



def add_arc_fxn(ax,plot_arc_flag,arc_r_pos=1.05,**kwargs):
    if plot_arc_flag:
        
        for crind,crarcpair in enumerate(kwargs['arc_positions']):
                #rvl=kwargs['plot_r_bnpds']
           
            polar_circle(ax,crarcpair,arc_r_pos,color=kwargs['arc_colors'][crind],linewidth=2)
            
            polar_circle(ax,[0,2*np.pi-.0001],1,color='0.5',linewidth=0.5) 
    else:
        #max_bnd=1
            #rpositions=[1.36,1.1,1.45,1.26]
        #rpositions=[1.26,1.03,1.35,1.16]
        if 'max_bnd' in kwargs:
            rvl=kwargs['max_bnd']
        else:
            rvl=1
        polar_circle(ax,[0,2*np.pi-.0001],rvl,color='0.5',linewidth=0.5) 



def make_colorbar(fig_flag,mesh,**kwargs):

    
    colorbar_ax=kwargs['cax']
    clim=kwargs['ticks']
    
    cb=fig_flag.colorbar(mesh,orientation='horizontal',**kwargs)
    colorbar_ax.get_xaxis().set_ticklabels(['0','%.3f'%clim[1]])
    #labels=colorbar_ax.xaxis.get_ticklabels()
        

        #labels[0]='0'
        #labels[1]=
    
    for l in colorbar_ax.xaxis.get_ticklabels():
        l.set_fontsize(4)
    colorbar_ax.tick_params(axis='both', which='both',length=0)
    colorbar_ax.spines['top'].set_color('none')
    colorbar_ax.spines['bottom'].set_color('none')
    colorbar_ax.spines['left'].set_color('none')
    colorbar_ax.spines['right'].set_color('none')
    colorbar_ax.set_xlabel("occupance probability",fontsize=4)
    colorbar_ax.xaxis.labelpad= -1 
    colorbar_ax.tick_params(axis='both', which='major', pad=0)
   

   
def adjust_polar_ax(ax,plot_power_value=False,withhold_vert_axis=False,withhold_horiz_axis=False,split_y_label=False,max_bnd=1.1,rpositions=[1.36,1.1,1.45,1.26],sub_flag=False,**kwargs):       
    
    if 'plot_mean' in kwargs:
        
        ax.plot([kwargs['plot_mean'],kwargs['plot_mean']],[0,5],'c')
        ax.plot([kwargs['plot_mean'][0]+np.pi,kwargs['plot_mean'][0]+np.pi],[0,5],'c')
    if plot_power_value:
        ax.set_ylim([1,5.5])
    else:
        ax.set_ylim([0,max_bnd])
    #ax.get_yaxis().set_ticks([])
    if not sub_flag:
        thetaticks=[0,np.pi/2,np.pi,3*np.pi/2]
        ax.get_xaxis().set_ticks(thetaticks)
        

        thetalabels=['270$^\circ$','0$^\circ$','90$^\circ$','180$^\circ$']

        ax.get_xaxis().set_ticklabels(thetalabels)
        #ax.set_thetagrids(thetaticks, frac=1.3)

        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])
        fpl.adjust_spines(ax,[])
        havl=['center','center','left','center']
        #rpositions=[1.36,1.1,1.45,1.26]
        for ind,crlabel in enumerate(thetalabels):
            #pdb.set_trace()
            ax.text(thetaticks[ind],rpositions[ind],crlabel,fontsize=6,ha=havl[ind])
    else:
        
        fpl.adjust_spines(ax,['left','bottom'])
        
        if withhold_vert_axis and withhold_horiz_axis:
            fpl.adjust_spines(ax,[])
            #ax.set_yticks([0,1])
            #ax.set_yticklabels([0,1],fontsize=5)
        #ax.get_xaxis().set_ticks([np.pi/2,np.pi/2+np.pi/4,np.pi])
        #ax.get_xaxis().set_ticklabels(['0$^\circ$','45$^\circ$','90$^\circ$'],fontsize=6)
        #ax.set_xlim([np.pi/2,np.pi])
        #ax.set_ylim([0,1])
        #ax.get_yaxis().set_ticks([0,0.2,0.4,0.6,0.8,1.0])
        #ax.get_yaxis().set_ticklabels(['0','0.2','0.4','0.6','0.8','1.0'],fontsize=6)
        else:
            if split_y_label:
                ax.set_ylabel('local\nvector strength',fontsize=5,multialignment='center')
            else: 
                ax.set_ylabel('local vector strength',fontsize=5)
            ax.set_xlabel('heading',fontsize=5)
            ax.set_yticks([0,1])
            ax.set_yticklabels([0,1],fontsize=5)
            ax.get_xaxis().set_ticks([np.pi/2,np.pi])
            ax.set_xticklabels([0,90],fontsize=5)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(5)
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(5)
        ax.set_ylim([0,1])
        
        ax.yaxis.labelpad=0
        
        ax.set_xlim([np.pi/2,np.pi])
        
        
        

        
        
        
        ax.xaxis.labelpad=-4
    

   
    



def polar_trajectory_plot(axh,inputdata):
    EXPCT=0
    BOUNDS=[0,1600]
    pts=[]
    for cr_experiment in inputdata:
        pts.append(get_points(cr_experiment))
    plot_trajectory(axh,pts)
    axh.get_yaxis().set_ticks([])
    axh.get_xaxis().set_ticks([0,np.pi/2,np.pi,3*np.pi/2])
    group_labels=['0','45','90','135']
    axh.set_xticklabels(group_labels)
    axh.set_ylim(BOUNDS)
    axh.set_xlim([0,np.pi])
    
#called by polar_trajectory_plot
#dependent variable is time
#independent variable is mean heading
def get_points(cr_experiment):
    FLOOR_VL=700
    xvl=cr_experiment['mnrad']
    exptime=int(cr_experiment['fname'].split('_')[-1][0:4])
    yvl=exptime-FLOOR_VL
    
    cr_pt=[xvl,yvl]
    return cr_pt

#plots combined headings in one row for a fly



def plot_combined_heading(crax,flydata):
    TIME_GAP_DIVIDER=20.0
    for index, exp in enumerate(flydata):
        crdegs=exp['deg_vls']
        crtime=exp['raw_time']
        if index is 0:
            TIME_OFFSET=0
            TOTAL_TIME=crtime[-1]-crtime[0]
            END_TIME=crtime[-1]
        else:
            TIME_OFFSET=TIME_OFFSET+TOTAL_TIME+(crtime[0]-END_TIME)/TIME_GAP_DIVIDER
            END_TIME=crtime[-1]
        timevls=TIME_OFFSET+np.array(crtime)-crtime[0]
        splitinds=treat_deg_vls(crdegs)
        
        for rotnum,crsplitinds in enumerate(splitinds):
            if np.size(crsplitinds):
                crax.plot(timevls[crsplitinds],crdegs[crsplitinds],'k')

        mnvl=exp['mean_deg']
        crax.plot([timevls[0],timevls[-1]],[mnvl, mnvl],c='c')
        crax.plot([timevls[0],timevls[-1]],[mnvl+180, mnvl+180],c='c')
        init_pol=exp['init_pol_angle']
        crax.plot(timevls[0],init_pol,marker='o',markersize=5, c='r',fillstyle='full')
#called by polar_trajectory_plot
#full revision
#takes 1. ax, 2. wing diff in degrees. motor_velocity in steps,4. steps per rotation, 5. list of linear gains to plot in rotations/degree wing diff, optional flag - downsample
def plot_wing_diff_rotation_speed(ax,wd_deg,motvel_steps,steps_per_rotation,**kwargs):
        COLVLS=['b','r']
        if 'downsample' in kwargs:
            downsample_flag=1
            downsample_value=kwargs['downsample']
        else:
            downsample_flag=0
            
        if 'linear_gain' in kwargs:
            linear_gain_flag=1
            linear_gain_vls=kwargs['linear_gain']
        else:
            linear_gain_flag=0
            
        if 'nonlinear_gain' in kwargs:
            nonlinear_gain_flag=1
            coeff=kwargs['nonlinear_gain'][0]
            asy_vl=kwargs['nonlinear_gain'][1]
        else:
            nonlinear_gain_flag=0
        
        if 'calc_fit' in kwargs:
            calc_fit_flag=1
            wd_cutoff=kwargs['wd_cutoff']
        else:
            calc_fit_flag=0
        
        if 'color_flag' in kwargs:
        	color_flag=1
        	colvl=kwargs['color_flag']
        else:
        	color_flag=0
        	colvl='k'
        
        #convert steps to degrees
        motvel_deg=(motvel_steps/steps_per_rotation)*360.0
        if downsample_flag:
            xvls=wd_deg[::downsample_value]
            yvls=-motvel_deg[::downsample_value]/360.0
        else:
            xvls=wd_deg
            yvls=-motvel_deg/360.0
        
        if color_flag:
        
        	ax.scatter(xvls,yvls,s=3,c='0.5',alpha=0.5,edgecolor=colvl)
        else:
        	ax.scatter(xvls,yvls,s=3,c='0.5',alpha=0.5,edgecolor='k')
        if calc_fit_flag:
            inds=np.where(np.abs(wd_deg)<wd_cutoff)
            fit_slope=calc.linear_ls_fit(np.abs(wd_deg[inds]),-np.abs(motvel_deg[inds]/360.0))
            max_wd_vl=np.nanmax(np.abs(xvls))
            linear_gain_xvls=np.linspace(-20,20,1000)
            
            if fit_slope:
                ax.plot(linear_gain_xvls,-linear_gain_xvls*fit_slope,'--',color='m')
        	
        if linear_gain_flag:
            for gainind,crgain in enumerate(linear_gain_vls):
                max_wd_vl=np.nanmax(np.abs(xvls))
                linear_gain_xvls=np.linspace(-max_wd_vl,max_wd_vl,1000)
                ax.plot(linear_gain_xvls,linear_gain_xvls*crgain,'--',color=COLVLS[gainind])
                
                
        if nonlinear_gain_flag:
            xvls_in_radians=calc.deg_to_rad(linear_gain_xvls)
            vel_in_steps=(2*asy_vl)/(1+np.exp(-coeff*xvls_in_radians))-asy_vl
           
            
            vel_in_degrees=(vel_in_steps/steps_per_rotation)*360.
            vel_in_rotations=vel_in_degrees/360.0
            ax.plot(linear_gain_xvls,vel_in_rotations,'c')
        fpl.adjust_spines(ax,['left','bottom'])
        #ax.set_aspect(.00007)
     
        #ax.set_xlim([-40,40])
        #ax.set_ylim([-0.4,0.4])
        
        ax.set_xlim([-20,20])
        ax.set_ylim([-.4,0.4])
        #ax.text(0.4,0,'coeff=%.1f,pk_vl=%.1f'%(coeff,asy_vel),fontsize=8)
        ax.set_ylabel('rotation rate, Hz',fontsize=8)
        ax.set_xlabel('wing difference deg/s',fontsize=8)
        #ax.get_yaxis().set_ticks([-.4,-.2,0,.2,.4])
        #ax.get_xaxis().set_ticks([-40,-20,0,20,40])
        ax.get_yaxis().set_ticks([-.4,-.3,.2,-.1,0,0.1,0.2,0.3,0.4])
        ax.get_xaxis().set_ticks([-20,-10,0,10,20])
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        
        if calc_fit_flag:
            return fit_slope

def treat_deg_vls(crdegs):
    MAXDEGPLOT=358
    MINDEGPLOT=2
    plotinds1=np.where(crdegs<MAXDEGPLOT)
    plotinds2=np.where(crdegs>MINDEGPLOT)
    allinds=np.intersect1d(np.array(plotinds1[0]),np.array(plotinds2[0]))
    splitinds=np.array_split(allinds,np.array(np.where(np.diff(allinds)!=1))[0]+1)
    return splitinds


def plot_trajectory(axh,pts):
    PTCT=0
    for crpt in pts:
        xvl=crpt[0]
        yvl=crpt[1]
        #multiply by 2 so circular
        axh.scatter(xvl*2,yvl, s=2, facecolor='0.5',markerfacecolor ='none',) 
        if PTCT>0:
            axh.plot([crpt[0]*2,prevpt[0]*2],[crpt[1],prevpt[1]],c='k')
        prevpt=crpt
        PTCT=PTCT+1


def plot_listvls(ax,xvls,yvls,**kwargs):
    for crind, crxvls in enumerate(xvls):
        if 'normalize_x' in kwargs :
            #pdb.set_trace()
            crxvls=np.array(crxvls)-crxvls[0]
        
        cryvls=yvls[crind]
        #pdb.set_trace()
        ax.plot(crxvls,cryvls,'o',color='k')

def plot_listscatter(ax,inlist):
    ax.scatter(inlist[0],inlist[1],s=20,c='0.5',alpha=0.5,edgecolor='k')
    
    #pdb.set_trace()
    #for crvls in inlist:
        #cryvls=yvls[crind]
        
        #ax.scatter(crvls[0],crvls[1], s=20, c='0.5',facecolor=None,edgecolor='k')
        
        #if len(crvls)>2:
            #for i in range(len(crvls)-2):
                
                #ax.scatter(crvls[0],crvls[i+2],s=20,c='r',facecolor='r',edgecolor=None)
                
        #fpl.adjust_spines(ax,['left', 'bottom'])

def raw_vector_plot(ax,indt):
    THRESHOLD=0.8
    cr_mn=indt['mn_vector_lst']
    cr_ln=np.array(indt['len_vector_lst'])
    
    ax.plot(cr_mn,cr_ln,'k',alpha=0.5,linewidth=0.3)
    threshinds=np.where(np.array(cr_ln)>THRESHOLD)[0]
    splitinds=np.array_split(threshinds,np.array(np.where(np.diff(threshinds)!=1))[0]+1)
    for rotnum,crsplitinds in enumerate(splitinds):
        if np.size(crsplitinds):
            
            ax.plot(cr_mn[crsplitinds],cr_ln[crsplitinds],'c',linewidth=0.3)
    ax.get_yaxis().set_ticks([])
    ax.title.set_visible(False)
    ax.get_xaxis().set_ticklabels([])
    ax.spines['polar'].set_color('none')
    ax.get_xaxis().set_ticks([0,np.pi/2,np.pi,3*np.pi/2])

def rand_scatter(ax, xvls, num_bins, yfloor,max_yrange):
    #algorithm is to bin the data into n bins, determine indices in those bins
    #for each bin calculate the relative number in bin as a function of the max size
    #then calculate the yvalues for those indices
    
    xedges=[np.min(xvls),np.max(xvls)]
    
    bins=np.linspace(xedges[0],xedges[1],10)
    
    hist, bins= np.histogram(xvls, bins=bins)
    
    normalized_hist=hist/float(np.max(hist))
    yvls=np.zeros(len(xvls))
    for crind,crnormvls in enumerate(normalized_hist):
        
        crinds=np.intersect1d(np.where(xvls>=bins[crind])[0],np.where(xvls<=bins[crind+1])[0])
        
        yvls[crinds]=yfloor+crnormvls*np.random.random_sample(len(crinds))*max_yrange

    #yrange=[0.07,.09]
    #yvls=np.random.random_sample(len(diff_vls))*(yrange[1]-yrange[0])+yrange[0]
    
    ax.scatter(np.array(xvls), yvls, facecolors='none', s=0.7,zorder=10,alpha=0.5,edgecolors='k')
    #ax.scatter(np.array(xvl), np.array(yvl),  facecolors='none', edgecolors='k',s=sizefactor,alpha=0.5,marker='o')


def rand_jitter(arr):
    
    stdev = .01*(np.max(arr)-np.min(arr))
    if stdev > 0:
        return arr + np.random.randn(len(arr)) * stdev
    else:
        return arr+ np.random.rand(len(arr))*.1

def jitter(ax,x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    
    ax.scatter(rand_jitter(x), y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None,  **kwargs)

def steppify(arr,isX=False,interval=0):
    """
    Converts an array to double-length for step plotting
    """
    if isX and interval==0:
        interval = abs(arr[1]-arr[0]) / 2.0
        
        newarr = np.array(zip(arr-interval,arr+interval)).ravel()
        return newarr

def plot_horizontal_lines(ax,start_vls,stop_vls,vert_vl):
    for i, crind in enumerate(start_vls):
        ax.plot([start_vls[i],stop_vls[i]],[vert_vl,vert_vl],'r')

def plot_sigmoid_compar(ax,wd,crmot,coeff,asy_vel):
     lincolvls=['c','g']
     steps_per_rotation=27500.
     deg_per_rot=360.
     wdrad=calc.deg_to_rad(wd)
     xvls=np.linspace(np.nanmin(wdrad),np.nanmax(wdrad),100)
     
     ax.plot(wdrad,crmot,'k.')
     
     ax.plot(xvls,(2*asy_vel)/(1+np.exp(-coeff*xvls))-asy_vel,'r')
     for i,crgain in enumerate([100.0,250.0]):
        gainStepsPerRadSec = crgain*steps_per_rotation/deg_per_rot
        ax.plot(xvls,xvls*gainStepsPerRadSec,'--',color=lincolvls[i])
     fpl.adjust_spines(ax,['left','bottom'])
     ax.set_aspect(.00007)
     
     ax.set_xlim([-0.7,0.7])
     ax.set_ylim([-6000,6000])
     ax.text(0.4,0,'coeff=%.1f,pk_vl=%.1f'%(coeff,asy_vel),fontsize=8)
     ax.set_xlabel('wd, radians',fontsize=8)
     ax.set_ylabel('mot. steps/s',fontsize=8)
     for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(8)
     for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(8)     

def add_mean_and_range(ax,mn,range,mn_height,range_height,plot_range=False,**kwargs):
    try:
        col=kwargs['color']
    except:
        col='r'
    ax.plot(mn,mn_height,color=col,marker='v',markersize=3,clip_on=False)
    if plot_range:
        ax.plot([range[0],range[1]],[range_height,range_height],color=col)

def plot_hist(axh,indata,**kwargs):
   
    data=np.array(indata)
    
    try:
        linewidth=kwargs['linewidth']
    except:
        linewidth=1
    try:
        invert_flag=kwargs['invert']
    except:
        invert_flag=False

    try:
        shift_vertical=kwargs['shift_vertical']
    except:
        shift_vertical=False

    if 'num_bins' in kwargs :
        NUM_BINS=kwargs['num_bins']
    else:
        NUM_BINS=10
    
    if 'no_plot' in kwargs:
        no_plot_flag=kwargs['no_plot']
    else:
        no_plot_flag=False

    if 'plt_type' in kwargs:
        cumulative_val=1
    else:
        cumulative_val=0
    if 'norm' in kwargs:
        normvl = kwargs['norm']
    else:
        normvl=False
    if 'hst_bnds' in kwargs:
        BNDS=kwargs['hst_bnds']
    else:
        BNDS=[np.floor(np.min(data[~np.isnan(data)])),np.ceil(np.max(data[~np.isnan(data)]))]
    if 'col' in kwargs:
        COL=kwargs['col']
    else:
        COL='k'
    if 'orientation' in kwargs:
        orient=kwargs['orientation']
    else:
        orient='vertical'    
    
    if 'plot_mean' in kwargs:
        if kwargs['plot_mean']:
            plot_mean=True
        else:
            plot_mean=False
    else:
        plot_mean=False
    if 'filled' in kwargs:
        if kwargs['filled']:
            his_type='stepfilled'
        else:
            his_type='step'
    else:
        his_type='step'
    if 'repeat_pi' in kwargs:
        repeat_pi_interval_flag=1
    else:
        repeat_pi_interval_flag=0
    if 'suppress_plot' in kwargs:
        suppress_plot=1
    else:
        suppress_plot=0
    if 'plot_text' in kwargs:
        text_flag=1
        text_str=kwargs['plot_text']
    else:
        text_flag=0
    if 'calc_conf_interval' in kwargs:
        calc_dist_flag=1

        conf_bnds=kwargs['confidence_bounds']
    else:
        calc_dist_flag=0
    
    if invert_flag:
        data=-data
    if shift_vertical:
       
        data=data+np.pi/2

    bin_width=(BNDS[1]-BNDS[0])/NUM_BINS

    
    bins=np.linspace(BNDS[0],BNDS[1],NUM_BINS+1)
    inarray=data[~np.isnan(data)]
    
    if calc_dist_flag==0:
        if normvl:
            weights = np.ones_like(inarray)/len(inarray)
            
            nvl,bins,patches = axh.hist(data[~np.isnan(data)], weights=weights,bins=bins, histtype=his_type, color=COL, orientation=orient,cumulative=cumulative_val,linewidth=linewidth)
                
     
        else:
            weights = np.ones_like(inarray)/len(inarray)
            hist, bins = np.histogram(data[~np.isnan(data)], bins=bins, weights=weights)

            offset = bins[1:]-bins[:-1]
            if no_plot_flag==False:
                if cumulative_val:
                    pltinds=np.where(np.cumsum(hist)<0.99)[0]
                    axh.step( (bins[:-1]+offset)[pltinds], np.cumsum(hist)[pltinds],color=COL,linewidth=linewidth )
                else:
                    axh.step( bins, np.insert(hist,0,0),color=COL )
                    if repeat_pi_interval_flag:
                        axh.step(bins+np.pi,np.insert(hist,0,0),color=COL,linewidth=linewidth)
        #nvl, bins, patches = axh.hist(data[~np.isnan(data)], bins=bins,normed=normvl, histtype=his_type, color=COL, cumulative=cumulative_val,orientation=orient,linewidth=1)
    else:
        cum_sum_hist=[]
        #calculate hist for each row
        crdata=np.array(indata)
        nrow=len(crdata[:,0])
        pdb.set_trace()
        for cr_row in np.arange(nrow):
            inarray=crdata[cr_row,:]
            weights = np.ones_like(inarray)/len(inarray)
            hist, bins = np.histogram(inarray[~np.isnan(inarray)], bins=bins, weights=weights)
            cum_sum_hist.append(np.cumsum(hist))
        cum_sum_array=np.array(cum_sum_hist)
        offset = bins[1:]-bins[:-1]
        if no_plot_flag==False:
            for crbnd in kwargs['confidence_bounds']:
                cr_array=np.percentile(cum_sum_array,crbnd,axis=0)
                
                pltinds=np.where(cr_array<0.99)[0]
                axh.step( (bins[:-1]+offset)[pltinds], cr_array[pltinds],color=COL,linewidth=linewidth)
                


    if suppress_plot:
       
        fpl.adjust_spines(axh,[])
        axh.get_yaxis().set_ticks([])
        axh.get_yaxis().set_ticklabels([],fontsize=8)
    
    
    #pylab.setp(patches,'facecolor', COL, 'alpha', 0.5)
    #if normvl:
    #    histout=histout/np.sum(histout)
    
    #bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])
    #xx=bin_edges
    #yy=np.insert(histout,0,0)
    #axh.step(xx,yy,color=COL,linewidth=2)
    
    #pdb.set_trace()
    #axh.get_yaxis().set_ticks([0,0.5,1])
    
    if 'plot_mean' in kwargs:
        
        colvls=['r','b']

        for addvl in [0,np.pi]:
            for i,crmnvl in enumerate(kwargs['mnvl']):

                axh.plot([kwargs['mn_xvl']],[crmnvl+addvl],'<',c=colvls[i],markersize=3)
    #if 'title' in kwargs:
     #   axh.set_title(kwargs['title'])
    #if 'plot_median' in kwargs:
     #   xvls=[0.15,0.65]
        #yvls=[0.5,0.2]
      #  for crind in [0,1]:
       #     axh.text(xvls[crind],0.5,str(kwargs['plot_median'][crind])[0:4],fontsize=7)
        #    axh.text(xvls[crind],0.2,str(kwargs['plot_frac'][crind])[0:4],fontsize=7)

def polar_circle(ax,thetavls,rvls,**kwargs):
    try:
        linewidth=kwargs['linewidth']
    except:
        linewidth=1

    xvls=np.linspace(thetavls[0],thetavls[1],50)
    try:
        rvls=rvls*np.ones(50)
    except:
        pdb.set_trace()
    crcol=kwargs['color']
    try:
        ax.plot(xvls,rvls,color=crcol,linewidth=linewidth)
    except:
        pdb.set_trace()

def plot_transects(axin,ave_heatmap_data,**kwargs):
    
    WRAP_VALS=True
    
    #SUM_INDS= [[0,2],[1,3]]
    if 'aligned' in kwargs:
        if kwargs['aligned']:
            dat_type='realigned_norm_heat_map_vls'
        else:
            dat_type='norm_heat_map_vls'
    else:
        dat_type='norm_heat_map_vls'
    
    if 'ind' in kwargs:
        crdt=ave_heatmap_data[dat_type][kwargs['ind']]
    else:
        crdt=ave_heatmap_data[dat_type]
        
   
    if 'theta' not in kwargs:
      
        kwargs['theta']=ave_heatmap_data['theta']

    if 'thetaedges' not in kwargs:
     
        kwargs['thetaedges']=ave_heatmap_data['thetaedges']


    try:
        sector_rvl=kwargs['sector_rvl']
    except:
        sector_rvl=1.05

    #sector_vls is list of radian pairs
    try:
        sector_vls=kwargs['sector_vls']
    except:
        number_of_sectors=6
    

    try:
        shift_vertical_flag=kwargs['shift_vertical_flag']
    except:
        shift_vertical_flag=True
   
    try:
        paired_flag=kwargs['paired_flag']
    except:
        paired_flag=False

    #for instance 72 r vls to make up whole space
    num_r_vls=np.shape(crdt)[0]
    
    redges=ave_heatmap_data['redges']
    
       
    #bnds=np.linspace(0,num_r_vls,number_of_sectors+1)-(num_r_vls/number_of_sectors)/2
    
    
    CTR=0
    
    
    if paired_flag:  
        
        for inds in [0,1]:
            plotdt=crdt[inds]
            pltax=axin[inds]
            if inds==0:
                determine_and_plot_transects(pltax,plotdt,redges,ymax=0.04, **kwargs)
            else:
                determine_and_plot_transects(pltax,plotdt,redges,ymax=0.04,no_legend=True, **kwargs)
    else:
        plotdt=crdt
        pltax=axin
        
        proportional_vls=determine_and_plot_transects(pltax,plotdt,redges,**kwargs)
        
    return proportional_vls
    #ax.grid()


def determine_and_plot_transects(pltax,crdt,redges,theta,offset=0,transect_x_type='vector',**kwargs):
    bnds=kwargs['bnds']
    proportional_vls={}
    try:
        colvls=kwargs['colvls']
    except:
        colvls=['r', 'k' ,'c' ,'b','m','g']
    summed_vls=[]
    
    try:
        ymax=kwargs['ymax']
    except:
        ymax=0.04
    try:
        no_legend=kwargs['no_legend']
    except:
        no_legend=False
    
    
    if transect_x_type=='vector':

        for crbnd in bnds:


            if crbnd[0]<0:
                #find how crbnd in radians maps to index on crdt
                
                pos_bnd=crbnd[0]+2*np.pi
                
                crind=np.argmin(np.abs(redges-pos_bnd))
                
                #find index of bnds[0]
                
                first_sum=np.sum(crdt[crind:,:],axis=0)
                #add to last segment to end
                
                crind=np.argmin(np.abs(redges-crbnd[1]))
                second_sum=np.sum(crdt[0:crind,:],axis=0)
                summed_vls.append(first_sum+second_sum)
                
            else:
                
                firstind=np.argmin(np.abs(redges-crbnd[0]))
                secondind=np.argmin(np.abs(redges-crbnd[1]))
                summed_vls.append(np.sum(crdt[firstind:secondind+1,:],axis=0))
                #plot sector
                
        array_transect_vls=np.array(summed_vls)
        xvls=kwargs['thetaedges']
        xvlsplt=np.append(xvls-xvls[0],1.0)
    #legend_text=['-20$^\circ$ to 20$^\circ$','70$^\circ$ to 110$^\circ$','160$^\circ$ to 200$^\circ$','250$^\circ$ to 290$^\circ$']
    
    elif transect_x_type == 'position':
        vec_thresh=kwargs['vec_threshold']
        vec_ind=np.min(np.where(kwargs['thetaedges']>=vec_thresh))
        
        
        summed_vls=np.sum(crdt[:,vec_ind:],axis=1)
        #assumes total width is 2*np.pi.
        #assumes taking 72 to 36 bins
        array_transect_vls=[]
        out_edges=[]
        if offset:
            ind_offset=np.where(redges==offset)[0][0]
            new_edges=redges-offset
            neg_inds=np.where(new_edges<0)
            new_edges[neg_inds]=new_edges[neg_inds]+2*np.pi
            xvlsplt=new_edges
        for crvl in np.linspace(0,70,36):
            
            array_transect_vls.append(summed_vls[crvl]+summed_vls[crvl+1])
            
        initxvls=redges
        
    if transect_x_type=='vector':

        for crind,cr_row in enumerate(array_transect_vls):

            yplt=np.append(cr_row,0)
            
            try:
                pltax.step(xvlsplt[:-1],yplt,color=kwargs['transect_colvls'][crind],linewidth=0.5)
            except:
                pdb.set_trace()
            position=[1.1,.01+.005*crind]
            #strvl=legend_text[crind]
       
    elif transect_x_type == 'position':     
        yplt=np.array(array_transect_vls)
            
        x_unsorted=xvlsplt[0:-1:2]
        
        sortinds=np.argsort(x_unsorted)
        pltax.step(x_unsorted[sortinds],yplt[sortinds],color=kwargs['transect_colvls'][0],linewidth=0.5)

    if transect_x_type=='vector':            
        fpl.adjust_spines(pltax,['left','bottom'])
        pltax.get_xaxis().set_ticks([0,1.0])
        pltax.get_xaxis().set_ticklabels([0,1],fontsize=5)
        pltax.get_yaxis().set_ticks([0,.03])
        pltax.get_yaxis().set_ticklabels([0,.03],fontsize=5)
        pltax.set_xlabel('local\nvector strength', fontsize=5,multialignment='center')
        #ax.yaxis.labelpad=-2
        pltax.xaxis.labelpad=0
        pltax.set_ylabel('probability',fontsize=5)
        pltax.yaxis.labelpad=-9
        pltax.xaxis.labelpad=0
        #ax.tick_params(direction='out', pad=1)
        #mpl.rcParams['xtick.major.size'] = 10
        #mpl.rcParams['xtick.major.width'] = 1
        pltax.set_xlim([0,1])
    
    elif transect_x_type=='position': 
        fpl.adjust_spines(pltax,['left','bottom'])
        pltax.get_xaxis().set_ticks([0,np.pi,2*np.pi])
        pltax.get_xaxis().set_ticklabels([0,180, 360],fontsize=5)
        pltax.get_yaxis().set_ticks([0,.01])
        pltax.get_yaxis().set_ticklabels([0,.01],fontsize=5)
        pltax.set_xlabel('heading $^\circ$', fontsize=5,multialignment='center')
        #ax.yaxis.labelpad=-2
        pltax.xaxis.labelpad=0
        pltax.set_ylabel('probability',fontsize=5)
        pltax.yaxis.labelpad=-9
        pltax.xaxis.labelpad=0
    #ax.set_ylim([0,.05])
    

    #ax.set_aspect(40)
        pltax.set_ylim([0,.01])
    return array_transect_vls
    #ax.get_yaxis().set_ticks(np.arange(0,ymax,0.01))
    #ax.get_yaxis().set_ticklabels(np.arange(0,ymax,0.01),fontsize=6)
    