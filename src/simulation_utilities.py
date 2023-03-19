import sys, os

from matplotlib.axes import Axes

sys.path.insert(0, "/Users/gregorio/Desktop/Thesis/code/python/src")

from matplotlib import projections
import matplotlib as mpl
import casadi  as ca
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.linalg
import scipy.spatial

#
from tudatpy.io               import save2txt
from tudatpy.util             import result2array
from conversions_tools        import classic2implicit_keplerian_elements
from dynamic_models           import AerodynamicSettings
from tudatpy.kernel.astro     import element_conversion,frame_conversion,two_body_dynamics
from tudatpy.kernel           import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel           import numerical_simulation # KC: newly added by me
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.math                 import interpolators
from tudatpy.kernel.numerical_simulation import propagation_setup
import sympy as sp



def show_trajectory2D(waypoints,state_history,xlabel:str = "x axis",ylabel:str="y axis") :
    '''
    plots the state history of a simple 2D dynamical model
    state_history -->array with x,y coordinates for a car
    '''
    
    fig,ax = plt.subplots()
    ax.plot(state_history[:,0],state_history[:,1],color='blue',marker='+',markersize=11,label="agent trajectory")
    ax.plot(waypoints[:,0],waypoints[:,1],marker='*',linestyle='dashed',markersize=12,color = 'red',label="waypoints")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    
def plot_quaternions_dynamics(time,quaternions_history,
                              xlabel:str = "time",
                              show_accuracy_measure:bool = [],
                              save_output_at:str=None) :
    
    """
    Parameters
    -----------             
                       
    time                (np.ndarray(N,))  : time stamps at which the quaternions are sampled
    quaternions_history (np.ndarray(N,4)) : matrix containing a quaternion instance for each row
                                            each row corresponds to a time stamp           
                                            The quaternion definition follows :
                                            q0     = real part
                                            q1,2,3 = vectorial part               
    
                  
    Returns
    -----------
    None
                                    
    Descriptiom
    -----------
    plots the quaternions state history
    """
    
    fig,ax_quat = plt.subplots(4,1,sharex=True)

    for jj,ax in enumerate(ax_quat) :
        ax.plot(time,quaternions_history[:,jj])
        ax.set_ylabel(r"q_{}".format(jj))
    
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    
    if show_accuracy_measure :
        try :
          fig,ax = plt.subplots()
          ax.semilogy(time,abs(1.-np.sqrt(np.sum(quaternions_history**2,axis=1))))
          ax.set_xlabel(xlabel)
          ax.set_ylabel(r"$abs(1-||q||_2)$")
        except : UserWarning
    
    if save_output_at != None :
       try :
         fig.savefig(save_output_at, format='svg', dpi=1200)
       except :
        raise ("The figure was not saved due to an error.Check that the output directory exists")

    return fig,ax
        
        
        
def show_control_trajectory(time,
                            control_history,
                            control_names:list=[],
                            xlabel:str = "instances",
                            single_figure:bool=True,
                            bounds:np.ndarray=[],
                            save_output_at:str=None,
                            title:str="control over time") :
    
    '''
    plots control trajectory

    bounds = [[u1_max,u1_min],
              [u2_max,u2_min],
              .....]
    '''
    
    _,cols        = np.shape(control_history)
    if len(control_names)==  0:
       control_names = [r'$u_{}$'.format(num) for num in range(cols)]
       
    if not single_figure :
        
        for col in range(cols) :
            fig,ax = plt.subplots()
            ax.step(time,control_history[:,col], where='post')
            if len(bounds) != 0 :
              ax.axhline(y=bounds[col,0], color='r', linestyle='--')
              ax.axhline(y=bounds[col,1], color='r', linestyle='--')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(control_names[cols])
            ax.set_title(title)
    else :
        fig,ax = plt.subplots(cols,1,sharex=True)
        for col,axis in zip(range(cols),ax) :
            axis.step(time,control_history[:,col], where='post')
            if len(bounds) != 0 : 
              axis.axhline(y=bounds[col,0], color='r', linestyle='--')
              axis.axhline(y=bounds[col,1], color='r', linestyle='--')
            axis.set_ylabel(control_names[col])
            axis.set_xlabel(xlabel)
            axis.set_title(title)

        plt.tight_layout()
    if save_output_at != None :
       try :
         fig.savefig(save_output_at, format='svg', dpi=1200)
       except :
        raise ("The figure was not saved due to an error.Check that the output directory exists")

    return fig,ax
        

def plot_relative_dynamics(relative_dynamics_state:np.ndarray,
                           waypoints: np.ndarray= None,
                           title: str ="relative orbit dynamics",
                           save_output_at:str=None,
                           units:str="m") :
    
    """
    Parameters
    -----------
    
    relative_dynamics_state   (np.ndarray(N,6)) : 
    array of N rows contaning the cartesian state history of N chaser states
    w.r.t the chief spacecraft in rsw frame 
              
    title       (str)             : title of the plots
    waypoints   (np.ndarray(N,6)) : set of waypoints to be followed by a controller 
    
    
    Note : rsw fame is defined as follows
    X =  radial direction                        (nadir direction)
    Y = cross(X,Y)
    Z = angular momentum direction             
    
    Returns
    -----------
    None
    
    Descriptiom
    -----------         
    plots three figures showing the cross-plane and in-plane relative dynamics of
    the chase spacecraft relative to the chief spacecraft
    
    """
    units_available = ["m","km","cm"]
    conversion = {"m":1,"km":10**-3,"cm":10**2}
    if units not in units_available :
        message = ("Invaid Units. Units available are {},{} and {}".format(*units_available))
        raise ValueError(message)

    rows,cols = np.shape(relative_dynamics_state)
    if cols == 7. :
        # you forgot to remove the time
        shift = 1
    else :
        shift = 0 
    
    
    figure,ax = plt.subplots(2,1)
    
    ax[0].plot(relative_dynamics_state[0 ,shift+1]*conversion[units],relative_dynamics_state[0 ,shift+2]*conversion[units],marker='*',markersize=10,color='r',label ="start")
    ax[0].plot(relative_dynamics_state[: ,shift+1]*conversion[units],relative_dynamics_state[: ,shift+2]*conversion[units])
    ax[0].plot(relative_dynamics_state[-1,shift+1]*conversion[units],relative_dynamics_state[-1,shift+2]*conversion[units],marker='.',markersize=10,color='g',label ="end")
    if np.all(waypoints != None) :
      for waypoint in waypoints :
        ax[0].plot(waypoint[1]*conversion[units],waypoint[2]*conversion[units],marker='*',markersize=10,color='k')
    
    
    ax[0].set_xlabel(" Y- along-track [{}]".format(units))
    ax[0].set_ylabel(" Z- cross-track [{}]".format(units))
    ax[0].set_title(title)
    
   
    ax[0].grid(color = 'green', linestyle = '--', linewidth = 0.5)
    ax[0].legend()
    

    ax[1].plot(relative_dynamics_state[0 ,shift+4]*conversion[units],relative_dynamics_state[0 ,shift+5]*conversion[units],marker='*',markersize=10,color='r',label ="start")
    ax[1].plot(relative_dynamics_state[-1,shift+4]*conversion[units],relative_dynamics_state[-1,shift+5]*conversion[units],marker='.',markersize=10,color='g',label ="end")
    ax[1].plot(relative_dynamics_state[: ,shift+4]*conversion[units],relative_dynamics_state[:, shift+5]*conversion[units])
    if np.all(waypoints != None) :
      for waypoint in waypoints :
        ax[1].plot(waypoint[4],waypoint[5],marker='^',markersize=10,color='k')
    
    ax[1].set_xlabel("  VY- along-track+ [{}/s]".format(units))
    ax[1].set_ylabel("  VZ- cross-track+ [{}/s]".format(units))
   
   
    ax[1].grid(color = 'green', linestyle = '--', linewidth = 0.5)
    ax[1].legend()
    
    plt.tight_layout()

    figure,ax = plt.subplots(2,1)
    ax[0].plot(relative_dynamics_state[0 ,shift+0]*conversion[units],relative_dynamics_state[0 ,shift+2]*conversion[units],marker='*',markersize=10,color='r',label ="start")
    ax[0].plot(relative_dynamics_state[-1,shift+0]*conversion[units],relative_dynamics_state[-1,shift+2]*conversion[units],marker='.',markersize=10,color='g',label ="end")
    ax[0].plot(relative_dynamics_state[: ,shift+0]*conversion[units],relative_dynamics_state[: ,shift+2]*conversion[units])
    if np.all(waypoints != None) :
      for waypoint in waypoints :
        ax[0].plot(waypoint[0]*conversion[units],waypoint[2]*conversion[units],marker='^',markersize=10,color='k')
    
    ax[0].set_xlabel(" X - radial     [{}]".format(units))
    ax[0].set_ylabel(" Z - cross-track[{}]".format(units))
    ax[0].set_title(title)
    
   
    ax[0].grid(color = 'green', linestyle = '--', linewidth = 0.5)
    ax[0].legend()
    
    
    ax[1].plot(relative_dynamics_state[0 ,shift+3]*conversion[units],relative_dynamics_state[0 ,shift+5]*conversion[units],marker='*',markersize=10,color='r',label ="start")
    ax[1].plot(relative_dynamics_state[-1,shift+3]*conversion[units],relative_dynamics_state[-1,shift+5]*conversion[units],marker='.',markersize=10,color='g',label ="end")
    ax[1].plot(relative_dynamics_state[: ,shift+3]*conversion[units],relative_dynamics_state[: ,shift+5]*conversion[units])
    if np.all(waypoints != None) :
      for waypoint in waypoints :
        ax[1].plot(waypoint[3],waypoint[5],marker='^',markersize=10,color='k')
    
    ax[1].set_xlabel(" VX - radial       [{}/s]".format(units))
    ax[1].set_ylabel(" VZ - cross-track  [{}/s]".format(units))
   
   
    ax[1].grid(color = 'green', linestyle = '--', linewidth = 0.5)
    ax[1].legend()
    
    plt.tight_layout()

    figure,ax = plt.subplots(2,1)
    ax[0].plot(relative_dynamics_state[0 ,shift+0]*conversion[units],relative_dynamics_state[0 ,shift+1]*conversion[units],marker='*',markersize=10,color='r',label ="start")
    ax[0].plot(relative_dynamics_state[-1,shift+0]*conversion[units],relative_dynamics_state[-1,shift+1]*conversion[units],marker='.',markersize=10,color='g',label ="end")
    ax[0].plot(relative_dynamics_state[: ,shift+0]*conversion[units],relative_dynamics_state[: ,shift+1]*conversion[units])
    if np.all(waypoints != None) :
      for waypoint in waypoints :
        ax[0].plot(waypoint[0]*conversion[units],waypoint[1]*conversion[units],marker='^',markersize=10,color='k')
        
    ax[0].set_xlabel(" X - radial      [{}]".format(units))
    ax[0].set_ylabel(" Y - along-track [{}]".format(units))
    ax[0].set_title(title)
    
   
    ax[0].grid(color = 'green', linestyle = '--', linewidth = 0.5)
    ax[0].legend()
    

    ax[1].plot(relative_dynamics_state[0 ,shift+3]*conversion[units],relative_dynamics_state[0 ,shift+4]*conversion[units],marker='*',markersize=10,color='r',label ="start")
    ax[1].plot(relative_dynamics_state[-1,shift+3]*conversion[units],relative_dynamics_state[-1,shift+4]*conversion[units],marker='.',markersize=10,color='g',label ="end")
    ax[1].plot(relative_dynamics_state[: ,shift+3]*conversion[units],relative_dynamics_state[: ,shift+4]*conversion[units])
    if np.all(waypoints != None) :
      for waypoint in waypoints :
        ax[1].plot(waypoint[3],waypoint[4],marker='^',markersize=10,color='k')
        
    ax[1].set_xlabel(" VX - radial      [{}/s]".format(units))
    ax[1].set_ylabel(" VY - cross-track [{}/s]".format(units))
   
   
    ax[1].grid(color = 'green', linestyle = '--', linewidth = 0.5)
    ax[1].legend()
    

    plt.tight_layout()

    if save_output_at != None :
        try :
          figure.savefig(save_output_at, format='svg', dpi=1200)
        except :
            raise ("The figure was not saved due to an error.Check that the output directory exists")

    return figure,ax

    
    

def plot3d_projected_relative_orbit(relative_dynamics_state: np.ndarray,
                                    title: str ="relative orbit dynamics",
                                    save_output_at:str=None):   

    rows,cols = np.shape(relative_dynamics_state)
    if cols == 7 :
        # if you forgot to remove the time
        shift = 1
    else :
        shift = 0 
        
    fig = plt.figure()
    ax  = plt.axes(projection="3d")

    x_projection =  relative_dynamics_state[:,shift+0]
    y_projection =  relative_dynamics_state[:,shift+1]
    z_projection =  relative_dynamics_state[:,shift+2]

    xmax_coordinate = np.max(x_projection);xmin_coordinate = np.min(x_projection)
    ymax_coordinate = np.max(y_projection);ymin_coordinate = np.min(y_projection)
    zmax_coordinate = np.max(z_projection);zmin_coordinate = np.min(z_projection)
    
    expansion_factor = 2
    
    xmax_coordinate += expansion_factor*abs(xmax_coordinate)
    xmin_coordinate += -expansion_factor*abs(xmin_coordinate)
    ymax_coordinate += expansion_factor*abs(ymax_coordinate)
    ymin_coordinate += -expansion_factor*abs(ymin_coordinate)
    zmax_coordinate += expansion_factor*abs(zmax_coordinate)
    zmin_coordinate += -expansion_factor*abs(zmin_coordinate)   
   

    # full orbit
    ax.plot(x_projection,y_projection,z_projection,color="blue",label="trajectory")
    ax.plot(x_projection[0],y_projection[0],z_projection[0],color="red",marker="*",label="start")
    ax.plot(x_projection[-1],y_projection[-1],z_projection[-1],color="green",marker="o",label="end")
    
    
    # projections
    ax.plot(x_projection,y_projection,np.ones(np.shape(y_projection))*zmin_coordinate,color="green")
    ax.plot(x_projection,np.ones(np.shape(z_projection))*ymin_coordinate,z_projection,color="green")
    ax.plot(np.ones(np.shape(x_projection))*xmin_coordinate,y_projection,z_projection,color="green")



    ax.set_xlabel("radial      [m]")
    ax.set_ylabel("along-track [m]")
    ax.set_zlabel("cross-track [m]")
    ax.set_xlim([xmin_coordinate,xmax_coordinate])
    ax.set_ylim([ymin_coordinate,ymax_coordinate])
    ax.set_zlim([zmin_coordinate,zmax_coordinate])
    ax.set_title(title)
    
    ax.legend()

    if save_output_at != None :
        try :
          fig.savefig(save_output_at, format='svg', dpi=1200)
        except :
            raise ("The figure was not saved due to an error.Check that the output directory exists")

    return fig,ax


def plot3d_multiagent_relative_orbit(state_history_set: np.ndarray,
                                     labels           :list =    [] ,
                                     title            : str ="relative orbit dynamics",
                                     save_output_at:str=None):   
    
    num_agents = len(state_history_set)
    rows,cols = np.shape(state_history_set[0])
    if cols == 7. :
        # if you forgot to remove the time
        shift = 1
    else :
        shift = 0 

    if len(labels) ==0 :
        labels=["agent{}".format(num) for num in range(num_agents)]   

    fig = plt.figure()
    ax  = plt.axes(projection="3d")
    
    for jj in range(num_agents):
         
        x_projection =  state_history_set[jj][:,shift+0]
        y_projection =  state_history_set[jj][:,shift+1]
        z_projection =  state_history_set[jj][:,shift+2]
    
        # full orbit
        if not jj== (num_agents-1) :
            ax.plot(x_projection,y_projection,z_projection,color="blue",label=labels[jj])
            ax.plot(x_projection[0],y_projection[0],z_projection[0],color="red",marker="*")
            ax.plot(x_projection[-1],y_projection[-1],z_projection[-1],color="green",marker="o")
        else :
            ax.plot(x_projection,y_projection,z_projection,color="blue",label=labels[jj])
            ax.plot(x_projection[0],y_projection[0],z_projection[0],color="red",marker="*",label="start")
            ax.plot(x_projection[-1],y_projection[-1],z_projection[-1],color="green",marker="o",label="end")


    ax.set_xlabel("radial      [m]")
    ax.set_ylabel("along-track [m]")
    ax.set_zlabel("cross-track [m]")
    ax.set_title(title)   
    ax.legend()

    if save_output_at != None :
        try :
          fig.savefig(save_output_at, format='svg', dpi=1200)
        except :
            raise ("The figure was not saved due to an error.Check that the output directory exists")

    return fig,ax

    
   
def  plot_classic_keplerian_elements(time:np.ndarray,state_history:np.ndarray,time_unit:str="s",title:str="keplerian elements",save_output_at:str=None):
      
      
      
      label_left  = [r"$a$ [m]",r"$e$",r"$i$"]
      label_right = [r"$\omega$",r"$\Omega$",r"$\theta$"]
      
      fig,axes = plt.subplots(3,2)
      
      state_history_left = state_history[:,:3]
      state_history_right = state_history[:,3:]
      
      for jj in range(3):
            axes[jj,0].plot(time,state_history_left[:,jj])
            axes[jj,0].set_ylabel(label_left[jj])
            axes[jj,0].set_xlabel("time [" + time_unit + " ]")
            
            axes[jj,1].plot(time,state_history_right[:,jj])
            axes[jj,1].set_ylabel(label_right[jj])
            axes[jj,1].set_xlabel("time [" + time_unit + " ]")
      
      fig.suptitle(title)
      plt.tight_layout()
      if save_output_at != None :
        try :
          fig.savefig(save_output_at, format='svg', dpi=1200)
        except :
            raise ("The figure was not saved due to an error.Check that the output directory exists")
      
      return fig,axes

def  plot_implicit_keplerian_elements(time:np.ndarray,state_history:np.ndarray,time_unit:str="s",title:str="implicit keplerian state",save_output_at:str=None):
      
      label_left   = [r"$\bar{\theta}$  [rad]",r"$i$ [rad]",r"$\Omega$ [rad]"]
      label_right = [r"$h$",r"$v_r$",r"$r$"]
      
      fig,axes = plt.subplots(3,2)
      
      state_history_left = state_history[:,:3]
      state_history_right = state_history[:,3:]
      
      for jj in range(3):
            axes[jj,0].plot(time,state_history_left[:,jj])
            axes[jj,0].set_ylabel(label_left[jj])
            axes[jj,0].set_xlabel("time [" + time_unit + " ]")
            
            axes[jj,1].plot(time,state_history_right[:,jj])
            axes[jj,1].set_ylabel(label_right[jj])
            axes[jj,1].set_xlabel("time [" + time_unit + " ]")
        
      fig.suptitle(title)
      
      plt.tight_layout()

      if save_output_at != None :
        try :
          fig.savefig(save_output_at, format='svg', dpi=1200)
        except :
            raise ("The figure was not saved due to an error.Check that the output directory exists")
      
      return fig,axes


def plot3D_orbit(bodies_list:list,save_output_at:str=None) :
    
    """
    Parameters
    ----------
    
    bodies_list (list): list of dictionaries for each body
                        
                        "state_Ahistory" -> cartesian states history np.ndarray((7,)) or np.ndarray((6,))
                                            The additional initial column is the time
                                            that will be neglected in case is present
                        "name"           -> Name of the body
                                           
    Returns
    -------
    None
    
    Description
    -----------
    plots the 3D orbit all the bodies inserted in the cartesina_state_Ahistories list                          
                                       
    """
    
    fig = plt.figure()
    ax  = plt.axes(projection="3d")
    number_of_bodies = len(bodies_list)
    cmap = plt.get_cmap('gnuplot2')
    colors = [cmap(i) for i in np.linspace(0, 0.5, number_of_bodies)]

    for jj,body in enumerate(bodies_list) :
        
        rows,cols = np.shape(body["state_Ahistory"])
        if cols != 6 :
            body["state_Ahistory"] = body["state_Ahistory"][:,1:]
        
        x_projection =  body["state_Ahistory"][:,0]
        y_projection =  body["state_Ahistory"][:,1]
        z_projection =  body["state_Ahistory"][:,2]
        
        ax.plot(x_projection,y_projection,z_projection,label=body["name"],color=colors[jj])
        ax.plot(x_projection[0],y_projection[0],z_projection[0],color="red",marker="o",markersize=5)
    # xmax_coordinate = np.max(np.max(x_projection));xmin_coordinate = np.min(np.min(x_projection))
    # ymax_coordinate = np.max(np.max(y_projection));ymin_coordinate = np.min(np.min(y_projection))
    # zmax_coordinate = np.max(np.max(z_projection));zmin_coordinate = np.min(np.min(z_projection))
    
    # upper = max(xmax_coordinate,ymax_coordinate,zmax_coordinate)
    # lower = min(xmin_coordinate,ymin_coordinate,zmin_coordinate)
    
    ax.set_title("J2000 frame")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    
    # ax.set_xlim([upper*1.5,lower*1.5])
    # ax.set_ylim([upper*1.5,lower*1.5])
    # ax.set_zlim([upper*1.5,lower*1.5])
    
    ax.legend()
    if save_output_at != None :
        try :
          fig.savefig(save_output_at, format='svg', dpi=1200)
        except :
            raise ("The figure was not saved due to an error.Check that the output directory exists")
      
    
    return fig,ax

    


def split_history(state_history:dict,n_bodies:int):
    
    """
    Parameters
    -----------
    
    state_history        (dict ) :  dictionary containing the state_history
                                    of N propagated bodies.
                                    The dictionary is defined as a standard 
                                    state_history following the TUdat convention. 
                                    
                                    the keys of the dictionary are the time stamps 
                                    corresponding to each state and the argument is 
                                    the state of the body at the given time stamp
                                    
                                    
    n_bodies            (int)     : number of propagated bodies
    
    Returns
    -----------
    
    state_hostory_book (list)   : list of state_history dictionaries for each propagated body.
                                  The order of the list is given directly by the order 
                                  in which the bodies are propagated.
    
    Description
    -----------
    Creates a list of state histories based on the unified state_history
    from the propagation of n_bodies 
    
    """
    
    state_history_book = []
    time               = [key for key in state_history.keys() ]
    n_variables        = len(state_history[time[0]])
    step               = int(n_variables/n_bodies)
    
    for n in range(n_bodies) :
        current_history = dict()
        
        for t in time:
           current_history[t]= state_history[t][step*n:step*n+step]
        
        state_history_book.append(current_history)
    
    return state_history_book




def simulate_keplerian_dynamics_rk4(bodies2simulate_list  : list,
                                    simulation_duration   : float,
                                    time_step             : float,
                                    activate_j2           : bool = False) :

    """
    Parameters
    -----------
    
    bodies2simulate_list (list)    :  list of dictionaries containing the following specification
                                      spaceraft[name]               = "name"
                                      spaceraft[initial_conditions] = np.ndarray(6,) # initial inertial state conditions
    simulation_start_epoch (float) : start epoch of the simulation
    simulation_end_epoch (float)   : end epoch of the simulation
    time_step            (flaot)   : time step for (runge kutta fixed step)                
    activate_j2          (bool)    : activate J2 harmonic term during state propagation
    
    
    
    Returns
    -----------
    
    results_per_object dict()      : dictionary with keys equals to spacecrafts name and
                                     items equal to a second dictionary with following specifications
                                     
                                    results_per_object["spacecraft name"]
                                         -spacecraft ["state_history"]      =  propagated state history np.ndarray(N,6)
                                         -spacecraft ["initial_conditions"] =  initial conditions of the spaceraft
                                         
                                    note N = int(time_end-time_end)/time_step
                                     
    
    
    Description
    -----------
    returns state history of a set of propagated bodies aroud the Earth using a simple 
    RK4 fixed step intergator
    
    """
    
    
    simulation_start_epoch = 0
    simulation_end_epoch =   simulation_start_epoch + simulation_duration 
    
    # Load SPICE.
    spice.load_standard_kernels() 

    # Create settings for celestial bodies
    bodies_to_create         = ["Earth"]   # this must have a list of all the planets to create
    global_frame_origin      = "Earth"             # this is the origin of the refernce system
    global_frame_orientation = "J2000"           # orinetation of the reference system (it is not ECI.it is ECi only at january 2000)

    # create a bodies
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation) # body settings taken from SPICE.

    # Create environment
    bodies = environment_setup.create_system_of_bodies(body_settings)


    # create full nonlinear simulation set up
    # Create vehicle object
    
    # todo : specilise for each spacecraft. Now all spaceraft are the same
    
    for vehicle in bodies2simulate_list :
       bodies.create_empty_body( vehicle["name"] )
       bodies.get_body( vehicle["name"]  ).mass = 2000.0

       
    bodies_to_propagate = [vehicle["name"] for vehicle in bodies2simulate_list]
    central_bodies      = ["Earth"]*len(bodies2simulate_list) 
    
    
    if not activate_j2 :
        acceleration_settings_for_spacecrafts = dict(
            Earth =[propagation_setup.acceleration.point_mass_gravity()]
        )
        
    else :
        acceleration_settings_for_spacecrafts = dict(
            Earth =[propagation_setup.acceleration.spherical_harmonic_gravity(2,0)]
        )
        # earth_normalized_c20   = body_settings.get( 'Earth' ).gravity_field_settings.normalized_cosine_coefficients[2,0]
        # earth_j2               = -earth_normalized_c20 * np.sqrt(5)
    
    acceleration_settings = {vehicle["name"] : acceleration_settings_for_spacecrafts for vehicle in bodies2simulate_list}
    acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    initial_conditions_list = tuple((vehicle["initial_conditions"] for vehicle in bodies2simulate_list))
    initial_system_state = np.hstack( initial_conditions_list)
   
    # Create propagation settings for the two cases
    termination_settings = propagation_setup.propagator.time_termination( simulation_end_epoch )

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_system_state ,
        termination_settings
    )

    fixed_step_size = time_step #s
    integrator_settings = propagation_setup.integrator.runge_kutta_4(
        simulation_start_epoch,
        fixed_step_size
    )

    dynamics_simulator = numerical_simulation.SingleArcSimulator(
        bodies, integrator_settings, propagator_settings)


    simulation_result_list        = split_history(dynamics_simulator.state_history,len(bodies2simulate_list))
    simulation_result_list        = [result2array(simulation_result) for simulation_result in simulation_result_list]
    results_per_object            = {}
      
    for jj,vehicle in enumerate(bodies2simulate_list) :
        
        result = {}
        result["state_history"]             =  simulation_result_list[jj]
        result["initial_conditions"]        =  vehicle["initial_conditions"]
        results_per_object[vehicle["name"]] = result
        
        
    return results_per_object


def vector2matrix(flat_matrix:np.ndarray) :
    
    """
    Parameters
    -----------
    
    flat_matrix     (np.ndarray ) : vector containing a flatttened rotation matrix 
                                    following the TuDat standards
                                    a rotation matrix is returned as a nine-entry 
                                    vector in the dependent variable output, where entry
                                    (i,j) of the matrix is stored in entry (3i+j)  of the vector
                                    with i,j = 0,1,2
                                    
    
                                    
    
    Returns
    -----------
    
    rotation_matrix      (np.ndarray)     : returns (3x3) Rotation matrix 
                                            (orthogonal matrix)
                                    
                                   
    
    Descriptiom
    -----------
    
    Retruns rotation matrix from its flattened vectorial version.
    Check for details :
    https://tudatpy.reaDthedocs.io/en/latest/dependent_variable.html#tudatpy.numerical_simulation.propagation_setup.dependent_variable.inertial_to_body_fixed_rotation_frame
    
    """
    
    rotations_matrix = np.empty((3,3))
    
    for i in range(3) :
        for j in range(3) :
           rotations_matrix[i,j]=flat_matrix[3*i+j]
   
    return rotations_matrix

def compute_inertial2rsw_DCM_from_inertial_state(cartesian_state:np.ndarray) :
    """
    Parameters
    -----------
    cartesian_state (np.ndarray(6,)) : cartesian state of orbiting body
    
    Returns
    -----------
    inertial2rsw_DCM      (np.ndarray)     : returns (3x3) DCM matrix (not a rotation)
                                             (orthogonal matrix)
                                    
    Description
    -----------
    
    Retruns DCM matrix from inertial coordinates to rsw coordinates
    
    """
    
    position = cartesian_state[:3]
    velocity = cartesian_state[3:]
    
    angular_momentum = np.cross(position,velocity)
    cross_track_direction = angular_momentum/np.linalg.norm(angular_momentum)
    radial_direction      = position/np.linalg.norm(position)
    along_track_direction = np.cross(cross_track_direction,radial_direction)
    inertial2rsw_DCM      = np.vstack((radial_direction,along_track_direction,cross_track_direction))
    
    return inertial2rsw_DCM
    

def obtain_chief_reference_dynamics(initial_keplerian_state_chief : np.ndarray,
                                    simulation_duration          : float,
                                    time_step                     : float,
                                    max_spherical_harmonics_order : int =0,
                                    aerodynamic_settings_chief   : AerodynamicSettings = None,) :
  

  
    simulation_start_epoch = 0
    simulation_end_epoch   = simulation_start_epoch + simulation_duration
    # Load SPICE.
    spice.load_standard_kernels() 

    # Create settings for celestial bodies
    bodies_to_create         = ["Earth"]   # this must have a list of all the planets to create
    global_frame_origin      = "Earth"             # this is the origin of the refernce system
    global_frame_orientation = "J2000"           # orinetation of the reference system

    # create a bodies
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation) # body settings taken from SPICE.
    
    if not  aerodynamic_settings_chief == None :
          density_scale_height = aerodynamic_settings_chief.scale_height
          density_at_zero_altitude =  aerodynamic_settings_chief.zero_level_density
          # Define the density as a function of altitude [in m]
          def density_function(h):
            # Return the density according to a modified exponential model
            return density_at_zero_altitude * np.exp(-(h-7128)/density_scale_height)
           
           # Define parameters for constant temperature and composition
          constant_temperature  = 250.0
          specific_gas_constant = 300.0
          ratio_of_specific_heats = 1.4
           # Create the custom constant temperature atmosphere settings
          custom_density_settings = environment_setup.atmosphere.custom_constant_temperature(
            density_function,
            constant_temperature,
            specific_gas_constant,
            ratio_of_specific_heats)
        # Add the custom density to the body settings of "Earth"
          body_settings.get("Earth").atmosphere_settings = custom_density_settings
      
    # Create environment
    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies.create_empty_body( "chief" )
    # Create vehicle object
    if not  aerodynamic_settings_chief == None :
        bodies.get_body( "chief" ).mass = aerodynamic_settings_chief.mass
        # Create aerodynamic coefficients interface (drag-only; zero side force and lift)
        reference_area = aerodynamic_settings_chief.Across
        drag_coefficient = aerodynamic_settings_chief.drag_coefficient
        

        aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area,[ drag_coefficient, 0.0 , 0.0 ] )
        environment_setup.add_aerodynamic_coefficient_interface(
                  bodies, "chief", aero_coefficient_settings )
    else :

        bodies.get_body( "chief"  ).mass = 2000
    
    bodies_to_propagate = ["chief"]
    central_bodies      = ["Earth"]
    
    mu_earth               = body_settings.get( 'Earth' ).gravity_field_settings.gravitational_parameter
    earth_normalized_c20   = body_settings.get( 'Earth' ).gravity_field_settings.normalized_cosine_coefficients[2,0]
    earth_j2               = -earth_normalized_c20 * np.sqrt(5)
    
    if not max_spherical_harmonics_order == 0 :                    
        acceleration_settings_on_chief = dict(
            Earth =[propagation_setup.acceleration.spherical_harmonic_gravity(max_spherical_harmonics_order ,0)]
        )
       
    else :
        acceleration_settings_on_chief = dict(
            Earth =[propagation_setup.acceleration.spherical_harmonic_gravity(0,0)]
        )
        
    # add aerodynamic interface
    if not  aerodynamic_settings_chief == None :
         acceleration_settings_on_chief["Earth"].append(propagation_setup.acceleration.aerodynamic())

    
    acceleration_settings = {"chief" : acceleration_settings_on_chief}
    acceleration_models  = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies)
    
    # save variables 
    dependent_variables_to_save = [
              propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration(
                       "chief","Earth",[(0,0)]),
              propagation_setup.dependent_variable.total_acceleration("chief")]
    
    #! IN CASE YOUR NUMERICAL SOLUTION AND YOUR ANALYTICAL SOLUTION are diverging 
    #! plot the density and see if it mateches your previsions. TuDat is using the real planet
    #! specifications to compute the altitude while you assume a round earth.

    # dependent_variables_to_save = [
    #           propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration(
    #                    "chief","Earth",[(0,0)]),
    #           propagation_setup.dependent_variable.total_acceleration("chief"),
    #           propagation_setup.dependent_variable.density("chief","Earth")]
                    
    # initial conditions convertion
    chief_initial_cartesian_state = element_conversion.keplerian_to_cartesian(keplerian_elements=initial_keplerian_state_chief,
                                                                              gravitational_parameter=mu_earth)
    # Create propagation settings for the two cases
    termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        chief_initial_cartesian_state ,
        termination_settings,
        output_variables = dependent_variables_to_save,
    )


    fixed_step_size     = time_step #s
    integrator_settings = propagation_setup.integrator.runge_kutta_4(
        simulation_start_epoch,
        fixed_step_size
    )

    dynamics_simulator = numerical_simulation.SingleArcSimulator(
        bodies, integrator_settings, propagator_settings)
    
    # obtain all the necessary information
    state_history               = dynamics_simulator.state_history
    dependent_variables_history = dynamics_simulator.dependent_variable_history
    

    keplerian_state_history              = {}
    perturbing_accelerations_rsw_history = {}
    inertial2rsw_DCM_history             = {}
    angular_velocity_rsw_frame_history   = {}
    
    for t in state_history.keys() :
        # obtain keplerian state 
        keplerian_state_history[t]= element_conversion.cartesian_to_keplerian(cartesian_elements=state_history[t] ,
                                                                               gravitational_parameter=mu_earth)
        implicit_keplerian_state = classic2implicit_keplerian_elements(keplerian_state_history[t],mu = mu_earth)    
        
        # obtain DCM inertial2rsw
        inertial2rsw                   = compute_inertial2rsw_DCM_from_inertial_state(state_history[t])
        inertial2rsw_DCM_history[t]    = inertial2rsw 
        
        # obtain perturbing_acceleration
        point_mass_gravity_acceleration     = dependent_variables_history[t][:3]
        total_acceleration                  = dependent_variables_history[t][3:6]
        
        #! PRINT THIS TO CHECK IF YOUR NUMERICAL SOLUTION DIVERGES A LOT FROM THE SYMBOLIC ONE
        # density = dependent_variables_history[t][-1]

        perturbing_accelerations_inertial       = total_acceleration - point_mass_gravity_acceleration
        perturbing_accelerations_rsw            = np.matmul(inertial2rsw,perturbing_accelerations_inertial)
        perturbing_accelerations_rsw_history[t] = perturbing_accelerations_rsw   
        # obtain angular speed in rsw frame
        
        r = implicit_keplerian_state[-1] # distance from the earth


        h = implicit_keplerian_state[3]  # angular momentum
        f_w = perturbing_accelerations_rsw[-1]
        omega_r = r/h * f_w
        omega_s = 0
        omega_w = h/r**2
        omega_rsw_frame = np.array([omega_r,omega_s,omega_w])
        
        angular_velocity_rsw_frame_history[t] = omega_rsw_frame
  
    result_dictionary = {}
    result_dictionary["state_Dhistory"]                        = state_history
    result_dictionary["keplerian_state_Dhistory"]              = keplerian_state_history             
    result_dictionary["perturbing_accelerations_rsw_Dhistory"] = perturbing_accelerations_rsw_history
    result_dictionary["inertial2rsw_DCM_Dhistory"]             = inertial2rsw_DCM_history             
    result_dictionary["angular_velocity_rsw_frame_Dhistory"]   = angular_velocity_rsw_frame_history  
    
        
    return result_dictionary

def rsw2inertial_state_transformation(relative_state_deputy_rsw,rsw2inertial_DCM,angular_velocity_rsw_wrt_inertial):
    
    relative_position_rsw =  relative_state_deputy_rsw[:3]
    relative_velocity_rsw =  relative_state_deputy_rsw[3:]
    
    relative_position_inertial             = np.matmul(rsw2inertial_DCM,relative_position_rsw)
    relative_velocity_inertial_in_rsw_coordinates  = relative_velocity_rsw + np.cross(angular_velocity_rsw_wrt_inertial,relative_position_rsw)
    relative_velocity_inertial             = np.matmul(rsw2inertial_DCM,relative_velocity_inertial_in_rsw_coordinates)
    relative_initial_state_deputy_inertial = np.hstack((relative_position_inertial,relative_velocity_inertial))
    
    return relative_initial_state_deputy_inertial

def inertial2rsw_state_transformation(relative_state_deputy_inertial,inertial2rsw_DCM,angular_velocity_inertial_wrt_rsw):
    
    relative_position_inertial =  relative_state_deputy_inertial[:3]
    relative_velocity_inertial =  relative_state_deputy_inertial[3:]
    
    relative_position_rsw             = np.matmul(inertial2rsw_DCM,relative_position_inertial)
    angular_velocity_inertial_wrt_rsw_in_inertial_coordinates = np.matmul(inertial2rsw_DCM.T,angular_velocity_inertial_wrt_rsw)
    relative_velocity_rsw_in_inertial_coordinates             = relative_velocity_inertial + np.cross(angular_velocity_inertial_wrt_rsw_in_inertial_coordinates,relative_position_inertial)
    relative_velocity_rsw             = np.matmul(inertial2rsw_DCM,relative_velocity_rsw_in_inertial_coordinates)
    relative_initial_state_deputy_rsw = np.hstack((relative_position_rsw,relative_velocity_rsw))
    
    return relative_initial_state_deputy_rsw


def simulate_relative_dynamics_rk4( initial_relative_state_deputy_rsw: np.ndarray,
                                    initial_keplerian_state_chief: np.ndarray,
                                    simulation_duration: float,
                                    time_step              : float,
                                    max_spherical_harmonics_order: int =0,
                                    aerodynamic_settings_chief  :AerodynamicSettings = None,
                                    aerodynamic_settings_deputy :AerodynamicSettings = None,) :
   

    """
    Parameters
    -----------
    
    initial_relative_state_deputy (np.ndarray) : initial relative state deputy in rsw
    initial_keplerian_state_chief (np.ndarray) : initial keplerian state chief
    simulation_start_epoch (float) : start epoch of the simulation
    simulation_end_epoch (float)   : end epoch of the simulation
    time_step            (flaot)   : time step for (runge kutta fixed step)                
    activate_j2          (bool)    : activate J2 harmonic term during state propagation
    
    
    
    Returns
    -----------
    
    results_per_obejct dict()      : dictionary with keys equals to spacecrafts name and
                                     items equal to a second dictionary with following specifications
                                     
                                    results_per_obejct["spacecraft name"]
                                         -spacecraft ["state_history"]      =  propagated state history np.ndarray(N,6)
                                         -spacecraft ["initial_conditions"] =  initial conditions of the spaceraft
                                         
                                    note N = int(time_end-time_end)/time_step
                                     
    
    
    Descriptiom
    -----------
    returns raltative state history of deputy in rsw frame propagated around the Earth using a simple 
    RK4 fixed step intergator
    
    """
    # check aerodyamic options consistency
    simulation_start_epoch = 0
    simulation_end_epoch   = simulation_start_epoch + simulation_duration

    if  ((aerodynamic_settings_chief==None) ^ (aerodynamic_settings_deputy==None)) :
        message = ("aerodynamic settings must be given for chief and deputy in case"
                   "one of the two is given.\n Add settings for both deputy and chief")
        raise ValueError(message)
    

    # Load SPICE.
    spice.load_standard_kernels() 

    # Create settings for celestial bodies
    bodies_to_create         = ["Earth"]   # this must have a list of all the planets to create
    global_frame_origin      = "Earth"             # this is the origin of the refernce system
    global_frame_orientation = "J2000"           # orinetation of the reference system

    # create a bodies
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation) # body settings taken from SPICE.

    # Create environment
    bodies = environment_setup.create_system_of_bodies(body_settings)
    # Add  exponential atmosphere
    if not  aerodynamic_settings_deputy == None :
        density_scale_height = aerodynamic_settings_deputy.scale_height
        density_at_zero_altitude =  aerodynamic_settings_deputy.zero_level_density
        #  body_settings.get( "Earth" ).atmosphere_settings = environment_setup.atmosphere.exponential( 
        # density_scale_height, density_at_zero_altitude)


        def density_function(h):
            # Return the density according to a modified exponential model
            return density_at_zero_altitude * np.exp(-(h-7128)/density_scale_height)
            
        # Define parameters for constant temperature and composition
        constant_temperature  = 250.0
        specific_gas_constant = 300.0
        ratio_of_specific_heats = 1.4
        # Create the custom constant temperature atmosphere settings
        custom_density_settings = environment_setup.atmosphere.custom_constant_temperature(
        density_function,
        constant_temperature,
        specific_gas_constant,
        ratio_of_specific_heats)

        # Add the custom density to the body settings of "Earth"
        body_settings.get("Earth").atmosphere_settings = custom_density_settings

    # Create environment
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # create full nonlinear simulation set up
    # Create vehicle object
    
    bodies.create_empty_body("deputy" )
    

    
    # propagate state chief first
    bodies_to_propagate = ["deputy"]
    central_bodies      = ["Earth"]

    if not max_spherical_harmonics_order == 0 :
        acceleration_settings_for_spacecrafts = dict(
            Earth =[propagation_setup.acceleration.spherical_harmonic_gravity(max_spherical_harmonics_order,0)]
        )
    else :
        acceleration_settings_for_spacecrafts = dict(
            Earth =[propagation_setup.acceleration.spherical_harmonic_gravity(0,0)]
        )
    
    # add aerodynamic interface
    if not  aerodynamic_settings_deputy == None :
        bodies.get_body( "deputy" ).mass = aerodynamic_settings_deputy.mass
        # Create aerodynamic coefficients interface (drag-only; zero side force and lift)
        reference_area = aerodynamic_settings_deputy.Across
        drag_coefficient = aerodynamic_settings_deputy.drag_coefficient
        aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area,[ drag_coefficient, 0.0 , 0.0 ] )
        environment_setup.add_aerodynamic_coefficient_interface(
                bodies, "deputy", aero_coefficient_settings )
    
        acceleration_settings_for_spacecrafts["Earth"].append( propagation_setup.acceleration.aerodynamic())

    
    acceleration_settings = {"deputy" : acceleration_settings_for_spacecrafts}
    acceleration_models   = propagation_setup.create_acceleration_models(
                             bodies, acceleration_settings, bodies_to_propagate, central_bodies)
    
    result_dict= obtain_chief_reference_dynamics( initial_keplerian_state_chief,
                                                  simulation_duration= simulation_duration,
                                                  time_step  = time_step,
                                                  max_spherical_harmonics_order=max_spherical_harmonics_order,
                                                  aerodynamic_settings_chief =aerodynamic_settings_chief ) 
    
    
    chief_state_Dhistory                   = result_dict["state_Dhistory"]                             
    perturbing_accelerations_rsw_Dhistory  = result_dict["perturbing_accelerations_rsw_Dhistory"] 
    inertial2rsw_DCM_Dhistory              = result_dict["inertial2rsw_DCM_Dhistory"]              
    angular_velocity_rsw_frame_Dhistory    = result_dict["angular_velocity_rsw_frame_Dhistory"]   
    
    # create inertial initial condition
    initial_inertial_deputy_relative_state = rsw2inertial_state_transformation(initial_relative_state_deputy_rsw,
                                                                               inertial2rsw_DCM_Dhistory[simulation_start_epoch].T,
                                                                               angular_velocity_rsw_frame_Dhistory[simulation_start_epoch],
                                                                               )
    initial_inertial_deputy_state = chief_state_Dhistory[simulation_start_epoch] + initial_inertial_deputy_relative_state
    
    # Create propagation settings for the two cases
    termination_settings = propagation_setup.propagator.time_termination( simulation_end_epoch )

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_inertial_deputy_state,
        termination_settings
    )

    fixed_step_size = time_step #s
    integrator_settings = propagation_setup.integrator.runge_kutta_4(
        simulation_start_epoch,
        fixed_step_size
    )

    dynamics_simulator = numerical_simulation.SingleArcSimulator(
        bodies, integrator_settings, propagator_settings)

    deputy_state_Dhistory  = dynamics_simulator.state_history
    
    # now convert everything to rsw again
    deputy_relative_state_rsw_Dhistory = {}
    deputy_kstate_Dhistory             = {}
    mu_earth=body_settings.get( 'Earth' ).gravity_field_settings.gravitational_parameter
    
    for t in deputy_state_Dhistory.keys() :
    
       deputy_relative_state_inertial       = deputy_state_Dhistory[t]-chief_state_Dhistory[t]
       deputy_relative_state_rsw_Dhistory[t] = inertial2rsw_state_transformation(deputy_relative_state_inertial,
                                                                    inertial2rsw_DCM_Dhistory[t],
                                                                    -angular_velocity_rsw_frame_Dhistory[t])
       deputy_kstate_Dhistory[t]    = element_conversion.cartesian_to_keplerian(deputy_state_Dhistory[t], gravitational_parameter=mu_earth)
    
    # They are all Dhistories
    simulation_results = {}
    simulation_results["chief_keplerian_state_Dhistory"]     = result_dict["keplerian_state_Dhistory"]     
    simulation_results["chief_inertial_state_Dhistory"]      = result_dict["state_Dhistory"]          
    simulation_results["deputy_inertial_state_Dhistory"]     = deputy_state_Dhistory                   
    simulation_results["deputy_relative_state_rsw_Dhistory"] = deputy_relative_state_rsw_Dhistory          
    simulation_results["deputy_keplerian_state_Dhistory"]    = deputy_kstate_Dhistory                   
    
    return simulation_results

    
def obtain_PRO_trajectory_from_geometry(rhox:float,rhoz:float,rhoy0:float,alphax:float,alphaz:float,mean_motion:float):
    """
    Parameters :
    
        rhox    (float): x-axis amplitude/ y-axis 2*x-axis amplitude [m]
        rhoz    (float): z-axis amplitude [m]
        rhoy0   (float): initial y-axis offset [m]
        alphax  (float): x/y-axis phase [rad]
        alphaz  (float): z-axis phase   [rad]
        mean_motion       (float): angular speed of the reference circular orbit [rad/s]
    
    Returns :
    
        PRO_trajectory (casadi.Fuction) : trajectory parameterised as function of time
                                          PRO_trajectory[t] -> [x(t),y(t),z(t)] RSW frame coordinates
    
    Description
        
        Returns casadi function to compute analytically the Passive Relative Orbit 
        given initial geometric parametrization of the relative orbit
    """
    
    t = ca.MX.sym("t")
    x = rhox*ca.sin(mean_motion*t+alphax)
    y = rhoy0 + 2*rhox*ca.cos(mean_motion*t+alphax)
    z = rhoz*ca.sin(mean_motion*t+alphaz)
    
    vx = mean_motion*rhox*ca.cos(mean_motion*t+alphax)
    vy = -2*mean_motion*rhox*ca.sin(mean_motion*t+alphax)
    vz = mean_motion* rhoz*ca.cos(mean_motion*t+alphaz)
    
    state = ca.vertcat(x,y,z,vx,vy,vz)
    PRO_trajectory = ca.Function("PRO_trajectory",[t],[state])
    
    return PRO_trajectory
    
def obtain_PRO_trajectory_from_initial_condions(initail_state_rsw:np.ndarray,mean_motion:float):
    """
    Parameters :
    
        initail_state_rsw (float) : initial relative state in RSW frame
        mean_motion                 (float) : angular speed of the reference circular orbit [rad/s]
    
    Returns :
    
        PRO_trajectory (casadi.Fuction) : trajectory parameterised as function of time
                                          PRO_trajectory[t] -> [x(t),y(t),z(t)] RSW frame coordinates
    
    Description
        
        Returns casadi function to compute analytically the Passive Relative Orbit 
        given initial geometric parametrization of the relative orbit
    """
    
    t = ca.MX.sym("t")
    try :
      x0,y0,z0,vx0,vy0,vz0 = initail_state_rsw
    except ValueError:
        message = ("make sure your state has the correct size\n"
                   "Now it is {},but it should be (6,)".format(np.shape(initail_state_rsw)))
        raise ValueError(message)
    
    
    x = (4*x0 + 2*vy0/mean_motion) + vx0/mean_motion*ca.sin(mean_motion*t) - (3*x0 + 2*vy0/mean_motion)*ca.cos(mean_motion*t)
    y = -(6*mean_motion*x0 + 3*vy0)*t + (y0- 2*vx0/mean_motion) + (6*x0 + 4*vy0/mean_motion)*ca.sin(mean_motion*t) + 2*vx0/mean_motion*ca.cos(mean_motion*t)
    z = vz0/mean_motion*ca.sin(mean_motion*t) + z0*ca.cos(mean_motion*t)
    
    vx = vx0*ca.cos(mean_motion*t) + (3*x0*mean_motion + 2*vy0)*ca.sin(mean_motion*t)
    vy = -(6*mean_motion*x0 + 3*vy0) + (6*x0*mean_motion + 4*vy0)*ca.cos(mean_motion*t) - 2*vx0*ca.sin(mean_motion*t)
    vz = vz0*ca.cos(mean_motion*t) - mean_motion*z0*ca.sin(mean_motion*t)
    
    state = ca.vertcat(x,y,z,vx,vy,vz)
    PRO_trajectory = ca.Function("PRO_trajectory",[t],[state])
    
    return PRO_trajectory

def visualize_docking_cone_approach(time_grid:np.ndarray,state_history:np.ndarray,docking_port_trajectory:ca.Function,cone_angle:float,docking_port_distance_from_COM:float) :
    
    
    fig = plt.figure()
    docking_port_width = np.tan(cone_angle)*docking_port_distance_from_COM
    line_coeff         = np.tan(cone_angle)
    max_later_distance_from_center = 20

    x_range   = np.linspace(0,max_later_distance_from_center,20)
    up_y      = np.array([x*line_coeff for x in x_range])
    down_y    = np.array([-x*line_coeff for x in x_range])
    

    ax = fig.add_subplot(111)
    ax.plot(x_range,up_y,c="r",linewidth=3)
    ax.plot(x_range,down_y,c="r",linewidth=3)
    ax.add_patch( Rectangle((-docking_port_width/2,-docking_port_width/2),
                             docking_port_width, docking_port_width,color ='yellow'))
    
    # transform the position

    position_agent = state_history[:,:3]
    position_transformed = np.zeros(np.shape(position_agent))

    # create rotation matrix for each time step
    # this function works only for 2D case. So the z direction is left unchanged
    for jj,t in enumerate(time_grid) :
        docking_port_position       = np.squeeze(docking_port_trajectory(t)[:3])
        docking_port_direction      = docking_port_position/np.linalg.norm(docking_port_position)
        orthogonal_port_direction   = np.array([-docking_port_direction[1],docking_port_direction[0],docking_port_direction[2]]) 
        z_dir = np.array([0,0,1])
        dcm_rsw_2_docking_port   = np.vstack((docking_port_direction,orthogonal_port_direction,z_dir))
        position_transformed[jj,:] = np.squeeze(np.matmul(dcm_rsw_2_docking_port,position_agent[jj,:][:,np.newaxis]))
    
    ax.plot(position_transformed[:,0],position_transformed[:,1])
    
    ax.set_aspect("equal")




def obtain_rotational_state_trajectory() :
    
    t = ca.MX.sym("t")
    q0   = ca.MX.sym("q0") 
    q1   = ca.MX.sym("q1")
    q2   = ca.MX.sym("q2")
    q3   = ca.MX.sym("q3")
    quaternion = ca.vertcat(q0,q1,q2,q3) 
    
    quaternion_trajectory = ca.Function("PRO_trajectory",[t],[ca.vertcat(1,0,0,0,0,0,0)])
    return quaternion_trajectory 
    
def obtain_state_Dhistory_from_trajectory(trajectory_function:ca.Function,time_vector:np.ndarray):
    
    state_Dhistory = {}
    
    for t in time_vector:
        state_Dhistory[t] = np.squeeze(np.asarray(trajectory_function(t)))
        
    return  state_Dhistory
        
def obtain_state_Ahistory_from_trajectory(trajectory_function:ca.Function,time_vector:np.ndarray):
    
    dim = len(np.squeeze(np.asarray(trajectory_function(0))))
    state_Ahistory = np.empty((len(time_vector),dim))

    for jj,t in enumerate(time_vector):
        state_Ahistory[jj,:] = np.squeeze(np.asarray(trajectory_function(t)))
    
    return  state_Ahistory

       
def history2array(state_Dhistory) :
    rows           =  len(state_Dhistory) 
    state_Ahistory = np.zeros((rows,7))

    for jj,t in enumerate(state_Dhistory.keys()) :
        
        state_Ahistory[jj,1:] = state_Dhistory[t]
        state_Ahistory[jj,0]  = t
    
    return state_Ahistory
        
def stream_classic_keplerian_state(keplerian_elements,output_file_name:str=None) :
    """
    Parameters
    -----------
    
    keplerian_elements    (np.ndarray(6,1))  : classic keplerian elements vectors
                                               The units are SI and the following order 
                                               must be followed 
                                               
                                               a -> semimajor axis
                                               e -> eccentricity
                                               i -> inclination
                                               omega -> argument of periapsis
                                               Raan  -> Right Ascension of the Ascending Node
                                               theta -> True Anomaly
    
                                       
    Returns
    -----------
    None
                                                            
    Descriptiom
    -----------
    print keplerian state to consol for fast checks
    
    """
    
    try :
      a,e,i,omega,Raan,theta = keplerian_elements 
    except ValueError :
        raise ValueError("Check the size of you input. It should be (6,) but it is {}".format(np.shape(keplerian_elements)))
    
    if output_file_name== None :
        print("\n Keplerian state \n")
        print("---------------------------------")
        print("semi-major axis : {} [m]".format(a))
        print("eccentricity    : {} ".format(e))
        print("inclination     : {} [deg]".format(i*180/np.pi))
        print("omega           : {} [deg]".format(omega*180/np.pi))
        print("Raan            : {} [deg]".format(Raan*180/np.pi))
        print("true anomaly    : {} [deg]".format(theta*180/np.pi))
        print("---------------------------------")
    else :
        with open(output_file_name,'w+') as f:
            f.write("\n Keplerian state \n")
            f.write("---------------------------------\n")
            f.write("semi-major axis : {} [m]  \n".format(a))
            f.write("eccentricity    : {}      \n".format(e))
            f.write("inclination     : {} [deg]\n".format(i*180/np.pi))
            f.write("omega           : {} [deg]\n".format(omega*180/np.pi))
            f.write("Raan            : {} [deg]\n".format(Raan*180/np.pi))
            f.write("True anomaly    : {} [deg]\n".format(theta*180/np.pi))
            f.write("---------------------------------")


def obtain_inplane_circular_trajectory(radius,angular_speed,initial_phase_angle) :

    t          = ca.MX.sym("t")
    pos        = ca.vertcat(radius*ca.cos(angular_speed*t+initial_phase_angle),radius*ca.sin(angular_speed*t+initial_phase_angle),0) 
    vel        = ca.vertcat(-angular_speed*radius*ca.sin(angular_speed*t+initial_phase_angle),angular_speed*radius*ca.cos(angular_speed*t+initial_phase_angle),0)
    state      = ca.vertcat(pos,vel)
    trajectory = ca.Function("circular_trajectory",[t],[state])
    
    return trajectory

def obtain_constant_angular_velocity_trajectory(angular_speed:float,direction:np.ndarray) :
    t  = ca.MX.sym("t")
    
    if not np.shape(direction) == (3,) :
        message = "incorrect direction vectors size. Expected {}, Given {}".format((3,),np.shape(direction))
        raise ValueError(message)
    
    if abs(np.linalg.norm(direction)-1) >= 10**-6 :
        message = "norm of direction vector is not unitary. Current vector norm is {}".format(np.linalg.norm(direction))
        raise ValueError(message)
    
    omega = ca.vertcat(angular_speed*direction[0],angular_speed*direction[1],angular_speed*direction[2])
    angular_velocity_trajectory = ca.Function("angular_velocity_constant_trajectory",[t],[omega])

    return angular_velocity_trajectory



def create_random_picosats_coordinates(number_of_picosats:int = 1,sampling_box:np.ndarray=np.array([[-1,1],[-1,1]]),min_dist:float=0.2) :
    
   x_min = sampling_box[0][0]
   x_max = sampling_box[0][1]
   y_min = sampling_box[1][0]
   y_max = sampling_box[1][1]

   x_width = x_max - x_min
   y_width = y_max - y_min
   scale = np.array([x_width,y_width]) 
   shift = np.array([x_min,y_min]) 
   positions = np.ones((number_of_picosats,2))*np.infty
   counter = 0
   safe_top = 0 

   while counter  < number_of_picosats and safe_top <1000 :
      
      pos      =  np.random.random_sample((1,2))*scale +shift

      dist     = scipy.spatial.distance.cdist(positions[:counter,:],pos )
      
      if not np.any(dist<min_dist) :
        positions[counter,:] = pos  
        counter+=1
      
      safe_top+=1
      if safe_top >1000 :
          message = ("Maximum iterations reached. Try to reduce minimum distance")
          raise ValueError(message)

   return positions


def create_random_picosats_coordinates3D(number_of_picosats:int = 1,sampling_box:np.ndarray=np.array([[-1,1],[-1,1],[-1,1]]),min_dist:float=0.2) :
    
   x_min = sampling_box[0][0]
   x_max = sampling_box[0][1]
   y_min = sampling_box[1][0]
   y_max = sampling_box[1][1]
   z_min = sampling_box[2][0]
   z_max = sampling_box[2][1]

   x_width = x_max - x_min
   y_width = y_max - y_min
   z_width = z_max - z_min

   scale = np.array([x_width,y_width,z_width]) 
   shift = np.array([x_min,y_min,z_min]) 
   positions = np.ones((number_of_picosats,3))*np.infty
   counter = 0
   safe_top = 0 

   while counter  < number_of_picosats and safe_top <1000 :
      
      pos      =  np.random.random_sample((1,3))*scale +shift

      dist     = scipy.spatial.distance.cdist(positions[:counter,:],pos )
      
      if not np.any(dist<min_dist) :
        positions[counter,:] = pos  
        counter+=1
      
      safe_top+=1
      if safe_top >1000 :
          message = ("Maximum iterations reached. Try to reduce minimum distance")
          raise ValueError(message)

   return positions



def obtain_LQR_cost_matrix(target_positions:np.ndarray,start_positons:np.ndarray,Ad:np.ndarray,Bd:np.ndarray,Q:np.ndarray,R:np.ndarray) :
    
    n_sats,_ = np.shape(target_positions)
    cost_matrix   = np.empty((n_sats,n_sats))
    # cols -> target loaction
    # rows -> angent number

    # LQR cost 
    P  =  np.matrix(scipy.linalg.solve_discrete_are(Ad,Bd,Q,R))


    for jj,agent_pos in enumerate(start_positons) :
        for kk,target in enumerate(target_positions) :
     
               distance           = agent_pos-target

               cost_matrix[jj,kk] = np.matmul(distance[np.newaxis,:],np.matmul(P,distance[:,np.newaxis]))
        
    return cost_matrix

def create_circular_formation(number_of_picosats:int = 1,radius:float=1) :

    theta_range = np.linspace(0,np.pi*2,number_of_picosats+1)[:-1]
    x,y         = radius*np.cos(theta_range),radius*np.sin(theta_range)
    return np.column_stack((x,y))


def LQR_state_and_control_prediction(number_of_steps:int,x0:np.ndarray,K:np.ndarray,Ad:np.ndarray,Bd:np.ndarray,u_max:float=np.infty) :

    
    state_prediction      = np.empty((number_of_steps+1,6))  
    control_prediction    = np.empty((number_of_steps,3))  

    state_prediction[0,:] = x0

    x0 = x0[:,np.newaxis]
    
    for step in range(0,number_of_steps) :
        
        u_lqr = np.clip(-K@x0,-u_max,u_max)
        control_prediction[step,:] = np.squeeze(u_lqr)
        x0 = Ad@x0 + Bd@u_lqr
        state_prediction[step+1,:] = np.squeeze(x0)

    return state_prediction,control_prediction


def compute_LQR_feedback_gain_and_optimal_cost_matrix(Ad:np.ndarray,Bd:np.ndarray,Q:np.ndarray,R:np.ndarray) :

    P = np.matrix(scipy.linalg.solve_discrete_are(Ad,Bd,Q, R))
    K = np.matrix(scipy.linalg.inv(Bd.T @ P @ Bd+ R) @ (Bd.T @ P @ Ad))

    return K,P


def compute_single_LQR_control_per_agent(target_state:np.ndarray,start_state:np.ndarray,K:np.ndarray,u_max:float=np.infty) :
    
    x0_array = target_state-start_state
    agents_n,_ = np.shape(x0_array)
    LQR_control_per_agent = np.empty((agents_n,3))
    
    for jj,x0 in enumerate(x0_array) :
        LQR_control_per_agent[jj,:] = np.squeeze((np.clip(-K@x0[:,np.newaxis],-u_max,u_max)))

    return LQR_control_per_agent

def compute_automata_state(transition_graph,state,event) :
    
  try :
    state = transition_graph[state][event]
  except:# if there is no corresponding action just stay where you are 
    return state
  return state

def compute_barriers_dictionary(time_grid:np.ndarray,
                                state_history:np.ndarray,
                                control_history:np.ndarray,
                                state_reference:np.ndarray,
                                CBF_position:ca.Function,
                                HOBF_position:ca.Function,
                                position_barrier_constraint:ca.Function,
                                CBF_velocity:ca.Function,
                                velocity_barrier_constraint,
                                implicit_keplerian_state_history:np.ndarray=[]):


# this function is used to save all the barrier values in a simple and intuitive dictionary

    CBF_velocity_history             = np.zeros((len(time_grid),))
    CBF_position_history             = np.zeros((len(time_grid),))
    HOBF_position_history            = np.zeros((len(time_grid),))
    position_constraint_history      = np.zeros((len(time_grid),))
    velocity_constraint_history      = np.zeros((len(time_grid),))
    ux       = np.zeros((len(time_grid),))
    uy       = np.zeros((len(time_grid),))
    uz       = np.zeros((len(time_grid),))


    if len(implicit_keplerian_state_history) == 0 :
        implicit_keplerian_state_history=np.zeros((len(time_grid),6))

    
    for jj,state,control,imp_kep,state_ref in zip(range(len(time_grid)),state_history,control_history,implicit_keplerian_state_history,state_reference) :
  
       CBF_velocity_history[jj]       = np.squeeze(np.asarray(CBF_velocity(state,state_ref)))
       CBF_position_history[jj]       = np.squeeze(np.asarray(CBF_position(state,state_ref)))
       HOBF_position_history[jj]       = np.squeeze(np.asarray(HOBF_position(state,state_ref)))
       position_constraint_history[jj] = np.squeeze(np.asarray(position_barrier_constraint(state,control,imp_kep,state_ref)))
       velocity_constraint_history[jj] = np.squeeze(np.asarray(velocity_barrier_constraint(state,control,imp_kep,state_ref)))
       ux[jj] = np.squeeze(control[0])
       uy[jj] = np.squeeze(control[1])
       uz[jj] = np.squeeze(control[2])

       
    CBF_dictionary = {
                    "time"               :time_grid,
                    "CBF_velocity"       :CBF_velocity_history,
                    "CBF_position"       :CBF_position_history   ,
                    "HOBF_position"      :HOBF_position_history  ,   
                    "position_constraint":position_constraint_history,
                    "velocity_constraint":velocity_constraint_history,
                    "ux" : ux,
                    "uy" : uy,
                    "uz" : uz}

    return CBF_dictionary


    
    
