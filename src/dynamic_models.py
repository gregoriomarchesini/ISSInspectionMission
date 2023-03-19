import sys, os
from xml.etree.ElementInclude import include
sys.path.insert(0, "/Users/gregorio/Desktop/Thesis/code/python/src")

import casadi  as ca
import numpy as np
from   conversions_tools import implicit2classic_keplerian_elements,classic2implicit_keplerian_elements,classic2implicit_keplerian_elements_pointwise
import matplotlib.pyplot as plt

J1 = 0
J2 = 0.0010826266835531513   # m5/s2
J3 = J2*-2.33936E-3 
J4 = J2*-1.49601E-3 
J5 = J2*-0.20995E-3 
J6 = J2* 0.49941E-3 
J7 = 0.3513684422E-6 

spherical_harmonic_coefficients_list = np.array([J1,J2,J3,J4,J5,J6,J7])
omega_earth = 7.2921E-5

class AerodynamicSettings():
    """ Auxiliary class :
        Defines aerodynamic and atmospheric settings for a given vehicle
        Note : most likely all vehicles in the same simulation will have the same
        atmospheric settings, but not the same aerodynamic settings.
    """
    def __init__(self, drag_coefficient:float,Across:float,mass:float,zero_level_density:float,scale_height:float) :
        self.drag_coefficient  = drag_coefficient
        self.Across = Across
        self.mass = mass
        self.zero_level_density = zero_level_density
        self.scale_height = scale_height
        self.drag_parameter = 0.5*drag_coefficient*Across/mass
    
    def compute_density_at_height(self,h:float) :
        # h -> altitude from the surface of the planet
        density = self.zero_level_density*np.exp(-h/self.scale_height)
        return density

def RK4_integrator(state_dim:int,
                   input_dim:int,
                   dynamics:ca.Function,
                   integration_time:float,
                   n_steps:int) :

    """
    Parameters
    -----------
    state_dim        (int)         : dimension of the state 
    input_dim        (int)         : dimension of the input 
    dynamics         (ca.Function) : continuous dynamis of the system to be discretised
    integration_time (float)       : discretisation time step
    n_steps          (int)         : number or Runge-Kutta steps inside the inteval "integration_time"
    
    Returns
    -----------
    discrete_dynamics (ca.Function) : discrete dynamis of the system defined by "dynamics"
                                                                                                              
    Description
    -----------
    retruns discrete time dynamics of the system defined in discrete_dynamics applying a Runge-Kutta method of the fourth 
    order and assuming constant constrol input during the inteval integration_time.

    continous dynamics :
       x_dot  = f(x,u)
    disctere dynamics :
       x(k+1) = F(x(k),u(k))
    
    ************************************
    *** CASADI FUNCTION DEFINITIONS  ***
    ************************************
    CasaDi functions accept both symbolic inputs ca.MX.sym and numpy.ndarray inputs

    -----------------------------------------------------------     
    dynamics  : input1 state   (array(state_dim,1))       output1 state_dt (array(state_dim,1))
                input2 control (array(input_dim,1))
    -----------------------------------------------------------
    disctere dynamics  : input1 state   (array(state_dim,1))       output1 state_next (array(state_dim,1))
                         input2 control (array(state_dim,1))

    """
    
    # input is part of the integration
    state_initial = ca.MX.sym("state",state_dim)
    state         = state_initial
    
    if not input_dim == None :
        u     = ca.MX.sym("input",input_dim)
        dt    = integration_time/n_steps

        for j in range(n_steps):

            k1 = dynamics(state,u)
            k2 = dynamics(state+dt/2*k1,u)
            k3 = dynamics(state+dt/2*k2,u)
            k4 = dynamics(state+dt*k3,u)

            state = state + dt/6*(k1 + 2*k2 + 2*k3 + k4)

        discrete_dynamics = ca.Function("rk4_integrator", [state_initial,u], [state],["x0","control"],["xf"])
    else :

        dt    = integration_time/n_steps

        for j in range(n_steps):

            k1 = dynamics(state)
            k2 = dynamics(state+dt/2*k1)
            k3 = dynamics(state+dt/2*k2)
            k4 = dynamics(state+dt*k3)

            state = state + dt/6*(k1 + 2*k2 + 2*k3 + k4)

        discrete_dynamics = ca.Function("rk4_integrator", [state_initial], [state],["x0"],["xf"])
        
    return discrete_dynamics

def obtain_symbolic_DCM(axis:int) :
    
    """
    Parameters
    -----------
    axis         (float) : principal axis of rotation (possible 1,2,3)
    
    Returns
    -----------
    DCM        (casadi.Function)) : symbolic DCM matrix function (not a rotation) around given axis
                                    axis = 1  -> x-axis
                                    axis = 2  -> y-axis
                                    axis = 3  -> z-axis
                                 
                                                                                                              
    Description
    -----------
    retruns symbolic DCM matrix (not a rotation) around given axis.

    ************************************
    *** CASADI FUNCTION DEFINITIONS  ***
    ************************************
    CasaDi functions accept both symbolic inputs ca.MX.sym and numpy.ndarray inputs

    -----------------------------------------------------------     
    DCM   : input1 state (array(1,))     output1 DCM (array(3,3))
  
    """

    angle = ca.MX.sym("angle ")
    C = ca.MX.sym("DCM",3,3)
    
    if axis == 1 :
        
        r1 =  ca.horzcat( 1,       0         ,    0)
        r2 =  ca.horzcat( 0,   ca.cos(angle) ,ca.sin(angle))  
        r3 =  ca.horzcat( 0,   -ca.sin(angle),ca.cos(angle) )
        
        C = ca.vertcat(r1,r2,r3)
    
    elif axis == 2 :
        
        r1 =  ca.horzcat( ca.cos(angle),  0,   -ca.sin(angle))
        r2 =  ca.horzcat(  0           ,  1,     0         )
        r3 =  ca.horzcat( ca.sin(angle),  0,   ca.cos(angle) )
        
        C = ca.vertcat(r1,r2,r3)
            
    
    elif axis == 3 :
        
       r1 =  ca.horzcat(ca.cos(angle) ,  ca.sin(angle)  ,  0)  
       r2 =  ca.horzcat(-ca.sin(angle), ca.cos(angle)   ,  0)  
       r3 =  ca.horzcat(    0    ,  0                   ,  1)
       
       C = ca.vertcat(r1,r2,r3)    
    
    else :
        raise ValueError("Axis definition is not supported. Only 3,2,1 are supported. your is {}".format(axis))
    
    DCM = ca.Function("DCM",[angle],[C])
    
    return DCM

def legendre_polynomial(degree = 0) :
    
    """
    Parameters
    -----------
    degree   (int) : degree of the legendre zero order polynomial
    
    Returns
    -----------
    legendre_polynomial    (casadi.Function)    : legendre zero order polinomial function
                      
    Description
    -----------
    returns symbolic legendre polynomial of order zero and degree equal to  "degree" 
    note : you can use Wikipidia to understand what are order and degree of a legendre polynomial.
           All polynomials produced by the function have zero order and the desired degree 

    ************************************
    *** CASADI FUNCTION DEFINITIONS  ***
    ************************************
    CasaDi functions accept both symbolic inputs ca.MX.sym and numpy.ndarray inputs

    -----------------------------------------------------------     
    legendre_polynomial   : input1 x(array(1,))     output1 legendre_polynomial(x) (array(1,)))
    
    """
    
    x             = ca.MX.sym("x",1)
    constant      = 1/(np.math.factorial(degree) * 2**degree)
    main_function = constant*(x**2-1)**degree
    
    for k in range(degree) :
        main_function = ca.jacobian(main_function,x)
        
    legendre_polynomial = ca.Function("legendre_polynomial",[x],[main_function])
    
    return legendre_polynomial

def load_chief_continuous_dynamics(planet_mean_radius  : float,                 
                                   gravitational_parameter : float,          
                                   max_spherical_harmonic_order: float=0,
                                   aerodynamic_settings_chief:AerodynamicSettings = None) :
    
    """
    Parameters
    -----------
    planet_mean_radius            (float)                : mean radius of the planet
    gravitational_parameter       (float)                : gravitational parameter of the planet
    max_spherical_harmonic_order  (float)                : max spherical harmonic order to be included in the model
    aerodynamic_settings_chief   (AerodynamicSettings)  : AerodynamicSettings defined on chief 
    
    
    Returns
    -----------
    chief_continuous_dynamics    (casadi.Function)    :  chief continuous dynamics function
                      
    Description
    -----------
    returns symbolic dynamics of the chief under the effect of aerodynamic acceleration and higher order spherical harmonics

    ************************************
    *** CASADI FUNCTION DEFINITIONS  ***
    ************************************
    CasaDi functions accept both symbolic inputs ca.MX.sym and numpy.ndarray inputs

    -----------------------------------------------------------     
    chief_continuous_dynamics   : input1 implicit_keplerian_state(array(6,))     output1 implicit_keplerian_state_dot (array(6,)))
    
    """
    
    
    # state of the chief
    theta_bar    = ca.MX.sym("theta_bar")  # chief true anomaly    
    inc          = ca.MX.sym("inc")        # chief inclination  
    raan         = ca.MX.sym("raan")       # chief Raan         
    h            = ca.MX.sym("h")          # chief angular momentum vector 
    vr_chief     = ca.MX.sym("vr_chief")   # chief radial speed  
    r_chief      = ca.MX.sym("r_chief")    # chief geocentric distance       
    
    # rsw state chief
    implicit_keplerian_state_chief = ca.vertcat(theta_bar,inc,raan,h,vr_chief,r_chief)
    
    f_total_pert_fun = obtain_total_perturbing_acceleration_on_chief(max_spherical_harmonic_order=max_spherical_harmonic_order,
                                                                     gravitational_parameter = gravitational_parameter,
                                                                     planet_mean_radius = planet_mean_radius,
                                                                     aerodynamic_settings_chief=aerodynamic_settings_chief)

    
    f_total_pert = f_total_pert_fun(implicit_keplerian_state_chief)

    # dynamics of the chief implicit keplerian elements
    dtheta_bar_dt  =   h/r_chief**2  - r_chief*ca.sin(theta_bar)/h/ca.tan(inc) * (f_total_pert[2])
    dinc_dt        =   (r_chief/h) * ca.cos(theta_bar) * (f_total_pert[2])
    draan_dt       =   r_chief*ca.sin(theta_bar)/h/ca.sin(inc) * (f_total_pert[2])
    dh_dt          =   r_chief *(f_total_pert[1])
    dvr_chief_dt   =  - gravitational_parameter/r_chief**2 + h**2/r_chief**3 + f_total_pert[0]
    dr_chief_dt    =    vr_chief 
    
    # create dynamics function
     
    implicit_keplerian_state_chief_dt    = ca.vertcat(dtheta_bar_dt,dinc_dt ,draan_dt,dh_dt,dvr_chief_dt,dr_chief_dt )
    chief_continuous_dynamics     = ca.Function("chief_dynamics",[implicit_keplerian_state_chief],[implicit_keplerian_state_chief_dt],['state_chief'],['chief_dynamics'])
    
    return chief_continuous_dynamics

def load_chief_discrete_dynamics(integration_time : float,
                                 n_steps          : int,
                                 planet_mean_radius         : float ,                 
                                 gravitational_parameter: float ,          
                                 max_spherical_harmonic_order: float=0,
                                 aerodynamic_settings_chief:AerodynamicSettings = None):
    

    """
    Parameters
    -----------
    integration_time              (float)                : time of integration
    n_steps                       (int)                  : number on RK4 steps taken inside the discretization time integration_time
    planet_mean_radius            (float)                : mean radius of the planet
    gravitational_parameter       (float)                : gravitational parameter of the planet
    max_spherical_harmonic_order  (float)                : max spherical harmonic order to be included in the model
    aerodynamic_settings_chief   (AerodynamicSettings)   : AerodynamicSettings defined on chief 
    
    
    Returns
    -----------
    discrete_chief_dynamics    (casadi.Function)    :  discrete chief dynamics symbolic function
                      
    Description
    -----------
    returns symbolic discrete dynamics of the chief under the effect of aerodynamic acceleration and higher order spherical harmonics

    ************************************
    *** CASADI FUNCTION DEFINITIONS  ***
    ************************************
    CasaDi functions accept both symbolic inputs ca.MX.sym and numpy.ndarray inputs

    -----------------------------------------------------------     
    discrete_chief_dynamics   : input1 implicit_keplerian_state(array(6,))     output1 implicit_keplerian_state_next (array(6,)))
    
    """
    
     # state of the chief
    theta_bar    = ca.MX.sym("theta_bar")  # chief true anomaly    
    inc          = ca.MX.sym("inc")        # chief inclination  
    raan         = ca.MX.sym("raan")       # chief Raan         
    h            = ca.MX.sym("h")          # chief angular momentum vector 
    vr_chief     = ca.MX.sym("vr_chief")   # chief radial speed  
    r_chief      = ca.MX.sym("r_chief")    # chief geocentric distance       
    
    # rsw state chief
    implicit_keplerian_state_chief = ca.vertcat(theta_bar,inc,raan,h,vr_chief,r_chief)
    
    chief_dynamics          = load_chief_continuous_dynamics(planet_mean_radius=planet_mean_radius,
                                                             gravitational_parameter=gravitational_parameter,
                                                             max_spherical_harmonic_order=max_spherical_harmonic_order,
                                                             aerodynamic_settings_chief = aerodynamic_settings_chief)
    
    discrete_chief_dynamics = RK4_integrator(state_dim=6,input_dim=None,dynamics=chief_dynamics,integration_time=integration_time,n_steps=n_steps) 
    
    
    # input is the implicit_keplerina_state_chief
    # output is the integrated implitic keplerian state at int_time later
    return   discrete_chief_dynamics 

def load_agent_linear_continuous_dynamics(orbital_period:float):

    """
    Parameters
    -----------
    orbital_period            (float)   : orbital period of the chief reference circular orbit
    
    Returns
    -----------
    agent_linear_continuous_dynamics    (casadi.Function)    :  agent linearised relative continuous dynamics function for the agent around the chief orbit
                      
    Description
    -----------
    returns symbolic dynamics of the agent around the chief under assuming a linearised model

    ************************************
    *** CASADI FUNCTION DEFINITIONS  ***
    ************************************
    CasaDi functions accept both symbolic inputs ca.MX.sym and numpy.ndarray inputs

    -----------------------------------------------------------     
    agent_linear_continuous_dynamics    : input1 agent_state_rsw(array(6,))          output1 agent_state_rsw_dot (array(6,)))
                                      
    """
    
    

    
    # mean anomaly 
    mean_motion = 2*np.pi/orbital_period
    x  = ca.MX.sym("x")         # m     
    y  = ca.MX.sym("y")         # m  
    z  = ca.MX.sym("z")         # m  
    vx = ca.MX.sym("vx")        # m/s   
    vy = ca.MX.sym("vy")        # m/s   
    vz = ca.MX.sym("vz")        # m/s  
    state_agent_translational      = ca.vertcat(x,y,z,vx,vy,vz)
    
    # control translational
    ux  = ca.MX.sym("u1")       # Control
    uy  = ca.MX.sym("u2")       # Control
    uz  = ca.MX.sym("u2")       # Control
    translational_control = ca.vertcat(ux,uy,uz)
    
    
    # define A matrix by row
    row1A = ca.horzcat(0,0,0,1,0,0)
    row2A = ca.horzcat(0,0,0,0,1,0)
    row3A = ca.horzcat(0,0,0,0,0,1)
    row4A = ca.horzcat(3*mean_motion**2,0  ,0              ,0             ,2*mean_motion,0)
    row5A = ca.horzcat(0               ,0  ,0              ,-2*mean_motion,0            ,0)
    row6A = ca.horzcat(0               ,0  ,-mean_motion**2,   0          ,0            ,0)
    A = ca.vertcat(row1A,
                   row2A,
                   row3A,
                   row4A,
                   row5A,
                   row6A,)
    
    # Define B by raw (input is always an acceleration)
    
    row1B = ca.horzcat(0,0,0)
    row2B = ca.horzcat(0,0,0)
    row3B = ca.horzcat(0,0,0)
    row4B = ca.horzcat(1,0,0)
    row5B = ca.horzcat(0,1,0)
    row6B = ca.horzcat(0,0,1)
    
    B = ca.vertcat(row1B,
                   row2B,
                   row3B,
                   row4B,
                   row5B,
                   row6B,)
    
    state_agent_translational_dt  = ca.mtimes(A,state_agent_translational) + ca.mtimes(B,translational_control)
    agent_linear_continuous_dynamics = ca.Function("agent_linear_continuous_dynamics",[state_agent_translational,translational_control],[state_agent_translational_dt])
    return agent_linear_continuous_dynamics

def load_continuous_linear_dynamics_matrices(orbital_period:float) :

    # mean anomaly 
    mean_motion = 2*np.pi/orbital_period
    
    # since the system is LTI, an analytical solution exist
    # So it is better to not integrate with RK4 and use the 
    # analytical solution
    
    # define A matrix by row
    row1A = ca.horzcat(0,0,0,1,0,0)
    row2A = ca.horzcat(0,0,0,0,1,0)
    row3A = ca.horzcat(0,0,0,0,0,1)
    row4A = ca.horzcat(3*mean_motion**2,0  ,0              ,0             ,2*mean_motion,0)
    row5A = ca.horzcat(0               ,0  ,0              ,-2*mean_motion,0            ,0)
    row6A = ca.horzcat(0               ,0  ,-mean_motion**2,   0          ,0            ,0)
    A = ca.vertcat(row1A,
                   row2A,
                   row3A,
                   row4A,
                   row5A,
                   row6A,)
    
    # Define B by raw (input is always an acceleration)
    
    row1B = ca.horzcat(0,0,0)
    row2B = ca.horzcat(0,0,0)
    row3B = ca.horzcat(0,0,0)
    row4B = ca.horzcat(1,0,0)
    row5B = ca.horzcat(0,1,0)
    row6B = ca.horzcat(0,0,1)
    
    B = ca.vertcat(row1B,
                   row2B,
                   row3B,
                   row4B,
                   row5B,
                   row6B,)
    return A,B


def load_discrete_linear_dynamics_matrices(orbital_period:float,
                                           integration_time:float) :

    # mean anomaly 
    mean_motion = 2*np.pi/orbital_period
    
    # since the system is LTI, an analytical solution exist
    # So it is better to not integrate with RK4 and use the 
    # analytical solution
    
    Ndt = np.cos(mean_motion*integration_time)
    sdt = np.sin(mean_motion*integration_time)
    ndt = integration_time*mean_motion

    # define A matrix by row
    row1A = np.array([4-3*Ndt                ,0                 ,0                 , 1/mean_motion*sdt         , 2/mean_motion*(1-Ndt)     ,  0                ])
    row2A = np.array([6*(sdt-ndt)            ,1                 ,0                 , -2/mean_motion*(1-Ndt)    ,1/mean_motion*(4*sdt-3*ndt),  0                ])
    row3A = np.array([0                      ,0                 ,Ndt               , 0                         ,0                          ,  1/mean_motion*sdt])
    row4A = np.array([3*mean_motion*sdt      ,0                 ,0                 ,Ndt                        ,2*sdt                      ,0                  ])
    row5A = np.array([-6*mean_motion*(1-Ndt) ,0                 ,0                 ,-2*sdt                     ,4*Ndt-3                    ,0                  ]) 
    row6A = np.array([0                      ,0                 ,-mean_motion*sdt  , 0                         ,0                          ,Ndt                ])
    
    A = np.vstack((row1A,
                   row2A,
                   row3A,
                   row4A,
                   row5A,
                   row6A))
    
    # Define B by raw (input is always an acceleration)
    
    row1B = np.array([1/mean_motion**2 *(1-Ndt)    , 2/mean_motion**2 * (ndt-sdt)                     ,  0                       ])
    row2B = np.array([-2/mean_motion**2 * (ndt-sdt),4/mean_motion**2 *(1-Ndt)- 3/2*integration_time**2,   0                      ])
    row3B = np.array([0                            ,0                                                 ,1/mean_motion**2 *(1-Ndt) ])
    row4B = np.array([1/mean_motion * sdt          ,2/mean_motion*(1-Ndt)                             ,0                         ])
    row5B = np.array([-2/mean_motion *(1-Ndt)      ,4/mean_motion*sdt-3*integration_time              ,0                         ])
    row6B = np.array([0                            ,0                                                 ,1/mean_motion*sdt         ])
    
    B = np.vstack((row1B,
                   row2B,
                   row3B,
                   row4B,
                   row5B,
                   row6B,))
    
    return A,B



def load_agent_linear_discrete_dynamics(orbital_period:float,
                                        integration_time:float):


                                        # mean anomaly 
    mean_motion = 2*np.pi/orbital_period
    x  = ca.MX.sym("x")         # m     
    y  = ca.MX.sym("y")         # m  
    z  = ca.MX.sym("z")         # m  
    vx = ca.MX.sym("vx")        # m/s   
    vy = ca.MX.sym("vy")        # m/s   
    vz = ca.MX.sym("vz")        # m/s  
    state_agent_translational      = ca.vertcat(x,y,z,vx,vy,vz)
    
    # control translational
    ux  = ca.MX.sym("u1")       # Control
    uy  = ca.MX.sym("u2")       # Control
    uz  = ca.MX.sym("u2")       # Control
    translational_control = ca.vertcat(ux,uy,uz)
    
    # since the system is LTI, an analytical solution exist
    # So it is better to not integrate with RK4 and use the 
    # analytical solution
    
    Ndt = ca.cos(mean_motion*integration_time)
    sdt = ca.sin(mean_motion*integration_time)
    ndt = integration_time*mean_motion
    # define A matrix by row
    row1A = ca.horzcat(4-3*Ndt                ,0                 ,0                 , 1/mean_motion*sdt         , 2/mean_motion*(1-Ndt)     ,  0      )
    row2A = ca.horzcat(6*(sdt-ndt)            ,1                 ,0                 , -2/mean_motion*(1-Ndt)    ,1/mean_motion*(4*sdt-3*ndt),  0      )
    row3A = ca.horzcat(0                      ,0                 ,Ndt               , 0                         ,0                          ,  1/mean_motion*sdt)
    row4A = ca.horzcat(3*mean_motion*sdt      ,0                 ,0                 ,Ndt                        ,2*sdt                      ,0        )
    row5A = ca.horzcat(-6*mean_motion*(1-Ndt) ,0                 ,0                 ,-2*sdt                     ,4*Ndt-3                    ,0        )
    row6A = ca.horzcat(0                      ,0                 ,-mean_motion*sdt  , 0                         ,0                          ,Ndt      )
    
    A = ca.vertcat(row1A,
                   row2A,
                   row3A,
                   row4A,
                   row5A,
                   row6A)
    
    # Define B by raw (input is always an acceleration)
    
    row1B = ca.horzcat(1/mean_motion**2 *(1-Ndt)    , 2/mean_motion**2 * (ndt-sdt)                     ,  0                       )
    row2B = ca.horzcat(-2/mean_motion**2 * (ndt-sdt),4/mean_motion**2 *(1-Ndt)- 3/2*integration_time**2,   0                      )
    row3B = ca.horzcat(0                            ,0                                                 ,1/mean_motion**2 *(1-Ndt) )
    row4B = ca.horzcat(1/mean_motion * sdt          ,2/mean_motion*(1-Ndt)                             ,0                         )
    row5B = ca.horzcat(-2/mean_motion *(1-Ndt)      ,4/mean_motion*sdt-3*integration_time              ,0                         )
    row6B = ca.horzcat(0                            ,0                                                 ,1/mean_motion*sdt         )
    
    B = ca.vertcat(row1B,
                   row2B,
                   row3B,
                   row4B,
                   row5B,
                   row6B,)
    
    state_agent_translational_dt  = ca.mtimes(A,state_agent_translational) + ca.mtimes(B,translational_control)
    agent_linear_discrete_dynamics = ca.Function("agent_linear_continuous_dynamics",[state_agent_translational,translational_control],[state_agent_translational_dt])
    return agent_linear_discrete_dynamics




def load_agent_rotational_continuous_dynamics(Inertia_matrix   :np.ndarray,):
    
    Inertia_inv = np.linalg.inv(Inertia_matrix )
    # quaternions
    q0 = ca.MX.sym("q0")        # m/s  
    q1 = ca.MX.sym("q1")        # m/s 
    q2 = ca.MX.sym("q2")        # m/s 
    q3 = ca.MX.sym("q3")        # m/s 
    quaternion  = ca.vertcat(q0,q1,q2,q3)
    
    # control torque
    taux  = ca.MX.sym("tau1")       # Control
    tauy  = ca.MX.sym("tau2")       # Control
    tauz  = ca.MX.sym("tau2")       # Control
    torque_control = ca.vertcat(taux,tauy,tauz) 
    
    # angular speed in body frame
    omegax = ca.MX.sym("omegax")        # m/s 
    omegay = ca.MX.sym("omegay")        # m/s 
    omegaz = ca.MX.sym("omegaz")        # m/s 
    omega  = ca.vertcat(omegax,omegay,omegaz) 
    
    state_agent_rotational  = ca.vertcat(quaternion ,omega)
    
    k = 0.01 # gain factor (leave it like that)
    quat_norm2 = (q0**2+q1**2+q2**2+q3**2)
    q0_dt     = 0.5*(-omegax*q1 -q2*omegay - q3*omegaz)  + k*(1-quat_norm2) *q0
    q1_dt     = 0.5*( omegax*q0 -q3*omegay + q2*omegaz)  + k*(1-quat_norm2) *q1
    q2_dt     = 0.5*( omegax*q3 +q0*omegay - q1*omegaz)  + k*(1-quat_norm2) *q2
    q3_dt     = 0.5*(-omegax*q2 +q1*omegay + q0*omegaz)  + k*(1-quat_norm2) *q3

    quaternion_dt  = ca.vertcat(q0_dt,q1_dt,q2_dt,q3_dt)
    omega_dt       = Inertia_inv@(-ca.cross(omega,Inertia_matrix@omega) + torque_control)
    
    state_agent_rotational    = ca.vertcat(quaternion ,omega)  
    state_agent_rotational_dt = ca.vertcat(quaternion_dt ,omega_dt )  
    
    continuous_rotational_dynamics = ca.Function("rotational_dynamics",[state_agent_rotational,torque_control],[state_agent_rotational_dt])
    
    return continuous_rotational_dynamics

def load_agent_rotational_discrete_dynamics(integration_time :float,
                                               n_steps          :int,
                                               Inertia_matrix   :np.ndarray,):
    
    Inertia_inv = np.linalg.inv(Inertia_matrix )
    # quaternions
    q0 = ca.MX.sym("q0")        # m/s  
    q1 = ca.MX.sym("q1")        # m/s 
    q2 = ca.MX.sym("q2")        # m/s 
    q3 = ca.MX.sym("q3")        # m/s 
    quaternion  = ca.vertcat(q0,q1,q2,q3)
    
    # control torque
    taux  = ca.MX.sym("tau1")       # Control
    tauy  = ca.MX.sym("tau2")       # Control
    tauz  = ca.MX.sym("tau2")       # Control
    torque_control = ca.vertcat(taux,tauy,tauz) 
    
    # angular speed in body frame
    omegax = ca.MX.sym("omegax")        # m/s 
    omegay = ca.MX.sym("omegay")        # m/s 
    omegaz = ca.MX.sym("omegaz")        # m/s 
    omega  = ca.vertcat(omegax,omegay,omegaz) 
    
    state_agent_rotational  = ca.vertcat(quaternion ,omega)
    
    k = 0.01 # gain factor (leave it like that)
    quat_norm2 = (q0**2+q1**2+q2**2+q3**2)
    q0_dt     = 0.5*(-omegax*q1 -q2*omegay - q3*omegaz)  + k*(1-quat_norm2) *q0
    q1_dt     = 0.5*( omegax*q0 -q3*omegay + q2*omegaz)  + k*(1-quat_norm2) *q1
    q2_dt     = 0.5*( omegax*q3 +q0*omegay - q1*omegaz)  + k*(1-quat_norm2) *q2
    q3_dt     = 0.5*(-omegax*q2 +q1*omegay + q0*omegaz)  + k*(1-quat_norm2) *q3

    quaternion_dt  = ca.vertcat(q0_dt,q1_dt,q2_dt,q3_dt)
    omega_dt       = Inertia_inv@(-ca.cross(omega,Inertia_matrix@omega) + torque_control)
    
    state_agent_rotational    = ca.vertcat(quaternion ,omega)  
    state_agent_rotational_dt = ca.vertcat(quaternion_dt ,omega_dt )  
    
    continuous_rotational_dynamics = ca.Function("rotational_dynamics",[state_agent_rotational,torque_control],[state_agent_rotational_dt])
    discrete_rotational_dynamics      = RK4_integrator(state_dim=7,input_dim=3,dynamics=continuous_rotational_dynamics,integration_time=integration_time,n_steps=n_steps) 
    
    return discrete_rotational_dynamics

def load_agent_translational_discrete_dynamics(integration_time :float,
                                               n_steps          :int,
                                               planet_mean_radius         :float ,                 
                                               gravitational_parameter :float ,          
                                               max_spherical_harmonic_order: float=0,
                                               aerodynamic_settings_chief:AerodynamicSettings = None,
                                               aerodynamic_settings_deputy:AerodynamicSettings = None):


    """
    Parameters
    -----------
    integration_time              (float)                : time of integration
    n_steps                       (int)                  : number on RK4 steps taken inside the discretization time integration_time
    planet_mean_radius            (float)                : mean radius of the planet
    gravitational_parameter       (float)                : gravitational parameter of the planet
    max_spherical_harmonic_order  (float)                : max spherical harmonic order to be included in the model
    aerodynamic_settings_chief    (AerodynamicSettings)  : AerodynamicSettings defined on chief 
    aerodynamic_settings_deputy    (AerodynamicSettings)  : AerodynamicSettings defined on deputy 
    
    
    Returns
    -----------
    agent_discrete_dynamics    (casadi.Function)    :  agent relative discrete dynamics function
                      
    Description
    -----------
    returns symbolic discrete dynamics of the agent around the chief under the effect of aerodynamic accelerations and higher order spherical harmonics

    ************************************
    *** CASADI FUNCTION DEFINITIONS  ***
    ************************************
    CasaDi functions accept both symbolic inputs ca.MX.sym and numpy.ndarray inputs

    -----------------------------------------------------------     
    agent_discrete_dynamics    : input1 agent_state_rsw(array(6,))          output1 agent_state_rsw_next (array(6,)))
                                 input2 implicit_keplerian_state(array(6,))
    """
    
    
    
    x  = ca.MX.sym("x")         # m     
    y  = ca.MX.sym("y")         # m  
    z  = ca.MX.sym("z")         # m  
    vx = ca.MX.sym("vx")        # m/s   
    vy = ca.MX.sym("vy")        # m/s   
    vz = ca.MX.sym("vz")        # m/s  
    state_agent_translational      = ca.vertcat(x,y,z,vx,vy,vz)
    
    # control translational
    ux  = ca.MX.sym("u1")       # Control
    uy  = ca.MX.sym("u2")       # Control
    uz  = ca.MX.sym("u2")       # Control
    translational_control = ca.vertcat(ux,uy,uz) 
    
    # state of the chief
    theta_bar  = ca.MX.sym("theta_bar")  # true anomaly    
    inc        = ca.MX.sym("inc")        # inclination  
    raan       = ca.MX.sym("raan")       # Raan         
    h          = ca.MX.sym("h")          # angular momentum vector 
    vr_chief   = ca.MX.sym("vr_chief")     # radial speed  
    r_chief    = ca.MX.sym("r_chief")      # geocentric distance 

    implicit_keplerian_state_chief = ca.vertcat(theta_bar,inc,raan,h,vr_chief,r_chief)

    agent_dynamics_fun = load_agent_translational_continuous_dynamics(planet_mean_radius                    = planet_mean_radius ,                 
                                                                      gravitational_parameter      = gravitational_parameter ,          
                                                                      max_spherical_harmonic_order = max_spherical_harmonic_order,
                                                                      aerodynamic_settings_chief   = aerodynamic_settings_chief,
                                                                      aerodynamic_settings_deputy  = aerodynamic_settings_deputy)
    chief_dynamics_fun = load_chief_continuous_dynamics(planet_mean_radius=planet_mean_radius,
                                                        gravitational_parameter=gravitational_parameter,
                                                        max_spherical_harmonic_order=max_spherical_harmonic_order,
                                                        aerodynamic_settings_chief = aerodynamic_settings_chief)
    
    agent_state_dt = agent_dynamics_fun(state_agent_translational,translational_control,implicit_keplerian_state_chief)
    chief_state_dt = chief_dynamics_fun(implicit_keplerian_state_chief)
    
    # add chief state to the state to be propagated 
    expanded_state_dt = ca.vertcat(agent_state_dt ,chief_state_dt )
    expanded_state    = ca.vertcat(state_agent_translational,implicit_keplerian_state_chief)
    
    expanded_state_dynamics = ca.Function("dynamics",[expanded_state,translational_control],[expanded_state_dt])
    discrete_expanded_state_dynamics   = RK4_integrator(state_dim=12,input_dim=3,dynamics=expanded_state_dynamics,integration_time=integration_time,n_steps=n_steps) 
    
    #take only the state of agent as propagated state and use chief as external parameter
    expanded_future_state   =  discrete_expanded_state_dynamics(expanded_state,translational_control)
    agent_future_state      =  expanded_future_state[:6]
    discrete_agent_dynamics = ca.Function("F", [state_agent_translational,translational_control,implicit_keplerian_state_chief], [agent_future_state],["state_initial","control","chief_initial_state"],["propagated_state"])

    return discrete_agent_dynamics 

                                                           
def load_agent_translational_continuous_dynamics(gravitational_parameter:float ,
                                                 planet_mean_radius :float ,                 
                                                 max_spherical_harmonic_order: float=0,
                                                 aerodynamic_settings_chief:AerodynamicSettings = None,
                                                 aerodynamic_settings_deputy:AerodynamicSettings = None) :

    """
    Parameters
    -----------
    planet_mean_radius            (float)                : mean radius of the planet
    gravitational_parameter       (float)                : gravitational parameter of the planet
    max_spherical_harmonic_order  (float)                : max spherical harmonic order to be included in the model
    aerodynamic_settings_chief    (AerodynamicSettings)  : AerodynamicSettings defined on chief 
    aerodynamic_settings_deputy   (AerodynamicSettings)  : AerodynamicSettings defined on chief
    
    Returns
    -----------
    agent_continuous_dynamics    (casadi.Function)    :  agent relative continuous dynamics function
                      
    Description
    -----------
    returns symbolic dynamics of the agent around the chief under the effect of aerodynamic accelerations and higher order spherical harmonics

    ************************************
    *** CASADI FUNCTION DEFINITIONS  ***
    ************************************
    CasaDi functions accept both symbolic inputs ca.MX.sym and numpy.ndarray inputs

    -----------------------------------------------------------     
    agent_continuous_dynamics    : input1 agent_state_rsw(array(6,))          output1 agent_state_rsw_dot (array(6,)))
                                  input2 implicit_keplerian_state(array(6,))
    """
    
    
    
    
    
    if  ((aerodynamic_settings_chief==None) ^ (aerodynamic_settings_deputy==None)) :
        message = ("aerodynamic settings must be given for chief and deputy in case"
                   "one of the two is given.\n Add settings for both deputy and chief")
        raise ValueError(message)
    
    # robot j state
    # symbolic state variables robot j [variable naming is convenint]
    # this is the state of the system in hill coordinates

    x  = ca.MX.sym("x")         # m     
    y  = ca.MX.sym("y")         # m  
    z  = ca.MX.sym("z")         # m  
    vx = ca.MX.sym("vx")        # m/s   
    vy = ca.MX.sym("vy")        # m/s   
    vz = ca.MX.sym("vz")        # m/s  
    state_agent_translational      = ca.vertcat(x,y,z,vx,vy,vz)
    
    # state of the chief
    theta_bar  = ca.MX.sym("theta_bar")  # true anomaly    
    inc        = ca.MX.sym("inc")        # inclination  
    raan       = ca.MX.sym("raan")       # Raan         
    h          = ca.MX.sym("h")          # angular momentum vector 
    vr_chief   = ca.MX.sym("vr_chief")     # radial speed  
    r_chief    = ca.MX.sym("r_chief")      # geocentric distance 

    implicit_keplerian_state_chief = ca.vertcat(theta_bar,inc,raan,h,vr_chief,r_chief)

    # control translational
    ux  = ca.MX.sym("u1")       # Control
    uy  = ca.MX.sym("u2")       # Control
    uz  = ca.MX.sym("u2")       # Control
    
    translational_control       = ca.vertcat(ux,uy,uz) 
    r_deputy                    = ca.sqrt((r_chief + x)**2 + y**2 + z**2)
    agent_global_position_rsw   = ca.vertcat(x+r_chief,y,z)
    chief_global_position_rsw   = ca.vertcat(r_chief,0,0) 
    agent_relative_velocity_rsw = ca.vertcat(vx,vy,vz) 
    agent_relative_position_rsw = ca.vertcat(x,y,z) 
    

    # Obtain acceleration on chief first  
    rsw_frame_angular_velocity_fun = obtain_RSW_frame_angular_velocity(gravitational_parameter=gravitational_parameter,
                                                                       planet_mean_radius= planet_mean_radius,
                                                                        max_spherical_harmonic_order=max_spherical_harmonic_order,
                                                                       aerodynamic_settings_chief=aerodynamic_settings_chief)
    
    rsw_frame_angular_acceleration_fun = obtain_RSW_angular_acceleration_vector(gravitational_parameter=gravitational_parameter,
                                                                                planet_mean_radius= planet_mean_radius,
                                                                                max_spherical_harmonic_order=max_spherical_harmonic_order,
                                                                                aerodynamic_settings_chief=aerodynamic_settings_chief)
    
    
    
    omega_rsw     = rsw_frame_angular_velocity_fun(implicit_keplerian_state_chief)
    omega_dot_rsw = rsw_frame_angular_acceleration_fun(implicit_keplerian_state_chief)
   
    C3      = obtain_symbolic_DCM(axis=3)
    C1      = obtain_symbolic_DCM(axis=1)
    DCM_inertial_to_rsw = ca.mtimes(ca.mtimes(C3(theta_bar),C1(inc)),C3(raan))
    
    f_pert_chief_fun  = obtain_total_perturbing_acceleration_on_chief(max_spherical_harmonic_order=max_spherical_harmonic_order,planet_mean_radius=planet_mean_radius,gravitational_parameter=gravitational_parameter,aerodynamic_settings_chief=aerodynamic_settings_chief)
    f_pert_deputy_fun = obtain_total_perturbing_acceleration_on_deputy(max_spherical_harmonic_order=max_spherical_harmonic_order,planet_mean_radius=planet_mean_radius,gravitational_parameter=gravitational_parameter,aerodynamic_settings_chief=aerodynamic_settings_chief,aerodynamic_settings_deputy=aerodynamic_settings_deputy)
   
    f_pert_chief  = f_pert_chief_fun (implicit_keplerian_state_chief)
    f_pert_deputy = f_pert_deputy_fun(state_agent_translational,implicit_keplerian_state_chief)
    
    point_mass_gravity_relative_acceleration = (- gravitational_parameter/r_deputy**3 * agent_global_position_rsw ) - ( -gravitational_parameter/r_chief**3 * chief_global_position_rsw )
    relative_acceleration_rsw = point_mass_gravity_relative_acceleration - 2*ca.cross(omega_rsw,agent_relative_velocity_rsw) - ca.cross(omega_rsw,ca.cross(omega_rsw,agent_relative_position_rsw)) - ca.cross(omega_dot_rsw,agent_relative_position_rsw) +  f_pert_deputy - f_pert_chief + translational_control 
   
    
    # relative_acceleration_rsw = point_mass_gravity_relative_acceleration - 2*ca.cross(omega_rsw,agent_relative_velocity_rsw) - ca.cross(omega_rsw,ca.cross(omega_rsw,agent_relative_position_rsw)) + (f_pert_deputy-f_pert_chief)  + translational_control 
    state_agent_translational_dt = ca.vertcat(vx,vy,vz,relative_acceleration_rsw)
    continuous_agent_dynamics    = ca.Function("dynamics",[state_agent_translational,translational_control,implicit_keplerian_state_chief],[state_agent_translational_dt])
  
    return  continuous_agent_dynamics

def load_energy_constraint(gravitational_parameter:float,planet_mean_radius:float,max_spherical_harmonic_order:float=0,aerodynamic_settings_chief:AerodynamicSettings=None) :
   
   
    """
    Parameters
    -----------
    planet_mean_radius            (float)                : mean radius of the planet
    gravitational_parameter       (float)                : gravitational parameter of the planet
    max_spherical_harmonic_order  (float)                : max spherical harmonic order to be included in the model
    aerodynamic_settings_chief    (AerodynamicSettings)  : AerodynamicSettings defined on chief 
    
    Returns
    -----------
    energy_match_function   (casadi.Function)    :  symbolic function defyining the energy difference between chief and deputy
                      
    Description
    -----------
    returns symboli cenergy matching  constraint function. This constraint could be directly inserted into an MPC formulation for example

    ************************************
    *** CASADI FUNCTION DEFINITIONS  ***
    ************************************
    CasaDi functions accept both symbolic inputs ca.MX.sym and numpy.ndarray inputs

    -----------------------------------------------------------     
    energy_match_function    : input1 agent_state_rsw(array(6,))            output1 energy_difference (array(1,)))
                              input2 implicit_keplerian_state(array(6,))
    
    """
    
    #! Update function to deal with advanced aerodynamic settings
    
    # state of the chief
    theta_bar= ca.MX.sym("theta_bar")  # chief true anomaly    
    inc      = ca.MX.sym("inc")        # chief inclination  
    Raan     = ca.MX.sym("raan")       # chief Raan         
    h        = ca.MX.sym("h")          # chief angular momentum vector 
    vr_chief = ca.MX.sym("vr_chief")     # chief radial speed  
    r_chief  = ca.MX.sym("r_chief")      # chief geocentric distance 
          
    implicit_keplerian_state_chief = ca.vertcat(theta_bar,inc,Raan,h,vr_chief,r_chief)
    
    x_rsw  = ca.MX.sym("x_rsw")         # m     
    y_rsw  = ca.MX.sym("y_rsw")         # m  
    z_rsw  = ca.MX.sym("z_rsw")         # m  
    vx_rsw = ca.MX.sym("vx_rsw")        # m/s   
    vy_rsw = ca.MX.sym("vy_rsw")        # m/s   
    vz_rsw = ca.MX.sym("vz_rsw")        # m/s  
    
    deputy_relative_state         = ca.vertcat(x_rsw,y_rsw,z_rsw,vx_rsw,vy_rsw,vz_rsw)
    deputy_relative_position_rsw  = ca.vertcat(x_rsw,y_rsw,z_rsw)
 

    deputy_global_position_rsw    = ca.vertcat(x_rsw+r_chief,y_rsw,z_rsw)
    deputy_global_velocity_rsw    = ca.vertcat(vr_chief+vx_rsw,vy_rsw,vz_rsw)
    r_deputy                      = ca.sqrt((x_rsw+r_chief)**2+(y_rsw)**2 +(z_rsw)**2)
    sin_phi_deputy                = ((x_rsw+r_chief)*ca.sin(theta_bar)*ca.sin(inc) + (y_rsw)*ca.cos(theta_bar)*ca.sin(inc)+ (z_rsw)*ca.cos(inc))/r_deputy
    sin_phi_chief                 = ca.sin(theta_bar)*ca.sin(inc)  

    # find inertial speed chief 
    chief_global_inertial_speed_square = vr_chief**2 + (h/r_chief)**2 
    #find deputy inertial speed
    omega_rsw_in    = obtain_RSW_frame_angular_velocity(gravitational_parameter=gravitational_parameter,
                                                        planet_mean_radius=planet_mean_radius,
                                                        max_spherical_harmonic_order=max_spherical_harmonic_order,
                                                        aerodynamic_settings_chief=aerodynamic_settings_chief) 


    deputy_global_inertial_velocity = deputy_global_velocity_rsw +  ca.cross(omega_rsw_in(implicit_keplerian_state_chief),deputy_global_position_rsw )
    deputy_global_inertial_speed_square = deputy_global_inertial_velocity[0]**2 + deputy_global_inertial_velocity[1]**2 + deputy_global_inertial_velocity[2]**2
    
    # find gravitational potential 
    potential_fun = obtain_spherical_harmonics_potential_function(gravitational_parameter=gravitational_parameter,
                                                                  planet_mean_radius = planet_mean_radius,
                                                                  max_spherical_harmonic_order= max_spherical_harmonic_order) 
    chief_potential = potential_fun(r_chief,sin_phi_chief)
    deputy_potential = potential_fun(r_deputy,sin_phi_deputy)


    total_energy_chief  = chief_potential  + 0.5*chief_global_inertial_speed_square 
    total_energy_deputy = deputy_potential + 0.5*deputy_global_inertial_speed_square
    
    # set energy matching condition
    energy_match_condition  = total_energy_chief  - total_energy_deputy 
    energy_match_function   = ca.Function("total_energy",[deputy_relative_state ,implicit_keplerian_state_chief],[energy_match_condition])
    
    return energy_match_function



def load_distributed_HOBF(epsilon_r:float,p0:float,p1:float,A:np.ndarray,B:np.ndarray,L:float=None,int_time:float=None) :
    # The dynamics is assumed to be linear
    
    # you have position of other agent but you don't have the full velocity
    # you only have the radial velocity
    
    # A abd B are state and contorl matrices in continuous time! the margin is then added to guarantee safety in between time steps

    x            = ca.MX.sym("x")         # m     
    y            = ca.MX.sym("y")         # m  
    z            = ca.MX.sym("z")         # m  
    vx           = ca.MX.sym("vx")         # m     
    vy           = ca.MX.sym("vy")         # m  
    vz           = ca.MX.sym("vz")         # m  

    
    
    
    x_opponent   = ca.MX.sym("x_opp")         # m     
    y_opponent   = ca.MX.sym("y_opp")         # m  
    z_opponent   = ca.MX.sym("z_opp")         # m  
    vx_opponent   = ca.MX.sym("vx_opp")         # m     
    vy_opponent   = ca.MX.sym("vy_opp")         # m  
    vz_opponent   = ca.MX.sym("vz_opp")         # m  
    
    state_opponent = ca.vertcat(x_opponent,y_opponent,z_opponent,vx_opponent,vy_opponent,vz_opponent)
    state          = ca.vertcat(x,y,z,vx,vy,vz)

    delta_u        = ca.MX.sym("delta_u",3,1)         
    
    eta_p = state_opponent[:3]-state[:3]
    eta_v = state_opponent[3:]-state[3:]
    delta_state  = state_opponent - state

    distance = ca.sqrt((eta_p[0])**2 + (eta_p[1])**2 + (eta_p[2])**2)
    
    eta_a = (ca.mtimes(A,delta_state) + ca.mtimes(B,delta_u))[3:] # acceleration only
    
    h      = distance**2 -  epsilon_r**2
    h_dot  = 2* eta_v.T@eta_p
    h_ddot = 2*ca.norm_2(eta_v)**2 + 2* ca.mtimes(eta_p.T,eta_a)

    H0  = h
    H1  = h_dot + p0*(H0)
    CBF_constraint_sym = h_ddot + (p1+p0)*h_dot + (p0*p1)*h

    H0    = ca.Function("CBF",[state ,state_opponent],[H0])
    H1    = ca.Function("CBF_order2",[state ,state_opponent],[H1])
    
    if  ((L==None) ^ (int_time==None)) :
        message = ("If you want to insert the discrete time margin you need to give both the Lipschitz constant and the integration time ")
        raise ValueError(message)
    else :
        if  not L ==None :
           barrier_constraint          = ca.Function("position_barrier_constraint",[state ,state_opponent,delta_u],[CBF_constraint_sym+L*int_time])
        else:
           barrier_constraint          = ca.Function("position_barrier_constraint",[state ,state_opponent,delta_u],[CBF_constraint_sym])
    
    return H0,H1,barrier_constraint
    
    




def obtain_spherical_harmonic_perturbing_acceleration_rsw(max_spherical_harmonic_order:float,gravitational_parameter:float,planet_mean_radius:float) :
    """
    model acceleration for spherical harmonics 
    given in rsw coordinates fixed at the chief location
    The input to the output function are the 
    """
    # cartesian coordinates
    x = ca.MX.sym("x")
    y = ca.MX.sym("y")
    z = ca.MX.sym("z")
    R = ca.sqrt(x**2+y**2+z**2) # norm
    X = ca.vertcat(x,y,z)       # vector state
    
    # state of the chief
    theta_bar    = ca.MX.sym("theta_bar")  # chief true anomaly    
    inc          = ca.MX.sym("inc")        # chief inclination  
    raan         = ca.MX.sym("raan")       # chief Raan         
    h            = ca.MX.sym("h")          # chief angular momentum vector 
    vr_chief     = ca.MX.sym("vr_chief")   # chief radial speed  
    r_chief      = ca.MX.sym("r_chief")    # chief geocentric distance       
    
    # rsw state chief
    implicit_keplerian_state_chief = ca.vertcat(theta_bar,inc,raan,h,vr_chief, r_chief )
    
    # rsw coordinates
    x_rsw = ca.MX.sym("x_rsw")
    y_rsw = ca.MX.sym("y_rsw")
    z_rsw = ca.MX.sym("z_rsw")
    
    agent_relative_position_rsw = ca.vertcat(x_rsw,y_rsw,z_rsw)
    agent_global_position_rsw = ca.vertcat(r_chief+x_rsw,y_rsw,z_rsw)
      
    C3      = obtain_symbolic_DCM(axis=3)
    C1      = obtain_symbolic_DCM(axis=1)
    DCM_inertial_to_rsw = ca.mtimes(ca.mtimes(C3(theta_bar),C1(inc)),C3(raan))
    
    # obtain rsw position, into inertial coordinate
    agent_global_position_in = ca.mtimes(DCM_inertial_to_rsw.T,agent_global_position_rsw )
    
    # define perturbing potential in cartesian coordinates
    potential = 0
    sin_pi    = z/R
    
    for kk,J in enumerate(spherical_harmonic_coefficients_list[:max_spherical_harmonic_order],start=1) :
        Pkk        = legendre_polynomial(kk)
        potential  =  potential + gravitational_parameter/R * J * (planet_mean_radius/R)**kk * Pkk(sin_pi)
    
    # this is still a function of the cartesian state and not rsw state
    total_gravity_acceleration_in     = - ca.gradient(potential,X) 
    total_gravity_acceleration_in_fun = ca.Function("grav_acc_in",[X],[total_gravity_acceleration_in  ])
    accelerarion_rsw                  = ca.mtimes(DCM_inertial_to_rsw,total_gravity_acceleration_in_fun(agent_global_position_in)) # perturbing acceleration in rsw coordinates and fiunction of implicit chief state and agent relative state
    
    return ca.Function("total_gravity_acceleration",[agent_relative_position_rsw,implicit_keplerian_state_chief],[accelerarion_rsw ])
# only for the chief and it will be a function of the chief state only

def obtain_aerodynamic_acceleration_on_chief(planet_mean_radius:float,
                                             aerodynamic_settings_chief:AerodynamicSettings) :
    
    """compute angular velocity and acceleration of the RSW frame of reference"""
    
    # state of the chief
    theta_bar    = ca.MX.sym("theta_bar")  # chief true anomaly    
    inc          = ca.MX.sym("inc")        # chief inclination  
    raan         = ca.MX.sym("raan")       # chief Raan         
    h            = ca.MX.sym("h")          # chief angular momentum vector 
    vr_chief     = ca.MX.sym("vr_chief")   # chief radial speed  
    r_chief      = ca.MX.sym("r_chief")    # chief geocentric distance       
    
    # rsw state chief
    implicit_keplerian_state_chief = ca.vertcat(theta_bar,inc,raan,h,vr_chief,r_chief)
    
    altitude                            = r_chief-planet_mean_radius 
    density                             = aerodynamic_settings_chief.zero_level_density*ca.exp(-altitude/aerodynamic_settings_chief.scale_height)
    inertial_atmospheric_speed_on_chief = ca.sqrt(vr_chief**2 + (h/r_chief-omega_earth*r_chief*ca.cos(inc))**2 + (omega_earth*ca.cos(theta_bar)*ca.sin(inc)*r_chief)**2)
    inertial_atmospheric_velocity_chief = ca.vertcat(vr_chief, (h/r_chief-omega_earth*r_chief*ca.cos(inc)),(omega_earth*ca.cos(theta_bar)*ca.sin(inc)*r_chief))
    
    f_drag = aerodynamic_settings_chief.drag_parameter*density* inertial_atmospheric_speed_on_chief *-inertial_atmospheric_velocity_chief
    return ca.Function("aerodynamic_acceleration_on_chief",[implicit_keplerian_state_chief ],[f_drag])
        
def obtain_RSW_frame_angular_velocity(gravitational_parameter:float,planet_mean_radius:float,max_spherical_harmonic_order:float=0,aerodynamic_settings_chief:AerodynamicSettings=None) :  
    
    """compute angular velocity and acceleration of the RSW frame of reference"""
    
    # state of the chief
    theta_bar    = ca.MX.sym("theta_bar")  # chief true anomaly    
    inc          = ca.MX.sym("inc")        # chief inclination  
    raan         = ca.MX.sym("raan")       # chief Raan         
    h            = ca.MX.sym("h")          # chief angular momentum vector 
    vr_chief     = ca.MX.sym("vr_chief")   # chief radial speed  
    r_chief      = ca.MX.sym("r_chief")    # chief geocentric distance       
    
    # rsw state chief
   
    implicit_keplerian_state_chief = ca.vertcat(theta_bar,inc,raan,h,vr_chief,r_chief)
    f_total_pert_fun = obtain_total_perturbing_acceleration_on_chief(max_spherical_harmonic_order=max_spherical_harmonic_order,
                                                                     gravitational_parameter = gravitational_parameter,
                                                                     planet_mean_radius = planet_mean_radius,
                                                                     aerodynamic_settings_chief=aerodynamic_settings_chief)
    
    f_total_pert = f_total_pert_fun(implicit_keplerian_state_chief)
    
    omega_r = r_chief/h * (f_total_pert[2])
    omega_s = 0
    omega_w = h/r_chief**2

    
    return ca.Function("omega_rsw", [implicit_keplerian_state_chief],[ca.vertcat(omega_r,omega_s,omega_w)])

def obtain_total_perturbing_acceleration_on_chief(max_spherical_harmonic_order:float,
                                                  gravitational_parameter:float,
                                                  planet_mean_radius:float,
                                                  aerodynamic_settings_chief:AerodynamicSettings) :

    """compute angular velocity and acceleration of the RSW frame of reference"""
    
    # state of the chief
    theta_bar    = ca.MX.sym("theta_bar")  # chief true anomaly    
    inc          = ca.MX.sym("inc")        # chief inclination  
    raan         = ca.MX.sym("raan")       # chief Raan         
    h            = ca.MX.sym("h")          # chief angular momentum vector 
    vr_chief     = ca.MX.sym("vr_chief")   # chief radial speed  
    r_chief      = ca.MX.sym("r_chief")    # chief geocentric distance       
    
    # rsw state chief
    chief_relative_position_rsw    = ca.vertcat(0,0,0)
    implicit_keplerian_state_chief = ca.vertcat(theta_bar,inc,raan,h,vr_chief,r_chief)
    
    # aerodynamic force on chief
    if  not aerodynamic_settings_chief == None  : 
       
        f_drag_chief_fun = obtain_aerodynamic_acceleration_on_chief(planet_mean_radius=planet_mean_radius,
                                                                    aerodynamic_settings_chief=aerodynamic_settings_chief)
        f_drag_chief = f_drag_chief_fun(implicit_keplerian_state_chief)
    else :
        
        f_drag_chief = ca.vertcat(0,0,0)
    
    # gravitational perturbing force 
    if not max_spherical_harmonic_order == 0 :
       f_sph_funtion = obtain_spherical_harmonic_perturbing_acceleration_rsw(max_spherical_harmonic_order=max_spherical_harmonic_order,
                                                                             gravitational_parameter= gravitational_parameter,
                                                                             planet_mean_radius=planet_mean_radius)
       f_sph   = f_sph_funtion(chief_relative_position_rsw,implicit_keplerian_state_chief)
    
    else :
        f_sph = ca.vertcat(0,0,0)
       
    f_pert_total = f_sph + f_drag_chief 

    return ca.Function("omega_rsw", [implicit_keplerian_state_chief],[f_pert_total])

def obtain_total_perturbing_acceleration_on_deputy(max_spherical_harmonic_order:float,
                                                   gravitational_parameter:float,
                                                   planet_mean_radius:float,
                                                   aerodynamic_settings_deputy:AerodynamicSettings,
                                                   aerodynamic_settings_chief:AerodynamicSettings) :
    
    """compute angular velocity and acceleration of the RSW frame of reference"""
    
    # state of the chief
    theta_bar    = ca.MX.sym("theta_bar")  # chief true anomaly    
    inc          = ca.MX.sym("inc")        # chief inclination  
    raan         = ca.MX.sym("raan")       # chief Raan         
    h            = ca.MX.sym("h")          # chief angular momentum vector 
    vr_chief     = ca.MX.sym("vr_chief")   # chief radial speed  
    r_chief      = ca.MX.sym("r_chief")    # chief geocentric distance       
    
    # rsw state chief
    implicit_keplerian_state_chief = ca.vertcat(theta_bar,inc,raan,h,vr_chief,r_chief)
    
    # deputy relative state 

    x_rsw  = ca.MX.sym("x_rsw ")               
    y_rsw  = ca.MX.sym("y_rsw ")             
    z_rsw  = ca.MX.sym("z_rsw ")             
    vx_rsw = ca.MX.sym("vx_rsw")             
    vy_rsw = ca.MX.sym("vy_rsw")   
    vz_rsw = ca.MX.sym("vz_rsw")

    deputy_relative_state         = ca.vertcat(x_rsw,y_rsw,z_rsw,vx_rsw,vy_rsw,vz_rsw)
    deputy_relative_position_rsw  = ca.vertcat(x_rsw,y_rsw,z_rsw)
 

    deputy_global_position_rsw    = ca.vertcat(x_rsw+r_chief,y_rsw,z_rsw)
    deputy_global_velocity_rsw    = ca.vertcat(vr_chief+vx_rsw,vy_rsw,vz_rsw)
    r_deputy = ca.sqrt((x_rsw+r_chief)**2 + (y_rsw)**2 + (z_rsw)**2)


    if aerodynamic_settings_deputy != None :
        earth_rotational_velocity_in = ca.vertcat(0,0,omega_earth)
        C3      = obtain_symbolic_DCM(axis=3)
        C1      = obtain_symbolic_DCM(axis=1)
        DCM_inertial_to_rsw = ca.mtimes(ca.mtimes(C3(theta_bar),C1(inc)),C3(raan))
        
        earth_rotational_velocity_rsw          = ca.mtimes(DCM_inertial_to_rsw ,earth_rotational_velocity_in)
        wind_inertial_velocity_at_deputy_rsw   = ca.cross(earth_rotational_velocity_rsw,deputy_global_position_rsw )
        
        
        omega_rsw_in_fun = obtain_RSW_frame_angular_velocity(gravitational_parameter=gravitational_parameter,
                                                            planet_mean_radius=planet_mean_radius,
                                                            max_spherical_harmonic_order=max_spherical_harmonic_order,
                                                            aerodynamic_settings_chief=aerodynamic_settings_chief) 

        omega_rsw_in                           = omega_rsw_in_fun(implicit_keplerian_state_chief)
        inertial_velocity_deputy_rsw           = deputy_global_velocity_rsw + ca.cross(omega_rsw_in,deputy_global_position_rsw)
        atmospheric_velocity_toward_deputy_rsw = wind_inertial_velocity_at_deputy_rsw - inertial_velocity_deputy_rsw 
        atmospheric_speed_rsw                  = ca.sqrt(ca.mtimes(atmospheric_velocity_toward_deputy_rsw.T,atmospheric_velocity_toward_deputy_rsw))

        altitude_deputy     = r_deputy-planet_mean_radius 
        density_deputy      = aerodynamic_settings_deputy.zero_level_density*ca.exp(-altitude_deputy /aerodynamic_settings_deputy.scale_height)
        f_drag_deputy       = aerodynamic_settings_deputy.drag_parameter * density_deputy * atmospheric_speed_rsw * atmospheric_velocity_toward_deputy_rsw
    
    else :
        f_drag_deputy = ca.vertcat(0,0,0)

        
    f_grav_deputy_fun = obtain_spherical_harmonic_perturbing_acceleration_rsw(max_spherical_harmonic_order=max_spherical_harmonic_order,gravitational_parameter=gravitational_parameter,planet_mean_radius=planet_mean_radius)
    f_grav            = f_grav_deputy_fun(deputy_relative_position_rsw,implicit_keplerian_state_chief)
    
    return ca.Function("aerodynamic_acceleration_on_chief",[deputy_relative_state,implicit_keplerian_state_chief ],[f_drag_deputy+f_grav])
        
def obtain_RSW_angular_acceleration_vector(gravitational_parameter:float,
                                              planet_mean_radius:float,
                                              max_spherical_harmonic_order:float=0,
                                              aerodynamic_settings_chief:AerodynamicSettings=None) :
    
    # state of the chief
    theta_bar    = ca.MX.sym("theta_bar")  # chief true anomaly    
    inc          = ca.MX.sym("inc")        # chief inclination  
    raan         = ca.MX.sym("raan")       # chief Raan         
    h            = ca.MX.sym("h")          # chief angular momentum vector 
    vr_chief     = ca.MX.sym("vr_chief")   # chief radial speed  
    r_chief      = ca.MX.sym("r_chief")    # chief geocentric distance    

    # rsw state chief
    chief_position_rsw             = ca.vertcat(r_chief,0,0)
    implicit_keplerian_state_chief = ca.vertcat(theta_bar,inc,raan,h,vr_chief, r_chief )
    inertial_velocity_chief_rsw    = ca.vertcat(vr_chief , h/r_chief,0)

    #  create rotation matrix inertial to rsw
    C3      = obtain_symbolic_DCM(axis=3)
    C1      = obtain_symbolic_DCM(axis=1)
    DCM_inertial_to_rsw = ca.mtimes(C3(theta_bar),ca.mtimes(C1(inc),C3(raan)))
    
    chief_velocity_in = ca.mtimes(DCM_inertial_to_rsw.T,inertial_velocity_chief_rsw) # inertial velocity in inertiaql coordinates
    chief_position_in = ca.mtimes(DCM_inertial_to_rsw.T,chief_position_rsw)
    
    # SPHERICAL HARMONICS ACCELERATION TIME DERIVATIVE 
    
    # built potential model staring from inertial cartesian state
    # cartesian coordinates of a point mass
    
    x_in = ca.MX.sym("x_in")
    y_in = ca.MX.sym("y_in")
    z_in = ca.MX.sym("z_in")
    vx_in = ca.MX.sym("vx_in")
    vy_in = ca.MX.sym("vy_in")
    vz_in = ca.MX.sym("vz_in")

    R    = ca.sqrt(x_in**2+y_in**2+z_in**2) # norm
    X_in = ca.vertcat(x_in,y_in,z_in)       # vector state
    V_in = ca.vertcat(vx_in,vy_in,vz_in)
    
    f_total_pert_fun = obtain_total_perturbing_acceleration_on_chief(max_spherical_harmonic_order=max_spherical_harmonic_order,
                                                                     gravitational_parameter = gravitational_parameter,
                                                                     planet_mean_radius = planet_mean_radius,
                                                                     aerodynamic_settings_chief=aerodynamic_settings_chief)

    f_total_pert = f_total_pert_fun(implicit_keplerian_state_chief)

    # dynamics of the chief implicit keplerian elements
    dtheta_bar_dt  =   h/r_chief**2  - r_chief*ca.sin(theta_bar)/h/ca.tan(inc) * (f_total_pert[2])
    dinc_dt        =   (r_chief/h) * ca.cos(theta_bar) * (f_total_pert[2])
    draan_dt       =   r_chief*ca.sin(theta_bar)/h/ca.sin(inc) * (f_total_pert[2])
    dh_dt          =   r_chief *(f_total_pert[1])
    dvr_chief_dt   =  - gravitational_parameter/r_chief**2 + h**2/r_chief**3 + f_total_pert[0]
    dr_chief_dt    =    vr_chief 


    if not max_spherical_harmonic_order == 0:
        # define perturbing potential in cartesian coordinates
        potential = 0
        sin_phi = z_in/R
        
        for kk,J in enumerate(spherical_harmonic_coefficients_list[:max_spherical_harmonic_order],start=1) :
            Pkk = legendre_polynomial(degree=kk)
            potential += gravitational_parameter/R * J * (planet_mean_radius/R)**kk * Pkk(sin_phi)
        
        # this is still a function of the cartesian state and not rsw state
        total_gravity_acceleration       = - ca.jacobian(potential,X_in) 
        # jacobian of the gravity acceleration
        jac_total_gravity_acceleration  = ca.jacobian(total_gravity_acceleration,X_in)
        # time derivative of the acceleration by chain rule
        # the coordinates are still inertial 
        total_gravity_acceleration_time_derivative = ca.mtimes(jac_total_gravity_acceleration,V_in)
        
        # now obtain the function
        total_gravity_acceleration_time_derivative_fun = ca.Function("inertial_time_derivative_of_the_acceleration",[X_in,V_in],[total_gravity_acceleration_time_derivative])
        
        # Now convert all in RSW coordinates and RSW state dependant function
        f_grav_dot = ca.mtimes(DCM_inertial_to_rsw,total_gravity_acceleration_time_derivative_fun(chief_position_in ,chief_velocity_in))
       
    
    else :
        f_grav_dot = ca.vertcat(0,0,0)


    # AERODYNAMIC ACCELERATION TIME DERIVATIVE

    if  not aerodynamic_settings_chief == None  : 
      altitude      =      r_chief-planet_mean_radius
      density       = aerodynamic_settings_chief.zero_level_density*ca.exp(-altitude/aerodynamic_settings_chief.scale_height) 
      ddensity_dt   = -1/aerodynamic_settings_chief.scale_height * vr_chief  * aerodynamic_settings_chief.zero_level_density*ca.exp(-altitude/aerodynamic_settings_chief.scale_height) 
      fw_drag_dot   = aerodynamic_settings_chief.drag_parameter*omega_earth*ca.sin(inc)*ca.sin(theta_bar)*r_chief*density*dtheta_bar_dt*((h/r_chief - omega_earth*ca.cos(inc)*r_chief)**2 + vr_chief**2 + omega_earth**2*ca.cos(theta_bar)**2*ca.sin(inc)**2*r_chief**2)**(1/2) - aerodynamic_settings_chief.drag_parameter*omega_earth*ca.cos(theta_bar)*ca.sin(inc)*density*vr_chief*((h/r_chief - omega_earth*ca.cos(inc)*r_chief)**2 + vr_chief**2 + omega_earth**2*ca.cos(theta_bar)**2*ca.sin(inc)**2*r_chief**2)**(1/2) - aerodynamic_settings_chief.drag_parameter*omega_earth*ca.cos(inc)*ca.cos(theta_bar)*r_chief*density*dinc_dt*((h/r_chief - omega_earth*ca.cos(inc)*r_chief)**2 + vr_chief**2 + omega_earth**2*ca.cos(theta_bar)**2*ca.sin(inc)**2*r_chief**2)**(1/2) - aerodynamic_settings_chief.drag_parameter*omega_earth*ca.cos(theta_bar)*ca.sin(inc)*r_chief*ddensity_dt*((h/r_chief - omega_earth*ca.cos(inc)*r_chief)**2 + vr_chief**2 + omega_earth**2*ca.cos(theta_bar)**2*ca.sin(inc)**2*r_chief**2)**(1/2) - (aerodynamic_settings_chief.drag_parameter*omega_earth*ca.cos(theta_bar)*ca.sin(inc)*r_chief*density*(2*vr_chief*dvr_chief_dt + 2*(h/r_chief - omega_earth*ca.cos(inc)*r_chief)*(dh_dt/r_chief - (h*vr_chief)/r_chief**2 - omega_earth*ca.cos(inc)*vr_chief + omega_earth*ca.sin(inc)*r_chief*dinc_dt) + 2*omega_earth**2*ca.cos(theta_bar)**2*ca.sin(inc)**2*r_chief*vr_chief + 2*omega_earth**2*ca.cos(inc)*ca.cos(theta_bar)**2*ca.sin(inc)*r_chief**2*dinc_dt - 2*omega_earth**2*ca.cos(theta_bar)*ca.sin(inc)**2*ca.sin(theta_bar)*r_chief**2*dtheta_bar_dt))/(2*((h/r_chief - omega_earth*ca.cos(inc)*r_chief)**2 + vr_chief**2 + omega_earth**2*ca.cos(theta_bar)**2*ca.sin(inc)**2*r_chief**2)**(1/2))
    else :
        fw_drag_dot =0

    
    
    omega_r_dot = dr_chief_dt/h * (f_total_pert[2]) -  dh_dt * r_chief/h**2 * (f_total_pert[2]) + r_chief/h * (fw_drag_dot + f_grav_dot[2])
    omega_s_dot = 0
    omega_w_dot = dh_dt/r_chief**2 - 2 * dr_chief_dt * h /r_chief**3

    return ca.Function("final_total_gravity_acceleration_time_derivative",[implicit_keplerian_state_chief],[ca.vertcat(omega_r_dot,omega_s_dot,omega_w_dot)]) 
  
def obtain_spherical_harmonics_potential_function(gravitational_parameter:float,
                                                  planet_mean_radius:float,
                                                  max_spherical_harmonic_order:float=0) : 

    r       =  ca.MX.sym("r") # distance from the planet
    sin_phi =  ca.MX.sym("sin_phi") # sine of the latitude
    
    potential = - gravitational_parameter/r # order zero potential

    for kk,J in enumerate(spherical_harmonic_coefficients_list[:max_spherical_harmonic_order],start=1) :
        Pkk = legendre_polynomial(degree=kk)
        potential += gravitational_parameter/r * J * (planet_mean_radius/r)**kk * Pkk(sin_phi)
    
    potential_fun = ca.Function("potential",[r,sin_phi],[potential])
    return potential_fun


def load_cone_barrier_constraint(p0:float,p1:float,cone_angle_ref :float,continuous_agent_dynamics:ca.Function) :
    # cone angle reference -> semicone angle in radiants
    # the dynamics of the agent is the linear dynamics only. It is a good approximton for circular orbits and 
    # it will fail for elliptocal orbits or for large separations. Considering that we sty close to the debris 
    # it is a good assumption. Further development can involve roboustness for eccentric orbits and change of the model 
    # through the Yamanaka-Ankersen state transition matrix for example, which is suitable for elliptic orbits. 
    # We can think about this.


    x  = ca.MX.sym("x")         # m     
    y  = ca.MX.sym("y")         # m  
    z  = ca.MX.sym("z")         # m  
    dr = ca.vertcat(x,y,z)

    vx = ca.MX.sym("vx")        # m/s   
    vy = ca.MX.sym("vy")        # m/s   
    vz = ca.MX.sym("vz")        # m/s 
    dvel = ca.vertcat(vx,vy,vz)

    state = ca.vertcat(dr,dvel)
    # docking port direction in the RSW frame not the body frame of the debris 
    # always remember which frame are you working in. Also note that here you should really add the 
    # the magnitude and direction of the docking port. Then the normalization is given inside the barrier definiton.
    # you don't need to give already a normalised diretion. So remember that bro!
    
    px  = ca.MX.sym("px")         #    
    py  = ca.MX.sym("py")         # 
    pz  = ca.MX.sym("pz")         # 
    
    p_dock = ca.vertcat(px,py,pz)
    pdir = p_dock/ca.sqrt(ca.mtimes(p_dock.T,p_dock))

    # angular velocity of the debris in the RSW frame of reference 
    # not the body frame. for 2D case this two may be the same 
    # because it is aligned with the z axis in both cases.
    # if you want to change the code to 3d you will need to create a 
    # a proper transformation for your needs

    omega_x  = ca.MX.sym("omega_x")   # m     
    omega_y  = ca.MX.sym("omega_y")   # m  
    omega_z  = ca.MX.sym("omega_z")   # m  
    
    omega = ca.vertcat(omega_x,omega_y,omega_z)

    # pdot 

    pdirdot = ca.cross(omega,pdir)

    # control translational
    ux  = ca.MX.sym("u1")       # Control
    uy  = ca.MX.sym("u2")       # Control
    uz  = ca.MX.sym("u2")       # Control
    control = ca.vertcat(ux,uy,uz)
    
    # it is the linear continuous dynamics x_dot = Ax + Bu
    dveldot = continuous_agent_dynamics(state,control)[3:]

    # defintion of the CBF
    dvel_dot_dr       = ca.mtimes(dvel.T,dr)
    dvel_dot_pdir     = ca.mtimes(dvel.T,pdir) 
    dvel_dot_pdirdot = ca.mtimes(dvel.T,pdirdot) 
    dr_dot_pdir      = ca.mtimes(dr.T,pdir)
    dr_dot_pdirdot   = ca.mtimes(dr.T,pdirdot)
    dr_norm           = ca.sqrt(ca.mtimes(dr.T,dr))
    dvel_norm_square  = ca.mtimes(dvel.T,dvel)
    dveldot_dot_pdir = ca.mtimes(dveldot.T,pdir)
    dveldot_dot_dr    = ca.mtimes(dveldot.T,dr)
    
    h      = dr_dot_pdir/dr_norm - ca.cos(cone_angle_ref)
    Q      = dr_dot_pdir/dr_norm
    h_dot  = (dvel_dot_pdir + dr_dot_pdirdot)/dr_norm - dvel_dot_dr/dr_norm**2 * Q
    h_ddot = (dveldot_dot_pdir + 2*dvel_dot_pdirdot)/dr_norm - (dvel_dot_pdir + dr_dot_pdirdot)*(dvel_dot_dr)/dr_norm**3 - ( Q * (-2*dvel_dot_dr**2/dr_norm**4 + (dveldot_dot_dr + dvel_norm_square)/dr_norm**2) + dvel_dot_dr/dr_norm**2 * h_dot)
    
    condition_1 = p0*h  + h_dot # this condition must be respected for the system to be in C safe at the beginning
    condition_2 = h                   # this condition must be respected for the system to be in C safe at the beginning
    
    initial_condition_check = ca.logic_and(condition_1>=0,condition_2>=0)

    CBF_sym            = h
    CBF1               = h_dot  + p0*h
    CBF_constraint_sym = h_ddot + p0*h_dot + p1*( CBF1 ) 
    

    CBF                         = ca.Function("cone_barrier_function",[state,p_dock],[CBF_sym])
    barrier_constraint          = ca.Function("cone_barrier_constraint",[state,p_dock,omega,control],[CBF_constraint_sym])
    safe_set_check              = ca.Function("safe_cone_check",[state,p_dock,omega],[initial_condition_check])

    return CBF,barrier_constraint,safe_set_check


def load_velocity_barrier_constraint(epsilon_v:float,p0:float,agent_dynamics:ca.Function,model:str="nonlinear"):
    
    x  = ca.MX.sym("x")         # m     
    y  = ca.MX.sym("y")         # m  
    z  = ca.MX.sym("z")         # m  
    vx = ca.MX.sym("vx")        # m/s   
    vy = ca.MX.sym("vy")        # m/s   
    vz = ca.MX.sym("vz")        # m/s 
     
    # state of the chief
    theta_bar  = ca.MX.sym("theta_bar")  # true anomaly    
    inc        = ca.MX.sym("inc")        # inclination  
    raan       = ca.MX.sym("raan")       # Raan         
    h          = ca.MX.sym("h")          # angular momentum vector 
    vr_chief   = ca.MX.sym("vr_chief")     # radial speed  
    r_chief    = ca.MX.sym("r_chief")      # geocentric distance 

    # control translational
    ux  = ca.MX.sym("u1")       # Control
    uy  = ca.MX.sym("u2")       # Control
    uz  = ca.MX.sym("u2")       # Control
    
  
    state_agent_translational  = ca.vertcat(x,y,z,vx,vy,vz)
    state_chief                = ca.vertcat(theta_bar,inc,raan,h,vr_chief,r_chief)
    
    translational_control          = ca.vertcat(ux,uy,uz) 
   
    velocity_reference      = ca.MX.sym("velocity_reference",3)
    position_reference      = ca.MX.sym("position_reference",3)
    reference_state         = ca.vertcat(position_reference,velocity_reference )
    
    velocity_agent          = ca.vertcat(vx,vy,vz)
    
    position_agent          = ca.vertcat(x,y,z)
 
    eta_v = (velocity_agent - velocity_reference)
    eta_r = (position_agent - position_reference)
    eta_v_normsquare =  ca.mtimes((velocity_agent-velocity_reference).T,(velocity_agent-velocity_reference))
    eta_r_normsquare =  ca.mtimes((position_agent - position_reference).T,(position_agent - position_reference)) 
    
    if model == "nonlinear":
        agent_dynamics_sym      = agent_dynamics(state_agent_translational,translational_control,state_chief)
    elif model == "linear" :
        agent_dynamics_sym      = agent_dynamics(state_agent_translational,translational_control)

    CBF_sym             =  epsilon_v**2 - eta_v_normsquare
    CBF_dot_sym         = - 2*ca.mtimes(eta_v.T,agent_dynamics_sym[3:6])                                                                           
                                                                     
    barrier_constraint_sym     = CBF_dot_sym + p0*CBF_sym


    CBF                         = ca.Function("CBF",[state_agent_translational,reference_state],[CBF_sym])
    barrier_constraint          = ca.Function("position_barrier_constraint",[state_agent_translational, translational_control,state_chief,reference_state],[barrier_constraint_sym])
    
    return CBF,barrier_constraint


def load_position_barrier_constraint(epsilon_r:float,p0:float,p1:float,agent_dynamics:ca.Function,model:str="nonlinear") :
    
    x  = ca.MX.sym("x")         # m     
    y  = ca.MX.sym("y")         # m  
    z  = ca.MX.sym("z")         # m  
    vx = ca.MX.sym("vx")        # m/s   
    vy = ca.MX.sym("vy")        # m/s   
    vz = ca.MX.sym("vz")        # m/s 
     
    # state of the chief
    theta_bar  = ca.MX.sym("theta_bar")  # true anomaly    
    inc        = ca.MX.sym("inc")        # inclination  
    raan       = ca.MX.sym("raan")       # Raan         
    h          = ca.MX.sym("h")          # angular momentum vector 
    vr_chief   = ca.MX.sym("vr_chief")     # radial speed  
    r_chief    = ca.MX.sym("r_chief")      # geocentric distance 

    # control translational
    ux  = ca.MX.sym("u1")       # Control
    uy  = ca.MX.sym("u2")       # Control
    uz  = ca.MX.sym("u2")       # Control
    
  
    state_agent_translational      = ca.vertcat(x,y,z,vx,vy,vz)
    state_chief                    = ca.vertcat(theta_bar,inc,raan,h,vr_chief,r_chief)
    
    translational_control          = ca.vertcat(ux,uy,uz) 
   
    velocity_reference      = ca.MX.sym("velocity_reference",3)
    position_reference      = ca.MX.sym("position_reference",3)
    reference_state         = ca.vertcat(position_reference,velocity_reference )
    
    velocity_agent          = ca.vertcat(vx,vy,vz)
    position_agent          = ca.vertcat(x,y,z)
 
    eta_v = (velocity_agent - velocity_reference)
    eta_r = (position_agent - position_reference)
    eta_v_normsquare =  ca.mtimes(eta_v .T,eta_v )
    eta_r_normsquare =  ca.mtimes(eta_r.T,eta_r) 
    
    if model == "nonlinear":
        agent_dynamics_sym      = agent_dynamics(state_agent_translational,translational_control,state_chief)
        agent_dynamics_unactuated_sym   = agent_dynamics(state_agent_translational,np.array([0,0,0]),state_chief)
    elif model == "linear" :
        agent_dynamics_sym      = agent_dynamics(state_agent_translational,translational_control)
        agent_dynamics_unactuated_sym   = agent_dynamics(state_agent_translational,np.array([0,0,0]))

    CBF_sym             =  epsilon_r**2 - eta_r_normsquare
    CBF_dot_sym         = - 2*ca.mtimes(eta_r.T,eta_v)                                                                         
    CBF_ddot_sym        = - 2*eta_v_normsquare - 2*ca.mtimes(eta_r.T,agent_dynamics_sym[3:6])    

    CBF_ddot_sym_unactuated  = - 2*eta_v_normsquare - 2*ca.mtimes(eta_r.T,agent_dynamics_unactuated_sym[3:6])                                                                 
    
 
    barrier_constraint_sym             = CBF_ddot_sym + (p0+p1)*CBF_dot_sym + (p0*p1)*CBF_sym
    un_actuated_barrier_constraint_sym =  CBF_ddot_sym_unactuated + (p0+p1)*CBF_dot_sym + (p0*p1)*CBF_sym

  
    CBF                         = ca.Function("CBF",[state_agent_translational,reference_state],[CBF_sym])
    HOBF                        = ca.Function("CBF_order2",[state_agent_translational,reference_state],[CBF_dot_sym    + CBF_sym*p0])
    barrier_constraint          = ca.Function("position_barrier_constraint",[state_agent_translational, translational_control,state_chief,reference_state],[barrier_constraint_sym])
    un_actuated_barrier_constraint     = ca.Function("position_barrier_constraint",[state_agent_translational,state_chief,reference_state],[un_actuated_barrier_constraint_sym])
  
    return CBF,HOBF ,barrier_constraint,un_actuated_barrier_constraint

def load_position_barrier_test1(epsilon_r:float,agent_dynamics:ca.Function,model:str="nonlinear") :
    
    x  = ca.MX.sym("x")         # m     
    y  = ca.MX.sym("y")         # m  
    z  = ca.MX.sym("z")         # m  
    vx = ca.MX.sym("vx")        # m/s   
    vy = ca.MX.sym("vy")        # m/s   
    vz = ca.MX.sym("vz")        # m/s 
     
    # state of the chief
    theta_bar  = ca.MX.sym("theta_bar")  # true anomaly    
    inc        = ca.MX.sym("inc")        # inclination  
    raan       = ca.MX.sym("raan")       # Raan         
    h          = ca.MX.sym("h")          # angular momentum vector 
    vr_chief   = ca.MX.sym("vr_chief")     # radial speed  
    r_chief    = ca.MX.sym("r_chief")      # geocentric distance 

    # control translational
    ux  = ca.MX.sym("u1")       # Control
    uy  = ca.MX.sym("u2")       # Control
    uz  = ca.MX.sym("u2")       # Control
    
  
    state_agent_translational      = ca.vertcat(x,y,z,vx,vy,vz)
    state_chief                    = ca.vertcat(theta_bar,inc,raan,h,vr_chief,r_chief)
    
    translational_control          = ca.vertcat(ux,uy,uz) 
   
    velocity_reference      = ca.MX.sym("velocity_reference",3)
    position_reference      = ca.MX.sym("position_reference",3)
    reference_state         = ca.vertcat(position_reference,velocity_reference )
    
    velocity_agent          = ca.vertcat(vx,vy,vz)
    position_agent          = ca.vertcat(x,y,z)
 
    eta_v = (velocity_agent - velocity_reference)
    eta_r = (position_agent - position_reference)
    eta_v_normsquare =  ca.mtimes(eta_v .T,eta_v )
    eta_r_normsquare =  ca.mtimes(eta_r.T,eta_r) 
    
    if model == "nonlinear":
        agent_dynamics_sym      = agent_dynamics(state_agent_translational,translational_control,state_chief)
    elif model == "linear" :
        agent_dynamics_sym      = agent_dynamics(state_agent_translational,translational_control)

    CBF_sym             =  epsilon_r**2 - eta_r_normsquare
    CBF_dot_sym         = - 2*ca.mtimes(eta_r.T,eta_v)                                                                         
    #CBF_ddot_sym        = - 2*eta_v_normsquare - 2*ca.mtimes(eta_r.T,agent_dynamics_sym[3:6])   
    CBF_ddot_sym        = - 2*eta_v_normsquare - 2*ca.mtimes(eta_r.T,translational_control)                                                                     
    
    
    # barrier_constraint_sym   = CBF_ddot_sym + (p0+p1)*CBF_dot_sym  + p0*p1*CBF_sym  
    barrier_constraint_sym     = CBF_ddot_sym + 3*CBF_dot_sym*CBF_sym**2  + (CBF_dot_sym  + CBF_sym**3)**3 

    barrier_constraint          = ca.Function("position_barrier_constraint",[state_agent_translational, translational_control,state_chief,reference_state],[barrier_constraint_sym])
  
    return barrier_constraint

def load_position_barrier_first_order(epsilon_r:float,agent_dynamics:ca.Function,model:str="nonlinear") :
    
    x  = ca.MX.sym("x")         # m     
    y  = ca.MX.sym("y")         # m  
    z  = ca.MX.sym("z")         # m  
    vx = ca.MX.sym("vx")        # m/s   
    vy = ca.MX.sym("vy")        # m/s   
    vz = ca.MX.sym("vz")        # m/s 
     
    # state of the chief
    theta_bar  = ca.MX.sym("theta_bar")  # true anomaly    
    inc        = ca.MX.sym("inc")        # inclination  
    raan       = ca.MX.sym("raan")       # Raan         
    h          = ca.MX.sym("h")          # angular momentum vector 
    vr_chief   = ca.MX.sym("vr_chief")     # radial speed  
    r_chief    = ca.MX.sym("r_chief")      # geocentric distance 

    # control translational
    ux  = ca.MX.sym("u1")       # Control
    uy  = ca.MX.sym("u2")       # Control
    uz  = ca.MX.sym("u2")       # Control
    
  
    state_agent_translational      = ca.vertcat(x,y,z,vx,vy,vz)
    state_chief                    = ca.vertcat(theta_bar,inc,raan,h,vr_chief,r_chief)
    
    translational_control          = ca.vertcat(ux,uy,uz) 
   
    velocity_reference      = ca.MX.sym("velocity_reference",3)
    position_reference      = ca.MX.sym("position_reference",3)
    reference_state         = ca.vertcat(position_reference,velocity_reference )
    
    velocity_agent          = ca.vertcat(vx,vy,vz)
    position_agent          = ca.vertcat(x,y,z)
 
    eta_v = (velocity_agent - velocity_reference)
    eta_r = (position_agent - position_reference)
    eta_v_normsquare =  ca.mtimes(eta_v .T,eta_v )
    eta_r_normsquare =  ca.mtimes(eta_r.T,eta_r) 
    
    if model == "nonlinear":
        agent_dynamics_sym      = agent_dynamics(state_agent_translational,translational_control,state_chief)
    elif model == "linear" :
        agent_dynamics_sym      = agent_dynamics(state_agent_translational,translational_control)

    CBF_sym             =  epsilon_r**2 - eta_r_normsquare
    CBF_dot_sym         = - 2*ca.mtimes(eta_r.T,eta_v)                                                                         
    
    # barrier_constraint_sym   = CBF_ddot_sym + (p0+p1)*CBF_dot_sym  + p0*p1*CBF_sym  
    barrier_constraint_sym     = CBF_dot_sym  +  CBF_sym  
    barrier_constraint          = ca.Function("position_barrier_constraint",[state_agent_translational, translational_control,state_chief,reference_state],[barrier_constraint_sym])
  
    return barrier_constraint







