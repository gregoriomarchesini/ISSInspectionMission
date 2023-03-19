
import sys, os
sys.path.insert(0, "/Users/gregorio/Desktop/Thesis/code/python/src")

from tudatpy.kernel.astro     import element_conversion
import numpy as np
import warnings

# list of abbreviations
# _in     --> inertial frame observer
# _h      --> h     frame observer
# _traget --> chief spacecraft
# _deputy --> deputy spacecraft
# _cstate --> cartesian
# _kstate --> keplerian state

def classic2implicit_keplerian_elements(keplerian_elements:np.ndarray,mu : float) :
    
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
    
    mu     (float)   : gravitational parameter of the planet.
                       Earth is assumed                                       

    Returns
    -----------
  
    implicit_keplerian_elements    (np.ndarray(6,1))  : Implicit keplerian elements vectors
                                                        The units are SI and the following order 
                                                        must be followed 
                                                        
                                                        theta_bar ->   argument of periapsis + true anomaly
                                                        i         ->   inclination
                                                        Raan      ->   Right Ascension of the Ascending Node
                                                        h         ->   angular momentum norm
                                                        vr        ->   radial velocity [component along the the position vector] 
                                                        r         ->   norm of the position vector
                                                            
    Descriptiom
    -----------
    Returns set of implicit keplerian parameters given the classic keplerian parameter.
    Only elliptical orbits are supported at the moment
    
    """
    
    try :
      a,e,i,omega,Raan,theta = keplerian_elements 
    except ValueError :
        raise ValueError("Check the size of you input. It should be (6,) but it is {}".format(np.shape(keplerian_elements)))
    
    cartesian_state = element_conversion.keplerian_to_cartesian(keplerian_elements=keplerian_elements ,
                                                                gravitational_parameter=mu)
    
    cartesian_position = cartesian_state[:3]
    cartesian_velocity = cartesian_state[3:]
    position_direction = cartesian_position/np.linalg.norm(cartesian_position)
    
    theta_bar = omega+theta
    h         = np.sqrt(mu*a*(1-e**2))
    r         = a* (1-e**2) * 1/(1+e*np.cos(theta))
    vr        = np.matmul(cartesian_velocity,position_direction)
   
    
    implicit_keplerian_elements = np.array([theta_bar,i,Raan,h,vr,r])
    
    return implicit_keplerian_elements

def implicit2classic_keplerian_elements(implicit_keplerian_elements:np.ndarray,mu : float ) :
    
    """
    Parameters
    -----------             
                       
    implicit_keplerian_elements    (np.ndarray(6,1))  : Implicit keplerian elements vectors
                                    The units are SI and the following order 
                                    must be followed 
                                    
                                    theta_bar ->   argument of periapsis + true anomaly
                                    i         ->   inclination
                                    Raan      ->   Right Ascension of the Ascending Node
                                    h         ->   angular momentum norm
                                    vr        ->   radial velocity [component along the the position vector] 
                                    r         ->   norm of the position vector
    
    mu     (float)   : gravitational parameter of the planet.
                       Earth is assumed 
                                            

    Returns
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
                                    
    Descriptiom
    -----------
    Returns set of classic keplerian parameters given the implicit keplerian parameters
    Only elliptical orbits are supported at the moment
    """
 
    try :
      theta_bar,i,Raan,h,vr,r = implicit_keplerian_elements 
    except ValueError :
        raise ValueError("Check the size of you input. It should be (6,) but it is {}".format(np.shape(implicit_keplerian_elements)))
    
    
    vs = h/r                       # horizontal component of the speed (s component in RSW frame)
    Vp2 = vr**2 + vs**2            # speed norm
    a  = 0.5*1/(1/r - 0.5*Vp2/mu)  # semimajor axis
    
    # for extremely close to circular orbits there could be issues of
    # eccentricity becoming negative. Here a trash hold is defined before the conversion
    # is defined and infeasible
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            e = np.sqrt(1-h**2/mu/a)
        except Warning as err:
            
            if abs(1-h**2/mu/a) <= 1E-9 and  np.sign(1-h**2/mu/a) == -1 :
              e = 0.0
            else :
              raise ValueError("infeasible conversion. Check the correctness of the parameters")
            
     

    # eccentricity vector in RSW components
    er = (vs*h/mu-1)  # radial component     (r component in RSW frame)
    es = - vr*h/mu    # horizontal component (s component in RSW frame)
    
    # norm of the eccentricity vector
    e_norm = np.sqrt(er**2 + es**2) 
    
    # unrotate of theta_bar and find coordinates
    cos_omega = np.cos(-theta_bar) * er/e_norm + np.sin(-theta_bar)*es/e_norm
    sin_omega = -np.sin(-theta_bar) * er/e_norm + np.cos(-theta_bar)*es/e_norm
    
    omega     = np.arctan2(sin_omega,cos_omega)
    theta     = theta_bar - omega
    
    keplerian_elements  = np.array([a,e,i,omega,Raan,theta])
    
    return keplerian_elements 

def classic2implicit_keplerian_elements_pointwise(a     :float,
                                                  e     :float,
                                                  i     :float, 
                                                  omega :float,
                                                  Raan  :float,
                                                  theta :float,
                                                  mu : float ) :
    
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
    
    mu     (float)   : gravitational parameter of the planet.
                       Earth is assumed                                       

    Returns
    -----------
    
    implicit_keplerian_elements    (np.ndarray(6,1))  : Implicit keplerian elements vectors
                                                        The units are SI and the following order 
                                                        must be followed 
                                                        
                                                        theta_bar ->   argument of periapsis + true anomaly
                                                        i         ->   inclination
                                                        Raan      ->   Right Ascension of the Ascending Node
                                                        h         ->   angular momentum norm
                                                        vr        ->   radial velocity [component along the the position vector] 
                                                        r         ->   norm of the position vector
                                                            
    Descriptiom
    -----------
    Returns set of implicit keplerian parameters given the classic keplerian parameter.
    Only elliptical orbits are supported at the moment
    
    """
    
   
    keplerian_elements = np.array([a,e,i,omega,Raan,theta]) 
    cartesian_state = element_conversion.keplerian_to_cartesian(keplerian_elements=keplerian_elements ,
                                                                gravitational_parameter=mu)
    
    cartesian_position = cartesian_state[:3]
    cartesian_velocity = cartesian_state[3:]
    position_direction = cartesian_position/np.linalg.norm(cartesian_position)
    
    theta_bar = omega+theta
    h         = np.sqrt(mu*a*(1-e**2))
    r         = a* (1-e**2) * 1/(1+e*np.cos(theta))
    vr        = np.matmul(cartesian_velocity,position_direction)
   
    
    implicit_keplerian_elements = np.array([theta_bar,i,Raan,h,vr,r])
    
    return implicit_keplerian_elements

def implicit2classic_keplerian_elements_pointwise( theta_bar:  float,
                                                    i       :  float, 
                                                    Raan    :  float,
                                                    h       :  float, 
                                                    vr      :  float, 
                                                    r       :  float, 
                                                    mu      :  float ) :
    
    """
    Parameters
    -----------             
                       
    implicit_keplerian_elements    (np.ndarray(6,1))  : Implicit keplerian elements vectors
                                    The units are SI and the following order 
                                    must be followed 
                                    
                                    theta_bar ->   argument of periapsis + true anomaly
                                    i         ->   inclination
                                    Raan      ->   Right Ascension of the Ascending Node
                                    h         ->   angular momentum norm
                                    vr        ->   radial velocity [component along the the position vector] 
                                    r         ->   norm of the position vector
    
    mu     (float)   : gravitational parameter of the planet.
                       Earth is assumed 
                                            

    Returns
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
                                    
    Descriptiom
    -----------
    Returns set of classic keplerian parameters given the implicit keplerian parameters
    Only elliptical orbits are supported at the moment
    """


    vs = h/r                        # horizontal component of the speed (s component in RSW frame)
    V  = np.sqrt(vr**2 + vs**2)     # speed norm
    a  = 0.5*1/(1/r - 0.5*V**2/mu)  # semimajor axis
    
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            e = np.sqrt(1-h**2/mu/a)
        except Warning as err:
            
            if abs(1-h**2/mu/a) <= 1E-9 and  np.sign(1-h**2/mu/a) == -1 :
              e = 0.0
            else :
              raise ValueError("infeasible conversion. Check the correctness of the parameters")
            
            
     
    # eccentricity vector in RSW components
    er = (vs*h/mu-1)  # radial component     (r component in RSW frame)
    es = - vr*h/mu    # horizontal component (s component in RSW frame)
    
    # norm of the eccentricity vector
    e_norm = np.sqrt(er**2 + es**2) 
    
    # unrotate of theta_bar and find coordinates
    cos_omega = np.cos(-theta_bar) * er/e_norm + np.sin(-theta_bar)*es/e_norm
    sin_omega = -np.sin(-theta_bar) * er/e_norm + np.cos(-theta_bar)*es/e_norm
    
    omega     = np.arctan2(sin_omega,cos_omega)
    theta     = theta_bar - omega
    
    keplerian_elements  = np.array([a,e,i,omega,Raan,theta])
    
    return keplerian_elements 



