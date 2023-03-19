import sys,os
sys.path.insert(0, "/Users/gregorio/Desktop/Thesis/code/python")
import  numpy as np
import casadi as ca
import  src.conversions_tools as conversions
import  matplotlib.pyplot as plt
import  src.dynamic_models as dynamic_models
import  src.simulation_utilities as sim_tool
from    src.CMPC_final import CMPC
from    tqdm import tqdm
import  scipy.linalg
import pandas as pd
import itertools

np.random.seed(2021)

save_simulation_results = True

# Initial Earth Parameters

R_earth     = 6378137.0               # m
mu_earth    = 398600441800000         # m3/2
max_spherical_harmonic_order_model = 2
max_spherical_harmonic_order_simulation = 6

density_at_ISS_altitude     = 3*1E-12 #kg/m^3  --> 3g/km^3  
scale_height_at_iss         = - 417000/(np.log(density_at_ISS_altitude/1.225))#

aerodynamic_settings_chief_model  = None 
aerodynamic_settings_deputy_model = None

aerodynamic_settings_chief_simulation  = dynamic_models.AerodynamicSettings(drag_coefficient=2.07,Across=130,mass=416E3,zero_level_density=1.225,scale_height=scale_height_at_iss )
aerodynamic_settings_deputy_simulation = dynamic_models.AerodynamicSettings(drag_coefficient=2.07,Across=0.0600,mass=10,zero_level_density=1.225,scale_height=scale_height_at_iss)

# DEFINE INITIAL CHIEF ORBIT

perigee_altitude = 417E3         # [m]
ecc              = 0.0004422  
omega            = 1E-8            # [rad]
Raan             = 1E-8            # [rad]
true_anomaly     = np.pi/4       # [rad]
inc              = 51 *np.pi/180 # [rad]
semimajor_a      = (R_earth+perigee_altitude)/(1-ecc)      #[m]
orbital_periods  = 2*np.pi*np.sqrt(semimajor_a**3/mu_earth)

chief_keplerian_state        = np.array([semimajor_a,ecc,inc,omega,Raan,true_anomaly])
implicit_keplerian_state_iss = conversions.classic2implicit_keplerian_elements(chief_keplerian_state, mu=mu_earth)

## OBTAIN TRAJECTORY TO FOLLOW
n = 2*np.pi/orbital_periods # circular orbit at 500km altitude is assumed fot linear trajectory 
                            # initialization

# set three identical trajectory offestted by some space on the along gtrack direction

rhoy0  = 0  ;   # aplitude in x/y plane (xy plane is always ellipse rhox:2*rhox) 
rhox   = 70;   # y axis offset
rhoz   = 0;   # amplitude in z-axis

alphax = -140*np.pi/180 # x-phase rad
alphaz = 0.0 # z-phase rad

trajectory    = sim_tool.obtain_PRO_trajectory_from_geometry(rhox,rhoz,rhoy0,alphax=alphax,alphaz=alphaz,mean_motion=n)

# in theory each orbit should revise the constraints since each orbit requires
# different speeds, but for similar PROS the speed and position reached are roughly the same
# you can think of grouping agents with the same similar PROS.

epsilon_d = 1E-5  # m/s^2 max aerodynamic and J2 effect
c_r       =  20.04   # max constant L_g Lie derivative position barrier
c_v       =  0.3096   # max constant L_g Lie derivative position barrier
epsilon_r = 10    # max alowed posotion difference
inner_d_safe = 3  # [m]
p0_dr = 0.01
p1_dr = 0.1

epsilon_v = p0_dr/2*(epsilon_r**2 -inner_d_safe**2)/(inner_d_safe)  # max predicted speed difference in the free drift phase from the reference orbit

p0_dv = 0.1
# max control from propulsion system per axis
u_max = 0.0200 # m/s^2

# obtained from matlab program
L_dr    = 0.0756
L_dv    = 0.0016
L_un_dr =  0.0115
L_un_dv = 3.6417e-05

## DISCRETIZED MODEL SETTINGS
int_time         = 0.15     # discretization time for the model. It also the actiation time of the engine
n_steps_rk4      = 1      # number of steps of RK4 inside the integration time (add precision at the integration)



Ad,Bd = dynamic_models.load_discrete_linear_dynamics_matrices(orbital_periods, int_time )

## RETRIVE CASADI-FUNCTION DISCRETISED MODELS
iss_discrete_dynamics_sim   = dynamic_models.load_chief_discrete_dynamics(integration_time=int_time ,
                                                                      n_steps=n_steps_rk4,
                                                                      gravitational_parameter=mu_earth,
                                                                      planet_mean_radius=R_earth,
                                                                      max_spherical_harmonic_order=max_spherical_harmonic_order_simulation ,
                                                                      aerodynamic_settings_chief=aerodynamic_settings_chief_simulation)

agent_discrete_dynamics_sim  = dynamic_models.load_agent_translational_discrete_dynamics(integration_time=int_time ,
                                                                                     n_steps=n_steps_rk4,
                                                                                     gravitational_parameter=mu_earth,
                                                                                     planet_mean_radius=R_earth,
                                                                                     max_spherical_harmonic_order=max_spherical_harmonic_order_simulation ,
                                                                                     aerodynamic_settings_chief=aerodynamic_settings_chief_simulation,
                                                                                     aerodynamic_settings_deputy=aerodynamic_settings_deputy_simulation)

agent_discrete_dynamics_model  = dynamic_models.load_agent_translational_discrete_dynamics(integration_time=int_time ,
                                                                                     n_steps=n_steps_rk4,
                                                                                     gravitational_parameter=mu_earth,
                                                                                     planet_mean_radius=R_earth,
                                                                                     max_spherical_harmonic_order=max_spherical_harmonic_order_model ,
                                                                                     aerodynamic_settings_chief=aerodynamic_settings_chief_model,
                                                                                     aerodynamic_settings_deputy=aerodynamic_settings_deputy_model)

iss_discrete_dynamics_model   = dynamic_models.load_chief_discrete_dynamics(integration_time=int_time ,
                                                                                    n_steps=n_steps_rk4,
                                                                                    gravitational_parameter=mu_earth,
                                                                                    planet_mean_radius=R_earth,
                                                                                    max_spherical_harmonic_order=max_spherical_harmonic_order_model ,
                                                                                    aerodynamic_settings_chief=aerodynamic_settings_chief_model)

agent_continuous_dynamics_model = dynamic_models.load_agent_translational_continuous_dynamics(gravitational_parameter=mu_earth,
                                                                                     planet_mean_radius=R_earth,
                                                                                     max_spherical_harmonic_order=max_spherical_harmonic_order_model ,
                                                                                     aerodynamic_settings_chief=aerodynamic_settings_chief_model,
                                                                                     aerodynamic_settings_deputy=aerodynamic_settings_deputy_model)
agent_continuous_dynamics_sim = dynamic_models.load_agent_translational_continuous_dynamics(gravitational_parameter=mu_earth,
                                                                                     planet_mean_radius=R_earth,
                                                                                     max_spherical_harmonic_order=max_spherical_harmonic_order_simulation,
                                                                                     aerodynamic_settings_chief=aerodynamic_settings_chief_simulation,
                                                                                     aerodynamic_settings_deputy=aerodynamic_settings_deputy_simulation)


CBF_position,HOBF_position,position_safety_constraint,un_actuated_barrier_constraint = dynamic_models.load_position_barrier_constraint(epsilon_r=epsilon_r,p0=p0_dr,p1=p1_dr,agent_dynamics=agent_continuous_dynamics_model,model="nonlinear")
CBF_velocity,velocity_safety_constraint  = dynamic_models.load_velocity_barrier_constraint(epsilon_v=epsilon_v,p0=p0_dv,agent_dynamics=agent_continuous_dynamics_model,model="nonlinear")
cbf_test = dynamic_models.load_position_barrier_test1(epsilon_r=epsilon_r,agent_dynamics=agent_continuous_dynamics_model,model="nonlinear")

## INITIALISE MPC CONTROLLER
controller = CMPC(state_dim = 6,input_dim = 3,dynamics=agent_discrete_dynamics_model)

max_control_norm    =  u_max  # N LINEAR CONTROL

# set penalties
# Q_cost =  np.diag(np.array([1/epsilon_r ,1/epsilon_r ,1/epsilon_r ,1/epsilon_v,1/epsilon_v,1/epsilon_v]))
# R_cost = np.diag(np.array([1/0.02,1/0.02,1/0.02]))

Q_cost =  np.diag(np.array([100,100,100,1000,1000,1000]))
R_cost = np.diag(np.array([10,10,10]))


max_speed_per_axis =   np.infty #  this is the absolute max speed limit given in the controller
min_speed_per_axis =  - np.infty 

position_lb = - np.infty 
position_ub =   np.infty 

# the same basic controller is used for all the agents.
# This should not be a problem

terminal_set_amplification_factor = 4
mpc_settings = {}
mpc_settings["n_steps_prediction"]                  = 25
mpc_settings["n_steps_actuation"]                   = 15
mpc_settings["state_cost_matrix"]                   = Q_cost
mpc_settings["control_cost_matrix"]                 = R_cost 
mpc_settings["terminal_cost_matrix"]                = terminal_set_amplification_factor * np.matrix(scipy.linalg.solve_discrete_are(Ad,Bd,Q_cost,R_cost))
mpc_settings["max_input_norm"]                      = max_control_norm 
mpc_settings["state_ub"]                            = np.array([position_ub,position_ub,position_ub,max_speed_per_axis,max_speed_per_axis,max_speed_per_axis,])
mpc_settings["state_lb"]                            = np.array([position_lb,position_lb,position_lb,min_speed_per_axis,min_speed_per_axis,min_speed_per_axis,])
mpc_settings["parameters_dynamics"]                 = iss_discrete_dynamics_model # let the mpc integrate the dynamics of the parameters from inside
mpc_settings["dynamic_parameters_vector_dimension"] = 6     # obtain input shape

simulation_parameters = {
"epsilon_r   " : epsilon_r   ,     
"inner_d_safe" : inner_d_safe,  
"p0_dr       " : p0_dr       , 
"p1_dr       " : p1_dr       , 
"epsilon_v   " : epsilon_v   , 
"p0_dv       " : p0_dv       , 
"u_max       " : u_max       , 
"L_dr        " : L_dr        , 
"L_dv        " : L_dv        , 
"L_un_dr     " : L_un_dr     , 
"L_un_dv     " : L_un_dv     , 
"n_steps_prediction":mpc_settings["n_steps_prediction"],
"n_steps_actuation" :mpc_settings["n_steps_actuation"] ,
}

controller.set_mpc_settings(mpc_settings)
controller.add_barrier_constraint(position_safety_constraint, L_dr*int_time + c_r*epsilon_d) 
controller.add_barrier_constraint(velocity_safety_constraint, L_dv*int_time + c_v*epsilon_d)
controller.set_constraints()
controller.initialise_mpc_solver()


activate_noise = True
number_of_test = 50;

for kk in range(number_of_test):
    
    initial_pos_error = epsilon_r*np.random.rand(1)
    if activate_noise :
        random_vector     = np.random.rand(3)
        random_direction  = random_vector/np.linalg.norm(random_vector)
        position_noise    = initial_pos_error*random_direction # random postion
        # find one of the infinite perpendicular directions to position error direction
        per_random_dir    = np.cross(np.array([0,0,1]),random_direction)
        # normalise
        per_random_dir    = per_random_dir /np.linalg.norm(per_random_dir)
        # find max radial speed
        max_radial_speed  = np.min([p0_dr/2*(epsilon_r**2-initial_pos_error **2)/(initial_pos_error),epsilon_v])
        # find minimum angle speed to radial direction (defined by ratio of maximum allowed speed and maximum allowed radial speed)
        min_angle_from_radial_dir = np.arccos(max_radial_speed/epsilon_v)
        # perpendiculr component of speed
        max_tangent_speed  = epsilon_v*np.sin(min_angle_from_radial_dir)
        velocity_noise     = random_direction * max_radial_speed  + per_random_dir* max_tangent_speed 
        state_noise        = np.hstack((position_noise,velocity_noise))
    else :
        state_noise  = np.full((6,),0)
        
    initial_state = np.squeeze(np.asarray(trajectory(0))) + state_noise


    current_state       = initial_state
    current_input       = np.zeros(3)

    print("prediction steps :")
    print(controller.n_steps_prediction)
    print("actuation steps :")
    print(controller.n_steps_actuation)
    
    time_sim = 200 # sec
    t0 = 0
    time_grid_prediction   = np.arange(t0,t0+time_sim,int_time)
    time_grid_simulation   = time_grid_prediction[:len(time_grid_prediction)-mpc_settings["n_steps_prediction"]]


    state_history                    = np.empty((len(time_grid_simulation),6))
    control_history                  = np.empty((len(time_grid_simulation),3))
    implicit_keplerian_state_history = np.empty((len(time_grid_simulation),6))
    position_barrier_value           = np.empty((len(time_grid_simulation),))
    automaton_state_history          = np.zeros((len(time_grid_simulation),))


    automata_state = 0

    for jj in tqdm(range(len(time_grid_simulation))) :
        time_shift                  = time_grid_prediction[jj:jj+(mpc_settings["n_steps_prediction"]+1)]
        
        state_history[jj,:]                     = current_state
        implicit_keplerian_state_history[jj,:]  = implicit_keplerian_state_iss
        
        current_reference = np.squeeze(np.asarray(trajectory(time_grid_simulation[jj])))
        linear_trajectory_reference = sim_tool.obtain_state_Ahistory_from_trajectory(trajectory,time_shift).flatten()
        
        controller.set_reference(linear_trajectory_reference)

        CBF_velocity_value    = np.squeeze(np.asarray(CBF_velocity(current_state,current_reference)))
        CBF_position_value    = np.squeeze(np.asarray(CBF_position(current_state,current_reference)))
        HOBF_position_value   = np.squeeze(np.asarray(HOBF_position(current_state,current_reference)))
        
        current_position_error = np.squeeze(np.asarray((current_reference[:3]-current_state[:3])))
        current_velocity_error = np.squeeze(np.asarray((current_reference[3:]-current_state[3:])))
        
        if  automata_state == 0:
            automaton_state_history[jj] = 0
            controller.set_barriers_activation_mask(np.array([1.,1.]))
            current_input = controller.mpc_controller(x0=current_state,initial_dynamic_parameters=implicit_keplerian_state_iss)

        elif automata_state == 1:
            automaton_state_history[jj] = 1
            # controller.set_barriers_activation_mask(np.array([0.,1.]))  # you don't need it for zero control
            current_input = np.zeros((3,))
        
        if not controller.is_feasible :
            time_grid_simulation             = time_grid_simulation[:jj+1]
            state_history                    = state_history[:jj+1,:] 
            implicit_keplerian_state_history = implicit_keplerian_state_history[:jj+1,:]       
            control_history                  = control_history[:jj+1,:] 
            automaton_state_history          = automaton_state_history[:jj+1] 
            break
        
        else :
            current_state                 = np.squeeze(np.asarray(agent_discrete_dynamics_sim(current_state ,current_input,implicit_keplerian_state_iss)))
            implicit_keplerian_state_iss  = np.squeeze(np.asarray(iss_discrete_dynamics_sim(implicit_keplerian_state_iss)))
            control_history[jj,:]         = np.squeeze(np.asarray(current_input))
            
        close_enough = bool(np.linalg.norm(current_position_error) <  0.001)  and  bool(np.linalg.norm(current_velocity_error)<0.001)
        pos_safe     = bool(position_safety_constraint(current_state, np.zeros((3,)), implicit_keplerian_state_iss,current_reference) >= L_un_dr*int_time + c_r*epsilon_d)
        vel_safe     = bool(velocity_safety_constraint(current_state, np.zeros((3,)), implicit_keplerian_state_iss,current_reference) >= L_un_dv*int_time + c_v*epsilon_d)
        
        if (automata_state == 0 and close_enough) :
            automata_state = 1
        elif (automata_state == 1 and (not pos_safe or not vel_safe)) :
            automata_state = 0


    trajectory_Dhistory   = sim_tool.obtain_state_Dhistory_from_trajectory(trajectory,time_grid_simulation)
    trajectory_Ahistory   = sim_tool.history2array(trajectory_Dhistory)[:,1:]

    speed_error =np.linalg.norm(state_history[:,3:] - trajectory_Ahistory[:,3:],axis=1)

    CBF_position,HOBF_position,position_safety_constraint,un_actuated_barrier_constraint = dynamic_models.load_position_barrier_constraint(epsilon_r=epsilon_r,p0=p0_dr,p1=p1_dr,agent_dynamics=agent_continuous_dynamics_model,model="nonlinear")
    CBF_velocity,velocity_safety_constraint  = dynamic_models.load_velocity_barrier_constraint(epsilon_v=epsilon_v,p0=p0_dv,agent_dynamics=agent_continuous_dynamics_model,model="nonlinear")

    CBF_dictionary = sim_tool.compute_barriers_dictionary(time_grid= time_grid_simulation,
                                                        state_history= state_history,
                                                        control_history=control_history ,
                                                        state_reference=trajectory_Ahistory,
                                                        CBF_position=CBF_position,
                                                        HOBF_position=HOBF_position,
                                                        position_barrier_constraint=position_safety_constraint,
                                                        CBF_velocity=CBF_velocity,
                                                        velocity_barrier_constraint=velocity_safety_constraint ,
                                                        implicit_keplerian_state_history=implicit_keplerian_state_history)

    CBF_dictionary["states_history"] =  automaton_state_history
    state_error = state_history-trajectory_Ahistory
   
    speed_error    = np.linalg.norm(state_error[:,3:],axis=1)
    position_error = np.linalg.norm(state_error[:,:3],axis=1)
    angle_postion_speed = np.arccos(np.sum(state_error[:,:3]*state_error[:,3:],axis=1)/np.linalg.norm(state_error[:,:3],axis=1)/np.linalg.norm(state_error[:,3:],axis=1))


 
    if save_simulation_results : 
        if not controller.is_feasible : 
           output_dir = "/Users/gregorio/Desktop/Thesis/code/matlab/python_simulations_to_plot/single_agent_simulation_robustness/test_{}_failure".format(kk) 
        else : 
           output_dir = "/Users/gregorio/Desktop/Thesis/code/matlab/python_simulations_to_plot/single_agent_simulation_robustness/test_{}_success".format(kk) 

        os.makedirs(output_dir,exist_ok=True)
        CBF_dataframe       = pd.DataFrame.from_dict(CBF_dictionary)
        CBF_dataframe.to_csv(output_dir +'/CBF_data_frame.csv', index=False)
        # control_data_frame  = pd.DataFrame(control_history, columns = ['u_x','u_y','u_z'])
        # control_data_frame.to_csv(output_dir +'/control_data_frame.csv', index=False)
        # state_data_frame  = pd.DataFrame(state_history, columns = ['x','y','z','vx','vy','vz'])
        # state_data_frame.to_csv(output_dir +'/state_data_frame.csv', index=False)
        # reference_state_data_frame  = pd.DataFrame(trajectory_Ahistory, columns = ['x','y','z','vx','vy','vz'])
        # reference_state_data_frame.to_csv(output_dir +'/reference_state_data_frame.csv', index=False)
        
        with open(output_dir+"/specifics.txt","w") as f :
                f.write("-------------------------------------------\n")
                f.write("------------   MPC settings  --------------\n")
                f.write("-------------------------------------------\n")

                for key in mpc_settings.keys():
                    f.write("{} :\n{}\n".format(key,str(mpc_settings[key])))
                f.write("-------------------------------------------\n")
                f.write("-----------simulation parameters----------\n")
                f.write("-------------------------------------------\n")
                
                for key in simulation_parameters .keys():
                    f.write("{} : {}\n".format(key,str(simulation_parameters[key])))
