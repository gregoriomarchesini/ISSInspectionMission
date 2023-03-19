import casadi as ca 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os,sys
print(sys.path)
# saving file specifications
output_dir      =  "./feasibility_simulations/"
agent_settings  = "/Users/gregorio/Desktop/standAlonePaperCode/python_simulations_to_plot/agent2/settings.csv"
save_simulation = True
     
# take the outer most agent which has the highest constraints for each parameter
settings = pd.read_csv(agent_settings,delimiter=",").to_dict()
settings = {key:value[0] for key,value in settings.items() }
  
epsilon_d = settings["epsilon_d"]
c_dr_l    = settings["c_dr_l"]   # max constant L_g Lie derivative position barrier
c_dv_l    = settings["c_dv_l"]   # max constant L_g Lie derivative position barrier
epsilon_r = settings["epsilon_r"]     # max alowed posotion difference
p0_dr     = settings["pdr0"]
p1_dr     = settings["pdr1"]
p0_dv     = settings["pdv0"]
epsilon_v = settings["epsilon_v"]
epsilon_f = settings["epsilon_f"]
L_dr_l    =  settings["L_dr_l"]
L_dv_l    =  settings["L_dv_l"]

L_un_dr   =  settings["L_dr_lZero"]
L_un_dv   =  settings["L_dv_lZero"]

print(settings)


# max control from propulsion system per axis
u_max =  settings["u_max"] # m/s^2

## DISCRETIZED MODEL SETTINGS
int_time         = settings["DeltaT"]     # discretization time for the model. It also the actiation time of the engine
n_steps_rk4      = 3      # number of steps of RK4 inside the integration time (add precision at the integration)

# Grid search 
# points_position_magnitude_grid = 150
# points_velocity_magnitude_grid = 10  
# points_velocity_angle_grid     = 180 # separation of two degrees

points_position_magnitude_grid = 150
points_velocity_magnitude_grid = 10
points_velocity_angle_grid     = 180


number_of_tests = points_position_magnitude_grid*points_velocity_magnitude_grid*points_velocity_angle_grid

position_magnitude_grid = np.linspace(0.0001,epsilon_r,points_position_magnitude_grid)
velocity_magnitude_grid = np.linspace(0,epsilon_v,points_velocity_magnitude_grid)
velocity_angle_grid     = np.linspace(0,np.pi,points_velocity_angle_grid)

red   = (1.,0.,0.,0.5)
green = (0.,1.,0.,1)
color = number_of_tests*[green]

pos_grid_plot,vel_grid_plot,angle_gird_plot = np.meshgrid(position_magnitude_grid,velocity_magnitude_grid,velocity_angle_grid,indexing='ij')
color  = np.ones(np.shape(pos_grid_plot))

# symbolic variable
t = ca.MX.sym("t",2,1) # one variable for each constraint
u = ca.MX.sym("u",2,1) # control variable
er_dir = np.array([1,0]) # position error direction

# define CBF constraints for position and velocity
ev_sym = ca.MX.sym("ev",2,1) # velocity error symbolic
er_sym = ca.MX.sym("er",2,1) # position error symbolic
parameters = ca.vertcat(er_sym ,ev_sym)

CBF_constraint_r =  -2*ev_sym.T@ev_sym - 2*ca.norm_2(er_sym)*epsilon_f - 2*er_sym.T@u - 2*(p0_dr+p1_dr)*(er_sym.T@ev_sym) + p0_dr*p1_dr*(epsilon_r**2 - er_sym.T@er_sym) - L_dr_l*int_time - c_dr_l*epsilon_d - t[0]
CBF_constraint_v =  -2*ca.norm_2(ev_sym)*epsilon_f - 2*ev_sym.T@u + p0_dv*(epsilon_v**2 - ev_sym.T@ev_sym) - L_dv_l*int_time - c_dv_l*epsilon_d - t[1]
norm_constraint  = u.T@u

ineq_constraints    = ca.vertcat(CBF_constraint_r,CBF_constraint_v,norm_constraint,t)
ineq_constraints_lb = [0.,0.,0.,0.,0.]
ineq_constraints_ub = [np.infty,np.infty,u_max**2,np.infty,np.infty]

obj   =  -t.T@t

nlp     = dict(x=ca.vertcat(u,t), f=obj, g = ineq_constraints, p = parameters)

options = {   "ipopt.print_level": 0,
               "print_time"       : False,
               "verbose"          : False,
               "expand"           : True,
          }

solver = ca.nlpsol('solver', 'ipopt', nlp , options)
count = 0
for ii,er_mag in enumerate(position_magnitude_grid) :
    for jj,ev_mag in enumerate(velocity_magnitude_grid) :
        for kk,angle in enumerate(velocity_angle_grid)    :
            
            er = er_dir*er_mag
            ev = np.array([np.cos(angle),np.sin(angle)])*ev_mag
            
            # this happens because you need to stay inside the secoind order safe set for 
            # for the position barrier. The allowed radial speed outward must be decreased as 
            # we go closer to the border

            if np.cos(angle) >= 0 :
                ev[0] = np.min([ev[0],p0_dr/2*(epsilon_r**2 - er_mag**2)/(er_mag)]) #test
                vel_grid_plot[ii,jj,kk] = np.sqrt(ev[0]**2 + ev[1]**2)

            current_parameters = np.hstack((er,ev))

            args  = dict(x0= ca.vertcat(0,0,0,0),
                     lbx=  4*[-np.infty],
                     ubx=  4*[np.infty],
                     lbg= ineq_constraints_lb,
                     ubg= ineq_constraints_ub,
                     p  = current_parameters)
            res   = solver(**args)

            solution = np.squeeze(np.asarray(res['x']))
            print("---------------------")
            print("Progress : ",count/number_of_tests*100)
            print("---------------------")
            print("Problem Parameters")
            print("position error norm : ", er_mag)
            print("velocity error norm : ", np.sqrt(ev[0]**2 + ev[1]**2))
            print("angle               : ", angle*180/np.pi)
            print("Solution to problem")
            print("u1: ", solution[0])
            print("u2: ", solution[1])
            print("u_norm: ", np.sqrt(solution[0]**2+solution[1]**2))
            print("t1: ", solution[2])
            print("t2: ", solution[3])
            print("Obtained minimum : ",res['f'])
            print("status:  ", solver.stats()['return_status'])       
            print(ii,jj,kk)
            if (solver.stats()['return_status'] == "Infeasible_Problem_Detected") :
                color[ii,jj,kk] =  0.   
            
            count += 1  


color = np.ravel(color)
color = [green if c == 1. else red for c in color]

succes_elements  = [True if c == green else False for c in color]
failure_elements = [False if c == green else True for c in color]
maker_size       = [3. if c == green else 3. for c in color]
succes_mask_to_save = np.ones(len(succes_elements))
succes_mask_to_save[failure_elements] = 0

if save_simulation :
    result_dict = {
        "pos_grid" : np.ravel(pos_grid_plot),
        "vel_grid" : np.ravel(vel_grid_plot), 
        "angle_grid" : np.ravel(angle_gird_plot),
        "success_mask" : succes_mask_to_save
    } 

    os.makedirs(output_dir,exist_ok=True)
    result_data_frame = pd.DataFrame.from_dict(result_dict)
    result_data_frame.to_csv(output_dir +'multiagent_feasibility.csv', index=False)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter( np.ravel(pos_grid_plot),
            np.ravel(vel_grid_plot), 
            np.ravel(angle_gird_plot)/np.pi*180,
            s = maker_size, 
            c = color)

ax.set_xlabel('pos magnitude ')
ax.set_ylabel('vel magnitude ')
ax.set_zlabel('angle [deg]')

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')

ax1.plot_trisurf( np.ravel(pos_grid_plot),
                  np.ravel(vel_grid_plot), 
                  np.ravel(angle_gird_plot)/np.pi*180,
                  )

ax1.set_xlabel('pos magnitude ')
ax1.set_ylabel('vel magnitude ')
ax1.set_zlabel('angle [deg]')

plt.show()
