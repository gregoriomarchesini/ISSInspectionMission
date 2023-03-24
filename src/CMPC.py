
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import casadi  as ca
import casadi.tools as ctools
import numpy as np


"""
Fully nonlinear corridor MPC formulation
1) state contraints are box cojnstarints
2) control set is a ball shaped non linear convex set in the form ||u||<= u_max
3) barrier functions can be added and they must be a function of the state and the reference trajectory 
3) possibility to define a barrier constraint margin such that CBF_dot + alpha(CBF) >= margin 
"""

class CMPC() :

    
    def __init__(self,state_dim:int = None,input_dim:int = None,dynamics = None) :
        
        '''
        state_dim (int)             : dimension of the state output state
        input_dim (int)             : dimension of the input state
        dynamics  (casadi.Function) : discrete dynamics model of the system (if you have parameterinc dynamics,Parameters 
                                      must be the last input to the dynamics )
        ''' 
        # set dimensions of model dynamics
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dynamics  = dynamics
        
        
        self.state_ref            = None 
        self.warm_start_solution  = None
        self.settings             = None
        
    
        self.n_steps_prediction                  =  1
        self.n_steps_actuation                   =  None 
        self.max_input_norm                      =   +np.inf
        self.state_lb                            =  np.full((self.state_dim), -np.inf)
        self.state_ub                            =  np.full((self.state_dim), np.inf)
        self.state_cost_matrix                   =  np.eye(self.state_dim)
        self.terminal_cost_matrix                =  np.zeros(self.state_dim)
        self.control_cost_matrix                 =  np.eye(self.input_dim)
        self.warm_start                          =  "on"
        self.parameters_dynamics                 = None
        self.dynamic_parameters_vector_dimension = None
        self.state_ref                           = None
        self.is_feasible                         = True  # does problem OP have a solution


        self.barrier_constraint_functions = [] 
        self.barrier_margin               = []
        self.activation_mask              = []
        self.constr_eq   = []
        self.constr_ineq = []
        self.constr_lb   = []
        self.constr_ub   = []
        self.param_s     = []
        self.obj_cost    = 0  # objective function cost
        self.cost        = 0  # current estimted cost by the solver
       
        
    def set_mpc_settings(self,settings :dict = None) :
        '''settings is a dictionary
        instructions : 
        settings["n_steps_prediction"]  (float) mpc steps predition
        settings["n_steps_actuation"]   (float) mpc steps of control
        settings["state_cost_matrix"]   (np.array(state_dim,state_dim)) MPC quadratic state cost matrix
        settings["control_cost_matrix"] (np.array(control_dim,control_dim)) MPC quadratic state cost matrix
        settings["control_lb"]          (np.ndarray(control_dim,))) control lowerbound
        settings["control_ub"]          (np.ndarray(control_dim,))) control upperbound
        settings["state_ub"]            (np.ndarray(state_dim,))) state upperbound
        settings["state_ub"]            (np.ndarray(state_dim,))) state upperbound
        settings["parameters_dynamics"] (casadi.Function)  Casadi Function object 
        

        # "parameters_dynamics" : tells the MPC controller how to propagate the parameters that are involved in the dynamics of the systems
          
          X(k+1)      = F(X(k),U(k),param(k))
          param(k+1)  = param(k)
        
        '''
        
        # change settings
        if not settings == None :
            for key in settings.keys() :
                if key in self.__dict__.keys():
                   setattr(self,key,settings[key])
                else:
                    raise AttributeError("No settings found corresponding to : {}".format(key))
        
        self.settings = settings   
        # default the control steps to actuatin steps   
        if self.n_steps_actuation == None :
            self.n_steps_actuation = self.n_steps_prediction
            
        # check inputs correctness :
        self.validate_input_settings()
        
        
        # Create optimization variables as a symbolic structure
        # (identical idea as matlab structure)
        
        # in n_steps_prediction you have n_steps_prediction control actions, but n_step+1 predicted states
        # since at the end you just move ahead but you don't control anymore
        
        # x0  x1  x2 ..  xN xN+1
        # u0  u1  u2 ..  uN
        opt_var = ctools.struct_symMX([(ctools.entry('u', shape=(self.input_dim,), repeat     = self.n_steps_actuation),
                                        ctools.entry('state', shape=(self.state_dim,), repeat = self.n_steps_prediction + 1),
                                        )])
        
        self.opt_var = opt_var
        self.num_var = opt_var.size
        
        # Decision variable boundaries
        # this creates two structure array equal to the original one, but the entries are now
        # all equal to the number you input
        
        # try 
        # print(opt_var(-np.inf)['state'])
        
        self.optvar_lb = opt_var(-np.inf)
        self.optvar_ub = opt_var(np.inf)
        self.worm_start_solution =  self.opt_var(0) 

        # for now the state itself has no bounds
        # we will express the bounds on the system
        # in the constraints, not a direct contraint
        
    def add_barrier_constraint(self,barrier_constraint:ca.Function,margin:float) :
        # barrier constraint to be inserted in the CMPC
        # they must be input as CasADi functins
        self.barrier_constraint_functions +=[barrier_constraint]
        self.barrier_margin               +=[margin] # it s positive number
        self.activation_mask              +=[1]
    
    def set_barriers_activation_mask(self,activation_mask:list):
        # here you can decide which barriers are active by settiong 0 or 1 in a list ordered as the 
        # the order in which the barriers where given 

        self.activation_mask = activation_mask
        if  not len(self.activation_mask ) == len( self.barrier_margin ):
            raise ValueError("length of the activation mask must be equal to length of available barriers list")



    def set_constraints(self) :
       '''
       define MPC constraints
       '''
       self.set_cost_functions()
       
       # Starting state parameters - add slack here
       # These will be considered parameters to be inserted in the optimization
       # They will be esplicited later once we optimize and
       # they will be replaced by fixed input values (not part of the parameters to be solved!)
       
       state0_sym      = ca.MX.sym('state0'    , self.state_dim) # SYMBOLIC initial state of the system
       state_ref_sym   = ca.MX.sym('state_ref' , self.state_dim*(self.n_steps_prediction+1)) # SYMBOLIC state reference trajectory
       u_ref_sym       = ca.MX.sym('u_ref'     , self.input_dim) # SYMBOLIC control ref
       active_barriers_mask =  ca.MX.sym('active_barriers_mask' , len(self.barrier_constraint_functions)) # 0 -> inactive, 1=> active
     

       if self.parameters_dynamics != None :
           
           dynamic_parameters_state0_sym     = ca.MX.sym('dynamic_parameters_state0', self.dynamic_parameters_vector_dimension) # inserted in the parameters
           dynamic_parameters_state_sym      = dynamic_parameters_state0_sym # propagated
           param_s                           = ca.vertcat(state0_sym, state_ref_sym, u_ref_sym,dynamic_parameters_state0_sym,active_barriers_mask)  # set of parameters stacked
       
       else :
           param_s                   = ca.vertcat(state0_sym, state_ref_sym, u_ref_sym,active_barriers_mask)
           
       # EQUALITY constaints are written in the form :
       #    0     <= equation <= 0     (equality)
       # INEQUALITY constaints are written in the form :
       #    -np.infty <= equation <= upper    (inequality)
       #       lower  <= equation <= np.infy  (inequality)
       # you have to define the contraints once for lower bound and once upper bound
       
       obj_cost        = ca.MX(0)    
       constr_eq       = []  # constraints equality target (alike x+y**2)    equality lhs
       constr_ineq     = []  # constraints inequality target (alike x+y**2)  equality lhs
       constr_ineq_lb  = []  # constraints inequality limits (lower bound like >= 50) rhs
       constr_ineq_ub  = []  # constraints inequality limits (upper bound like <= 50) rhs
       self.name_eq         = []
       self.name_ineq       = []
       # define contraint for initial state
       constr_eq += [self.opt_var['state', 0] - state0_sym]
       self.name_eq   += [["initial state"]*self.state_dim]
       # Generate MPC Problem
       # set contraints for each step
       
       for step in range(self.n_steps_prediction):
            
            state_step = self.opt_var['state', step] 
            
            # control constraint
            if  step < self.n_steps_actuation  :
                u_step          = self.opt_var['u', step]
                constr_ineq    += [u_step.T@u_step]           # object of the contraint
                constr_ineq_ub += [self.max_input_norm**2]    # right side
                constr_ineq_lb += [-np.infty] # left side
                self.name_ineq      += [["max control"]]

                # add barrier constraint if given
                translational_state_ref   = state_ref_sym[self.state_dim*step:self.state_dim*(step+1)]
                translational_state_agent = self.opt_var["state", step]
                
                if step == 0 :
                    for jj,barrier_constraint,margin in zip(range(len(self.barrier_constraint_functions)),self.barrier_constraint_functions,self.barrier_margin) :
                        translational_state_ref   = state_ref_sym[self.state_dim*step:self.state_dim*(step+1)]
                        translational_state_agent = self.opt_var["state", step]
                        
                        if self.parameters_dynamics != None:
                            constr_ineq              += [barrier_constraint(translational_state_agent, u_step,dynamic_parameters_state_sym,translational_state_ref) + 1000000*(1.-active_barriers_mask[jj])]
                            constr_ineq_lb           += [margin] 
                            constr_ineq_ub           += [ca.inf]
                            self.name_ineq                += [["barrier{}".format(jj)]]
                        else:
                            constr_ineq              += [barrier_constraint(translational_state_agent, u_step,np.array([0,0,0,0,0,0]),translational_state_ref)+ 1000000*(1.-active_barriers_mask[jj])]
                            constr_ineq_lb           += [margin] 
                            constr_ineq_ub           += [ca.inf]
                            self.name_ineq                += [["barrier{}".format(jj)]]
                    
                # if you want to try to smooth the input you can try this. but it must be tuned better the cost
                # if step!=0 :
                #   obj_cost +=  self.smoothing_cost(u_step,self.opt_var['u', step-1],self.control_cost_matrix*3000)
            
            
   
            # State constraints
            if self.state_ub is not None:
                constr_ineq    +=[state_step]
                constr_ineq_ub +=[self.state_ub]            # right side
                constr_ineq_lb +=[np.full((self.state_dim,), -ca.inf)]  # left side
                self.name_ineq      += [["state upper"]*self.state_dim]
            
            if self.state_lb is not None:
                constr_ineq    +=[state_step]
                constr_ineq_lb +=[self.state_lb]            # left side
                constr_ineq_ub +=[np.full((self.state_dim,), +ca.inf)]  # right side
                self.name_ineq      += [["state lower"]*self.state_dim]
            
                 
            # dynamics constraint - future step is given by the dynamics
            
            if self.parameters_dynamics != None :
                   state_step_next = self.dynamics(state_step, u_step,dynamic_parameters_state_sym)      # propagate state
                   dynamic_parameters_state_sym = self.parameters_dynamics(dynamic_parameters_state_sym) # propagate parameters

            else :
                state_step_next = self.dynamics(state_step, u_step)
 
            constr_eq      += [state_step_next - self.opt_var["state", step + 1]] # dynamics constraint definition
            self.name_eq        += [["dynamic constarint"]*self.state_dim]
            
            # Objective Function / Cost Function
            # here if yiou want a single point to be tracked yiou need to put the state repeated for the number of time steps
            if step !=0 : # you don't pay for initial state
               obj_cost += self.running_cost((self.opt_var["state", step] - state_ref_sym[self.state_dim*step:self.state_dim*(step+1)]), self.state_cost_matrix, u_step, self.control_cost_matrix) 

       obj_cost += self.terminal_cost((self.opt_var["state", self.n_steps_prediction] - state_ref_sym[self.state_dim*self.n_steps_prediction:self.state_dim*(self.n_steps_prediction+1)]),self.terminal_cost_matrix ) 
       num_eq_constr   = ca.vertcat(*constr_eq).size1()
       num_ineq_constr = ca.vertcat(*constr_ineq).size1()
       self.number_contr = num_ineq_constr+num_eq_constr 
       constr_eq_lb    = np.zeros((num_eq_constr,)) # left side equality =0
       constr_eq_ub    = np.zeros((num_eq_constr,)) # right side equality =0
       
       # stack constraints rhs/lhs/argument
       self.constr_eq   = constr_eq
       self.constr_ineq = constr_ineq
       self.constr_lb   = ca.vertcat(constr_eq_lb , *constr_ineq_lb)
       self.constr_ub   = ca.vertcat(constr_eq_ub  , *constr_ineq_ub)
       self.param_s     = param_s
       self.obj_cost    = obj_cost

       
    def initialise_mpc_solver(self):
        
       constr       = ca.vertcat(*self.constr_eq, *self.constr_ineq)    # arguments
       self.constr_names = self.name_eq  + self.name_ineq


       # Build NLP Solver (can also solve QP)
       nlp     = dict(x=self.opt_var, f=self.obj_cost, g=constr, p=self.param_s)
       options = {
               "ipopt.print_level": 0,
               "print_time"       : False,
               "verbose"          : False,
               "expand"           : True,
           }
           
           # if solver_opts is not None:
           #     options.update(solver_opts)
           
       self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, options)
       
       num_eq_constr   = ca.vertcat(*self.constr_eq).size1()
       num_ineq_constr = ca.vertcat(*self.constr_ineq).size1()
       
       print('\n________________________________________')
       print('# Number of variables: %d' % self.num_var)
       print('# Number of equality constraints: %d' % num_eq_constr)
       print('# Number of inequality constraints: %d' % num_ineq_constr)
       print('----------------------------------------')
    
    #    kk =0
    #    jj =0
    
    #    for kk in range(len(self.constr_names)):
    #     print("-----------------------------------------------------------------------------")
    #     print("name  :{}".format(self.constr_names[kk])) 
    #     print("upper :{}".format(self.constr_ub[jj:jj+len(self.constr_names[kk])]))  
    #     print("lower :{}".format(self.constr_lb[jj:jj+len(self.constr_names[kk])])) 
    #     print("-----------------------------------------------------------------------------")
    #     jj = jj+len(self.constr_names[kk])
    #     kk +=1

    def set_cost_functions(self):
        """
        Helper method to create CasADi functions for the MPC cost objective.
        """
        # Create function and function variables for calculating the cost
        Q = ca.MX.sym('Q', self.state_dim, self.state_dim)
        R = ca.MX.sym('R', self.input_dim, self.input_dim)
        R_smoothing = ca.MX.sym('R_smoothing', self.input_dim, self.input_dim)
        P = ca.MX.sym('P', self.state_dim, self.state_dim)
    
        state = ca.MX.sym('state', self.state_dim)
        u     = ca.MX.sym('u', self.input_dim)
        u_previous  = ca.MX.sym('u_prev', self.input_dim)
    
        # Instantiate function
        self.running_cost = ca.Function('Jstage', [state, Q, u, R],
                                        [state.T @ Q @ state + u.T @ R @ u])

        self.terminal_cost = ca.Function('terminal_cost', [state, P],
                                            [state.T @ P @ state])

        self.smoothing_cost = ca.Function('smoothing_cost', [u,u_previous, R_smoothing],
                                            [(u-u_previous).T @ R_smoothing @ (u-u_previous)])
    
    def solve_mpc(self, initial_state,initial_dynamic_parameters=None, u_guess=None,):
        """
        Solve the optimal control problem
    
        initial_state (np.ndarray(state_dim,)) : initial state of the system
        u_guess       (np.ndarray(input_dim,)) : control input guess (you can use on of the previous MPC solution for example)
        # note : at start it will be left un_guessed
        """
    
        # Initial state
        if u_guess is None:
            u_guess = np.zeros(self.input_dim)
        if self.state_ref is None:
            # initialise reference to zero in case None is given
            self.state_ref = np.zeros(self.state_dim)

        # These parameters will be inserted directly in the MPC as fixed parameters
        # TODO : check input shape and check that input parameters are actally given
        if  self.dynamic_parameters_vector_dimension != None :
            param = ca.vertcat(initial_state, self.state_ref, u_guess,initial_dynamic_parameters,self.activation_mask) 
        else:
            param = ca.vertcat(initial_state, self.state_ref, u_guess,self.activation_mask) 
        
        self.worm_start_solution["state",0] = initial_state

        args  = dict(x0=self.worm_start_solution  ,
                     lbx=self.optvar_lb,
                     ubx=self.optvar_ub,
                     lbg=self.constr_lb,
                     ubg=self.constr_ub,
                     p=param)
    
        # Solve NLP
       
        sol    = self.solver(**args)
        status = self.solver.stats()['return_status']
        
        optsol = self.opt_var(sol["x"])
    
        if status == "Infeasible_Problem_Detected":
            print("Infeasible_Problem_Detected")
            constraint_matrix = sol['g']
            kk =0
            jj =0
    
            for kk in range(len(self.constr_names)):
                print("-----------------------------------------------------------------------------")
                print("name  :{}".format(self.constr_names[kk])) 
                print("value :{}".format(constraint_matrix[jj:jj+len(self.constr_names[kk])]))  
                print("upper :{}".format(self.constr_ub[jj:jj+len(self.constr_names[kk])]))  
                print("lower :{}".format(self.constr_lb[jj:jj+len(self.constr_names[kk])])) 
                print("-----------------------------------------------------------------------------")
                jj = jj+len(self.constr_names[kk])
                kk +=1
            
            
            
            self.is_feasible = False
            return None,None
      
        self.cost = np.squeeze(np.asarray(sol['f']))
       
        return optsol, optsol['u']
    
    def mpc_controller(self, x0:np.ndarray,initial_dynamic_parameters:np.ndarray=None, u_guess:np.ndarray=None):
        
        """
        MPC controller wrapper.
        Gets first control input to apply to the system
        :return: control input
        :rtype: ca.DM
        """
        
        if  self.dynamic_parameters_vector_dimension != None :
                full_solution, u_pred = self.solve_mpc(x0,initial_dynamic_parameters,u_guess)    
        else:
            full_solution, u_pred = self.solve_mpc(x0,u_guess)
        
        self.worm_start_solution = full_solution # update worm start
        if u_pred == None:
            return None
        else :
           return u_pred[0]
    
    def set_reference(self, state_ref:np.ndarray):
        """
        Parameters
        ----------
        state_ref (np.ndarray) : state reference trajectory. It can be either one 
                                 single state or a trajectory. In case a trajectory
                                 is designed, the trajectory must be given as a long
                                 1D vector of dimension state_dim*n_steps_prediction
                                                       
        """
        self.state_ref = state_ref
        self.validate_input_trajectory()
    
    
    def validate_input_settings(self) :
         
        if not np.shape(self.state_cost_matrix) == (self.state_dim,self.state_dim) :
            message = ("State cost matrix incorrect. State dimension is {}" 
                       ", but Q matrix dimension is {}".format(self.state_dim,np.shape(self.state_cost_matrix)))
            raise ValueError(message)
        
        if not np.shape(self.control_cost_matrix) == (self.input_dim,self.input_dim) :
            message = ("Input cost matrix incorrect. Input dimension is {}" 
                       ", but R matrix dimension is {}".format(self.input_dim,np.shape(self.control_cost_matrix)))
            raise ValueError(message)
       
        if not np.shape(self.state_lb) == (self.state_dim,) :
            message = ("Input constraint array incorrect. State dimension is {}" 
                       ", but state bound vector is dimension is {}".format((self.state_dim,),np.shape(self.state_lb)))
            raise ValueError(message)
        if not np.shape(self.state_ub) == (self.state_dim,) :
            message = ("Input constraint array incorrect. Input dimension is {}" 
                       ", but state bound vector is dimension is {}".format((self.state_dim,),np.shape(self.state_ub)))
            raise ValueError(message)
        
        if not isinstance(self.n_steps_prediction,int) or  self.n_steps_prediction <= 0:
            message = ("number of steps must be strictly positive an integer. {} was given".format(self.n_steps_prediction))
            raise ValueError(message)
        
        if not isinstance(self.n_steps_actuation,int) or  self.n_steps_actuation <= 0 or self.n_steps_actuation>self.n_steps_prediction:
            message = ("number of actuation steps must be a strictly positive integer lower that prediction steps. {} was given".format(self.n_steps_actuation))
            raise ValueError(message)
        
        if not self.parameters_dynamics == None and self.dynamic_parameters_vector_dimension == None :
            message = ("The parameter vector dimension needs to be specified if the parameter dynamics is given."
                       "Make sure to specify the setting \"dynamic_parameters_vector_dimension\" ")
            raise ValueError(message)
        
        if not self.parameters_dynamics == None and not isinstance(self.dynamic_parameters_vector_dimension,int) :
             message = ("dynamic_parameters_vector_dimension  must be an integer. {} was given".format(self.dynamic_parameters_vector_dimension))
             raise ValueError(message)

    def validate_input_trajectory(self) :
        
        possibilities = [(self.state_dim,),(self.state_dim*(self.n_steps_prediction+1),)]
        
        if not (np.shape(self.state_ref) in possibilities) :
                message = ("Input reference trajectory incorrect. State reference dimension is {}" 
                           ", but valid trajectories dimensions are {} and {}".format(np.shape(self.state_ref,),*possibilities))
                raise ValueError(message)

