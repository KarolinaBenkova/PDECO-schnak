from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import minres
from timeit import default_timer as timer
from datetime import timedelta
import csv
from numpy import genfromtxt

import data_helpers
from SV_functions import *
import helpers
from helpers import *
# ---------------------------------------------------------------------------
### PDE-constrained optimisation problem for the Schnakenberg model
# min_{u,v,a,b} beta1/2*||u-\hat{u}||^2 + beta2/2*||v-\hat{v}||^2 + alpha1/2*||a||^2 + alpha2/2*||b||^2  (norms in L^2)
# subject to:
#   -Du grad^2 u + gamma(u-u^2v-a) = f       in Ω
#   -Dv grad^2 v + gamma(u^2v-b)   = g       in Ω
#                        zero flux BCs       on ∂Ω
# ---------------------------------------------------------------------------

## Define the parameters
a1 = 0
a2 = 1
T = 1
dx0 = 0.2
dxs = [dx0/2/2/2]
beta = 1
alpha = 0.001
Du = 1
Dv = 10
gamma = 2

# for the exact solution
## frequencies
k1 = 2
k2 = 1
k3 = 2
k4 = 1
# # ## rates
e1 = 0.2/2
e2 = 0.3/2
e3 = 0.2/2
e4 = 0.3/2

# ## rates
# e1 = 0.2/2/2
# e2 = 0.3/2/2
# e3 = 0.001*5
# e4 = 0.002*5*2

tol_minres = 1e-9

# Tolerance for the relative error in outer iterations
tol_st = 1e-5
tol_ad = 1e-5

show_plots = False
show_error_plots = False
show_minres = True

file_path = "SV_alpha_" + str(alpha) +  ".csv"
# folder_name = 'T_1_solutions_may2_tolmin1e-9'
folder_name = 'T_1_solutions_IG_as_BWE'

with open(file_path, mode='w', newline='') as file:   
    writer = csv.writer(file)
    # Writing parameters to CSV file
    writer.writerow(["T","alpha", "beta", "Du", "Dv", "k1", "k2", "k3", "k4",
                     "e1", "e2", "e3", "e4"])
    writer.writerow([T, alpha, beta, Du, Dv, k1, k2, k3, k4, e1, e2, e3, e4])
    writer.writerow(["tol_minres", "tol_st", "tol_ad"])  # Write parameter names
    writer.writerow([tol_minres, tol_st, tol_ad])
    
    # Write header for results table
    writer.writerow(["dimension", "dx", "dt", "Nx", "Nt", 
                     "RE_u", "RE_v", "RE_p", "RE_q",
                     "WE_u", "WE_v", "WE_p", "WE_q",
                     "MINRES_meanits", "SQP_its", "CPU time"])
    file.flush()
    
    for deltax in dxs:
        print('Starting SV solver for dx=', deltax)

        intervals_line = round((a2-a1)/deltax)
        t0 = 0
        dt = deltax/5
        num_steps = round((T-t0)/dt)

        # mesh
        X = np.arange(a1, a2 + deltax, deltax)
        Y = np.arange(a1, a2 + deltax, deltax)
        X, Y = np.meshgrid(X,Y)

        mesh = UnitSquareMesh(intervals_line,intervals_line) 
        V = FunctionSpace(mesh, 'CG', 1)
        nodes = V.dim()                 
        elems = mesh.num_cells()        
        sqnodes = int(np.sqrt(nodes))

        u = TrialFunction(V)
        w1 = TestFunction(V)

        # Mapping to convert boundary node indices to dof indices
        vertextodof = vertex_to_dof_map(V)

        # ---------------------------------------------------------------------------
        # start_time = timer()   

        ###############################################################################
        ###################### Initialise matrices and vectors ########################
        ###############################################################################

        ## General (same for all methods)

        #  Stiffness matrix
        a = dot(grad(u), grad(w1)) * dx
        K = assemble_sparse(a)

        # Mass matrix
        m = u*w1*dx
        M = assemble_sparse(m)

        # Initial conditions
        u0_orig = uex(0,X,Y,e1,k1).reshape(nodes)
        v0_orig = vex(0,X,Y,e2,k2).reshape(nodes)

        # Define the time vectors for desired states and source functions
        shift = num_steps*nodes
        u_target_alltime_orig = np.zeros(shift+nodes)
        v_target_alltime_orig = np.zeros(shift+nodes)
        for i in range(0,num_steps+1): # includes states at time zero
            start = i*nodes
            end = (i+1)*nodes
            u_target_alltime_orig[start:end] = uhat(i*dt,T,X,Y,e1,e2,e3,e4,k1,k2,k3,k4,Du,beta,gamma).reshape(nodes)
            v_target_alltime_orig[start:end] = vhat(i*dt,T,X,Y,e1,e2,e3,e4,k1,k2,k3,k4,Dv,beta,gamma).reshape(nodes)
        
        ############# Stormer-Verlet system matrix blocks (stationary) ################
        
        # (1,1)-block 
        B11_elem = dt*gamma**2/alpha*M

        # (2,2)-block of the (2,2)-block (diagonal)
        B22_22 = -dt*beta*M
        Block22_22_diag = []
        for n in range(num_steps-1):
            Block22_22_diag.append(B22_22)
        Block22_22_diag.append(1/2*B22_22)

        # preconditioner
        D_elem = dt*gamma*np.sqrt(beta/alpha)*M

        ################################ RHS functions  ################################

        f_alltime_orig = np.zeros(shift)
        g_alltime_orig = np.zeros(shift)
        for i in range(1,num_steps+1):
            start = (i-1)*nodes
            end = i*nodes
            f_alltime_orig[start:end] = fex((i-0.5)*dt,T,X,Y,e1,e2,e3,k1,k2,k3,Du,alpha,gamma).reshape(nodes)
            g_alltime_orig[start:end] = gex((i-0.5)*dt,T,X,Y,e1,e2,e4,k1,k2,k4,Dv,alpha,gamma).reshape(nodes)

        ###############################################################################
        ########################### Reorder vector elements ###########################
        ###### from vertex to dof ordering for compatibility with dolfin matrices #####
        ###############################################################################

        u_target_alltime = reorder_vector_to_dof_time(u_target_alltime_orig, num_steps+1, nodes, vertextodof)
        v_target_alltime = reorder_vector_to_dof_time(v_target_alltime_orig, num_steps+1, nodes, vertextodof)
             
        f_alltime = reorder_vector_to_dof_time(f_alltime_orig, num_steps, nodes, vertextodof)
        g_alltime = reorder_vector_to_dof_time(g_alltime_orig, num_steps, nodes, vertextodof)

        u0 = reorder_vector_to_dof(u0_orig, nodes, vertextodof)
        v0 = reorder_vector_to_dof(v0_orig, nodes, vertextodof)
            
        ###############################################################################
        #################### Initial guesses for SQP method ###########################
        ################ (should be in accordance with the BCs) #######################
        ###############################################################################
        
        if deltax == dx0:
            # exclude target states at time zero
            # u_old_times = u_target_alltime[nodes:]
            # v_old_times = v_target_alltime[nodes:]
            # p_old_times = np.zeros(shift)
            # q_old_times = np.zeros(shift)   
            # p0_old = np.zeros(nodes)
            # q0_old = np.zeros(nodes)
            
            con = 0.4
            u_old_times = con*u_target_alltime[nodes:]
            v_old_times = con*v_target_alltime[nodes:]
            p_old_times = np.zeros(shift)
            q_old_times = np.zeros(shift)   
            p0_old = np.zeros(nodes)
            q0_old = np.zeros(nodes)   
            
        else:
            u_old_times = np.zeros(shift)
            v_old_times = np.zeros(shift)
            p_old_times = np.zeros(shift)
            q_old_times = np.zeros(shift)
            
            # get details for the coarser mesh
            dx_coarse = deltax*2
            intervals_line_coarse = round((a2-a1)/dx_coarse)
            dt_coarse = dx_coarse/5
            num_steps_coarse = round((T-t0)/dt_coarse)
            mesh_coarse = UnitSquareMesh(intervals_line_coarse,intervals_line_coarse) 
            V_coarse = FunctionSpace(mesh_coarse, 'CG', 1)
            nodes_coarse = V_coarse.dim()   
            
            u_coarse = genfromtxt(folder_name + '/u_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')
            v_coarse = genfromtxt(folder_name + '/v_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')
            p_coarse = genfromtxt(folder_name + '/p_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')
            q_coarse = genfromtxt(folder_name + '/q_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')
            p0_coarse = genfromtxt(folder_name + '/p0_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')
            q0_coarse = genfromtxt(folder_name + '/q0_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')

            u_coarse_interpmesh = np.zeros(nodes*num_steps_coarse)
            v_coarse_interpmesh = np.zeros(nodes*num_steps_coarse)
            p_coarse_interpmesh = np.zeros(nodes*num_steps_coarse)
            q_coarse_interpmesh = np.zeros(nodes*num_steps_coarse)

            # first interpolate the coarser mesh solution to the current mesh
            # over the original time interval
            for i in range(1,num_steps_coarse+1):
                    start_c = (i-1)*nodes_coarse
                    end_c = i*nodes_coarse
                    start = (i-1)*nodes
                    end = i*nodes
                    u_c_fun = vec_to_function(u_coarse[start_c:end_c], V_coarse)
                    v_c_fun = vec_to_function(v_coarse[start_c:end_c], V_coarse)
                    p_c_fun = vec_to_function(p_coarse[start_c:end_c], V_coarse)
                    q_c_fun = vec_to_function(q_coarse[start_c:end_c], V_coarse)
                    
                    u_coarse_interpmesh[start:end] = interpolate(u_c_fun, V).vector().get_local()
                    v_coarse_interpmesh[start:end] = interpolate(v_c_fun, V).vector().get_local()
                    p_coarse_interpmesh[start:end] = interpolate(p_c_fun, V).vector().get_local()
                    q_coarse_interpmesh[start:end] = interpolate(q_c_fun, V).vector().get_local()
            p0_c_fun = vec_to_function(p0_coarse, V_coarse)
            q0_c_fun = vec_to_function(q0_coarse, V_coarse)
            p0_coarse_interpmesh = interpolate(p0_c_fun, V).vector().get_local()
            q0_coarse_interpmesh = interpolate(q0_c_fun, V).vector().get_local()
            
            # now interpolate between the time intervals
            # (use dt from the coarser mesh and the fact that dx is halving 
            # when refining the mesh)
            
            ## state variables - at full time steps
            # first set u^(2*i*dt) = u_coarse_interpmesh^(i*dt)
            for i in range(num_steps_coarse+1):
                start_c = i*nodes
                end_c = (i+1)*nodes
                start_IG = (2*i+1)*nodes
                end_IG = (2*i+2)*nodes
                u_old_times[start_IG:end_IG] = u_coarse_interpmesh[start_c:end_c]
                v_old_times[start_IG:end_IG] = v_coarse_interpmesh[start_c:end_c]
            
            # for the steps in-between, average over two time step values 
            u_old_times[:nodes] = 0.5*(u0 + u_coarse_interpmesh[:nodes])
            v_old_times[:nodes] = 0.5*(v0 + v_coarse_interpmesh[:nodes])
            for i in range(num_steps_coarse-1):
                start_left = i*nodes
                end_left = (i+1)*nodes
                start_right = (i+1)*nodes
                end_right = (i+2)*nodes
                start_IG = (2*i+2)*nodes
                end_IG = (2*i+3)*nodes
                u_old_times[start_IG:end_IG] = 0.5*(u_coarse_interpmesh[start_left:end_left] + u_coarse_interpmesh[start_right:end_right])
                v_old_times[start_IG:end_IG] = 0.5*(v_coarse_interpmesh[start_left:end_left] + v_coarse_interpmesh[start_right:end_right])


            # adjoint variables - at half-time steps
            p0_old = p0_coarse_interpmesh
            q0_old = q0_coarse_interpmesh
            
            ## first half-time step: 
            p_old_times[:nodes] = 0.5*(p_coarse_interpmesh[:nodes] + p0_coarse_interpmesh)
            q_old_times[:nodes] = 0.5*(q_coarse_interpmesh[:nodes] + q0_coarse_interpmesh)
            
            p_old_times[nodes:2*nodes] = 1/4*(3*p_coarse_interpmesh[:nodes] + p_coarse_interpmesh[nodes:2*nodes])
            q_old_times[nodes:2*nodes] = 1/4*(3*q_coarse_interpmesh[:nodes] + q_coarse_interpmesh[nodes:2*nodes])
            
            p_old_times[2*nodes:3*nodes] = 1/4*(p_coarse_interpmesh[:nodes] + 3*p_coarse_interpmesh[nodes:2*nodes])
            q_old_times[2*nodes:3*nodes] = 1/4*(q_coarse_interpmesh[:nodes] + 3*q_coarse_interpmesh[nodes:2*nodes])         
            
            ## last half-time step (averaging with final-time condition, i.e. zero):
            p_old_times[(num_steps-1)*nodes:] = 0.5*p_coarse_interpmesh[(num_steps_coarse-1)*nodes:]
            q_old_times[(num_steps-1)*nodes:] = 0.5*q_coarse_interpmesh[(num_steps_coarse-1)*nodes:]
           
            for i in range(num_steps_coarse-2):
                start_left = (i+1)*nodes
                end_left = (i+2)*nodes
                start_right = (i+2)*nodes
                end_right = (i+3)*nodes
 
                start_IG_left = (2*i+3)*nodes
                end_IG_left = (2*i+4)*nodes
                start_IG_right = (2*i+4)*nodes
                end_IG_right = (2*i+5)*nodes
                
                p_old_times[start_IG_left:end_IG_left] = 1/4*(3*p_coarse_interpmesh[start_left:end_left] + p_coarse_interpmesh[start_right:end_right])
                q_old_times[start_IG_left:end_IG_left] = 1/4*(3*q_coarse_interpmesh[start_left:end_left] + q_coarse_interpmesh[start_right:end_right])
                
                p_old_times[start_IG_right:end_IG_right] = 1/4*(p_coarse_interpmesh[start_left:end_left] + 3*p_coarse_interpmesh[start_right:end_right])
                q_old_times[start_IG_right:end_IG_right] = 1/4*(q_coarse_interpmesh[start_left:end_left] + 3*q_coarse_interpmesh[start_right:end_right])
                
            u_old_times = 0.8*u_old_times
            v_old_times = 0.8*v_old_times
            p_old_times = 0.8*p_old_times
            q_old_times = 0.8*q_old_times
            p0_old = 0.8*p0_old
            q0_old = 0.8*q0_old

        ###############################################################################
        ########################## Convert data to functions ##########################
        ###############################################################################

        u0_fun = vec_to_function(u0, V)
        v0_fun = vec_to_function(v0, V)

        ###############################################################################
        ##################################### SQP #####################################
        ###############################################################################

        # Set initial relative approximation error to be above tolerance (arbitrary)
        rel_err_u = 10
        rel_err_v = 10
        rel_err_p = 10
        rel_err_q = 10
        rel_err_p0 = 10
        rel_err_q0 = 10

        it = 0
        minres_its = []
        start_time = timer()
        while (((rel_err_u > tol_st) or (rel_err_v > tol_st) or (rel_err_p > tol_ad) or (rel_err_q > tol_ad)) and it<30):
            
            print('SQP iter:', it)
            RHS_vec = np.zeros(4*nodes*num_steps)   
            Block12_11_diag, Block12_12_diag, Block12_21_diag, Block12_22_diag = ([] for _ in range(4))
            Block12_11_subdiag, Block12_12_subdiag, Block12_21_subdiag, Block12_22_subdiag = ([] for _ in range(4))
            Block22_11_diag, Block22_12_diag = ([] for _ in range(2))

            ## Initialise matrices and vectors changing in time
            for i in range(1,num_steps+1): # loop over the time steps 1 to N
                # solve for (u,v) at time steps 1 to N
                # and for (p,q) at time steps 1/2 to N-1/2
                start = (i-1)*nodes
                end = i*nodes
                
                # Get old vectors for state and adjoint equation
                # (same time step, previous iteration)
                u_old = u_old_times[start:end]
                v_old = v_old_times[start:end]
                p_old = p_old_times[start:end]
                q_old = q_old_times[start:end]
                
                # To use iterative solution at the previous time step in the RHS:
                if i!=1:
                    # (previous time step, previous iteration)
                    u_old_prev = u_old_times[start-nodes:end-nodes]
                    v_old_prev = v_old_times[start-nodes:end-nodes]
                else: # for the 1st (u,v) / (1/2)th (p,q) timestep, set the previous time step data to 0
                    u_old_prev = u0         # u^0   #np.zeros(nodes)
                    v_old_prev = v0         # v^0   #np.zeros(nodes)

                # Define vectors as functions
                u_old_fun = vec_to_function(u_old, V)
                v_old_fun = vec_to_function(v_old, V)
                p_old_fun = vec_to_function(p_old, V)
                q_old_fun = vec_to_function(q_old, V)

                u_old_fun_prev = vec_to_function(u_old_prev, V)
                v_old_fun_prev = vec_to_function(v_old_prev, V)
                    
                u_target_fun = vec_to_function(u_target_alltime[start+nodes:end+nodes], V)
                v_target_fun = vec_to_function(v_target_alltime[start+nodes:end+nodes], V)
                u_target_fun_prev = vec_to_function(u_target_alltime[start:end], V)
                v_target_fun_prev = vec_to_function(v_target_alltime[start:end], V)

                f_fun = vec_to_function(f_alltime[start:end], V)
                g_fun = vec_to_function(g_alltime[start:end], V)

                # Create the mass matrices using values at the previous iteration
                M_u2 = assemble_sparse(u_old_fun**2 *u*w1*dx)
                M_uv = assemble_sparse(u_old_fun*v_old_fun *u*w1*dx)
                M_uqp_minus = assemble_sparse(u_old_fun*(q_old_fun-p_old_fun) *u*w1*dx)
                M_vqp_minus = assemble_sparse(v_old_fun*(q_old_fun-p_old_fun) *u*w1*dx)

                # collect main diagonals of (1,2)-block
                Block12_11 = M + dt*(Du/2*K + gamma/2*M - gamma*M_uv)   # M+dt*L_i^(1)
                Block12_12 = -dt*gamma/2*M_u2
                Block12_21 = dt*gamma*M_uv  
                Block12_22 = M + dt*(Dv/2*K + gamma/2*M_u2)             # M+dt*L_i^(2)

                Block12_11_diag.append(Block12_11)
                Block12_12_diag.append(Block12_12)
                Block12_21_diag.append(Block12_21)
                Block12_22_diag.append(Block12_22)

                if i != num_steps: 
                    # (next time step, previous iteration)
                    p_old_next = p_old_times[start+nodes:end+nodes]
                    q_old_next = q_old_times[start+nodes:end+nodes]
                    p_old_fun_next = vec_to_function(p_old_next, V)
                    q_old_fun_next = vec_to_function(q_old_next, V)
                    
                    m_uqp_plus = u_old_fun*(q_old_fun_next-p_old_fun_next) *u*w1*dx
                    m_vqp_plus = v_old_fun*(q_old_fun_next-p_old_fun_next) *u*w1*dx
                    
                    M_uqp_plus = assemble_sparse(m_uqp_plus)
                    M_vqp_plus = assemble_sparse(m_vqp_plus)
                    
                    # collect subdiagonals of (1,2)-block
                    Block12_11_sub = dt*(Du/2*K + gamma/2*M - gamma*M_uv) -M # -M+dt*L_i^(1)
                    Block12_12_sub = -dt*gamma/2*M_u2
                    Block12_21_sub = dt*gamma*M_uv  
                    Block12_22_sub = dt*(Dv/2*K + gamma/2*M_u2) -M    # -M+dt*L_i^(2)
                    
                    Block12_11_subdiag.append(Block12_11_sub)
                    Block12_12_subdiag.append(Block12_12_sub)
                    Block12_21_subdiag.append(Block12_21_sub)
                    Block12_22_subdiag.append(Block12_22_sub)
                    
                    # collect main diagonals of (2,2)-block
                    Block22_11 = -dt*(beta*M + gamma*(M_vqp_minus + M_vqp_plus)) # -dt*(beta1*M+A_i^(1))
                    Block22_12 = -dt*gamma*(M_uqp_minus + M_uqp_plus) # -dt*A_i^(12)
                    
                    Block22_11_diag.append(Block22_11)
                    Block22_12_diag.append(Block22_12)
                    
                    if i==1:
                        r1 = dt*gamma/2*u_old_fun_prev**2*v_old_fun_prev* w1*dx -\
                            dt*gamma*u_old_fun**2*v_old_fun*w1*dx +\
                            (1-dt*gamma/2)*u_old_fun_prev*w1*dx -\
                            dt*Du/2*dot(grad(u_old_fun_prev),grad(w1))*dx +\
                            dt*f_fun*w1*dx
                            
                        r2 = -dt*gamma/2*u_old_fun_prev**2*v_old_fun_prev* w1*dx +\
                            dt*gamma*u_old_fun**2*v_old_fun* w1*dx +\
                            v_old_fun_prev*w1*dx -\
                            dt*Dv/2*dot(grad(v_old_fun_prev),grad(w1))*dx +\
                            dt*g_fun*w1*dx
        
                    else:
                        r1 = -dt*gamma*(u_old_fun**2*v_old_fun + u_old_fun_prev**2*v_old_fun_prev)* w1*dx +\
                            dt*f_fun*w1*dx
                        r2 = dt*gamma*(u_old_fun**2*v_old_fun + u_old_fun_prev**2*v_old_fun_prev) * w1*dx +\
                            dt*g_fun*w1*dx
                
                    r3 = dt*beta*u_target_fun*w1*dx +\
                            2*dt*gamma*u_old_fun*v_old_fun*(q_old_fun-p_old_fun) * w1*dx +\
                                2*dt*gamma*u_old_fun*v_old_fun*(q_old_fun_next-p_old_fun_next) * w1*dx
                    r4 = dt*beta*v_target_fun*w1*dx + \
                            dt*gamma*u_old_fun**2 *(q_old_fun-p_old_fun) * w1*dx + \
                                dt*gamma*u_old_fun**2 *(q_old_fun_next-p_old_fun_next) * w1*dx
                        
                else: # i=num_steps
                      # collect main diagonals of (2,2)-block
                    Block22_11 = -dt*(beta/2*M + gamma*M_vqp_minus) # -dt*(beta1*M+A_nt^(1))
                    Block22_12 = -dt*gamma*M_uqp_minus              # -dt*A_nt^(12)
                    
                    Block22_11_diag.append(Block22_11)
                    Block22_12_diag.append(Block22_12)
                    
                    r1 = -dt*gamma*(u_old_fun**2*v_old_fun + u_old_fun_prev**2*v_old_fun_prev)* w1 * dx +\
                            dt*f_fun*w1*dx
                    r2 = dt*gamma*(u_old_fun**2*v_old_fun + u_old_fun_prev**2*v_old_fun_prev) * w1 * dx +\
                            dt*g_fun*w1*dx
                    r3 = dt/2*beta*u_target_fun*w1*dx +\
                        2*dt*gamma*u_old_fun*v_old_fun*(q_old_fun-p_old_fun) * w1*dx
                    r4 = dt/2*beta*v_target_fun*w1*dx + \
                        dt*gamma*u_old_fun**2 *(q_old_fun-p_old_fun) * w1*dx
                # -------------------------------------------------------------------
                R1 = np.asarray(assemble(r1))
                R2 = np.asarray(assemble(r2))
                R3 = -np.asarray(assemble(r3))
                R4 = -np.asarray(assemble(r4))
                
                RHS_vec[start:end] = R1  
                RHS_vec[shift+start:shift+end] = R2
                RHS_vec[2*shift+start:2*shift+end] = R3
                RHS_vec[3*shift+start:3*shift+end] = R4
                
                # -------------------------------------------------------------------
            P_LO = preconditioner_matrix(B11_elem, 
                                          Block12_11_diag, Block12_21_diag, Block12_12_diag, Block12_22_diag, 
                                          Block12_11_subdiag, Block12_21_subdiag, Block12_12_subdiag, Block12_22_subdiag,
                                          D_elem,
                                          num_steps, nodes, dt, gamma, alpha, beta) 
            
            C_LO = system_matrix(B11_elem, 
                                Block12_11_diag, Block12_21_diag, Block12_12_diag, Block12_22_diag, 
                                Block12_11_subdiag, Block12_21_subdiag, Block12_12_subdiag, Block12_22_subdiag,
                                Block22_11_diag,Block22_12_diag,Block22_22_diag,
                                num_steps, nodes, dt, gamma, alpha, beta)

            counter = MinresCounter()
            sol, info = minres(C_LO,RHS_vec, tol=tol_minres, M=P_LO, maxiter=150, show=show_minres, callback = counter)
            print('MINRES sover finished in', counter.num_iterations, 'iterations')
            minres_its.append(counter.num_iterations)
        
            p_new_times = -sol[:shift]
            q_new_times = -sol[shift:2*shift]
            u_new_times = sol[2*shift:3*shift]
            v_new_times = sol[3*shift:]

            rel_err_u = rel_err(u_new_times,u_old_times)
            rel_err_v = rel_err(v_new_times,v_old_times)
            rel_err_p = rel_err(p_new_times,p_old_times)
            rel_err_q = rel_err(q_new_times,q_old_times)

            print('Relative error for u: ', rel_err_u)
            print('Relative error for v: ', rel_err_v)
            print('Relative error for p: ', rel_err_p)
            print('Relative error for q: ', rel_err_q)

            # Assign new values to old
            u_old_times = u_new_times
            v_old_times = v_new_times
            p_old_times = p_new_times
            q_old_times = q_new_times
            
            #Solve for the adjoints at time zero using the previously computed values
            p_half_fun = vec_to_function(p_new_times[:nodes], V)
            q_half_fun = vec_to_function(q_new_times[:nodes], V)
            u_target0_fun = vec_to_function(u_target_alltime[:nodes], V)
            v_target0_fun = vec_to_function(v_target_alltime[:nodes], V)
            u0_fun = vec_to_function(u0, V)
            v0_fun = vec_to_function(v0, V)
            M_u2 = assemble_sparse(u0_fun**2 *u*w1*dx)
            M_uv = assemble_sparse(u0_fun*v0_fun *u*w1*dx)
            
            rhs_p0 = np.asarray(assemble(dt*beta/2*(u0_fun-u_target0_fun)*w1*dx))
            rhs_p0 -= dt*gamma*M_uv*q_new_times[:nodes]
            rhs_p0 -= (-M + dt*(Du/2*K + gamma/2*M - gamma*M_uv))*p_new_times[:nodes]
            p0_new = ChebSI(rhs_p0, M, M.diagonal(), 20, 0.5, 2)
            
            rhs_q0 = np.asarray(assemble(dt*beta/2*(v0_fun-v_target0_fun)*w1*dx))
            rhs_p0 += dt*gamma/2*M_u2*p_new_times[:nodes]
            rhs_q0 -= (-M + dt*(Dv/2*K + gamma/2*M_u2))*q_new_times[:nodes]
            q0_new = ChebSI(rhs_q0, M, M.diagonal(), 20, 0.5, 2)
            
            rel_err_p0 = rel_err(p0_new,p0_old)
            rel_err_q0 = rel_err(q0_new,q0_old)
            print('Relative error for p0: ', rel_err_p0)
            print('Relative error for q0: ', rel_err_q0)
            
            p0_old = p0_new
            q0_old = q0_new

            it +=1
            print('------------------------------------------------------')

        print('SQP solver finished in', it, 'iterations')
        end_time = timer()
        elapsed_time = end_time-start_time
        
        # Mapping to order the solution vectors based on vertex indices
        u_new_times_re = reorder_vector_from_dof_time(u_new_times, num_steps, nodes, vertextodof)
        v_new_times_re = reorder_vector_from_dof_time(v_new_times, num_steps, nodes, vertextodof)
        p_new_times_re = reorder_vector_from_dof_time(p_new_times, num_steps, nodes, vertextodof)
        q_new_times_re = reorder_vector_from_dof_time(q_new_times, num_steps, nodes, vertextodof)
        p0_new_re = reorder_vector_from_dof_time(p0_new, 1, nodes, vertextodof)
        q0_new_re = reorder_vector_from_dof_time(q0_new, 1, nodes, vertextodof)
        
        rel_errs_u, rel_errs_v, rel_errs_p, rel_errs_q = ([] for _ in range(4))
        werrs_u, werrs_v, werrs_p, werrs_q = ([] for _ in range(4))
        errs_u, errs_v, errs_p, errs_q = ([] for _ in range(4))

        # Create target data from the images 
        u_target = uhat(T,T,X,Y,e1,e2,e3,e4,k1,k2,k3,k4,Du,beta,gamma).reshape(nodes)
        v_target = vhat(T,T,X,Y,e1,e2,e3,e4,k1,k2,k3,k4,Dv,beta,gamma).reshape(nodes)

        min_u = min(np.amin(u_target), np.amin(u_new_times))
        max_u = max(np.amax(u_target), np.amax(u_new_times))
        min_v = min(np.amin(v_target), np.amin(v_new_times))
        max_v = max(np.amax(v_target), np.amax(v_new_times))

        min_a = gamma/alpha*min(np.amin(pex(T,T,X,Y,e3,k3)), np.amin(p_new_times))
        max_a = gamma/alpha*max(np.amax(pex(0,T,X,Y,e3,k3)), np.amax(p_new_times))
        min_b = gamma/alpha*min(np.amin(qex(T,T,X,Y,e4,k4)), np.amin(q_new_times))
        max_b = gamma/alpha*max(np.amax(qex(0,T,X,Y,e4,k4)), np.amax(q_new_times))
        
        
        # adjoints at t=0
        a0_num = (p0_new_re * gamma / alpha).reshape(((sqnodes,sqnodes)))
        b0_num = (q0_new_re * gamma / alpha).reshape(((sqnodes,sqnodes)))
        p_ex = pex(0,T,X,Y,e3,k3).reshape(nodes)
        q_ex = qex(0,T,X,Y,e4,k4).reshape(nodes)
        err_p0 = np.linalg.norm(p_ex - p0_new_re)
        err_q0 = np.linalg.norm(q_ex - q0_new_re)
        rel_err_p0 =  err_p0 / np.linalg.norm(p_ex)
        rel_err_q0 =  err_q0 / np.linalg.norm(q_ex)
        werr_p0 = deltax * err_p0
        werr_q0 = deltax * err_q0

        if show_plots is True:
            fig2 = plt.figure(figsize=(10,5))
            ax2 = plt.subplot(1,2,1)
            im1 = plt.imshow(a0_num, cmap ="gray", vmin=min_a, vmax=max_a, extent =[a1,a2,a1,a2])
            fig2.colorbar(im1)
            plt.title('Computed control $a$ at t=0')
            ax2 = plt.subplot(1,2,2)
            im2 = plt.imshow(b0_num, cmap="gray", vmin=min_a, vmax=max_a, extent =[a1,a2,a1,a2])
            fig2.colorbar(im2)
            plt.title('Computed control $b at t=0')

        for i in range(1,num_steps+1):
            start = (i-1)*nodes
            end = i*nodes

            u_num = u_new_times_re[start:end]
            v_num = v_new_times_re[start:end]
            p_num = p_new_times_re[start:end]
            q_num = q_new_times_re[start:end]
                    
            # Desired states for u and v
            u_target_time = uhat(i*dt,T,X,Y,e1,e2,e3,e4,k1,k2,k3,k4,Du,beta,gamma).reshape((sqnodes,sqnodes))
            v_target_time = vhat(i*dt,T,X,Y,e1,e2,e3,e4,k1,k2,k3,k4,Dv,beta,gamma).reshape((sqnodes,sqnodes))
            
            u_ex = uex(i*dt,X,Y,e1,k1).reshape(nodes)
            v_ex = vex(i*dt,X,Y,e2,k2).reshape(nodes)
            p_ex = pex((i-0.5)*dt,T,X,Y,e3,k3).reshape(nodes)
            q_ex = qex((i-0.5)*dt,T,X,Y,e4,k4).reshape(nodes)
               
            # Error in 2-norm
            err_u = np.linalg.norm(u_ex - u_num)
            err_v = np.linalg.norm(v_ex - v_num)
            err_p = np.linalg.norm(p_ex - p_num)
            err_q = np.linalg.norm(q_ex - q_num)
            
            errs_u.append(err_u)
            errs_v.append(err_v)
            errs_p.append(err_p)
            errs_q.append(err_q)
            
            # Relative error in 2-norm with exact solution
            norm_2u = err_u / np.linalg.norm(u_ex)
            norm_2v = err_v /  np.linalg.norm(v_ex)
            norm_2p = err_p / np.linalg.norm(p_ex)
            norm_2q = err_q /  np.linalg.norm(q_ex)
            
            rel_errs_u.append(norm_2u)
            rel_errs_v.append(norm_2v)
            rel_errs_p.append(norm_2p)
            rel_errs_q.append(norm_2q)
            
            norm_2wu = deltax * err_u
            norm_2wv = deltax * err_v
            norm_2wp = deltax * err_p
            norm_2wq = deltax * err_q
            
            werrs_u.append(norm_2wu)
            werrs_v.append(norm_2wv)
            werrs_p.append(norm_2wp)
            werrs_q.append(norm_2wq)

            u_num = u_num.reshape((sqnodes,sqnodes))
            v_num = v_num.reshape((sqnodes,sqnodes))
            u_ex = u_ex.reshape((sqnodes,sqnodes))
            v_ex = v_ex.reshape((sqnodes,sqnodes))
            p_ex = p_ex.reshape((sqnodes,sqnodes))
            q_ex = q_ex.reshape((sqnodes,sqnodes))
            
            # Compute the controls
            a_num = (p_num * gamma / alpha).reshape(((sqnodes,sqnodes)))
            b_num = (q_num * gamma / alpha).reshape(((sqnodes,sqnodes)))
            a_ex = gamma / alpha * p_ex
            b_ex = gamma / alpha * q_ex
        
            if show_plots is True:
                fig2 = plt.figure(figsize=(10,5))
                fig2.tight_layout(pad=3.0)
                ax2 = plt.subplot(1,2,1)
                im1 = plt.imshow(u_ex, cmap ="gray", vmin=min_u, vmax=max_u, extent =[a1,a2,a1,a2])
                fig2.colorbar(im1)
                plt.title(f'Exact solution $u$ at t={round(i*dt,5)}')
                ax2 = plt.subplot(1,2,2)
                im2 = plt.imshow( u_num, cmap="gray", vmin=min_u, vmax=max_u, extent =[a1,a2,a1,a2])
                fig2.colorbar(im2)
                plt.title(f'Computed state $u$ at t={round(i*dt,5)}')
                plt.show()
                
                fig2 = plt.figure(figsize=(10,5))
                fig2.tight_layout(pad=3.0)
                ax2 = plt.subplot(1,2,1)
                im1 = plt.imshow(v_ex, cmap="gray", vmin=min_v, vmax=max_v, extent =[a1,a2,a1,a2])
                fig2.colorbar(im1)
                plt.title(f'Exact solution $v$ at t={round(i*dt,5)}')
                ax2 = plt.subplot(1,2,2)
                im2 = plt.imshow( v_num, cmap="gray", vmin=min_v, vmax=max_v, extent =[a1,a2,a1,a2])
                fig2.colorbar(im2)
                plt.title(f'Computed state $v$ at t={round(i*dt,5)}')
                plt.show()
                
                fig2 = plt.figure(figsize=(10,5))
                fig2.tight_layout(pad=3.0)
                ax2 = plt.subplot(1,2,1)
                im1 = plt.imshow(a_ex, cmap ="gray", vmin=min_a, vmax=max_a, extent =[a1,a2,a1,a2])
                fig2.colorbar(im1)
                plt.title(f'Exact solution $a$ at t={round((i-0.5)*dt,5)}')
                ax2 = plt.subplot(1,2,2)
                im3 = plt.imshow( a_num, cmap="gray", vmin=min_a, vmax=max_a, extent =[a1,a2,a1,a2])
                fig2.colorbar(im3)
                plt.title(f'Computed control $a$ at t={round((i-0.5)*dt,5)}')
                plt.show()
                
                fig2 = plt.figure(figsize=(10,5))
                fig2.tight_layout(pad=3.0)
                ax2 = plt.subplot(1,2,1)
                im1 = plt.imshow(b_ex, cmap ="gray", vmin=min_b, vmax=max_b, extent =[a1,a2,a1,a2])
                fig2.colorbar(im1)
                plt.title(f'Exact solution $b$ at t={round((i-0.5)*dt,5)}')
                ax2 = plt.subplot(1,2,2)
                im3 = plt.imshow( b_num, cmap="gray", vmin=min_b, vmax=max_b, extent =[a1,a2,a1,a2])
                fig2.colorbar(im3)
                plt.title(f'Computed control $b$ at t={round((i-0.5)*dt,5)}')
                plt.show()

        if show_error_plots is True:
            plt.plot(np.arange(1,num_steps+1),rel_errs_u,'-o',label='relative error')
            plt.plot(np.arange(1,num_steps+1),werrs_u,'--o',label='weighted error')
            plt.title('Relative error with exact solution, u')
            plt.legend()
            plt.show()

            plt.plot(np.arange(1,num_steps+1),rel_errs_v,'-o',label='relative error')
            plt.plot(np.arange(1,num_steps+1),werrs_v,'--o',label='weighted error')
            plt.title('Relative error with exact solution, v')
            plt.legend()
            plt.show()

            plt.plot(np.arange(0.5,num_steps+0.5),rel_errs_p,'-o',label='relative error')
            plt.plot(np.arange(0.5,num_steps+0.5),werrs_p,'--o',label='weighted error')
            plt.plot(np.arange(0.5,num_steps+0.5),errs_p,'--x',label='err in 2-norm')
            plt.title('Errors with exact solution, p')
            plt.legend()
            plt.show()

            plt.plot(np.arange(0.5,num_steps+0.5),rel_errs_q,'-o',label='relative error')
            plt.plot(np.arange(0.5,num_steps+0.5),werrs_q,'--o',label='weighted error')
            plt.plot(np.arange(0.5,num_steps+0.5),errs_q,'--x',label='err in 2-norm')
            plt.title('Errors with exact solution, q')
            plt.legend()
            plt.show()

        # print('Finished running Stormer-Verlet with Neumann BCs')
        print('dx=', deltax, 'dt=', dt, 'T=', T, 'alpha=', alpha)
        print('Relative errors')
        print('u:', max(rel_errs_u))
        print('v:', max(rel_errs_v))
        print('p:', max(rel_errs_p))
        print('q:', max(rel_errs_q))
        # print('p0:', rel_err_p0)
        # print('q0:', rel_err_q0)
        print('Weighted errors')
        print('u:', max(werrs_u))
        print('v:', max(werrs_v))
        print('p:', max(werrs_p))
        print('q:', max(werrs_q))
        # print('p0:', werr_p0)
        # print('q0:', werr_q0)
        print(max(rel_errs_u),',',max(rel_errs_v),',',max(rel_errs_p),',',max(rel_errs_q),',',
              max(werrs_u),',',max(werrs_v),',',max(werrs_p),',',max(werrs_q),
              ',',np.mean(minres_its),',',it,',',elapsed_time)
        print(minres_its)

        writer.writerow([4*nodes*num_steps, deltax, dt, nodes, num_steps, 
                          max(rel_errs_u), max(rel_errs_v), max(rel_errs_p), max(rel_errs_q),
                          max(werrs_u), max(werrs_v), max(werrs_p), max(werrs_q),
                          np.mean(minres_its), it, elapsed_time])

        print('Elapsed time [s]: ', elapsed_time)
        print(f'{tol_minres=}', f'{tol_ad=}', f'{tol_st=}' )

        file.flush()
       
        u_new_times.tofile(folder_name + '/u_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
        v_new_times.tofile(folder_name + '/v_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
        p_new_times.tofile(folder_name + '/p_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
        q_new_times.tofile(folder_name + '/q_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
        p0_new.tofile(folder_name + '/p0_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
        q0_new.tofile(folder_name + '/q0_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
