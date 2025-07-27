from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import minres, spsolve
from scipy.sparse import vstack, hstack
from timeit import default_timer as timer
from datetime import timedelta
import csv
from numpy import genfromtxt

import data_helpers
from BWE_functions import *
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
dxs = [dx0/2/2]
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

tol_minres = 1e-9

# Tolerance for the relative error in outer iterations
tol_st = 1e-5
tol_ad = 1e-5


show_plots = False
show_error_plots = False
show_minres = True

file_path = "BWE_output_alpha_" + str(alpha) +  ".csv"
folder_name = 'BWE_T_1_solutions' 


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
        print('Starting BWE solver for dx=', deltax)

        intervals_line = round((a2-a1)/deltax)
        t0 = 0
        dt = 2* deltax**2
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
        shift = (num_steps-1)*nodes
        u_target_alltime_orig = np.zeros(shift+nodes)
        v_target_alltime_orig = np.zeros(shift+nodes)
        for i in range(1,num_steps+1): # includes states at time zero as we need to compute adjoints at time zero
            start = (i-1)*nodes
            end = i*nodes
            u_target_alltime_orig[start:end] = uhat((i-1)*dt,T,X,Y,e1,e2,e3,e4,k1,k2,k3,k4,Du,beta,gamma).reshape(nodes)
            v_target_alltime_orig[start:end] = vhat((i-1)*dt,T,X,Y,e1,e2,e3,e4,k1,k2,k3,k4,Dv,beta,gamma).reshape(nodes)

        ############# Backward Euler system matrix blocks (stationary) ################

        # (1,1)-block 
        B11_elem = dt*gamma**2/alpha*M

        # (2,2)-block of the (2,2)-block (diagonal)
        B22_22 = -dt*beta*M

        # preconditioner
        D_elem = dt*gamma/np.sqrt(alpha*beta) * M

        ################################ RHS functions  ################################

        f_alltime_orig = np.zeros(shift+nodes)
        g_alltime_orig = np.zeros(shift+nodes)
        for i in range(1,num_steps+1):
            start = (i-1)*nodes
            end = i*nodes
            f_alltime_orig[start:end] = fex(i*dt,T,X,Y,e1,e2,e3,k1,k2,k3,Du,alpha,gamma).reshape(nodes)
            g_alltime_orig[start:end] = gex(i*dt,T,X,Y,e1,e2,e4,k1,k2,k4,Dv,alpha,gamma).reshape(nodes)

        ###############################################################################
        ########################### Reorder vector elements ###########################
        ###### from vertex to dof ordering for compatibility with dolfin matrices #####
        ###############################################################################

        u_target_alltime = reorder_vector_to_dof_time(u_target_alltime_orig, num_steps, nodes, vertextodof)
        v_target_alltime = reorder_vector_to_dof_time(v_target_alltime_orig, num_steps, nodes, vertextodof)
             
        f_alltime = reorder_vector_to_dof_time(f_alltime_orig, num_steps, nodes, vertextodof)
        g_alltime = reorder_vector_to_dof_time(g_alltime_orig, num_steps, nodes, vertextodof)

        u0 = reorder_vector_to_dof(u0_orig, nodes, vertextodof)
        v0 = reorder_vector_to_dof(v0_orig, nodes, vertextodof)
            
        ###############################################################################
        #################### Initial guesses for SQP method ###########################
        ################ (should be in accordance with the BCs) #######################
        ###############################################################################
        
        if deltax == dx0:
            con = 0.4
            u_old_times = con*u_target_alltime[nodes:]
            v_old_times = con*v_target_alltime[nodes:]
            u_Nt_old = con*u_target_alltime[(num_steps-1)*nodes:]
            v_Nt_old = con*v_target_alltime[(num_steps-1)*nodes:]
            p_old_times = np.zeros(shift)
            q_old_times = np.zeros(shift)   
            p0_old = np.zeros(nodes)
            q0_old = np.zeros(nodes)   
                      
        else:
            u_old_times = np.zeros(shift)
            v_old_times = np.zeros(shift)
            p_old_times = np.zeros(shift)
            q_old_times = np.zeros(shift)
            
            p0_old = np.zeros(nodes)
            q0_old = np.zeros(nodes) 
            u_Nt_old = u_target_alltime[(num_steps-1)*nodes:]
            v_Nt_old = v_target_alltime[(num_steps-1)*nodes:]
            
            
            # get details for the coarser mesh
            dx_coarse = deltax*2
            intervals_line_coarse = round((a2-a1)/dx_coarse)
            dt_coarse = 2*dx_coarse**2
            num_steps_coarse = round((T-t0)/dt_coarse)
            mesh_coarse = UnitSquareMesh(intervals_line_coarse,intervals_line_coarse) 
            V_coarse = FunctionSpace(mesh_coarse, 'CG', 1)
            nodes_coarse = V_coarse.dim()       
            
            u_coarse = genfromtxt(folder_name + '/u_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')
            v_coarse = genfromtxt(folder_name + '/v_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')
            p_coarse = genfromtxt(folder_name + '/p_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')
            q_coarse = genfromtxt(folder_name + '/q_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')
            u_Nt_coarse = genfromtxt(folder_name + '/uNt_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')
            v_Nt_coarse = genfromtxt(folder_name + '/vNt_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')
            p0_coarse = genfromtxt(folder_name + '/p0_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')
            q0_coarse = genfromtxt(folder_name + '/q0_' + str(alpha) + '_' + str(dx_coarse) + '.csv', delimiter=',')

            u_coarse_interpmesh = np.zeros(nodes*(num_steps_coarse-1))
            v_coarse_interpmesh = np.zeros(nodes*(num_steps_coarse-1))
            p_coarse_interpmesh = np.zeros(nodes*(num_steps_coarse-1))
            q_coarse_interpmesh = np.zeros(nodes*(num_steps_coarse-1))

            # first interpolate the coarser mesh solution to the current mesh
            # over the original time interval
            for i in range(1,num_steps_coarse):
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
                    
            u_Nt_c_fun = vec_to_function(u_Nt_coarse, V_coarse)
            v_Nt_c_fun = vec_to_function(v_Nt_coarse, V_coarse)
            u_Nt_coarse_interpmesh = interpolate(u_Nt_c_fun, V).vector().get_local()
            v_Nt_coarse_interpmesh = interpolate(v_Nt_c_fun, V).vector().get_local()
            
            p0_c_fun = vec_to_function(p0_coarse, V_coarse)
            q0_c_fun = vec_to_function(q0_coarse, V_coarse)
            p0_coarse_interpmesh = interpolate(p0_c_fun, V).vector().get_local()
            q0_coarse_interpmesh = interpolate(q0_c_fun, V).vector().get_local()
            
            # now interpolate between the time intervals
            # (use dt from the coarser mesh and the fact that dx is halving 
            # when refining the mesh)
            
            p_old_times[:nodes] = p_coarse_interpmesh[:nodes]
            q_old_times[:nodes] = q_coarse_interpmesh[:nodes]
            
            for i in range(num_steps_coarse-1):
                start_c = i*nodes
                end_c = (i+1)*nodes
                start_IG_st = (4*i+3)*nodes
                end_IG_st = (4*i+4)*nodes
                start_IG_ad = (4*i+4)*nodes
                end_IG_ad = (4*i+5)*nodes
                u_old_times[start_IG_st:end_IG_st] = u_coarse_interpmesh[start_c:end_c]
                v_old_times[start_IG_st:end_IG_st] = v_coarse_interpmesh[start_c:end_c]
                p_old_times[start_IG_ad:end_IG_ad] = p_coarse_interpmesh[start_c:end_c]
                q_old_times[start_IG_ad:end_IG_ad] = q_coarse_interpmesh[start_c:end_c]
                
            u_Nt_old = u_Nt_coarse_interpmesh
            v_Nt_old = v_Nt_coarse_interpmesh
            p0_old = p0_coarse_interpmesh
            q0_old = q0_coarse_interpmesh
            
            # for the steps in-between, interpolate between steps
            # first 3 steps use the initial condition
            u_old_times[:nodes] = 1/4*(3*u0 + u_coarse_interpmesh[:nodes])
            u_old_times[nodes:2*nodes] = 1/2*(u0 + u_coarse_interpmesh[:nodes])
            u_old_times[2*nodes:3*nodes] = 1/4*(u0 + 3*u_coarse_interpmesh[:nodes])
            
            v_old_times[:nodes] = 1/4*(3*v0 + v_coarse_interpmesh[:nodes])
            v_old_times[nodes:2*nodes] = 1/2*(v0 + v_coarse_interpmesh[:nodes])
            v_old_times[2*nodes:3*nodes] = 1/4*(v0 + 3*v_coarse_interpmesh[:nodes])
            
            # loop for u,v
            for i in range(1,num_steps_coarse-1):
                start_left = (i-1)*nodes
                end_left = i*nodes
                start_right = i*nodes
                end_right = (i+1)*nodes

                start_IG_1 = 4*i*nodes
                end_IG_1 = (4*i+1)*nodes
                end_IG_2 = (4*i+2)*nodes
                end_IG_3 = (4*i+3)*nodes
                
                u_old_times[start_IG_1:end_IG_1] = 1/4*(3*u_coarse_interpmesh[start_left:end_left] + u_coarse_interpmesh[start_right:end_right])
                u_old_times[end_IG_1:end_IG_2] = 1/2*(u_coarse_interpmesh[start_left:end_left] + u_coarse_interpmesh[start_right:end_right])
                u_old_times[end_IG_2:end_IG_3] = 1/4*(u_coarse_interpmesh[start_left:end_left] + 3*u_coarse_interpmesh[start_right:end_right])
                
                v_old_times[start_IG_1:end_IG_1] = 1/4*(3*v_coarse_interpmesh[start_left:end_left] + v_coarse_interpmesh[start_right:end_right])
                v_old_times[end_IG_1:end_IG_2] = 1/2*(v_coarse_interpmesh[start_left:end_left] + v_coarse_interpmesh[start_right:end_right])
                v_old_times[end_IG_2:end_IG_3] = 1/4*(v_coarse_interpmesh[start_left:end_left] + 3*v_coarse_interpmesh[start_right:end_right])
                
            # final 3 steps
            u_old_times[(num_steps-4)*nodes:(num_steps-3)*nodes] = 1/4*(3*u_coarse_interpmesh[(num_steps_coarse-2)*nodes:] + u_Nt_coarse_interpmesh)
            u_old_times[(num_steps-3)*nodes:(num_steps-2)*nodes] = 1/2*(u_coarse_interpmesh[(num_steps_coarse-2)*nodes:] + u_Nt_coarse_interpmesh)
            u_old_times[(num_steps-2)*nodes:] = 1/4*(u_coarse_interpmesh[(num_steps_coarse-2)*nodes:] + 3*u_Nt_coarse_interpmesh)
            
            v_old_times[(num_steps-4)*nodes:(num_steps-3)*nodes] = 1/4*(3*v_coarse_interpmesh[(num_steps_coarse-2)*nodes:] + v_Nt_coarse_interpmesh)
            v_old_times[(num_steps-3)*nodes:(num_steps-2)*nodes] = 1/2*(v_coarse_interpmesh[(num_steps_coarse-2)*nodes:] + v_Nt_coarse_interpmesh)
            v_old_times[(num_steps-2)*nodes:] = 1/4*(v_coarse_interpmesh[(num_steps_coarse-2)*nodes:] + 3*v_Nt_coarse_interpmesh)

            # first 3 steps
            p_old_times[nodes:2*nodes] = 1/4*(3*p0_coarse_interpmesh + p_coarse_interpmesh[:nodes])
            p_old_times[2*nodes:3*nodes] = 1/2*(p0_coarse_interpmesh + p_coarse_interpmesh[:nodes])
            p_old_times[3*nodes:4*nodes] = 1/4*(p0_coarse_interpmesh + 3*p_coarse_interpmesh[:nodes])
            
            q_old_times[nodes:2*nodes] = 1/4*(3*q0_coarse_interpmesh + q_coarse_interpmesh[:nodes])
            q_old_times[2*nodes:3*nodes] = 1/2*(q0_coarse_interpmesh + q_coarse_interpmesh[:nodes])
            q_old_times[3*nodes:4*nodes] = 1/4*(q0_coarse_interpmesh + 3*q_coarse_interpmesh[:nodes])
            
            # loop for p,q
            for i in range(1,num_steps_coarse-2):
                start_left = (i-1)*nodes
                end_left = i*nodes
                start_right = i*nodes
                end_right = (i+1)*nodes
                
                start_IG_1 = (4*i+1)*nodes
                end_IG_1 = (4*i+2)*nodes
                end_IG_2 = (4*i+3)*nodes
                end_IG_3 = (4*i+4)*nodes
            
                p_old_times[start_IG_1:end_IG_1] = 1/4*(3*p_coarse_interpmesh[start_left:end_left] + p_coarse_interpmesh[start_right:end_right])
                p_old_times[end_IG_1:end_IG_2] = 1/2*(p_coarse_interpmesh[start_left:end_left] + p_coarse_interpmesh[start_right:end_right])
                p_old_times[end_IG_2:end_IG_3] = 1/4*(p_coarse_interpmesh[start_left:end_left] + 3*p_coarse_interpmesh[start_right:end_right])
                
                q_old_times[start_IG_1:end_IG_1] = 1/4*(3*q_coarse_interpmesh[start_left:end_left] + q_coarse_interpmesh[start_right:end_right])
                q_old_times[end_IG_1:end_IG_2] = 1/2*(q_coarse_interpmesh[start_left:end_left] + q_coarse_interpmesh[start_right:end_right])
                q_old_times[end_IG_2:end_IG_3] = 1/4*(q_coarse_interpmesh[start_left:end_left] + 3*q_coarse_interpmesh[start_right:end_right])
               
            # final 3 steps use the zero final-time condition
            p_old_times[(num_steps-4)*nodes:(num_steps-3)*nodes] = 3/4*p_coarse_interpmesh[(num_steps_coarse-2)*nodes:] 
            p_old_times[(num_steps-3)*nodes:(num_steps-2)*nodes] = 1/2*p_coarse_interpmesh[(num_steps_coarse-2)*nodes:]
            p_old_times[(num_steps-2)*nodes:] = 1/4*p_coarse_interpmesh[(num_steps_coarse-2)*nodes:]
            
            q_old_times[(num_steps-4)*nodes:(num_steps-3)*nodes] = 3/4*q_coarse_interpmesh[(num_steps_coarse-2)*nodes:] 
            q_old_times[(num_steps-3)*nodes:(num_steps-2)*nodes] = 1/2*q_coarse_interpmesh[(num_steps_coarse-2)*nodes:]
            q_old_times[(num_steps-2)*nodes:] = 1/4*q_coarse_interpmesh[(num_steps_coarse-2)*nodes:]
           
            u_old_times = 0.8*u_old_times
            v_old_times = 0.8*v_old_times
            p_old_times = 0.8*p_old_times
            q_old_times = 0.8*q_old_times
            u_Nt_old = 0.8*u_Nt_old
            v_Nt_old = 0.8*v_Nt_old
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

        it = 0
        minres_its = []
        start_time = timer()   # timer moved
        while (((rel_err_u > tol_st) or (rel_err_v > tol_st) or (rel_err_p > tol_ad) or (rel_err_q > tol_ad)) and it<30):
            print('SQP iter:', it)
            RHS_vec = np.zeros(4*nodes*(num_steps-1))
            Rhs_uv = np.zeros(2*nodes)
            Rhs_pq = np.zeros(2*nodes)
            Block12_11_diag, Block12_12_diag, Block12_21_diag, Block12_22_diag = ([] for _ in range(4))
            Block22_11_diag, Block22_12_diag = ([] for _ in range(2))

            ## Initialise matrices and vectors changing in time
            for i in range(1,num_steps): # loop over the time steps 1 to N-1
                start = (i-1)*nodes
                end = i*nodes
                
                # Define vectors at current time step as functions
                u_old_fun = vec_to_function(u_old_times[start:end], V)
                v_old_fun = vec_to_function(v_old_times[start:end], V)
                p_old_fun = vec_to_function(p_old_times[start:end], V)
                q_old_fun = vec_to_function(q_old_times[start:end], V)

                # f_fun = vec_to_function(f_alltime[start+nodes:end+nodes],V)
                # g_fun = vec_to_function(g_alltime[start+nodes:end+nodes],V)
                f_fun = vec_to_function(f_alltime[start:end],V)
                g_fun = vec_to_function(g_alltime[start:end],V)

                # u_target_fun = vec_to_function(u_target_alltime[start+nodes:end+nodes], V)
                # v_target_fun = vec_to_function(v_target_alltime[start+nodes:end+nodes], V)
                u_target_fun = vec_to_function(u_target_alltime[start:end], V)
                v_target_fun = vec_to_function(v_target_alltime[start:end], V)
                
                # Define the matrices
                M_uv = assemble_sparse(u_old_fun * v_old_fun * u * w1 *dx)
                M_u2 = assemble_sparse(u_old_fun * u_old_fun * u * w1 *dx)
                M_uqp = assemble_sparse(u_old_fun * (q_old_fun - p_old_fun)* u * w1 *dx)
                M_vqp = assemble_sparse(v_old_fun * (q_old_fun - p_old_fun)* u * w1 *dx)
                
                # (1,2)-block
                B12_11 = (1+dt*gamma)*M + dt*Du*K - 2*dt*gamma*M_uv
                B12_12 = -dt*gamma*M_u2
                B12_21 = 2*dt*gamma*M_uv
                B12_22 = M + dt*Dv*K + dt*gamma*M_u2
                # (2,2)-block
                B22_11 = -dt*beta*M - 2*dt*gamma*M_vqp
                B22_12 = -2*dt*gamma*M_uqp
                
                r1 = -2*dt*gamma*u_old_fun**2 *v_old_fun*w1*dx + dt*f_fun*w1*dx
                r2 = 2*dt*gamma*u_old_fun**2 *v_old_fun*w1*dx + dt*g_fun*w1*dx
                r3 = -1*(dt*beta*u_target_fun + 4*dt*gamma*u_old_fun*v_old_fun*(q_old_fun-p_old_fun)) *w1*dx
                r4 = -1*(dt*beta*v_target_fun + 2*dt*gamma*u_old_fun**2 *(q_old_fun-p_old_fun)) *w1*dx
                
                # Add initial condition to the first time step
                if i==1:
                    r1 += u0_fun*w1*dx
                    r2 += v0_fun*w1*dx
                    
                R1 = np.asarray(assemble(r1))
                R2 = np.asarray(assemble(r2))
                R3 = np.asarray(assemble(r3))
                R4 = np.asarray(assemble(r4))
                
                Block12_11_diag.append(B12_11)
                Block12_12_diag.append(B12_12)
                Block12_21_diag.append(B12_21)
                Block12_22_diag.append(B12_22)
                Block22_11_diag.append(B22_11)
                Block22_12_diag.append(B22_12)

                RHS_vec[start:end] = R1
                RHS_vec[start+shift:end+shift] = R2
                RHS_vec[start+2*shift:end+2*shift] = R3
                RHS_vec[start+3*shift:end+3*shift] = R4
                # -------------------------------------------------------------------
            P_LO = preconditioner_matrix(B11_elem, M, Block12_11_diag, Block12_12_diag, Block12_21_diag, 
                                          Block12_22_diag, D_elem,
                                          num_steps, nodes, dt, gamma, alpha, beta)             

            C_LO = system_matrix(B11_elem, B22_22, M, Block12_11_diag, Block12_12_diag, Block12_21_diag, 
                                  Block12_22_diag, Block22_11_diag, Block22_12_diag,
                                  num_steps, nodes, dt, gamma, alpha, beta) 



            # Solve the system for the new values
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
            
            # Solve for u,v at the last time step
            u_prev_fun = vec_to_function(u_new_times[(num_steps-2)*nodes:],V)
            v_prev_fun = vec_to_function(v_new_times[(num_steps-2)*nodes:],V)
            u_Nt_fun = vec_to_function(u_Nt_old,V)
            v_Nt_fun = vec_to_function(v_Nt_old,V)
            f_Nt_fun = vec_to_function(f_alltime[(num_steps-1)*nodes:],V)
            g_Nt_fun = vec_to_function(g_alltime[(num_steps-1)*nodes:],V)
            
            M_uv = assemble_sparse(u_Nt_fun * v_Nt_fun * u * w1 *dx)
            M_u2 = assemble_sparse(u_Nt_fun * u_Nt_fun * u * w1 *dx)
            
            B11_Nt = (1+dt*gamma)*M + dt*Du*K - 2*dt*gamma*M_uv
            B12_Nt = -dt*gamma*M_u2
            B21_Nt = 2*dt*gamma*M_uv
            B22_Nt = M + dt*Dv*K + dt*gamma*M_u2
            
            rhs_u = -2*dt*gamma*u_Nt_fun**2 *v_Nt_fun*w1*dx + dt*f_Nt_fun*w1*dx + u_prev_fun*w1*dx
            rhs_v = 2*dt*gamma*u_Nt_fun**2 *v_Nt_fun*w1*dx + dt*g_Nt_fun*w1*dx + v_prev_fun*w1*dx
            Rhs_uv[:nodes] = np.asarray(assemble(rhs_u))
            Rhs_uv[nodes:] = np.asarray(assemble(rhs_v))
            
            Mat_uv = vstack([hstack([B11_Nt, B12_Nt]), 
                            hstack([B21_Nt, B22_Nt])])
            sol_uv = spsolve(Mat_uv, Rhs_uv)
            u_Nt_new = sol_uv[:nodes]
            v_Nt_new = sol_uv[nodes:]
            
            rel_err_u_Nt = rel_err(u_Nt_new,u_Nt_old)
            rel_err_v_Nt = rel_err(v_Nt_new,v_Nt_old)
            print('Relative error for u_Nt: ', rel_err_u_Nt)
            print('Relative error for v_Nt: ', rel_err_v_Nt)
            
            u_Nt_old = u_Nt_new
            u_Nt_old = u_Nt_new

            # Solve for the adjoints at time zero using the previously computed values
            p1_fun = vec_to_function(p_new_times[:nodes], V)
            q1_fun = vec_to_function(q_new_times[:nodes], V)
            p0_fun = vec_to_function(p0_old, V)
            q0_fun = vec_to_function(q0_old, V)
            
            u_target0_fun = vec_to_function(u_target_alltime[:nodes], V)
            v_target0_fun = vec_to_function(v_target_alltime[:nodes], V)
            u0_fun = vec_to_function(u0, V)
            v0_fun = vec_to_function(v0, V)
            
            M_uv = assemble_sparse(u0_fun*v0_fun *u*w1*dx)
            M_u2 = assemble_sparse(u0_fun**2 *u*w1*dx)
            
            B11_0 = (1+dt*gamma)*M + dt*Du*K - 2*dt*gamma*M_uv
            B12_0 = 2*dt*gamma*M_uv 
            B21_0 = -dt*gamma*M_u2
            B22_0 = M + dt*Dv*K + dt*gamma*M_u2
            
            rhs_p = dt*alpha*(u_target0_fun-u0_fun)*w1*dx + p1_fun*w1*dx
            rhs_q = dt*alpha*(v_target0_fun-v0_fun)*w1*dx + q1_fun*w1*dx
            Rhs_pq[:nodes] = np.asarray(assemble(rhs_p))
            Rhs_pq[nodes:] = np.asarray(assemble(rhs_q))
            
            Mat_pq = vstack([hstack([B11_0, B12_0]), 
                            hstack([B21_0, B22_0])])
            sol_pq = spsolve(Mat_pq, Rhs_pq)
            p0_new = sol_pq[:nodes]
            q0_new = sol_pq[nodes:]

            rel_err_p0 = rel_err(p0_new,p0_old)
            rel_err_q0 = rel_err(q0_new,q0_old)
            print('Relative error for p0: ', rel_err_p0)
            print('Relative error for q0: ', rel_err_q0)
            
            p0_old = p0_new
            q0_old = q0_new

            it +=1
            print('------------------------------------------------------')

        print('Newton solver finished in', it, 'iterations')
        end_time = timer()
        elapsed_time = end_time-start_time
        
        u_new_times = u_old_times
        v_new_times = v_old_times
        p_new_times = p_old_times
        q_new_times = q_old_times
        
        # Mapping to order the solution vectors based on vertex indices
        u_new_times_re = reorder_vector_from_dof_time(u_new_times, num_steps-1, nodes, vertextodof)
        v_new_times_re = reorder_vector_from_dof_time(v_new_times, num_steps-1, nodes, vertextodof)
        p_new_times_re = reorder_vector_from_dof_time(p_new_times, num_steps-1, nodes, vertextodof)
        q_new_times_re = reorder_vector_from_dof_time(q_new_times, num_steps-1, nodes, vertextodof)

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

        for i in range(1,num_steps):
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
            p_ex = pex(i*dt,T,X,Y,e3,k3).reshape(nodes)
            q_ex = qex(i*dt,T,X,Y,e4,k4).reshape(nodes)
               
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
                plt.title(f'Exact solution $u$ at t={round(i*dt,4)}')
                ax2 = plt.subplot(1,2,2)
                im2 = plt.imshow( u_num, cmap="gray" , vmin=min_u, vmax=max_u, extent =[a1,a2,a1,a2])
                fig2.colorbar(im2)
                plt.title(f'Computed state $u$ at t={round(i*dt,4)}')
                plt.show()
                
                fig2 = plt.figure(figsize=(10,5))
                fig2.tight_layout(pad=3.0)
                ax2 = plt.subplot(1,2,1)
                im1 = plt.imshow(v_ex, cmap="gray", vmin=min_v, vmax=max_v, extent =[a1,a2,a1,a2])
                fig2.colorbar(im1)
                plt.title(f'Exact solution $v$ at t={round(i*dt,4)}')
                ax2 = plt.subplot(1,2,2)
                im2 = plt.imshow( v_num, cmap="gray", vmin=min_v, vmax=max_v, extent =[a1,a2,a1,a2])
                fig2.colorbar(im2)
                plt.title(f'Computed state $v$ at t={round(i*dt,4)}')
                plt.show()
                
                fig2 = plt.figure(figsize=(10,5))
                fig2.tight_layout(pad=3.0)
                ax2 = plt.subplot(1,2,1)
                im1 = plt.imshow(a_ex, cmap ="gray", vmin=min_a, vmax=max_a, extent =[a1,a2,a1,a2])
                fig2.colorbar(im1)
                plt.title(f'Exact solution $a$ at t={round(i*dt,4)}')
                ax2 = plt.subplot(1,2,2)
                im3 = plt.imshow( a_num, cmap="gray", vmin=min_a, vmax=max_a, extent =[a1,a2,a1,a2])
                fig2.colorbar(im3)
                plt.title(f'Computed control $a$ at t={round(i*dt,4)}')
                plt.show()
                
                fig2 = plt.figure(figsize=(10,5))
                fig2.tight_layout(pad=3.0)
                ax2 = plt.subplot(1,2,1)
                im1 = plt.imshow(b_ex, cmap ="gray", vmin=min_b, vmax=max_b, extent =[a1,a2,a1,a2])
                fig2.colorbar(im1)
                plt.title(f'Exact solution $b$ at t={round(i*dt,4)}')
                ax2 = plt.subplot(1,2,2)
                im3 = plt.imshow( b_num, cmap="gray", vmin=min_b, vmax=max_b, extent =[a1,a2,a1,a2])
                fig2.colorbar(im3)
                plt.title(f'Computed control $b$ at t={round(i*dt,4)}')
                plt.show()

        if show_error_plots is True:
            plt.plot(np.arange(1,num_steps),rel_errs_u,'-o',label='relative error')
            plt.plot(np.arange(1,num_steps),werrs_u,'--o',label='weighted error')
            plt.title('Relative error with exact solution, u')
            plt.legend()
            plt.show()

            plt.plot(np.arange(1,num_steps),rel_errs_v,'-o',label='relative error')
            plt.plot(np.arange(1,num_steps),werrs_v,'--o',label='weighted error')
            plt.title('Relative error with exact solution, v')
            plt.legend()
            plt.show()

            plt.plot(np.arange(1,num_steps),rel_errs_p,'-o',label='relative error')
            plt.plot(np.arange(1,num_steps),werrs_p,'--o',label='weighted error')
            plt.plot(np.arange(1,num_steps),errs_p,'--x',label='err in 2-norm')
            plt.title('Errors with exact solution, p')
            plt.legend()
            plt.show()

            plt.plot(np.arange(1,num_steps),rel_errs_q,'-o',label='relative error')
            plt.plot(np.arange(1,num_steps),werrs_q,'--o',label='weighted error')
            plt.plot(np.arange(1,num_steps),errs_q,'--x',label='err in 2-norm')
            plt.title('Errors with exact solution, q')
            plt.legend()
            plt.show()
        print('Finished running Backward Euler with Neumann BCs')
        print('dx=', deltax, 'dt=', dt, 'T=', T)
        print('Relative errors')
        print('u:', max(rel_errs_u))
        print('v:', max(rel_errs_v))
        print('p:', max(rel_errs_p[:-1]))
        print('q:', max(rel_errs_q[:-1]))
        # print('p0:', rel_err_p0)
        # print('q0:', rel_err_q0)
        print('Weighted errors')
        print('u:', max(werrs_u))
        print('v:', max(werrs_v))
        print('p:', max(werrs_p))#[:-1]))
        print('q:', max(werrs_q))#[:-1]))
        # print('p0:', werr_p0)
        # print('q0:', werr_q0)
        print(max(rel_errs_u),',',max(rel_errs_v),',',max(rel_errs_p),',',max(rel_errs_q),',',
              max(werrs_u),',',max(werrs_v),',',max(werrs_p),',',max(werrs_q),
              ',',np.mean(minres_its),',',it,',',elapsed_time)
        print(minres_its)
	
        writer.writerow([4*nodes*(num_steps-1), deltax, dt, nodes, num_steps, 
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
        u_Nt_new.tofile(folder_name + '/uNt_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
        v_Nt_new.tofile(folder_name + '/vNt_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
        p0_new.tofile(folder_name + '/p0_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
        q0_new.tofile(folder_name + '/q0_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
