from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import minres
from timeit import default_timer as timer
from datetime import timedelta

import data_helpers
from SV_functions import *
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
deltax = 0.05/2/2
intervals_line = round((a2-a1)/deltax)
t0 = 0
dt = 0.01
T = 2
num_steps = round((T-t0)/dt) 
beta = 1
alpha = 0.001
Du = 1
Dv = 10
gamma = 1000

if gamma==1000:
    a_gen_T5 = 0.126779
    b_gen_T5 = 0.792366
elif gamma==100 or gamma==80 or gamma==50:
    a_gen_T5 = 0.1
    b_gen_T5 = 0.9
elif gamma==10:
    ## set 1
    a_gen_T5 = 0.3
    b_gen_T5 = 0.7
    ## set 2
    # a_gen_T5 = 0.25
    # b_gen_T5 = 3.35

tol_minres = 1e-9

# Tolerance for the relative error in outer iterations
tol_st = 1e-5
tol_ad = 1e-5

bctype = "Neumann"

mesh = UnitSquareMesh(intervals_line,intervals_line) 
V = FunctionSpace(mesh, 'CG', 1)
nodes = V.dim()                 
elems = mesh.num_cells()        
sqnodes = round(np.sqrt(nodes))

u = TrialFunction(V)
w1 = TestFunction(V)

show_plots = False
show_minres = True


# Get node indices that lie on the boundary of the mesh
boundary_nodes = []
for n in range(nodes):
    if n%sqnodes in [0,sqnodes-1] or n<sqnodes or n>=nodes-sqnodes: 
        boundary_nodes.append(n)        
boundary_nodes = np.array(boundary_nodes)

# Mapping to convert boundary node indices to dof indices
vertextodof = vertex_to_dof_map(V)
boundary_nodes_dof = [] 
for i in range(len(boundary_nodes)):
    j = int(vertextodof[boundary_nodes[i]])
    boundary_nodes_dof.append(j)

# Mapping to convert boundary node indices to dof indices
vertextodof = vertex_to_dof_map(V)
       
# Generate target images for the chosen mesh
data_helpers.generate_image('u', bctype, nodes, gamma)
data_helpers.generate_image('v', bctype, nodes, gamma)

# ---------------------------------------------------------------------------

start_time = timer()    

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
u0_orig = np.zeros(nodes)
v0_orig = np.zeros(nodes)

# Define the time vectors for desired states and source functions
shift = num_steps*nodes
u_target_alltime_orig = np.zeros(shift+nodes)
v_target_alltime_orig = np.zeros(shift+nodes)
for i in range(0,num_steps+1): # includes states at time zero
    start = i*nodes
    end = (i+1)*nodes
    u_target_alltime_orig[start:end] = data_helpers.uhat(i*dt,T,nodes,boundary_nodes,bctype,gamma).reshape(nodes)
    v_target_alltime_orig[start:end] = data_helpers.vhat(i*dt,T,nodes,boundary_nodes,bctype,gamma).reshape(nodes)

############# Stormer-Verlet system matrix blocks (stationary) ################

# (1,1)-block 
Block11_elem = dt*gamma**2/alpha*M

# (2,2)-block of the (2,2)-block (diagonal)
Block22_22_elem = -dt*beta*M
Block22_22_diag = []
for n in range(num_steps-1):
    Block22_22_diag.append(Block22_22_elem)
Block22_22_diag.append(1/2*Block22_22_elem)

# preconditioner 
D_elem = dt*gamma*np.sqrt(beta/alpha)*M

###############################################################################
#################### Initial guesses for SQP method ###########################
################ (should be in accordance with the BCs) #######################
###############################################################################

u_old_times_orig = np.zeros(shift)
v_old_times_orig = np.zeros(shift)
p_old_times_orig = np.zeros(shift)
q_old_times_orig = np.zeros(shift)

for i in range(1,num_steps+1):
    start = (i-1)*nodes
    end = i*nodes
    u_old_times_orig[start:end] = data_helpers.uhat(i*dt,T,nodes,boundary_nodes,bctype,gamma).reshape(nodes)
    v_old_times_orig[start:end] = data_helpers.vhat(i*dt,T,nodes,boundary_nodes,bctype,gamma).reshape(nodes)
    # p_old_times_orig[start:end] = 0.8*pex((i-1)*dt,X,Y).reshape(nodes)
    # q_old_times_orig[start:end] = 0.8*qex((i-1)*dt,X,Y).reshape(nodes)
p0_old = np.zeros(nodes)
q0_old = np.zeros(nodes)

###############################################################################
########################### Reorder vector elements ###########################
###### from vertex to dof ordering for compatibility with dolfin matrices #####
###############################################################################

u_old_times = reorder_vector_to_dof_time(u_old_times_orig, num_steps, nodes, vertextodof)
v_old_times = reorder_vector_to_dof_time(v_old_times_orig, num_steps, nodes, vertextodof)
p_old_times = reorder_vector_to_dof_time(p_old_times_orig, num_steps, nodes, vertextodof)
q_old_times = reorder_vector_to_dof_time(q_old_times_orig, num_steps, nodes, vertextodof)

u_target_alltime = reorder_vector_to_dof_time(u_target_alltime_orig, num_steps+1, nodes, vertextodof)
v_target_alltime = reorder_vector_to_dof_time(v_target_alltime_orig, num_steps+1, nodes, vertextodof)

u0 = reorder_vector_to_dof(u0_orig, nodes, vertextodof)
v0 = reorder_vector_to_dof(v0_orig, nodes, vertextodof)
      
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

while (((rel_err_u > tol_st) or (rel_err_v > tol_st) or (rel_err_p > tol_ad) 
        or (rel_err_q > tol_ad) or (rel_err_p0 > tol_ad) or (rel_err_q0 > tol_ad)) and it<30):
  
    RHS_vec = np.zeros(4*nodes*num_steps)
    Block12_11_diag = []
    Block12_12_diag = []
    Block12_21_diag = []
    Block12_22_diag = []
    
    Block12_11_subdiag = []
    Block12_12_subdiag = []
    Block12_21_subdiag = []
    Block12_22_subdiag = []
    
    Block22_11_diag = []
    Block22_12_diag = []
    
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
        
        u_target_time_prev = u_target_alltime[start:end]
        v_target_time_prev = v_target_alltime[start:end]
        u_target_time = u_target_alltime[start+nodes:end+nodes]
        v_target_time = v_target_alltime[start+nodes:end+nodes]
        
        u_target_fun = vec_to_function(u_target_time, V)
        v_target_fun = vec_to_function(v_target_time, V)
        u_target_fun_prev = vec_to_function(u_target_time_prev, V)
        v_target_fun_prev = vec_to_function(v_target_time_prev, V)
        
        u_target_time = u_target_alltime
        
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
            
            M_uqp_plus = assemble_sparse(u_old_fun*(q_old_fun_next-p_old_fun_next) *u*w1*dx)
            M_vqp_plus = assemble_sparse(v_old_fun*(q_old_fun_next-p_old_fun_next) *u*w1*dx)
            
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
                    dt*Du/2*dot(grad(u_old_fun_prev),grad(w1))*dx
                    
                r2 = -dt*gamma/2*u_old_fun_prev**2*v_old_fun_prev* w1*dx +\
                    dt*gamma*u_old_fun**2*v_old_fun* w1*dx +\
                    v_old_fun_prev*w1*dx -\
                    dt*Dv/2*dot(grad(v_old_fun_prev),grad(w1))*dx

            else:
                r1 = -dt*gamma*(u_old_fun**2*v_old_fun + u_old_fun_prev**2*v_old_fun_prev)* w1*dx
                r2 = dt*gamma*(u_old_fun**2*v_old_fun + u_old_fun_prev**2*v_old_fun_prev) * w1*dx
        
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
            
            r1 = -dt*gamma*(u_old_fun**2*v_old_fun + u_old_fun_prev**2*v_old_fun_prev)* w1 * dx
            r2 = dt*gamma*(u_old_fun**2*v_old_fun + u_old_fun_prev**2*v_old_fun_prev) * w1 * dx
            r3 = dt/2*beta*u_target_fun*w1*dx +\
                2*dt*gamma*u_old_fun*v_old_fun*(q_old_fun-p_old_fun) * w1*dx
            r4 = dt/2*beta*v_target_fun*w1*dx + \
                dt*gamma*u_old_fun**2 *(q_old_fun-p_old_fun) * w1*dx
        # -------------------------------------------------------------------
        R1 = assemble(r1)
        R2 = assemble(r2)
        R3 = assemble(r3)
        R4 = assemble(r4)
        R1 = np.asarray(R1)
        R2 = np.asarray(R2)
        R3 = -np.asarray(R3)
        R4 = -np.asarray(R4)
        
        RHS_vec[start:end] = R1  
        RHS_vec[shift+start:shift+end] = R2
        RHS_vec[2*shift+start:2*shift+end] = R3
        RHS_vec[3*shift+start:3*shift+end] = R4
        
        # -------------------------------------------------------------------
    P_LO = preconditioner_matrix(Block11_elem, 
                                  Block12_11_diag, Block12_21_diag, Block12_12_diag, Block12_22_diag, 
                                  Block12_11_subdiag, Block12_21_subdiag, Block12_12_subdiag, Block12_22_subdiag,
                                  D_elem,
                                  num_steps, nodes, dt, gamma, alpha, beta) 
    
    C_LO = system_matrix(Block11_elem, 
                        Block12_11_diag, Block12_21_diag, Block12_12_diag, Block12_22_diag, 
                        Block12_11_subdiag, Block12_21_subdiag, Block12_12_subdiag, Block12_22_subdiag,
                        Block22_11_diag,Block22_12_diag,Block22_22_diag,
                        num_steps, nodes, dt, gamma, alpha, beta)
          
    # Solve the system for the new values
    counter = MinresCounter()
    sol, info = minres(C_LO,RHS_vec, tol=tol_minres, M=P_LO, maxiter=150, show=show_minres, callback = counter)
    minres_its.append(counter.num_iterations)

    p_new_times = -sol[:shift]
    q_new_times = -sol[shift:2*shift]
    u_new_times = sol[2*shift:3*shift]
    v_new_times = sol[3*shift:]

    rel_err_u = rel_err(u_new_times,u_old_times)
    rel_err_v = rel_err(v_new_times,v_old_times)
    rel_err_p = rel_err(p_new_times,p_old_times)
    rel_err_q = rel_err(q_new_times,q_old_times)

    print(f'Iter: {it}')
    print('Relative error for u: ', rel_err_u)
    print('Relative error for v: ', rel_err_v)
    print('Relative error for p: ', rel_err_p)
    print('Relative error for q: ', rel_err_q)

    # Assign new values to old
    u_old_times = u_new_times
    v_old_times = v_new_times
    p_old_times = p_new_times
    q_old_times = q_new_times


    # Solve for the adjoints at time zero using the previously computed values
    p_half_fun = vec_to_function(p_new_times[:nodes], V)
    q_half_fun = vec_to_function(q_new_times[:nodes], V)
    u_target0_fun = vec_to_function(data_helpers.uhat(0,T,nodes,boundary_nodes,bctype,gamma).reshape(nodes), V)
    v_target0_fun = vec_to_function(data_helpers.vhat(0,T,nodes,boundary_nodes,bctype,gamma).reshape(nodes), V)
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
    print('Relative error for p: ', rel_err_p0)
    print('Relative error for q: ', rel_err_q0)
    
    p0_old = p0_new
    q0_old = q0_new
    
    it +=1
    print('------------------------------------------------------')


print('SQP solver finished in', it, 'iterations')
end_time = timer()

# Mapping to order the solution vectors based on vertex indices
u_new_times_re = reorder_vector_from_dof_time(u_new_times, num_steps, nodes, vertextodof)
v_new_times_re = reorder_vector_from_dof_time(v_new_times, num_steps, nodes, vertextodof)
p_new_times_re = reorder_vector_from_dof_time(p_new_times, num_steps, nodes, vertextodof)
q_new_times_re = reorder_vector_from_dof_time(q_new_times, num_steps, nodes, vertextodof)
p0_new_re = reorder_vector_from_dof_time(p0_new, 1, nodes, vertextodof)
q0_new_re = reorder_vector_from_dof_time(q0_new, 1, nodes, vertextodof)

# Create target data from the images 
u_target = data_helpers.uhat(T,T,nodes,boundary_nodes,bctype,gamma).reshape(nodes)
v_target = data_helpers.vhat(T,T,nodes,boundary_nodes,bctype,gamma).reshape(nodes)

min_u = min(np.amin(u_target), np.amin(u_new_times))
max_u = max(np.amax(u_target), np.amax(u_new_times))
min_v = min(np.amin(v_target), np.amin(v_new_times))
max_v = max(np.amax(v_target), np.amax(v_new_times))

min_a = gamma/alpha*np.amin(p_new_times)
max_a = gamma/alpha*np.amax(p_new_times)
min_b = gamma/alpha*np.amin(q_new_times)
max_b = gamma/alpha*np.amax(q_new_times)

a2_norms = dt*np.ones(num_steps)
b2_norms = dt*np.ones(num_steps)

udif2_norms = dt*np.ones(num_steps)
udif2_norms[0] = udif2_norms[0]/2
udif2_norms[-1] = udif2_norms[-1]/2

vdif2_norms = dt*np.ones(num_steps)
vdif2_norms[0] = vdif2_norms[0]/2
vdif2_norms[-1] = vdif2_norms[-1]/2

cost_functional = np.zeros(num_steps)

a_means = np.zeros(num_steps)
b_means = np.zeros(num_steps)

a0_num = (p0_new_re * gamma / alpha).reshape(((sqnodes,sqnodes)))
b0_num = (q0_new_re * gamma / alpha).reshape(((sqnodes,sqnodes)))

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
    u_target_time = data_helpers.uhat(i*dt,T,nodes,boundary_nodes,bctype,gamma).reshape((sqnodes,sqnodes))
    v_target_time = data_helpers.vhat(i*dt,T,nodes,boundary_nodes,bctype,gamma).reshape((sqnodes,sqnodes))
    
    u_num = u_num.reshape((sqnodes,sqnodes))
    v_num = v_num.reshape((sqnodes,sqnodes))
    
    # Compute the controls
    a_num = (p_num * gamma / alpha).reshape(((sqnodes,sqnodes)))
    b_num = (q_num * gamma / alpha).reshape(((sqnodes,sqnodes)))
    
    a2_norm = L2(M,a_num.reshape(nodes))
    b2_norm = L2(M,b_num.reshape(nodes))
    a2_norms[i-1] *= a2_norm
    b2_norms[i-1] *= b2_norm
    
    dif_u = u_num - u_target_time
    dif_v = v_num - v_target_time
    difu2 = L2(M,dif_u.reshape(nodes))
    difv2 = L2(M,dif_v.reshape(nodes))
    
    udif2_norms[i-1] *= difu2
    vdif2_norms[i-1] *= difv2
        
    a_means[i-1] = np.mean(a_num)
    b_means[i-1] = np.mean(b_num)
    
    if show_plots is True and i%10==0:
        fig2 = plt.figure(figsize=(15,10))
        ax2 = plt.subplot(2,3,1)
        im1 = plt.imshow(u_target_time, cmap ="gray", vmin=min_u, vmax=max_u, extent =[a1,a2,a1,a2])
        fig2.colorbar(im1)
        plt.title(f'Desired state for $u$ at t={round(i*dt,4)}')
        ax2 = plt.subplot(2,3,2)
        im2 = plt.imshow( u_num, cmap="gray", vmin=min_u, vmax=max_u, extent =[a1,a2,a1,a2])
        fig2.colorbar(im2)
        plt.title(f'Computed state $u$ at t={round(i*dt,4)}')
        ax2 = plt.subplot(2,3,3)
        im3 = plt.imshow( a_num, cmap="gray", vmin=min_a, vmax=max_a, extent =[a1,a2,a1,a2])
        fig2.colorbar(im3)
        plt.title(f'Computed control $a$ at t={round((i-0.5)*dt,4)}')
        ax2 = plt.subplot(2,3,4)
        im1 = plt.imshow(v_target_time, cmap="gray",vmin=min_v, vmax=max_v, extent =[a1,a2,a1,a2])
        fig2.colorbar(im1)
        plt.title(f'Desired state for $v$ at t={round(i*dt,4)}')
        ax2 = plt.subplot(2,3,5)
        im2 = plt.imshow( v_num, cmap="gray", vmin=min_v, vmax=max_v, extent =[a1,a2,a1,a2])
        fig2.colorbar(im2)
        plt.title(f'Computed state $v$ at t={round(i*dt,4)}')
        ax2 = plt.subplot(2,3,6)
        im3 = plt.imshow( b_num, cmap="gray", vmin=min_b, vmax=max_b, extent =[a1,a2,a1,a2])
        fig2.colorbar(im3)
        plt.title(f'Computed control $b$ at t={round((i-0.5)*dt,4)}')
        # fig2.savefig(folder_name + f'/SV_alpha_{alpha}_T2_step_{i}.png', dpi=500)
        plt.show()

    # print('------------------------------------------------------')

folder_name = 'data_driven_results' 
u_new_times.tofile(folder_name + '/u_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
v_new_times.tofile(folder_name + '/v_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
p_new_times.tofile(folder_name + '/p_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
q_new_times.tofile(folder_name + '/q_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
p0_new.tofile(folder_name + '/p0_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')
q0_new.tofile(folder_name + '/q0_' + str(alpha) + '_' + str(deltax) + '.csv', sep = ',')


fig = plt.figure(figsize=(18,4.5))
ax = plt.subplot(1,4,1)
im = plt.plot(a2_norms)
plt.title('$\|a\|^2$ in $L^2(\Omega)$ throughout [0,T]')
plt.xlabel('timestep')
plt.ylabel('norm')

ax = plt.subplot(1,4,2)
im = plt.plot(b2_norms)
plt.title('$\|b\|^2$ in $L^2(\Omega)$ throughout [0,T]')
plt.xlabel('timestep')

ax = plt.subplot(1,4,3)
im = plt.plot(udif2_norms)
plt.title('$\|u-\hat{u}\|^2$ in $L^2(\Omega)$ throughout [0,T]')
plt.xlabel('timestep')

ax = plt.subplot(1,4,4)
im = plt.plot(vdif2_norms)
plt.title('$\|v-\hat{v}\|^2$ in $L^2(\Omega)$ throughout [0,T]')
plt.xlabel('timestep')
plt.show()

plt.plot(a_means, linestyle = '-', label='a')
plt.plot(b_means, linestyle = '-.', label ='b')
plt.plot(a_gen_T5*np.ones(num_steps),linestyle = '--', label='$a_G$')
plt.plot(b_gen_T5*np.ones(num_steps), linestyle = ':', label='$b_G$')
# plt.title(f'Means of controls at each time step, $\gamma$={gamma}')
plt.xlabel('Timestep')
plt.ylabel('Mean value')
plt.legend()
plt.savefig(folder_name + f'/SV_alpha_{alpha}_T2_means.png', dpi=500)
plt.show()

L2Q_u = L2Q(M, u_target_alltime[nodes:]-u_new_times, num_steps, dt, nodes, 'state')
L2Q_v = L2Q(M, v_target_alltime[nodes:]-v_new_times, num_steps, dt, nodes, 'state')
L2Q_a = L2Q(M, gamma/alpha*p_new_times, num_steps, dt, nodes, 'adjoint')
L2Q_b = L2Q(M, gamma/alpha*q_new_times, num_steps, dt, nodes, 'adjoint')

print('Elapsed time [s]: ', end_time-start_time)
print(f'mean(a(T)) = {np.mean(a_num)}')
print(f'mean(b(T)) = {np.mean(b_num)}')
print(f'RE(u,uhat) = {rel_err(u_num,u_target_time)}')
print(f'RE(v,vhat) = {rel_err(v_num,v_target_time)}')
print(minres_its)   
print('L^2(Q) [\hat{u}-u]^2 =', L2Q_u)
print('L^2(Q) [\hat{v}-v] ^2=', L2Q_v)
print('L^2(Q) [a] ^2=', L2Q_a)
print('L^2(Q) [b] ^2=', L2Q_b)