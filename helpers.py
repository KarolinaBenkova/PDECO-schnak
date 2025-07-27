import dolfin as df
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

# Contains functions used in all scripts

def reorder_vector_to_dof(vec, nodes, vertextodof):
    vec_dof = np.zeros(vec.shape)
    for i in range(nodes):
        j = int(vertextodof[i])
        vec_dof[j] = vec[i]
    return vec_dof

def reorder_vector_to_dof_time(vec, num_steps, nodes, vertextodof):
    vec_dof = np.zeros(vec.shape)
    for n in range(num_steps):
        temp = vec[n*nodes:(n+1)*nodes]
        for i in range(nodes):
            j = int(vertextodof[i])
            vec_dof[n*nodes + j] = temp[i] 
    return vec_dof

def reorder_vector_from_dof_time(vec, num_steps, nodes, vertextodof):
    vec_dof = np.zeros(vec.shape)
    for n in range(num_steps):
        temp = vec[n*nodes:(n+1)*nodes]
        for i in range(nodes):
            j = int(vertextodof[i])
            vec_dof[n*nodes + i] = temp[j] 
    return vec_dof

def rel_err(new,old):
    '''
    Calculates the relative error between the new and the old value.
    '''
    return np.linalg.norm(new-old)/np.linalg.norm(old)

def assemble_sparse(a):
    '''
    a is an integral that can be assembled to dolfin.cpp.la.Matrix.
    The function converts it to a sparse, csr matrix.
    '''
    A = df.assemble(a)
    mat = df.as_backend_type(A).mat()
    csr = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    return csr

def assemble_sparse_lil(a):
    '''
    a is an integral that can be assembled to dolfin.cpp.la.Matrix.
    The function converts it to a sparse, csr matrix of the form lil.
    '''
    csr = assemble_sparse(a)
    return lil_matrix(csr)

def vec_to_function(vec, V):
    out = df.Function(V)
    out.vector().set_local(vec)
    return out

class MinresCounter:
    def __init__(self):
        self.num_iterations = 0

    def __call__(self, x=None):
        self.num_iterations += 1

def ChebSI(vec,M,Md,cheb_iter,lmin,lmax):
    ymid = 0*vec
    yold = ymid
    omega = 0
    
    rho = (lmax - lmin) / (lmax + lmin)
    Md = (lmin + lmax) / 2 * Md
    
    for k in range(1,cheb_iter + 1):
        if k==2:
            omega = 1 / (1 - rho**2 / 2)
        else:
            omega = 1 / (1 -(omega * rho**2) / 4);
        r = vec - M*ymid #np.dot(M,ymid)                    #?  residual = b - Ax
        z = r / Md  # z = Md\r ? 
        ynew = omega * (z + ymid - yold) + yold
        yold = ymid
        ymid = ynew
    
    return ynew

def boundary(x, on_boundary):
    return on_boundary

def uex(t,X,Y,e1,k1):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions)
    '''
    out = np.exp(e1*t)*(np.cos(k1*X*np.pi)*np.cos(k1*Y*np.pi)+1)
    return out 

def vex(t,X,Y,e2,k2):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions)
    '''
    out = np.exp(e2*t)*(np.cos(k2*X*np.pi)*np.cos(k2*Y*np.pi)+1)
    return out

def pex(t,T,X,Y,e3,k3):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions)
    '''
    out = (np.exp(e3*t)-np.exp(e3*T))*(np.cos(k3*X*np.pi)*np.cos(k3*Y*np.pi)+1)
    return out 

def qex(t,T,X,Y,e4,k4):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions)
    '''
    out = (np.exp(e4*t)-np.exp(e4*T))*(np.cos(k4*X*np.pi)*np.cos(k4*Y*np.pi)+1)
    return out 


def fex(t,T,X,Y,e1,e2,e3,k1,k2,k3,Du,alpha,gamma):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions)
    '''

    ux = uex(t,X,Y,e1,k1)
    vx = vex(t,X,Y,e2,k2)
    px = pex(t,T,X,Y,e3,k3)
    
    out =  (e1+gamma)*ux+ 2*Du*k1**2*np.pi**2*np.exp(e1*t)*np.cos(k1*X*np.pi)*np.cos(k1*Y*np.pi) - \
        gamma*ux**2*vx - gamma**2/alpha*px
    return out

def gex(t,T,X,Y,e1,e2,e4,k1,k2,k4,Dv,alpha,gamma):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions)
    '''

    ux = uex(t,X,Y,e1,k1)
    vx = vex(t,X,Y,e2,k2)
    qx = qex(t,T,X,Y,e4,k4)

    out = e2*vx+ 2*Dv*k2**2*np.pi**2*np.exp(e2*t)*np.cos(k2*X*np.pi)*np.cos(k2*Y*np.pi) + \
        gamma*ux**2*vx - gamma**2/alpha*qx
    return out

def uhat(t,T,X,Y,e1,e2,e3,e4,k1,k2,k3,k4,Du,beta,gamma):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions)
    '''
    ux = uex(t,X,Y,e1,k1)
    vx = vex(t,X,Y,e2,k2)
    px = pex(t,T,X,Y,e3,k3)
    qx = qex(t,T,X,Y,e4,k4)
    
    out = 1/beta*(-e3*np.exp(e3*t)*(np.cos(k3*X*np.pi)*np.cos(k3*Y*np.pi) + 1) +\
                2*Du*np.pi**2*k3**2*(np.exp(e3*t)-np.exp(e3*T))*np.cos(k3*X*np.pi)*np.cos(k3*Y*np.pi) +\
                beta*ux +\
                2*gamma*ux*vx*(qx-px) +\
                gamma*px)
    return out

def vhat(t,T,X,Y,e1,e2,e3,e4,k1,k2,k3,k4,Dv,beta,gamma):
    '''
    Function for the true solution.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions)
    '''
    ux = uex(t,X,Y,e1,k1)
    vx = vex(t,X,Y,e2,k2)
    px = pex(t,T,X,Y,e3,k3)
    qx = qex(t,T,X,Y,e4,k4)
    
    out = 1/beta*(-e4*np.exp(e4*t)*(np.cos(k4*X*np.pi)*np.cos(k4*Y*np.pi) + 1) +\
               2*Dv*np.pi**2*k4**2*(np.exp(e4*t)-np.exp(e4*T))*np.cos(k4*X*np.pi)*np.cos(k4*Y*np.pi) +\
               beta*vx +\
               gamma*ux**2*(qx-px))

    return out

def fex_p(t,T,X,Y,e1,e2,k1,k2,Du,alpha,gamma):
    '''
    Function for the true solution for state equation with zero control.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions)
    '''

    ux = uex(t,X,Y,e1,k1)
    vx = vex(t,X,Y,e2,k2)
    
    out =  (e1+gamma)*ux+ 2*Du*k1**2*np.pi**2*np.exp(e1*t)*np.cos(k1*X*np.pi)*np.cos(k1*Y*np.pi) - \
        gamma*ux**2*vx 
    return out

def gex_p(t,T,X,Y,e1,e2,k1,k2,Dv,alpha,gamma):
    '''
    Function for the true solution for state equation with zero control.
    Input = mesh grid (X,Y = square 2D arrays with the same dimensions)
    '''

    ux = uex(t,X,Y,e1,k1)
    vx = vex(t,X,Y,e2,k2)

    out = e2*vx+ 2*Dv*k2**2*np.pi**2*np.exp(e2*t)*np.cos(k2*X*np.pi)*np.cos(k2*Y*np.pi) + \
        gamma*ux**2*vx
    return out

def L2(mat,vec):
    '''
    Function to calculate the L^2-norm squared of the input vector with the 
    input (mass) matrix.
    '''
    p1 = vec @ mat
    return p1 @ vec

def L2Q(M,vec,num_steps,dt,nodes,var):
    ''' 
    Function to calculate the L^2 norm over the spatio-temporal domain Q.
    The quadrature formula differs for variable type,
    var = {'state', 'adjoint'}
    '''
    n = 0
    for i in range(1,num_steps):
        start = (i-1)*nodes
        end = i*nodes
        n += dt*L2(M, vec[start:end])
    if var=='state':
        n += dt/2*L2(M, vec[(num_steps-1)*nodes:])
    elif var=='adjoint':
        n += dt*L2(M, vec[(num_steps-1)*nodes:])
    return n