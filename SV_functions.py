import numpy as np
from scipy.sparse import vstack, hstack, csr_matrix
from helpers import *
from scipy.sparse.linalg import LinearOperator
import pyamg

def system_matrix(Q, B11, B12, B21, B22, B11s, B12s, B21s, B22s, C11, C12, C22,
                  num_steps, nodes, dt, gamma, alpha, beta):
    """Constructs a Linear Operator that multiplies a vector with the system
    matrix of the nonlinear problem.
    
    Parameters:
        Q (numpy.ndarray): Matrix Q used in the system matrix.
        B11, B12, B21, B22 (list of numpy.ndarray): Lists of matrices B11, B12,
            B21, B22 used in the system.
        B11s, B12s, B21s, B22s (list of numpy.ndarray): Lists of matrices B11s,
            B12s, B21s, B22s used in the system.
        C11, C12, C22 (list of numpy.ndarray): Lists of matrices C11, C12, C22
            used in the system.
        num_steps (int): Number of time steps.
        nodes (int): Number of nodes.
        dt (float): Time step size.
        gamma (float): Parameter gamma.
        alpha (float): Parameter alpha.
        beta (float): Parameter beta.
    
    Returns:
        LinearOperator: A linear operator that represents the system matrix.
    """
    n = 4 * num_steps * nodes

    def mv(vec):
        """Function for matrix-vector multiplication with the system matrix.
        Multiplies the input vector with the system matrix.
        
        Parameters:
        vec (numpy.ndarray): Input vector of length `n`, where `n` is assumed
            to be divisible by 4.
        
        Returns:
        numpy.ndarray: A 2D array containing four vectors [x1, x2, x3, x4] as
            rows, each of length `m` where `m = n/4`.
        """
        
        m = int(n / 4)  # = nodes
   
        z1 = vec[:m]
        z2 = vec[m:2 * m]
        z3 = vec[2 * m:3 * m]
        z4 = vec[3 * m:4 * m]
        
        x1 = np.zeros(m)
        x2 = np.zeros(m)
        x3 = np.zeros(m)
        x4 = np.zeros(m)
        a = nodes
        
        x1[:a] = Q * z1[:a] + B11[0] * z3[:a] + B21[0] * z4[:a]
        x2[:a] = Q * z2[:a] + B12[0] * z3[:a] + B22[0] * z4[:a]

        for i in range(2, num_steps + 1):  # i=2:N
            s_i = (i - 1) * a  # start ith entry
            e_i = i * a  # end ith entry = start (i+1)th entry
            x1[s_i:e_i] = Q * z1[s_i:e_i] +\
                B11s[i - 2] * z3[(i - 2) * a:s_i] + B11[i - 1] * z3[s_i:e_i] +\
                B21s[i - 2] * z4[(i - 2) * a:s_i] + B21[i - 1] * z4[s_i:e_i]
            
            x2[s_i:e_i] = Q * z2[s_i:e_i] +\
                B12s[i - 2] * z3[(i - 2) * a:s_i] + B12[i - 1] * z3[s_i:e_i] +\
                B22s[i - 2] * z4[(i - 2) * a:s_i] + B22[i - 1] * z4[s_i:e_i]

        for i in range(1, num_steps):  # i=1:N-1
            s_i = (i - 1) * a  # start ith entry
            e_i = i * a  # end ith entry = start (i+1)th entry
            x3[s_i:e_i] = B11[i - 1] * z1[s_i:e_i] +\
                B11s[i - 1] * z1[e_i:(i + 1) * a] +\
                B12[i - 1] * z2[s_i:e_i] +\
                B12s[i - 1] * z2[e_i:(i + 1) * a] +\
                C11[i - 1] * z3[s_i:e_i] + C12[i - 1] * z4[s_i:e_i]
            
            x4[s_i:e_i] = B21[i - 1] * z1[s_i:e_i] +\
                B21s[i - 1] * z1[e_i:(i + 1) * a] +\
                B22[i - 1] * z2[s_i:e_i] +\
                B22s[i - 1] * z2[e_i:(i + 1) * a] +\
                C12[i - 1] * z3[s_i:e_i] + C22[i - 1] * z4[s_i:e_i]
        
        b = num_steps - 1
        x3[b * a:] = B11[-1] * z1[b * a:] + B12[-1] * z2[b * a:] +\
            C11[-1] * z3[b * a:] + C12[-1] * z4[b * a:]
            
        x4[b * a:] = B21[-1] * z1[b * a:] + B22[-1] * z2[b * a:] +\
            C12[-1] * z3[b * a:] + C22[-1] * z4[b * a:] 
            
        return np.array([x1, x2, x3, x4])

    return LinearOperator((n, n), matvec=mv)

def preconditioner_matrix(Q, B11, B12, B21, B22, Bs11, Bs12, Bs21, Bs22, Fm,
                          num_steps, nodes, dt, gamma, alpha, beta):
    """Constructs a Linear Operator for the preconditioner to be used with
    scipy.sparse.linalg.minres or scipy.sparse.linalg.gmres which require that 
    the supplied preconditioner should approximate the inverse of the system
    matrix. The Linear Operator multiplies the given vector by the inverse of 
    the preconditioner by solving for x in Px = y where P is the preconditioner 
    and y is a given vector.
    
    Parameters:
        Q (numpy.ndarray): Matrix Q used in the preconditioning matrix.
        B11, B12, B21, B22 (list of numpy.ndarray): Lists of matrices B11, B12,
            B21, B22 used in the preconditioning matrix.
        Bs11, Bs12, Bs21, Bs22 (list of numpy.ndarray): Lists of subdiagonal
            matrices Bs11, Bs12, Bs21, Bs22 used in the preconditioning matrix.
        Fm (numpy.ndarray): Matrix Fm used in the preconditioning matrix.
        num_steps (int): Number of time steps.
        nodes (int): Number of nodes.
        dt (float): Time step size.
        gamma (float): Parameter gamma.
        alpha (float): Parameter alpha.
        beta (float): Parameter beta.
    
    Returns:
        LinearOperator: A linear operator that represents the preconditioning
            matrix.
    """
    n = 4 * num_steps * nodes
    
    def mv(vec):
        """Function for matrix-vector multiplication with the preconditioning
        matrix. Multiplies the input vector with the preconditioning matrix.

        Parameters:
            vec (numpy.ndarray): Input vector.

        Returns:
            numpy.ndarray: Resulting vector after multiplication.
        """
        m = int(n / 4) 
        # RHS
        z1 = vec[:2 * m]
        z21 = vec[2 * m:3 * m]
        z22 = vec[3 * m:]
        # Unknowns
        x1 = np.zeros(2 * m)
        x21 = np.zeros(m)
        x22 = np.zeros(m)
        y1 = np.zeros(m)
        y2 = np.zeros(m)
        
        a = nodes
        ### Solve for x1
        Qdiag = Q.diagonal()
        for i in range(2 * num_steps):
            x1[i * a:(i + 1) * a] = ChebSI(z1[i * a:(i + 1) * a], Q, Qdiag, 20,
                                           0.5, 2)  # originally 20 iter
        
        ### Solve for x2 
   
        # Calculate y1_{N}, y2_{N}
        mat = vstack([hstack([B11[-1] + Fm / 2, B12[-1]]),
                      hstack([B21[-1], B22[-1] + Fm / 2])])
        
        rhs = np.concatenate((z21[(num_steps - 1) * a:], z22[(num_steps - 1) * a:]))
        ml1 = pyamg.smoothed_aggregation_solver(mat)
        y_sol = ml1.solve(rhs, tol=0.0, maxiter=6, cycle="V")
        
        y1[(num_steps - 1) * a:], y2[(num_steps - 1) * a:] = np.split(y_sol, 2)
   
        # Solve for y1, y2
        for i in reversed(range(1, num_steps)):
            mat = vstack([hstack([B11[i - 1] + Fm, B12[i - 1]]),
                          hstack([B21[i - 1], B22[i - 1] + Fm])])
            
            rhs = np.concatenate((z21[(i - 1) * a:i * a] - Bs11[i - 1] * y_sol[:a] - Bs12[i - 1] * y_sol[a:], 
                                  z22[(i - 1) * a:i * a] - Bs21[i - 1] * y_sol[:a] - Bs22[i - 1] * y_sol[a:])) 
            
            ml1 = pyamg.smoothed_aggregation_solver(mat)
            y_sol = ml1.solve(rhs, tol=0.0, maxiter=6, cycle="V")
            
            y1[(i - 1) * a:i * a], y2[(i - 1) * a:i * a] = np.split(y_sol, 2)

        # Calculate x21_1, x22_1
        mat = vstack([hstack([B11[0] + Fm, B21[0]]),
                      hstack([B12[0], B22[0] + Fm])])
        rhs = np.concatenate((Q * y1[:a], Q * y2[:a]))
        
        ml1 = pyamg.smoothed_aggregation_solver(mat)
        x_sol = ml1.solve(rhs, tol=0.0, maxiter=6, cycle="V")
        
        x21[:a], x22[:a] = np.split(x_sol, 2)
        
        # Solve for x21, x22
        for i in range(1, num_steps):
            if i != num_steps - 1:
                F = Fm
            else:
                F = Fm / 2
            mat = vstack([hstack([B11[i] + F, B21[i]]),
                          hstack([B12[i], B22[i] + F])])
            rhs = np.concatenate((Q * y1[i * a:(i + 1) * a] - Bs11[i - 1] * x_sol[:a] - Bs21[i - 1] * x_sol[a:],
                                  Q * y2[i * a:(i + 1) * a] - Bs12[i - 1] * x_sol[:a] - Bs22[i - 1] * x_sol[a:]))
           
            ml1 = pyamg.smoothed_aggregation_solver(mat)
            x_sol = ml1.solve(rhs, tol=0.0, maxiter=6, cycle="V")
            
            x21[i * a:(i + 1) * a], x22[i * a:(i + 1) * a] = np.split(x_sol, 2)
        return np.array([x1, np.concatenate((x21, x22))])

    return LinearOperator((n, n), matvec=mv)

def rhs_from_matrix(block, inner_nodes, boundary_nodes, bc_vec):
    """Creates a vector to be subtracted from the RHS in the Dirichlet BC case.
    This function sets both boundary rows and columns to zero to keep the
    system matrix symmetric.
    
    Parameters:
        block (scipy.sparse matrix): The original block before setting boundary
            rows and columns to zero.
        inner_nodes (list): List of inner nodes in degree of freedom (dof)
            ordering.
        boundary_nodes (list): List of boundary nodes in dof ordering.
        bc_vec (numpy.ndarray): Vector of boundary values for a certain
            variable.
    
    Returns:
        numpy.ndarray: Vector to be subtracted from the RHS. Calculated as the
            product of the rows corresponding to inner nodes with the columns
            non-zero only for boundary nodes, with the vector of boundary nodes.
    """
    mat = np.zeros(block.shape)
    vec = block[inner_nodes, :].toarray()
    mat[inner_nodes, :] = vec
    mat[:, inner_nodes] = 0
    out_vec = mat @ bc_vec
    return out_vec
