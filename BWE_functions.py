import numpy as np
from scipy.sparse import vstack, hstack, csr_matrix
from helpers import *
from scipy.sparse.linalg import LinearOperator
import pyamg

def preconditioner_matrix(B11, M, L1, L2, L3, L4, Em,
                          num_steps, nodes, dt, gamma, alpha, beta): 
    """Constructs a Linear Operator for the preconditioner to be used with
    scipy.sparse.linalg.minres or scipy.sparse.linalg.gmres which require that 
    the supplied preconditioner should approximate the inverse of the system
    matrix. The Linear Operator multiplies the given vector by the inverse of 
    the preconditioner by solving for x in Px = y where P is the preconditioner 
    and y is a given vector.
    
    Parameters:
        B11 (scipy.sparse.csr_matrix): Sparse matrix B11 (the (1,1)-block).
        M (scipy.sparse.csr_matrix): Sparse matrix M (the mass matrix).
        L1, L2, L3, L4 (scipy.sparse.csr_matrix): Sparse matrices L1, L2, L3, L4.
        Em (scipy.sparse.csr_matrix): Sparse matrix Em.
        num_steps (int): Number of time steps.
        nodes (int): Number of nodes.
        dt (float): Time step size.
        gamma (float): Parameter gamma.
        alpha (float): Parameter alpha.
        beta (float): Parameter beta.

    Returns:
        scipy.sparse.linalg.LinearOperator: Linear operator representing the 
            preconditioning matrix.
    """
    n = 4 * (num_steps - 1) * nodes
    
    def mv(vec):
        """Function for matrix-vector multiplication with the preconditioning 
        matrix. Multiplies the input vector with the preconditioning matrix.

        Parameters:
            vec (numpy.ndarray): Input vector.

        Returns:
            numpy.ndarray: Resulting vector after multiplication.
        """
        m = int(n / 4)  # (N-1)(M-1)^2
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
        B11diag = B11.diagonal()
        ### Solve for x1
        for i in range(1, 2 * num_steps - 1):  # loop i=1:2*(num_steps-1)
            x1[(i - 1) * a:i * a] = ChebSI(
            z1[(i - 1) * a:i * a], B11, B11diag, 20, 0.5, 2
            )

        ### Solve for x2 in 2 steps:

        ## Solve for y (Back substitution)
        # Calculate y1_{N-1}, y2_{N-1}
        mat = vstack([
            hstack([L1[-1] + Em, L3[-1]]),
            hstack([L2[-1], L4[-1] + Em])
        ])
        rhs = np.concatenate((z21[(num_steps - 2) * a:], z22[(num_steps - 2) * a:]))
        ml1 = pyamg.smoothed_aggregation_solver(mat)
        y_sol = ml1.solve(rhs, tol=0.0, maxiter=6, cycle="V")
        y1[(num_steps - 2) * a:], y2[(num_steps - 2) * a:] = np.split(y_sol, 2)

        # Solve for y1, y2
        for i in reversed(range(1, num_steps - 1)):
            mat = vstack([
            hstack([L1[i - 1] + Em, L3[i - 1]]),
            hstack([L2[i - 1], L4[i - 1] + Em])
            ])
            rhs = np.concatenate((
            z21[(i - 1) * a:i * a] + M * y_sol[:a],  # z21^i + My21^{i+1}
            z22[(i - 1) * a:i * a] + M * y_sol[a:]   # z22^i + My22^{i+1}
            ))
            ml1 = pyamg.smoothed_aggregation_solver(mat)
            y_sol = ml1.solve(rhs, tol=0.0, maxiter=6, cycle="V")
            y1[(i - 1) * a:i * a], y2[(i - 1) * a:i * a] = np.split(y_sol, 2)
            
        ## Solve for x2=[x21,x22] using y1,y1 (Forward substitution)
        
        # Calculate x21_1, x22_1
        mat = vstack([hstack([L1[0] + Em, L2[0]]),
                      hstack([L3[0], L4[0] + Em])])
        rhs = np.concatenate((B11 * y1[:a], B11 * y2[:a]))
        ml1 = pyamg.smoothed_aggregation_solver(mat)
        x_sol = ml1.solve(rhs, tol=0.0, maxiter=6, cycle="V")
        x21[:a], x22[:a] = np.split(x_sol, 2)
        
        # Solve for x21, x22
        for i in range(1, num_steps - 1):
            mat = vstack([hstack([L1[i] + Em, L2[i]]),
                  hstack([L3[i], L4[i] + Em])])
            rhs = np.concatenate(
            (B11 * y1[i * a:(i + 1) * a] + M * x_sol[:a],  # B11*y1_i + M*x21_{i-1}
             B11 * y2[i * a:(i + 1) * a] + M * x_sol[a:]))  # B11*y2_i + M*x22^{i-1}
            ml1 = pyamg.smoothed_aggregation_solver(mat)
            x_sol = ml1.solve(rhs, tol=0.0, maxiter=6, cycle="V")
            x21[i * a:(i + 1) * a], x22[i * a:(i + 1) * a] = np.split(x_sol, 2)
        return np.array([x1, np.concatenate((x21, x22))])

    return LinearOperator((n, n), matvec=mv)


def system_matrix(B11, B22, M, L1, L2, L3, L4, L5, L6,
                  num_steps, nodes, dt, gamma, alpha, beta): 
    """Constructs a Linear Operator that multiplies a vector with the
    system matrix of the nonlinear problem.

    Parameters:
        B11 (scipy.sparse.csr_matrix): Sparse matrix B11 (the (1,1)-block).
        B22 (scipy.sparse.csr_matrix): Sparse matrix B22 (the (2,2)-block).
        M (scipy.sparse.csr_matrix): Sparse matrix M.
        L1, L2, L3, L4, L5, L6 (list of scipy.sparse.csr_matrix): Lists of 
            sparse matrices L1, L2, L3, L4, L5, and L6.
        num_steps (int): Number of time steps.
        nodes (int): Number of nodes.
        dt (float): Time step size.
        gamma (float): Parameter gamma.
        alpha (float): Parameter alpha.
        beta (float): Parameter beta.

    Returns:
        scipy.sparse.linalg.LinearOperator: Linear operator representing the 
            system matrix.
    """
    n = 4 * (num_steps - 1) * nodes
    def mv(vec):
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
        
        x1[:a] = B11 * z1[:a] + L1[0] * z3[:a] + L2[0] * z4[:a]
        x2[:a] = B11 * z2[:a] + L3[0] * z3[:a] + L4[0] * z4[:a]

        for i in range(2, num_steps):  # i=2:N-1
            s_i = (i - 1) * a  # start ith entry
            e_i = i
            x1[s_i:e_i] = B11*z1[s_i:e_i] -\
                M*z3[(i-2)*a:s_i] + L1[i-1]*z3[s_i:e_i] +\
                    L2[i-1]*z4[s_i:e_i]
            
            x2[s_i:e_i] = B11*z2[s_i:e_i] +\
                L3[i-1]*z3[s_i:e_i] -\
                    M*z4[(i-2)*a:s_i] + L4[i-1]*z4[s_i:e_i]
            

        for i in range(1,num_steps-1): # i=1:N-2
            s_i = (i-1)*a    # start ith entry
            e_i = i*a        # end ith entry = start (i+1)th entry
            x3[s_i:e_i] = L1[i-1]*z1[s_i:e_i] - M*z1[e_i:(i+1)*a] +\
                L3[i-1]*z2[s_i:e_i] + L5[i-1]*z3[s_i:e_i] +\
                    L6[i-1]*z4[s_i:e_i]
            
            x4[s_i:e_i] = L2[i-1]*z1[s_i:e_i] +\
                L4[i-1]*z2[s_i:e_i] - M*z2[e_i:(i+1)*a] +\
                L6[i-1]*z3[s_i:e_i] + B22*z4[s_i:e_i]
        
        b = num_steps-2
        x3[b*a:] = L1[-1]*z1[b*a:] + L3[-1]*z2[(num_steps-2)*a:] +\
            L5[-1]*z3[b*a:] + L6[-1]*z4[b*a:]
            
        x4[b*a:] = L2[-1]*z1[b*a:] + L4[-1]*z2[(num_steps-2)*a:] +\
            L6[-1]*z3[b*a:] + B22*z4[b*a:] 
        return np.array([x1, x2, x3, x4])

    return LinearOperator((n,n), matvec = mv)
