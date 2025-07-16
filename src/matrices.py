import numpy as np

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def omega_matrix(omega: np.ndarray) -> np.ndarray:
    '''
    Returns the skew-symmetric matrix for a given angular velocity.
    '''
    w1, w2, w3 = omega.squeeze()
    return np.array([
        [0.0,   w3,  -w2,  w1],
        [-w3,  0.0,   w1,  w2],
        [w2,  -w1,  0.0,   w3],
        [-w1, -w2,  -w3,  0.0],
    ])

def cross_matrix(v: np.ndarray) -> np.ndarray:
    '''
    Returns the cross-product matrix for a given vector.
    '''
    v1, v2, v3 = v.squeeze()
    return np.array([
        [0.0, -v3,  v2],
        [v3,  0.0, -v1],
        [-v2,  v1,  0.0],
    ]) # shape (3,3)

def attitude_matrix(q: np.ndarray) -> np.ndarray:
    '''
    Returns the attitude matrix for a given quaternion.
    '''
    e = q[:3].reshape(3,1)
    q4 = q[3]
    return (q4**2 - e @ e.T) @ np.eye(3) + 2.0 * e@e.T + 2.0 * q4 * cross_matrix(e)

def H_q(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    '''
    Returns the Jacobian of the attitude matrix with respect to the quaternion.
    '''
    e = q[:3].reshape(3, 1)   # shape (3, 1)
    q4 = q[3].squeeze()
    r = r.reshape(3, 1)       # shape (3, 1)
    
    eTrI = e @ r.T * np.eye(3)
    erT = e @ r.T
    reT = r @ e.T
    qrx = q4 * cross_matrix(r)
    qr = q4 * r
    rxe = cross_matrix(r) @ e

    term_vec = eTrI + (erT - reT) + qrx
    term_scalar = qr + rxe
    return 2.0 * np.hstack([term_vec, term_scalar]) # shape (3,4)

def Gt_matrix(q: np.ndarray) -> np.ndarray:
    '''
    Returns the G matrix - which is the coefficient matrix for the noise vector w.
    '''
    
    top_left = squeezed_Gt_matrix(q)
    top_right = np.zeros((4,3))
    bottom_left = np.zeros((3,3))
    bottom_right = np.eye(3)
    return np.vstack([np.hstack([top_left, top_right]), np.hstack([bottom_left, bottom_right])]) # shape (7,6)
    
    
    
def squeezed_Gt_matrix(q: np.ndarray) -> np.ndarray:
    '''
    Returns the noise matrix components of Gt matrix (only the topleft part)
    Also denoted as ksi-matrix
    '''
    e1, e2 ,e3 ,w = q.squeeze()
    
    G = np.array([
        [w, -e3, e2],
        [e3, w, -e1],
        [-e2, e1, w],
        [-e1, -e2, -e3],
        ])
    return -0.5 * G # shape (4,3)

def F_matrix(x: np.ndarray, omega_m: np.ndarray) -> np.ndarray:
    '''
    Returns the Jacobian of the state transition matrix with respect to the state.
    '''
    q = x[:4] # dims (4,1)
    mu = x[4:] # dims (3,1)
    omega_tilde = omega_m - mu # dims (3,1)
    Omega_half = 0.5 * omega_matrix(omega_tilde) # dims (4,4)
    G_eps = squeezed_Gt_matrix(q)     # dims (4,3)
    upper = np.hstack([Omega_half, G_eps])
    lower = np.zeros((3, 7))
    F_mat =  np.vstack([upper, lower])
    return F_mat # dims (7,7)

def additive_Q(Q: np.ndarray, Gamma: np.ndarray, dt: float) -> np.ndarray:
        '''
        Returns the additive covariance matrix Q.
        '''
        return (Gamma @ Q @ Gamma.T) * dt
    
    
def mekf_F_matrix(omega_tilde: np.ndarray) -> np.ndarray:
    '''
    Returns the Jacobian of the state transition matrix with respect to the state.
    '''
    top_left = -cross_matrix(omega_tilde) # dims (3,3)
    zero_33 = np.zeros((3,3))
    I3 = -np.eye(3)
    result = np.vstack([np.hstack([top_left, I3]), np.hstack([zero_33, zero_33])])
    return result # shape (6,6)


def mekf_Gt_matrix() -> np.ndarray:
    '''
    Returns the Jacobian of the state transition matrix with respect to the state.
    '''
    zero_33 = np.zeros((3,3))
    I3 = np.eye(3)
    result = np.vstack([np.hstack([-I3, zero_33]), np.hstack([zero_33, I3])])
    
    return result # shape (6,6)


def mekf_H_q(b_k: np.ndarray) -> np.ndarray:
    '''
    Returns the Jacobian of the attitude matrix with respect to the quaternion.
    b_k is the nominal measurement, which is the attitude matrix @ r, and shape is (3,1)
    '''
    Hq = -cross_matrix(b_k) # shape (3,3)
    return Hq # shape (3,3)


def mekf_H(b_k: np.ndarray) -> np.ndarray:
    '''
    Returns the Jacobian of the measurement matrix with respect to the state.
    '''
    Hq = mekf_H_q(b_k) # shape (3,3)
    H = np.hstack([Hq, np.zeros((3, 3))]) # dims (3,6)
    return H # shape (3,6)


def _mekf_H(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    3×6 Jacobian  H = [ -A(q) [r]_x , 0 ]   for the MEKF.
    given by the chat - need to check
    """
    A_q = attitude_matrix(q)                 # 3×3
    Hq  = -A_q @ cross_matrix(r)             # 3×3
    return np.hstack([Hq, np.zeros((3,3))])  # 3×6



def quaternion_multiply(q2: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """
    Hamilton product  q = q2 ⊗ q1.
    Both quaternions are column-vectors [x, y, z, w]ᵀ with the scalar part last.
    Returns a 4×1 column-vector.
    """
    x1, y1, z1, w1 = q1.flatten()
    x2, y2, z2, w2 = q2.flatten()

    x =  w2*x1 + x2*w1 + y2*z1 - z2*y1
    y =  w2*y1 - x2*z1 + y2*w1 + z2*x1
    z =  w2*z1 + x2*y1 - y2*x1 + z2*w1
    w =  w2*w1 - x2*x1 - y2*y1 - z2*z1

    return np.array([[x], [y], [z], [w]])




def g_nor(q: np.ndarray) -> np.ndarray:
    return q.T@q # shape (1,1)


def white_noise(cov: np.ndarray) -> np.ndarray:
    """
    Generate white noise with zero mean and covariance matrix R.
    
    Args:
        R: Diagonal covariance matrix of any size (n x n)
        
    Returns:
        Random vector of size n with zero mean and covariance R
    """
    n = cov.shape[0]  # Get the dimension from the covariance matrix
    return np.random.multivariate_normal(np.zeros(n), cov)