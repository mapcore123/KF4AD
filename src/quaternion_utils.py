import numpy as np

# Quaternion utilities following scalar-last convention [e1, e2, e3, q4]

EPS = 1e-12

def normalize(q: np.ndarray) -> np.ndarray:
    """Return a unit‐norm quaternion."""
    n = np.linalg.norm(q)
    if n < EPS:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n

def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (scalar‐last)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w])

def omega_matrix(omega: np.ndarray) -> np.ndarray:
    """Return 4x4 Omega matrix such that q_dot = 0.5*Omega(q)*q."""
    wx, wy, wz = omega
    return np.array([
        [ 0.0,  wz, -wy,  wx],
        [-wz,  0.0,  wx,  wy],
        [ wy, -wx,  0.0,  wz],
        [-wx, -wy, -wz,  0.0],
    ])

def propagate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Propagate quaternion given angular velocity (rad/s) for dt seconds using first‐order integration."""
    # Use quaternion exponential mapping for better accuracy
    angle = np.linalg.norm(omega) * dt
    if angle < EPS:
        dq = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        axis = omega / np.linalg.norm(omega)
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        dq = np.concatenate((axis * sin_half, [np.cos(half_angle)]))
    q_new = multiply(q, dq)
    return normalize(q_new)

def to_dcm(q: np.ndarray) -> np.ndarray:
    """Return 3x3 direction cosine matrix (DCM) from inertial to body frame."""
    x, y, z, w = q
    # Precompute products
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ])

def small_angle_quat(delta_theta: np.ndarray) -> np.ndarray:
    """Convert small rotation vector to quaternion (first order)."""
    half_theta = 0.5 * delta_theta
    return normalize(np.concatenate((half_theta, [1.0])))

def delta_angle(q_true: np.ndarray, q_est: np.ndarray) -> float:
    """Compute angular error δφ between true and estimated quaternion (rad)."""
    A_true = to_dcm(q_true)
    A_est = to_dcm(q_est)
    delta_A = A_est @ A_true.T
    trace = np.trace(delta_A)
    val = 0.5 * (trace - 1.0)
    # Clamp due to numerical errors
    val = np.clip(val, -1.0, 1.0)
    return np.arccos(val) 