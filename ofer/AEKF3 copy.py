import numpy as np
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Create a synthetic curved surface: z = 0.05 x^2 + 0.03 y^2 + noise
rng = np.random.default_rng(42)
n_points = 800
x = rng.uniform(-0.05, 0.05, n_points)
y = rng.uniform(-0.05, 0.05, n_points)
z_clean = 0.05 * x**2 + 0.03 * y**2
z = z_clean + rng.normal(0, 0.0003, n_points)  # small measurement noise (±0.3 mm)

points = np.vstack([x, y, z]).T

# 2. Pick the contact point P (the "touch" location of the board)
P = np.array([0.01, -0.015, 0.05 * 0.01**2 + 0.03 * (-0.015)**2])  # known on the analytic surface

# 3. Extract a small neighbourhood (r = 8 mm)
r = 0.008
d2 = np.sum((points - P)**2, axis=1)
patch = points[d2 < r**2]

# Translate neighbourhood so that P is at origin
xyz = patch - P
x_loc, y_loc, z_loc = xyz.T

# 4. Quadric model z = a x + b y + c x^2 + d xy + e y^2  (no constant term after translation)
def quadric(B, data):
    a, b, c, d, e = B
    X, Y = data
    return a*X + b*Y + c*X**2 + d*X*Y + e*Y**2

data = RealData(np.vstack([x_loc, y_loc]), z_loc)
model = Model(quadric)
beta0 = np.zeros(5)
out = ODR(data, model, beta0).run()
a, b, c_, d_, e_ = out.beta
sigma_a, sigma_b = out.sd_beta[:2]

# 5. Tangent plane normal at P
n = np.array([-a, -b, 1.0])
n /= np.linalg.norm(n)

# Compute roll, pitch, inclination
roll = np.degrees(np.arctan(-a))     # rotation about y‑axis
pitch = np.degrees(np.arctan(-b))    # rotation about x‑axis
incl = np.degrees(np.arctan(np.hypot(a, b)))

print(f"Tangent‑plane normal  : {n}")
print(f"Roll  (about Y)       : {roll:.3f}°   ±{np.degrees(sigma_a):.3f}°")
print(f"Pitch (about X)       : {pitch:.3f}°   ±{np.degrees(sigma_b):.3f}°")
print(f"Overall inclination   : {incl:.3f}°")

# 6. Visualisation
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter the neighbourhood points
ax.scatter(patch[:, 0], patch[:, 1], patch[:, 2], s=6)

# Draw the fitted tangent plane (mesh)
grid_x = np.linspace(P[0]-r, P[0]+r, 20)
grid_y = np.linspace(P[1]-r, P[1]+r, 20)
GX, GY = np.meshgrid(grid_x, grid_y)
# plane z = a (x-Px) + b (y-Py) + Pz   (remove translation)
GZ = a*(GX - P[0]) + b*(GY - P[1]) + P[2]

ax.plot_surface(GX, GY, GZ, alpha=0.4)

# Mark point P
ax.scatter(P[0], P[1], P[2], s=60, marker='*')

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Neighbourhood points & tangent plane')
plt.tight_layout()
plt.show()
