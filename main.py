import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import differential_evolution

# --------- Vector Class ---------
class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z
    def __add__(self, v): return Vector(self.x + v.x, self.y + v.y, self.z + v.z)
    def __sub__(self, v): return Vector(self.x - v.x, self.y - v.y, self.z - v.z)
    def __mul__(self, scalar): return Vector(self.x * scalar, self.y * scalar, self.z * scalar)
    def __truediv__(self, scalar): 
        if scalar == 0:
            raise ValueError("Division by zero in Vector.__truediv__")
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)
    def get_norm(self): return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    def normalize(self):
        norm = self.get_norm()
        return self / norm if norm != 0 else Vector(0, 0, 0)
    def dot(self, v): return self.x * v.x + self.y * v.y + self.z * v.z
    def copy(self): return Vector(self.x, self.y, self.z)
    def as_tuple(self): return (self.x, self.y, self.z)

# --------- Kepler Orbit ---------
class KeplerOrbit:
    def __init__(self, G, M, a, e, i, Omega, omega, M0=0):
        self.G = G
        self.M = M
        self.a = a
        self.e = e
        self.i = i
        self.Omega = Omega
        self.omega = omega
        self.M0 = M0
        if a <= 0:
            raise ValueError("Semi-major axis 'a' must be positive for KeplerOrbit.")
        self.n = math.sqrt(G * M / a**3)

    def solve_kepler(self, M_anom):
        E = M_anom
        for _ in range(20):
            f = E - self.e * math.sin(E) - M_anom
            f_prime = 1 - self.e * math.cos(E)
            if f_prime == 0:
                break
            E = E - f / f_prime
        return E

    def get_state(self, t):
        M_anom = (self.M0 + self.n * t) % (2 * math.pi)
        E = self.solve_kepler(M_anom)
        x_orb = self.a * (math.cos(E) - self.e)
        y_orb = self.a * math.sqrt(1 - self.e**2) * math.sin(E)
        mu = self.G * self.M
        r = math.sqrt(x_orb**2 + y_orb**2)
        if r == 0:
            v_x_orb = 0
            v_y_orb = 0
        else:
            v_x_orb = -math.sqrt(mu / self.a) / r * math.sin(E)
            v_y_orb = math.sqrt(mu / self.a) / r * math.sqrt(1 - self.e**2) * math.cos(E)
        cos_Omega, sin_Omega = math.cos(self.Omega), math.sin(self.Omega)
        cos_i, sin_i = math.cos(self.i), math.sin(self.i)
        cos_omega, sin_omega = math.cos(self.omega), math.sin(self.omega)
        x = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * x_orb + \
            (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i) * y_orb
        y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * x_orb + \
            (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i) * y_orb
        z = (sin_omega * sin_i) * x_orb + (cos_omega * sin_i) * y_orb
        vx = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * v_x_orb + \
             (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i) * v_y_orb
        vy = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * v_x_orb + \
             (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i) * v_y_orb
        vz = (sin_omega * sin_i) * v_x_orb + (cos_omega * sin_i) * v_y_orb
        return Vector(x, y, z), Vector(vx, vy, vz)

# --------- Euler Rotation for Mirror Orbit ---------
def euler_rotate(pos, angles):
    x, y, z = pos.x, pos.y, pos.z
    alpha, beta, gamma = angles
    x0 = x * math.cos(alpha) - y * math.sin(alpha)
    y0 = x * math.sin(alpha) + y * math.cos(alpha)
    z0 = z
    x1 = x0
    y1 = y0 * math.cos(beta) - z0 * math.sin(beta)
    z1 = y0 * math.sin(beta) + z0 * math.cos(beta)
    x2 = x1 * math.cos(gamma) - y1 * math.sin(gamma)
    y2 = x1 * math.sin(gamma) + y1 * math.cos(gamma)
    z2 = z1
    return Vector(x2, y2, z2)

# --------- Mirror Simulation ---------
class MirrorSimulation:
    def __init__(self):
        self.dt = 0.1
        self.t = 0
        self.G = 1
        self.sun_mass = 1e6
        self.mars_mass = 1e3
        self.mars_orbit = KeplerOrbit(
            G=self.G, M=self.sun_mass,
            a=50, e=0.1, i=math.radians(1),
            Omega=0, omega=0, M0=0
        )
        self.mirror_orbit_radius = 5
        self.alpha_dot = 0.05
        self.beta_dot = 0.02
        self.gamma_dot = 0.03
        self.trace = []
        self.energy_trace = []
        self.cumulative_energy = 0
        self.fig = None
        self.ax_orbit = None
        self.ax_energy = None

    def compute_reflected_energy(self, mirror_abs_pos, mars_pos):
        sun_pos = Vector(0, 0, 0)
        sun_dir = (sun_pos - mirror_abs_pos).normalize()
        distance_to_sun = (sun_pos - mirror_abs_pos).get_norm()
        normal = (mars_pos - mirror_abs_pos).normalize()
        incidence_angle_cos = max(0, normal.dot(sun_dir))
        mirror_area = 10
        solar_constant = 1361
        reference_distance_au = 1
        effective_solar_constant = solar_constant * (reference_distance_au / distance_to_sun) ** 2 if distance_to_sun != 0 else 0
        return effective_solar_constant * mirror_area * incidence_angle_cos

    def update(self, record_trace=False):
        self.t += self.dt
        mars_pos, _ = self.mars_orbit.get_state(self.t)
        base_pos = Vector(self.mirror_orbit_radius, 0, 0)
        alpha = self.alpha_dot * self.t
        beta = self.beta_dot * self.t
        gamma = self.gamma_dot * self.t
        mirror_rel_pos = euler_rotate(base_pos, (alpha, beta, gamma))
        mirror_abs_pos = mars_pos + mirror_rel_pos
        energy = self.compute_reflected_energy(mirror_abs_pos, mars_pos)
        self.cumulative_energy += energy * self.dt
        if record_trace:
            self.trace.append(mirror_abs_pos.copy())
            self.energy_trace.append(energy)
            if len(self.trace) > 2000:
                self.trace.pop(0)
            if len(self.energy_trace) > 2000:
                self.energy_trace.pop(0)
        self.mars_pos = mars_pos
        self.mirror_abs_pos = mirror_abs_pos

    def run_simulation(self, total_steps, record_trace=False):
        self.t = 0
        self.cumulative_energy = 0
        if record_trace:
            self.trace = []
            self.energy_trace = []
        for _ in range(total_steps):
            self.update(record_trace)
        return self.cumulative_energy

    def draw(self):
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
            self.ax_orbit = self.fig.add_subplot(gs[0], projection='3d')
            self.ax_energy = self.fig.add_subplot(gs[1])
            plt.ion()
        self.ax_orbit.clear()
        self.ax_energy.clear()
        self.ax_orbit.scatter(0, 0, 0, color='yellow', s=200, label='Sun')
        self.ax_orbit.scatter(self.mars_pos.x, self.mars_pos.y, self.mars_pos.z, color='red', s=100, label='Mars')
        self.ax_orbit.scatter(self.mirror_abs_pos.x, self.mirror_abs_pos.y, self.mirror_abs_pos.z, color='blue', s=50, label='Mirror')
        if self.trace:
            xs = [p.x for p in self.trace]
            ys = [p.y for p in self.trace]
            zs = [p.z for p in self.trace]
            self.ax_orbit.plot(xs, ys, zs, color='blue', linewidth=0.8, label='Mirror Trace')
        lim = max(self.mars_orbit.a * 1.5, self.mirror_orbit_radius * 2 + self.mars_orbit.a * 0.5)
        self.ax_orbit.set_xlim(-lim, lim)
        self.ax_orbit.set_ylim(-lim, lim)
        self.ax_orbit.set_zlim(-lim, lim)
        self.ax_orbit.set_title('Mirror Orbiting Mars & Mars around Sun')
        self.ax_orbit.set_xlabel('X')
        self.ax_orbit.set_ylabel('Y')
        self.ax_orbit.set_zlabel('Z')
        self.ax_orbit.legend()
        if self.energy_trace:
            self.ax_energy.plot(self.energy_trace, color='orange')
            self.ax_energy.set_title('Reflected Solar Energy (W)')
            self.ax_energy.set_xlabel('Time step')
            self.ax_energy.set_ylabel('Energy (W)')
        plt.tight_layout()
        plt.pause(0.01)

    def run_visual_simulation(self, steps=2000):
        self.t = 0
        self.cumulative_energy = 0
        self.trace = []
        self.energy_trace = []
        for step in range(steps):
            self.update(record_trace=True)
            self.draw()
            if step % 100 == 0:
                print(f"Step {step}: Energy = {self.energy_trace[-1]:.2f} W, Cumulative = {self.cumulative_energy:.2f} J")
        plt.ioff()
        plt.show()

# --------- Objective Function for Differential Evolution ---------
def objective_function(params, total_simulation_steps):
    a, e, i, Omega, omega, M0, mirror_orbit_radius, alpha_dot, beta_dot, gamma_dot = params
    if not (0 <= e < 1) or a <= 0:
        return float('inf')
    sim_instance = MirrorSimulation()
    try:
        sim_instance.mars_orbit = KeplerOrbit(
            G=sim_instance.G, M=sim_instance.sun_mass,
            a=a, e=e, i=i, Omega=Omega, omega=omega, M0=M0
        )
    except ValueError:
        return float('inf')
    sim_instance.mirror_orbit_radius = mirror_orbit_radius
    sim_instance.alpha_dot = alpha_dot
    sim_instance.beta_dot = beta_dot
    sim_instance.gamma_dot = gamma_dot
    return sim_instance.run_simulation(total_simulation_steps, record_trace=False)

# --------- Execution ---------
if __name__ == "__main__":
    print("Starting Differential Evolution optimization...")
    OPTIMIZATION_SIM_STEPS = 500
    bounds = [
        (10, 100), (0.0, 0.9), (0, math.pi), (0, 2 * math.pi), (0, 2 * math.pi),
        (0, 2 * math.pi), (1, 15), (0.001, 0.5), (0.001, 0.5), (0.001, 0.5)
    ]
    result = differential_evolution(
        func=objective_function, bounds=bounds, args=(OPTIMIZATION_SIM_STEPS,),
        strategy='best1bin', maxiter=20, popsize=20, tol=0.01, disp=True,
        polish=True, workers=-1, mutation=(0.5, 1.0), recombination=0.7
    )
    print("\n-------------------------------------------------")
    print("Differential Evolution optimization completed!")
    print("-------------------------------------------------")
    print("Optimal parameters (a, e, i, Omega, omega, M0, mirror_radius, alpha_dot, beta_dot, gamma_dot):")
    print(f"  a: {result.x[0]:.4f}")
    print(f"  e: {result.x[1]:.4f}")
    print(f"  i: {math.degrees(result.x[2]):.4f} deg ({result.x[2]:.4f} rad)")
    print(f"  Omega: {math.degrees(result.x[3]):.4f} deg ({result.x[3]:.4f} rad)")
    print(f"  omega: {math.degrees(result.x[4]):.4f} deg ({result.x[4]:.4f} rad)")
    print(f"  M0: {math.degrees(result.x[5]):.4f} deg ({result.x[5]:.4f} rad)")
    print(f"  mirror_radius: {result.x[6]:.4f}")
    print(f"  alpha_dot: {result.x[7]:.4f}")
    print(f"  beta_dot: {result.x[8]:.4f}")
    print(f"  gamma_dot: {result.x[9]:.4f}")
    optimal_cumulative_energy = -result.fun
    print(f"\nMax cumulative energy: {optimal_cumulative_energy:.2f} J")
    print("-------------------------------------------------")
    print("\nStarting simulation visualization with optimized parameters...")
    final_sim = MirrorSimulation()
    final_sim.mars_orbit = KeplerOrbit(
        G=final_sim.G, M=final_sim.sun_mass,
        a=result.x[0], e=result.x[1], i=result.x[2],
        Omega=result.x[3], omega=result.x[4], M0=result.x[5]
    )
    final_sim.mirror_orbit_radius = result.x[6]
    final_sim.alpha_dot = result.x[7]
    final_sim.beta_dot = result.x[8]
    final_sim.gamma_dot = result.x[9]
    VISUAL_SIM_STEPS = 2000
    final_sim.run_visual_simulation(VISUAL_SIM_STEPS)
    print("Simulation visualization completed.")