import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import differential_evolution

# defining vector
class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z
    def __add__(self, v):
        return Vector(self.x+v.x, self.y+v.y, self.z+v.z)
    def __sub__(self, v):
        return Vector(self.x-v.x, self.y-v.y, self.z-v.z)
    def __mul__(self, scalar):
        return Vector(self.x*scalar, self.y*scalar, self.z*scalar)
    def __truediv__(self, scalar):
        return Vector(self.x/scalar, self.y/scalar, self.z/scalar)
    def get_norm(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    def normalize(self):
        norm = self.get_norm()
        return self / norm if norm != 0 else Vector(0, 0, 0)
    def dot(self, v):
        return self.x*v.x + self.y*v.y + self.z*v.z
    def copy(self):
        return Vector(self.x, self.y, self.z)

# define orbit
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
        self.n = math.sqrt(G*M/a**3)

    def solve_kepler(self, M):
        E = M
        for j in range(10):
            f = E-self.e*math.sin(E)-M
            f_prime = 1-self.e*math.cos(E)
            E -= f/f_prime
        return E

    # perifocal reference frame -> cartesian(inertial) frame
    def get_state(self, t):
        M = (self.M0 + self.n*t) % (2*math.pi)
        E = self.solve_kepler(M)
        x_orb = self.a*(math.cos(E)-self.e)
        y_orb = self.a*math.sqrt(1-self.e**2)*math.sin(E)
        mu = self.G*self.M
        r = math.sqrt(x_orb**2 + y_orb**2)
        v_x_orb = -math.sqrt(mu/self.a)/r*math.sin(E)
        v_y_orb = math.sqrt(mu/self.a)/r*math.sqrt(1-self.e**2)*math.cos(E)

        cos_Omega = math.cos(self.Omega)
        sin_Omega = math.sin(self.Omega)
        cos_i = math.cos(self.i)
        sin_i = math.sin(self.i)
        cos_omega = math.cos(self.omega)
        sin_omega = math.sin(self.omega)

        x = (cos_Omega*cos_omega-sin_Omega*sin_omega*cos_i)*x_orb + (-cos_Omega*sin_omega-sin_Omega*cos_omega*cos_i)*y_orb
        y = (sin_Omega*cos_omega+cos_Omega*sin_omega*cos_i)*x_orb + (-sin_Omega*sin_omega+cos_Omega*cos_omega*cos_i)*y_orb
        z = (sin_omega*sin_i)*x_orb + (cos_omega*sin_i)*y_orb

        vx = (cos_Omega*cos_omega-sin_Omega*sin_omega*cos_i)*v_x_orb + (-cos_Omega*sin_omega-sin_Omega*cos_omega*cos_i)*v_y_orb
        vy = (sin_Omega*cos_omega+cos_Omega*sin_omega*cos_i)*v_x_orb + (-sin_Omega*sin_omega+cos_Omega*cos_omega*cos_i)*v_y_orb
        vz = (sin_omega*sin_i)*v_x_orb + (cos_omega*sin_i)*v_y_orb

        return Vector(x, y, z), Vector(vx, vy, vz)

class MirrorSimulation:
    def __init__(self, mirror_orbit_params, visualize=True):
        self.visualize = visualize
        self.dt = 0.01
        self.t = 0
        self.G = 5000
        self.sun_mass = 1e10
        self.mars_mass = 1e8

        self.mars_orbit = KeplerOrbit(
            G = self.G,
            M = self.sun_mass,
            a = 500,
            e = 0.0934,
            i = math.radians(1.85),
            Omega = math.radians(49.57854),
            omega = math.radians(286.502),
            M0 = 0
        )

        self.mirror_orbit = KeplerOrbit(
            G=self.G,
            M=self.mars_mass,
            a=mirror_orbit_params['a'],
            e=mirror_orbit_params['e'],
            i=mirror_orbit_params['i'],
            Omega=mirror_orbit_params['Omega'],
            omega=mirror_orbit_params['omega'],
            M0=mirror_orbit_params['M0']
        )

        self.mirror_radius = 3.4062
        self.alpha_dot = 0.4972
        self.beta_dot = 0.0296
        self.gamma_dot = 0.4990

        self.mars_pos = Vector()
        self.mirror_pos = Vector()
        self.trace = []
        self.mars_trace = []
        self.energy_trace = []
        self.cumulative_energy = 0
        self.last_print_time = 0

        if self.visualize:
            self.fig = plt.figure(figsize=(12, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
            self.ax_orbit = self.fig.add_subplot(gs[0], projection='3d')
            self.ax_energy = self.fig.add_subplot(gs[1])

    def get_mars_position(self, t):
        pos, _ = self.mars_orbit.get_state(t)
        return pos

    def compute_reflected_energy(self):
        sun_pos = Vector(0, 0, 0)
        sun_direction = (sun_pos-self.mirror_pos).normalize()
        distance = (sun_pos-self.mirror_pos).get_norm()
        normal = (self.mars_pos-self.mirror_pos).normalize()
        incidence_angle = max(0, normal.dot(sun_direction))

        mirror_area = 10
        solar_constant_at_1AU = 1361
        reference_distance = 1e6
        attenuation = (reference_distance / distance) ** 2
        return solar_constant_at_1AU * mirror_area * incidence_angle * attenuation

    def update(self):
        self.mars_pos = self.get_mars_position(self.t)
        pos_rel, _ = self.mirror_orbit.get_state(self.t)
        self.mirror_pos = self.mars_pos+pos_rel

        self.trace.append(self.mirror_pos.copy())
        if len(self.trace) > 500:
            self.trace.pop(0)

        self.mars_trace.append(self.mars_pos.copy())
        if len(self.mars_trace) > 500:
            self.mars_trace.pop(0)

        energy = self.compute_reflected_energy()
        self.cumulative_energy += energy * self.dt
        self.energy_trace.append(energy)
        if len(self.energy_trace) > 500:
            self.energy_trace.pop(0)

        self.t += self.dt

    def draw(self):
        if not self.visualize:
            return
        self.ax_orbit.clear()
        self.ax_energy.clear()
        self.ax_orbit.scatter(0, 0, 0, color='orange', s=200, label='Sun')
        self.ax_orbit.scatter(self.mars_pos.x, self.mars_pos.y, self.mars_pos.z, color='red', s=100, label='Mars')
        self.ax_orbit.scatter(self.mirror_pos.x, self.mirror_pos.y, self.mirror_pos.z, color='blue', s=30, label='Mirror')

        xs = [p.x for p in self.trace]
        ys = [p.y for p in self.trace]
        zs = [p.z for p in self.trace]
        self.ax_orbit.plot(xs, ys, zs, color='blue', linewidth=0.5, alpha=0.7)

        xs_mars = [p.x for p in self.mars_trace]
        ys_mars = [p.y for p in self.mars_trace]
        zs_mars = [p.z for p in self.mars_trace]
        self.ax_orbit.plot(xs_mars, ys_mars, zs_mars, color='red', linewidth=0.5, alpha=0.7)

        margin = 150
        orbit_radius = self.mars_orbit.a*(1+self.mars_orbit.e)
        self.ax_orbit.set_xlim(-orbit_radius-margin, orbit_radius+margin)
        self.ax_orbit.set_ylim(-orbit_radius-margin, orbit_radius+margin)
        self.ax_orbit.set_zlim(-margin/2, margin/2)
        self.ax_orbit.set_xlabel('X')
        self.ax_orbit.set_ylabel('Y')
        self.ax_orbit.set_zlabel('Z')
        self.ax_orbit.set_title('Orbits')
        self.ax_orbit.legend(loc='upper right')

        self.ax_energy.plot(self.energy_trace, color='black')
        self.ax_energy.set_title('태양광 도달량')
        self.ax_energy.set_xlabel('t')
        self.ax_energy.set_ylabel('E')
        plt.pause(0.001)

    def run(self, steps=2000):
        for j in range(steps):
            self.update()
            self.draw()

def objective(params):
    a, e, i, Omega, omega, M0 = params
    mirror_orbit_params = {
        'a': a,
        'e': e,
        'i': i,
        'Omega': Omega,
        'omega': omega,
        'M0': M0
    }
    sim = MirrorSimulation(mirror_orbit_params, visualize=False)
    for j in range(1000):
        sim.update()
    return -sim.cumulative_energy

bounds = [
    (50, 150),
    (0.0, 0.9),
    (math.radians(0), math.radians(30)),
    (0, 2*math.pi),
    (0, 2*math.pi),
    (0, 2*math.pi)
]

result = differential_evolution(objective, bounds, maxiter=50, popsize=15)
best_params = result.x
best_param_dict = {
    'a': best_params[0],
    'e': best_params[1],
    'i': best_params[2],
    'Omega': best_params[3],
    'omega': best_params[4],
    'M0': best_params[5],
}

print(best_param_dict)
final_sim = MirrorSimulation(best_param_dict, visualize=True)
final_sim.run(steps=2000)