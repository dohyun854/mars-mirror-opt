import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --------- Vector Class ---------
class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z
    def __add__(self, v): return Vector(self.x + v.x, self.y + v.y, self.z + v.z)
    def __sub__(self, v): return Vector(self.x - v.x, self.y - v.y, self.z - v.z)
    def __mul__(self, scalar): return Vector(self.x * scalar, self.y * scalar, self.z * scalar)
    def __truediv__(self, scalar): return Vector(self.x / scalar, self.y / scalar, self.z / scalar)
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
        self.n = math.sqrt(G * M / a**3)

    def solve_kepler(self, M):
        E = M
        for _ in range(10):
            f = E - self.e * math.sin(E) - M
            f_prime = 1 - self.e * math.cos(E)
            E = E - f / f_prime
        return E

    def get_state(self, t):
        M = (self.M0 + self.n * t) % (2 * math.pi)
        E = self.solve_kepler(M)

        x_orb = self.a * (math.cos(E) - self.e)
        y_orb = self.a * math.sqrt(1 - self.e**2) * math.sin(E)

        mu = self.G * self.M
        r = math.sqrt(x_orb**2 + y_orb**2)
        v_x_orb = - math.sqrt(mu / self.a) / r * math.sin(E)
        v_y_orb = math.sqrt(mu / self.a) / r * math.sqrt(1 - self.e**2) * math.cos(E)

        cos_Omega = math.cos(self.Omega)
        sin_Omega = math.sin(self.Omega)
        cos_i = math.cos(self.i)
        sin_i = math.sin(self.i)
        cos_omega = math.cos(self.omega)
        sin_omega = math.sin(self.omega)

        x = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * x_orb + (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i) * y_orb
        y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * x_orb + (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i) * y_orb
        z = (sin_omega * sin_i) * x_orb + (cos_omega * sin_i) * y_orb

        vx = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * v_x_orb + (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i) * v_y_orb
        vy = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * v_x_orb + (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i) * v_y_orb
        vz = (sin_omega * sin_i) * v_x_orb + (cos_omega * sin_i) * v_y_orb

        pos = Vector(x, y, z)
        vel = Vector(vx, vy, vz)
        return pos, vel

# --------- Euler Rotation for Mirror Orbit ---------
def euler_rotate(pos, angles):
    """pos: Vector, angles: (alpha, beta, gamma) in radians"""
    x, y, z = pos.x, pos.y, pos.z
    alpha, beta, gamma = angles

    # Rotation matrix from ZXZ Euler angles:
    # R = Rz(gamma) * Rx(beta) * Rz(alpha)

    # Rz(gamma)
    x1 = x * math.cos(gamma) - y * math.sin(gamma)
    y1 = x * math.sin(gamma) + y * math.cos(gamma)
    z1 = z

    # Rx(beta)
    x2 = x1
    y2 = y1 * math.cos(beta) - z1 * math.sin(beta)
    z2 = y1 * math.sin(beta) + z1 * math.cos(beta)

    # Rz(alpha)
    x3 = x2 * math.cos(alpha) - y2 * math.sin(alpha)
    y3 = x2 * math.sin(alpha) + y2 * math.cos(alpha)
    z3 = z2

    return Vector(x3, y3, z3)

# --------- Mirror Simulation ---------
class MirrorSimulation:
    def __init__(self):
        # Constants & Params
        self.dt = 0.1
        self.t = 0

        self.G = 1  # 중력상수 간소화
        self.sun_mass = 1e6
        self.mars_mass = 1e3

        # 태양 중심 케플러 화성 궤도 (단위 임의)
        self.mars_orbit = KeplerOrbit(
            G=self.G, M=self.sun_mass,
            a=50, e=0.1, i=math.radians(1),
            Omega=0, omega=0, M0=0
        )

        # 화성 중심 오일러 각을 이용한 거울 궤도 반경 (반지름 고정)
        self.mirror_orbit_radius = 5

        # 오일러 각 속도(rad/s)
        self.alpha_dot = 0.05  # z축 회전 속도
        self.beta_dot = 0.02   # x축 회전 속도
        self.gamma_dot = 0.03  # z축 회전 속도

        self.trace = []
        self.energy_trace = []
        self.cumulative_energy = 0
        self.last_print_time = 0

        # 플롯 셋업
        self.fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        self.ax_orbit = self.fig.add_subplot(gs[0], projection='3d')
        self.ax_energy = self.fig.add_subplot(gs[1])

    def compute_reflected_energy(self, mirror_abs_pos, mars_pos):
        sun_pos = Vector(0, 0, 0)
        sun_dir = (sun_pos - mirror_abs_pos).normalize()
        distance = (sun_pos - mirror_abs_pos).get_norm()
        normal = (mars_pos - mirror_abs_pos).normalize()
        incidence_angle = max(0, normal.dot(sun_dir))

        mirror_area = 10  # m^2
        solar_constant = 1361  # W/m^2
        reference_distance = 1  # 1 unit distance for scaling
        attenuation = (reference_distance / distance) ** 2
        reflected_energy = solar_constant * mirror_area * incidence_angle * attenuation
        return reflected_energy

    def update(self):
        self.t += self.dt

        # 1) 화성 태양 중심 위치
        mars_pos, _ = self.mars_orbit.get_state(self.t)

        # 2) 거울 위치 계산 - 화성 중심 기준 원 궤도(반지름 고정)
        # 초기 위치 거울을 x축에 두고 시작
        base_pos = Vector(self.mirror_orbit_radius, 0, 0)

        # 현재 오일러 각 계산
        alpha = self.alpha_dot * self.t
        beta = self.beta_dot * self.t
        gamma = self.gamma_dot * self.t

        # 오일러 회전 적용
        mirror_rel_pos = euler_rotate(base_pos, (alpha, beta, gamma))

        # 거울 절대 위치 = 화성 위치 + 상대 위치
        mirror_abs_pos = mars_pos + mirror_rel_pos

        # 위치 기록
        self.trace.append(mirror_abs_pos.copy())
        if len(self.trace) > 500:
            self.trace.pop(0)

        # 반사 에너지 계산
        energy = self.compute_reflected_energy(mirror_abs_pos, mars_pos)
        self.cumulative_energy += energy * self.dt
        self.energy_trace.append(energy)
        if len(self.energy_trace) > 500:
            self.energy_trace.pop(0)

        # 상태 출력 (1초 간격)
        if int(self.t) > self.last_print_time:
            print(f"Time: {int(self.t)} s, Cumulative Reflected Energy: {self.cumulative_energy:.2f} J")
            self.last_print_time = int(self.t)

        # 현재 위치 상태 저장
        self.mars_pos = mars_pos
        self.mirror_abs_pos = mirror_abs_pos

    def draw(self):
        self.ax_orbit.clear()
        self.ax_energy.clear()

        # 태양
        self.ax_orbit.scatter(0, 0, 0, color='yellow', s=200, label='Sun')

        # 화성
        self.ax_orbit.scatter(self.mars_pos.x, self.mars_pos.y, self.mars_pos.z, color='red', s=100, label='Mars')

        # 거울
        self.ax_orbit.scatter(self.mirror_abs_pos.x, self.mirror_abs_pos.y, self.mirror_abs_pos.z, color='blue', s=50, label='Mirror')

        # 궤적
        xs = [p.x for p in self.trace]
        ys = [p.y for p in self.trace]
        zs = [p.z for p in self.trace]
        self.ax_orbit.plot(xs, ys, zs, color='blue', linewidth=0.8)

        # 축 한계
        lim = 70
        self.ax_orbit.set_xlim(-lim, lim)
        self.ax_orbit.set_ylim(-lim, lim)
        self.ax_orbit.set_zlim(-lim, lim)
        self.ax_orbit.set_title('Mirror Orbiting Mars (Euler Rotation) & Mars around Sun')
        self.ax_orbit.legend()

        # 반사 에너지 그래프
        self.ax_energy.plot(self.energy_trace, color='orange')
        self.ax_energy.set_title('Reflected Solar Energy (W)')
        self.ax_energy.set_xlabel('Time step')
        self.ax_energy.set_ylabel('Energy')

        plt.pause(0.001)

    def run(self, steps=2000):
        plt.ion()
        for _ in range(steps):
            self.update()
            self.draw()
        plt.ioff()
        plt.show()

# --------- 실행 ---------
if __name__ == "__main__":
    sim = MirrorSimulation()
    sim.run()
