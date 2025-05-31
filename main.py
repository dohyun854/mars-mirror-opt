    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # ─────────────────────────────
    # 기본 상수 및 화성 궤도 계산 (실제 데이터 기반)
    # ─────────────────────────────
    AU = 149597870.0                # 1 AU (km)
    a_mars = 1.523 * AU             # 화성의 반장축 (km)
    e_mars = 0.0934                 # 화성의 이심률

    frames = 360
    # 0 ~ 2π 범위의 true anomaly (θ)
    theta = np.linspace(0, 2 * np.pi, frames)
    # 케플러 궤도 공식: r(θ) = a*(1-e²)/(1+e*cosθ)
    mars_r = a_mars * (1 - e_mars**2) / (1 + e_mars * np.cos(theta))
    mars_x = mars_r * np.cos(theta)
    mars_y = mars_r * np.sin(theta)

    # 화성의 공전 주기 (초 단위): 약 686.98일
    mars_orbit_period = 686.98 * 24 * 60 * 60
    # 시뮬레이션 전체 시간 배열 (화성 1년 동안)
    t = np.linspace(0, mars_orbit_period, frames)


    # ─────────────────────────────
    # 우주 거울이 화성 주위를 도는 궤도 설정
    # ─────────────────────────────
    # (여기서는 우주 거울이 화성 중심에서 일정 거리(예: 3×10⁷ km)를 두고,
    #  100일 주기로 도는 것으로 설정합니다.)
    mirror_orbit_distance = 3e7  # 화성 중심으로부터의 거리 (km)
    mirror_orbit_period = 100 * 24 * 60 * 60  # 100일 (초 단위)

    # 우주 거울의 궤도 각도: 시뮬레이션 시간 t에 따라 변화
    mirror_angle = 2 * np.pi * t / mirror_orbit_period


    # ─────────────────────────────
    # 시각적 크기 (실제 크기와 관계없이 우리 눈에 보이는 크기로 설정)
    # ─────────────────────────────
    sun_visual_radius = 5e7          # 태양의 시각적 반지름 (km)
    mars_visual_radius = 2e7         # 화성의 시각적 반지름 (km)
    mirror_visual_radius = mars_visual_radius / 3  # 우주 거울은 화성 크기의 1/3


    # ─────────────────────────────
    # 플롯 및 초기 패치 설정
    # ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-a_mars * 1.2, a_mars * 1.2)
    ax.set_ylim(-a_mars * 1.2, a_mars * 1.2)
    ax.set_aspect('equal', 'box')

    # 태양: 고정되어 있음
    sun_patch = plt.Circle((0, 0), sun_visual_radius, color='yellow', label='태양')
    ax.add_patch(sun_patch)

    #화성에서의 평균 태양 복사량
    solar_irradiance_mars = 590  # 화성에서의 평균 태양 복사량

    # 화성 궤적: 태양 기준 전체 궤도를 파선으로 표시
    ax.plot(mars_x, mars_y, 'r--', alpha=0.5, label="화성 궤도")

    # 화성: 최초 위치를 frame 0의 값으로 설정
    mars_patch = plt.Circle((mars_x[0], mars_y[0]), mars_visual_radius, color='red', label='화성')
    ax.add_patch(mars_patch)

    # 우주 거울: 화성을 중심으로 하는 궤도상 위치
    mirror_initial_x = mars_x[0] + mirror_orbit_distance * np.cos(mirror_angle[0])
    mirror_initial_y = mars_y[0] + mirror_orbit_distance * np.sin(mirror_angle[0])
    mirror_patch = plt.Circle((mirror_initial_x, mirror_initial_y),
                            mirror_visual_radius, color='blue', label='우주 거울')
    ax.add_patch(mirror_patch)


    #우주 거울 반사율 & 우주 거울 면적 &반사된 총 에너지 
    mirror_reflectivity = 0.9  # 90% 반사
    mirror_area = 1e6  # 1 km²
    reflected_energy_mars = solar_irradiance_mars * mirror_reflectivity * mirror_area
    # ─────────────────────────────
    # 애니메이션 업데이트 함수
    # ─────────────────────────────
    def update(frame):
        # 화성 위치 업데이트 (태양 주위를 도는 궤도)
        mars_patch.center = (mars_x[frame], mars_y[frame])
        
        # 우주 거울은 화성 위치를 중심으로, 지정한 거리와 각도로 이동
        new_mirror_x = mars_x[frame] + mirror_orbit_distance * np.cos(mirror_angle[frame])
        new_mirror_y = mars_y[frame] + mirror_orbit_distance * np.sin(mirror_angle[frame])
        mirror_patch.center = (new_mirror_x, new_mirror_y)
        
        return mars_patch, mirror_patch

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

    print(f"energy: {reflected_energy_mars:.2f} W")

    plt.title("태양 기준 화성 및 우주 거울 공전 시뮬레이션\n(우주 거울은 화성 주위를 도는, 크기는 화성의 1/3)")
    plt.legend()
    plt.show()



