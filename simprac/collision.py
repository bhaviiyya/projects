import taichi as ti
try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)
    print("Running on CPU, expect lower FPS. Lower the body count if it crawls.")

import os
import math
import numpy as np

# Ensure local cache directory to prevent macOS sandbox permission errors
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = os.path.abspath(".cache")

# -----------------------------------------------------------------------------
# ASTROPHYSICAL CONSTANTS & SIMULATION CONSTANTS
# -----------------------------------------------------------------------------
# Physical Unit System:
#   1 Length Unit (LU)  = 5.0 kiloparsecs (~16,300 light-years)
#   1 Mass Unit (MU)    = 1.0 x 10^11 Solar Masses (typical spiral galaxy disk)
#   1 Time Unit (TU)    = ~50 Million Years
#   Velocity scale      = 1 LU / 1 TU ~ 97.8 km/s
# In these scaled astronomical units:
#   Gravitational constant G = 4 * pi^2 (~39.4784), so circular speed at r=1 is 2*pi
G = 4.0 * (math.pi ** 2)

# Gravitational Softening (Plummer potential: Phi(r) = -G*M / sqrt(r^2 + eps^2))
# Physical interpretation: models the finite spatial extent of stellar bulges
# and dark matter halos, eliminating artificial 1/r singularities.
EPS_CORE_CORE = 0.28             # Core-core gravitational softening (~1.4 kpc)
EPS_CORE_STAR = 0.18             # Core-star gravitational softening (~0.9 kpc)
EPS_DIV_ZERO = 1e-3              # Numerical divide-by-zero protection in all divisions

# Body Counts (Tuned for dense stellar filament detail at steady 50+ FPS)
NUM_STARS_A = 37000              # Primary Galaxy (Andromeda-like cyan/azure disk)
NUM_STARS_B = 28000              # Companion Perturber (Solar amber/coral disk)
NUM_STARS = NUM_STARS_A + NUM_STARS_B
NUM_CORES = 2

# Core Trajectory Trail Buffer
TRAIL_LEN = 240
NUM_TRAIL_VERTS = NUM_CORES * (TRAIL_LEN - 1) * 2

# Numerical Integration
SUBSTEPS = 3                     # Symplectic Velocity-Verlet substeps per frame
DT_DEFAULT = 0.0022              # Base time step per substep

# Distant Ambient Cosmic Dust (provides deep 3D perspective)
NUM_DUST = 2500

# -----------------------------------------------------------------------------
# TAICHI FIELDS (All state lives on GPU)
# -----------------------------------------------------------------------------
# Star fields
star_pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_STARS)
star_vel = ti.Vector.field(3, dtype=ti.f32, shape=NUM_STARS)
star_acc = ti.Vector.field(3, dtype=ti.f32, shape=NUM_STARS)
star_color = ti.Vector.field(3, dtype=ti.f32, shape=NUM_STARS)
star_galaxy_id = ti.field(dtype=ti.i32, shape=NUM_STARS)
star_init_rad = ti.field(dtype=ti.f32, shape=NUM_STARS)

# Core fields
core_pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_CORES)
core_vel = ti.Vector.field(3, dtype=ti.f32, shape=NUM_CORES)
core_acc = ti.Vector.field(3, dtype=ti.f32, shape=NUM_CORES)
core_mass = ti.field(dtype=ti.f32, shape=NUM_CORES)
core_color = ti.Vector.field(3, dtype=ti.f32, shape=NUM_CORES)

# Core Halo Bulge particles (luminous cluster around each nucleus)
NUM_HALO_PER_CORE = 180
NUM_HALO = NUM_CORES * NUM_HALO_PER_CORE
halo_pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_HALO)
halo_color = ti.Vector.field(3, dtype=ti.f32, shape=NUM_HALO)
halo_offset = ti.Vector.field(3, dtype=ti.f32, shape=NUM_HALO)

# Distant Cosmic Dust
dust_pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_DUST)
dust_color = ti.Vector.field(3, dtype=ti.f32, shape=NUM_DUST)

# Core Trajectory Trail fields
trail_verts = ti.Vector.field(3, dtype=ti.f32, shape=NUM_TRAIL_VERTS)
trail_colors = ti.Vector.field(3, dtype=ti.f32, shape=NUM_TRAIL_VERTS)
core_history = ti.Vector.field(3, dtype=ti.f32, shape=(NUM_CORES, TRAIL_LEN))

# Reference Grid fields (celestial plane)
GRID_RINGS = 6
GRID_SEGS_PER_RING = 64
NUM_GRID_VERTS = GRID_RINGS * GRID_SEGS_PER_RING * 2 + 8  # concentric circles + 4 crosshair axes
grid_verts = ti.Vector.field(3, dtype=ti.f32, shape=NUM_GRID_VERTS)
grid_colors = ti.Vector.field(3, dtype=ti.f32, shape=NUM_GRID_VERTS)

# Global Simulation Control Parameters
color_mode = ti.field(dtype=ti.i32, shape=())       # 0: Dual Galaxy, 1: Kinetic Temp, 2: Tidal Strain
trail_head = ti.field(dtype=ti.i32, shape=())       # Ring buffer head pointer
trail_count = ti.field(dtype=ti.i32, shape=())      # Total samples recorded
dt_scale = ti.field(dtype=ti.f32, shape=())         # Time dilation multiplier
dyn_friction = ti.field(dtype=ti.f32, shape=())     # Halo dynamical friction strength
barycenter_out = ti.Vector.field(3, dtype=ti.f32, shape=())
core_metrics = ti.Vector.field(2, dtype=ti.f32, shape=())  # [core_dist, v_rel]

# -----------------------------------------------------------------------------
# SYMPLECTIC VELOCITY-VERLET KERNELS
# -----------------------------------------------------------------------------
@ti.kernel
def kick_velocity(dt: ti.f32):
    """Kick step: v(t + dt) = v(t) + a(t) * dt.
    In Velocity Verlet, two half-kicks surround the drift step."""
    for k in range(NUM_CORES):
        core_vel[k] += core_acc[k] * dt

    for i in range(NUM_STARS):
        star_vel[i] += star_acc[i] * dt

@ti.kernel
def drift_position(dt: ti.f32):
    """Drift step: x(t + dt) = x(t) + v(t + dt/2) * dt."""
    for k in range(NUM_CORES):
        core_pos[k] += core_vel[k] * dt

    for i in range(NUM_STARS):
        star_pos[i] += star_vel[i] * dt

@ti.kernel
def compute_accelerations():
    """Compute gravitational accelerations for cores and stars.
    Uses Plummer softening to realistically model extended stellar bulges."""
    # 1. Mutual Core-Core Interaction
    r01 = core_pos[1] - core_pos[0]
    dist_sq_c = r01.dot(r01) + (EPS_CORE_CORE ** 2) + EPS_DIV_ZERO
    inv_dist_c = 1.0 / ti.sqrt(dist_sq_c)
    inv_dist3_c = inv_dist_c * inv_dist_c * inv_dist_c

    # Newtonian gravity between cores
    f_grav = G * inv_dist3_c
    a0 = f_grav * core_mass[1] * r01
    a1 = -f_grav * core_mass[0] * r01

    # Chandrasekhar Dynamical Friction between overlapping dark matter halos:
    v_rel = core_vel[1] - core_vel[0]
    v_norm = v_rel.norm()
    dist_c = ti.sqrt(r01.dot(r01) + EPS_DIV_ZERO)
    halo_overlap = ti.exp(-dist_c / 2.2)
    df_mag = dyn_friction[None] * halo_overlap / (v_norm * v_norm * v_norm + 0.15 + EPS_DIV_ZERO)
    df_acc0 = df_mag * core_mass[1] * v_rel
    df_acc1 = -df_mag * core_mass[0] * v_rel

    core_acc[0] = a0 + df_acc0
    core_acc[1] = a1 + df_acc1

    # 2. Acceleration on each test-particle star from BOTH cores
    for i in range(NUM_STARS):
        p = star_pos[i]
        acc = ti.Vector([0.0, 0.0, 0.0])

        # Acceleration from Core 0
        d0 = p - core_pos[0]
        dist_sq_0 = d0.dot(d0) + (EPS_CORE_STAR ** 2) + EPS_DIV_ZERO
        inv_d0 = 1.0 / ti.sqrt(dist_sq_0)
        acc -= (G * core_mass[0] * inv_d0 * inv_d0 * inv_d0) * d0

        # Acceleration from Core 1
        d1 = p - core_pos[1]
        dist_sq_1 = d1.dot(d1) + (EPS_CORE_STAR ** 2) + EPS_DIV_ZERO
        inv_d1 = 1.0 / ti.sqrt(dist_sq_1)
        acc -= (G * core_mass[1] * inv_d1 * inv_d1 * inv_d1) * d1

        star_acc[i] = acc

@ti.kernel
def compute_metrics():
    total_m = core_mass[0] + core_mass[1]
    barycenter_out[None] = (core_pos[0] * core_mass[0] + core_pos[1] * core_mass[1]) / (total_m + EPS_DIV_ZERO)
    dist01 = (core_pos[1] - core_pos[0]).norm()
    v_rel = (core_vel[1] - core_vel[0]).norm()
    core_metrics[None] = ti.Vector([dist01, v_rel])

@ti.kernel
def recycle_escaped_stars():
    """Continuous simulation guardrail: if a star is flung out into the void (>36 LU),
    recycle it gracefully back into its parent galaxy disk on a stable circular orbit."""
    bary = barycenter_out[None]
    for i in range(NUM_STARS):
        dist_bary = (star_pos[i] - bary).norm()
        if dist_bary > 36.0:
            gid = star_galaxy_id[i]
            c_pos = core_pos[gid]
            c_vel = core_vel[gid]

            # Generate random respawn radius in mid-disk
            u = ti.random(ti.f32)
            theta = ti.random(ti.f32) * 2.0 * math.pi
            r_spawn = 0.6 + u * 2.2
            z_spawn = (ti.random(ti.f32) - 0.5) * 0.08

            # In-plane offset
            dx = r_spawn * ti.cos(theta)
            dy = r_spawn * ti.sin(theta)
            star_pos[i] = c_pos + ti.Vector([dx, dy, z_spawn])

            # Circular orbital velocity
            dist_sq = r_spawn * r_spawn + (EPS_CORE_STAR ** 2) + EPS_DIV_ZERO
            v_circ = ti.sqrt(G * core_mass[gid] * r_spawn * r_spawn / (dist_sq * ti.sqrt(dist_sq)))
            star_vel[i] = c_vel + ti.Vector([-v_circ * ti.sin(theta), v_circ * ti.cos(theta), 0.0])

@ti.kernel
def update_core_halos():
    """Update luminous bulge halo particles around each core."""
    for k in range(NUM_CORES):
        c_pos = core_pos[k]
        for h in range(NUM_HALO_PER_CORE):
            idx = k * NUM_HALO_PER_CORE + h
            halo_pos[idx] = c_pos + halo_offset[idx]
            if k == 0:
                halo_color[idx] = ti.Vector([0.90, 0.96, 1.0])
            else:
                halo_color[idx] = ti.Vector([1.0, 0.90, 0.70])

# -----------------------------------------------------------------------------
# CINEMATIC COLORING KERNEL
# -----------------------------------------------------------------------------
@ti.kernel
def update_colors():
    """Computes physically meaningful colors for all stars.
    Mode 0: Galactic Identity (Cyan vs Amber) modulated by velocity & core distance.
    Mode 1: Kinetic Temperature Heatmap (Doppler / Virial heating).
    Mode 2: Tidal Disruption Strain (Hill sphere rupture indicator)."""
    mode = color_mode[None]

    for i in range(NUM_STARS):
        gid = star_galaxy_id[i]
        pos = star_pos[i]
        vel = star_vel[i]
        speed = vel.norm()

        c0 = core_pos[0]
        c1 = core_pos[1]
        d0 = (pos - c0).norm() + EPS_DIV_ZERO
        d1 = (pos - c1).norm() + EPS_DIV_ZERO

        # Tidal strain ratio: gravitational pull of perturber vs parent
        f0 = core_mass[0] / (d0 * d0 + EPS_CORE_STAR ** 2)
        f1 = core_mass[1] / (d1 * d1 + EPS_CORE_STAR ** 2)
        tidal_ratio = f1 / (f0 + EPS_DIV_ZERO) if gid == 0 else f0 / (f1 + EPS_DIV_ZERO)

        col = ti.Vector([1.0, 1.0, 1.0])

        if mode == 0:
            # DUAL GALACTIC IDENTITY WITH KINETIC BOOST
            # Galaxy A (Andromeda): Cyan / Cerulean / Electric Azure -> White
            # Galaxy B (Companion): Solar Gold / Amber / Fiery Magenta -> White
            dist_parent = d0 if gid == 0 else d1
            r_norm = dist_parent / 3.0

            # Kinetic acceleration boost: stars whipped around by close encounters glow brighter
            speed_boost = ti.min(ti.max((speed - 4.5) / 5.0, 0.0), 1.0)

            if gid == 0:
                # Galaxy A: Cerulean to Electric Cyan gradient
                core_c = ti.Vector([0.96, 0.98, 1.0])
                mid_c = ti.Vector([0.18, 0.85, 1.0])
                outer_c = ti.Vector([0.24, 0.32, 0.95])
                tail_c = ti.Vector([0.48, 0.15, 0.90])

                base = ti.Vector([0.0, 0.0, 0.0])
                if r_norm < 0.4:
                    t = r_norm / 0.4
                    base = (1.0 - t) * core_c + t * mid_c
                elif r_norm < 1.0:
                    t = (r_norm - 0.4) / 0.6
                    base = (1.0 - t) * mid_c + t * outer_c
                else:
                    t = ti.min((r_norm - 1.0) / 1.5, 1.0)
                    base = (1.0 - t) * outer_c + t * tail_c

                col = (1.0 - speed_boost * 0.75) * base + speed_boost * 0.75 * ti.Vector([1.0, 1.0, 1.0])

            else:
                # Galaxy B: Solar Amber to Fiery Coral gradient
                core_c = ti.Vector([1.0, 0.98, 0.92])
                mid_c = ti.Vector([1.0, 0.75, 0.18])
                outer_c = ti.Vector([1.0, 0.30, 0.08])
                tail_c = ti.Vector([0.96, 0.12, 0.45])

                base = ti.Vector([0.0, 0.0, 0.0])
                if r_norm < 0.4:
                    t = r_norm / 0.4
                    base = (1.0 - t) * core_c + t * mid_c
                elif r_norm < 1.0:
                    t = (r_norm - 0.4) / 0.6
                    base = (1.0 - t) * mid_c + t * outer_c
                else:
                    t = ti.min((r_norm - 1.0) / 1.5, 1.0)
                    base = (1.0 - t) * outer_c + t * tail_c

                col = (1.0 - speed_boost * 0.75) * base + speed_boost * 0.75 * ti.Vector([1.0, 1.0, 0.92])

        elif mode == 1:
            # KINETIC TEMPERATURE HEATMAP (Blackbody spectrum)
            s_norm = ti.min(speed / 9.0, 1.0)
            if s_norm < 0.25:
                t = s_norm / 0.25
                col = (1.0 - t) * ti.Vector([0.08, 0.05, 0.45]) + t * ti.Vector([0.1, 0.45, 0.95])
            elif s_norm < 0.5:
                t = (s_norm - 0.25) / 0.25
                col = (1.0 - t) * ti.Vector([0.1, 0.45, 0.95]) + t * ti.Vector([0.15, 0.92, 0.65])
            elif s_norm < 0.75:
                t = (s_norm - 0.5) / 0.25
                col = (1.0 - t) * ti.Vector([0.15, 0.92, 0.65]) + t * ti.Vector([1.0, 0.75, 0.15])
            else:
                t = (s_norm - 0.75) / 0.25
                col = (1.0 - t) * ti.Vector([1.0, 0.75, 0.15]) + t * ti.Vector([1.0, 1.0, 1.0])

        else:
            # TIDAL DISRUPTION STRAIN
            strain = ti.min(tidal_ratio / 1.2, 1.0)
            calm_col = ti.Vector([0.15, 0.65, 0.95]) if gid == 0 else ti.Vector([0.2, 0.85, 0.75])
            rip_col = ti.Vector([1.0, 0.12, 0.45])
            col = (1.0 - strain) * calm_col + strain * rip_col

        star_color[i] = col

# -----------------------------------------------------------------------------
# CORE TRAJECTORY TRAILS
# -----------------------------------------------------------------------------
@ti.kernel
def record_core_history():
    """Stores the latest core positions into the history ring buffer."""
    head = trail_head[None]
    for k in range(NUM_CORES):
        core_history[k, head] = core_pos[k]

    trail_head[None] = (head + 1) % TRAIL_LEN
    if trail_count[None] < TRAIL_LEN:
        trail_count[None] += 1

@ti.kernel
def update_trail_vertices():
    """Constructs line segments for scene.lines connecting sequential core positions."""
    head = trail_head[None]
    count = trail_count[None]
    num_segs = count - 1

    for k in range(NUM_CORES):
        for s in range(TRAIL_LEN - 1):
            vert_idx = (k * (TRAIL_LEN - 1) + s) * 2
            if s < num_segs:
                idx0 = (head - count + s + TRAIL_LEN) % TRAIL_LEN
                idx1 = (head - count + s + 1 + TRAIL_LEN) % TRAIL_LEN

                p0 = core_history[k, idx0]
                p1 = core_history[k, idx1]

                trail_verts[vert_idx] = p0
                trail_verts[vert_idx + 1] = p1

                alpha = ti.cast(s + 1, ti.f32) / ti.cast(count, ti.f32)
                base_c = ti.Vector([0.2, 0.85, 1.0]) if k == 0 else ti.Vector([1.0, 0.75, 0.25])
                trail_colors[vert_idx] = base_c * (alpha * 0.8)
                trail_colors[vert_idx + 1] = base_c * (alpha * 0.9)
            else:
                trail_verts[vert_idx] = core_pos[k]
                trail_verts[vert_idx + 1] = core_pos[k]
                trail_colors[vert_idx] = ti.Vector([0.0, 0.0, 0.0])
                trail_colors[vert_idx + 1] = ti.Vector([0.0, 0.0, 0.0])

# -----------------------------------------------------------------------------
# SETUP & INITIALIZATION (NumPy disk construction & 3D rotation)
# -----------------------------------------------------------------------------
def build_disk(num_stars, core_m, r_scale, r_min, r_max, z_scale, normal_vec, spin_dir=1.0):
    """Constructs an exponential disk of stars with circular rotation curve and 3D tilt."""
    u = np.random.uniform(0.0, 1.0, size=num_stars).astype(np.float32)
    e_min = np.exp(-r_min / r_scale)
    e_max = np.exp(-r_max / r_scale)
    r = -r_scale * np.log(e_min - u * (e_min - e_max))

    theta = np.random.uniform(0.0, 2.0 * np.pi, size=num_stars).astype(np.float32)
    z = np.random.normal(0.0, z_scale, size=num_stars).astype(np.float32)

    x_local = r * np.cos(theta)
    y_local = r * np.sin(theta)
    pos_local = np.stack([x_local, y_local, z], axis=-1)

    # Circular velocity from Plummer core + extended halo potential
    r_sq = r * r
    dist_cube = (r_sq + (EPS_CORE_STAR ** 2)) ** 1.5
    v_sq_core = G * core_m * r_sq / dist_cube
    # Halo component ensures flat rotation curve in outer disk, stabilizing spiral arms
    v_halo_sq = 0.35 * G * core_m
    v_sq_halo = v_halo_sq * r_sq / (r_sq + 1.2 ** 2)
    v_circ = np.sqrt(v_sq_core + v_sq_halo)

    vx_local = -spin_dir * v_circ * np.sin(theta)
    vy_local = spin_dir * v_circ * np.cos(theta)
    v_disp = 0.04 * v_circ
    vz_local = np.random.normal(0.0, v_disp * 0.4, size=num_stars).astype(np.float32)
    vel_local = np.stack([vx_local, vy_local, vz_local], axis=-1)

    # Compute orthonormal basis (u, w, n) for arbitrary 3D disk tilt
    normal = normal_vec / (np.linalg.norm(normal_vec) + 1e-6)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if np.abs(np.dot(normal, ref)) > 0.92:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u_axis = np.cross(ref, normal)
    u_axis /= np.linalg.norm(u_axis)
    w_axis = np.cross(normal, u_axis)

    rot_matrix = np.stack([u_axis, w_axis, normal], axis=1) # 3x3

    pos_3d = pos_local @ rot_matrix.T
    vel_3d = vel_local @ rot_matrix.T

    return pos_3d, vel_3d, r

def setup_simulation(randomize=False):
    """Initializes or resets the two colliding galaxies.
    Calculates hyperbolic/parabolic encounter orbit with exact pericenter distance."""
    m0 = 1.5
    m1 = 1.0 if not randomize else float(np.random.uniform(0.75, 1.35))

    # Initial separation D ~ 14.0 LU
    # Pericenter distance q ~ 1.9 - 3.2 LU (promotes dramatic tidal bridges & tails)
    D = 14.0
    q = 2.4 if not randomize else float(np.random.uniform(1.85, 3.2))

    # Orbital encounter plane orientation
    orbit_inclination = 0.22 if not randomize else float(np.random.uniform(-0.6, 0.6))
    encounter_angle = 0.35 if not randomize else float(np.random.uniform(0.0, 2.0 * np.pi))

    # Two-body celestial mechanics: parabolic/slightly hyperbolic orbit
    total_m = m0 + m1
    L_orbit = np.sqrt(2.0 * G * total_m * q)
    v_r = -np.sqrt(np.maximum(2.0 * G * total_m * (1.0 / D - q / (D ** 2)), 0.01))
    v_theta = L_orbit / D

    cos_a, sin_a = np.cos(encounter_angle), np.sin(encounter_angle)
    r_rel_2d = np.array([D * cos_a, D * sin_a, 0.0], dtype=np.float32)
    v_rel_2d = np.array([v_r * cos_a - v_theta * sin_a, v_r * sin_a + v_theta * cos_a, 0.0], dtype=np.float32)

    # Rotate orbit by inclination around x-axis
    cos_i, sin_i = np.cos(orbit_inclination), np.sin(orbit_inclination)
    rot_orbit = np.array([[1, 0, 0], [0, cos_i, -sin_i], [0, sin_i, cos_i]], dtype=np.float32)
    r_rel = rot_orbit @ r_rel_2d
    v_rel = rot_orbit @ v_rel_2d

    # Center of mass placement (Barycenter at origin)
    r0 = -(m1 / total_m) * r_rel
    r1 = (m0 / total_m) * r_rel
    v0 = -(m1 / total_m) * v_rel
    v1 = (m0 / total_m) * v_rel

    c_pos_np = np.stack([r0, r1], axis=0).astype(np.float32)
    c_vel_np = np.stack([v0, v1], axis=0).astype(np.float32)
    c_mass_np = np.array([m0, m1], dtype=np.float32)
    c_col_np = np.array([[0.9, 0.95, 1.0], [1.0, 0.9, 0.7]], dtype=np.float32)

    core_pos.from_numpy(c_pos_np)
    core_vel.from_numpy(c_vel_np)
    core_mass.from_numpy(c_mass_np)
    core_color.from_numpy(c_col_np)

    # Disk 3D Orientations:
    norm_A = np.array([0.15, 0.25, 1.0], dtype=np.float32) if not randomize else np.random.normal(0, 0.5, size=3).astype(np.float32)
    norm_A[2] = np.abs(norm_A[2]) + 0.3
    norm_A /= np.linalg.norm(norm_A)

    norm_B = np.array([-0.3, 0.4, 0.9], dtype=np.float32) if not randomize else np.random.normal(0, 0.6, size=3).astype(np.float32)
    norm_B[2] = np.abs(norm_B[2]) + 0.2
    norm_B /= np.linalg.norm(norm_B)

    # 75% prograde encounters for massive sweeping tidal tails, 25% retrograde for rings
    spin_B = 1.0 if (not randomize or np.random.rand() > 0.25) else -1.0

    pos_A, vel_A, r_A = build_disk(NUM_STARS_A, m0, r_scale=1.1, r_min=0.25, r_max=3.8, z_scale=0.045, normal_vec=norm_A, spin_dir=1.0)
    pos_B, vel_B, r_B = build_disk(NUM_STARS_B, m1, r_scale=0.9, r_min=0.22, r_max=3.0, z_scale=0.038, normal_vec=norm_B, spin_dir=spin_B)

    pos_A += r0
    vel_A += v0
    pos_B += r1
    vel_B += v1

    stars_pos_np = np.concatenate([pos_A, pos_B], axis=0).astype(np.float32)
    stars_vel_np = np.concatenate([vel_A, vel_B], axis=0).astype(np.float32)
    stars_gid_np = np.concatenate([np.zeros(NUM_STARS_A, dtype=np.int32), np.ones(NUM_STARS_B, dtype=np.int32)])
    stars_rad_np = np.concatenate([r_A, r_B], axis=0).astype(np.float32)

    star_pos.from_numpy(stars_pos_np)
    star_vel.from_numpy(stars_vel_np)
    star_galaxy_id.from_numpy(stars_gid_np)
    star_init_rad.from_numpy(stars_rad_np)

    # Initialize Bulge Halos
    halo_offset_np = np.random.normal(0.0, 0.05, size=(NUM_HALO, 3)).astype(np.float32)
    halo_offset.from_numpy(halo_offset_np)
    update_core_halos()

    # Reset trajectory trail ring buffer
    trail_head[None] = 0
    trail_count[None] = 0
    hist_np = np.zeros((NUM_CORES, TRAIL_LEN, 3), dtype=np.float32)
    for k in range(NUM_CORES):
        hist_np[k, :] = c_pos_np[k]
    core_history.from_numpy(hist_np)

    # Compute initial accelerations
    compute_accelerations()
    compute_metrics()
    update_colors()

    if randomize:
        print(f"[COLLISION RESET] Impact Pericenter q={q:.2f} LU (~{q * 16.3:.1f} kly) | M0={m0:.2f}, M1={m1:.2f} | Spin={'Prograde (Sweeping Tails)' if spin_B > 0 else 'Retrograde (Ripples)'}")

def init_environment():
    """Generates a faint, elegant reference grid and distant ambient stars for cosmic depth."""
    verts = []
    cols = []
    radii = [2.0, 4.0, 6.0, 8.0, 10.0, 13.0]
    for r in radii:
        for s in range(GRID_SEGS_PER_RING):
            th0 = s * 2.0 * np.pi / GRID_SEGS_PER_RING
            th1 = (s + 1) * 2.0 * np.pi / GRID_SEGS_PER_RING
            p0 = [r * np.cos(th0), r * np.sin(th0), -0.05]
            p1 = [r * np.cos(th1), r * np.sin(th1), -0.05]
            verts.extend([p0, p1])
            c = [0.07, 0.11, 0.20] if r != 10.0 else [0.12, 0.18, 0.30]
            cols.extend([c, c])

    axes = [
        ([-15.0, 0.0, -0.05], [15.0, 0.0, -0.05]),
        ([0.0, -15.0, -0.05], [0.0, 15.0, -0.05]),
    ]
    for p0, p1 in axes:
        verts.extend([p0, p1])
        c = [0.09, 0.15, 0.26]
        cols.extend([c, c])

    verts_np = np.array(verts, dtype=np.float32)
    cols_np = np.array(cols, dtype=np.float32)
    grid_verts.from_numpy(verts_np)
    grid_colors.from_numpy(cols_np)

    # Distant cosmic background dust
    dust_r = np.random.uniform(25.0, 48.0, size=NUM_DUST).astype(np.float32)
    dust_th = np.random.uniform(0.0, 2.0 * np.pi, size=NUM_DUST).astype(np.float32)
    dust_phi = np.random.uniform(-0.5, 0.5, size=NUM_DUST).astype(np.float32)

    dust_x = dust_r * np.cos(dust_phi) * np.cos(dust_th)
    dust_y = dust_r * np.cos(dust_phi) * np.sin(dust_th)
    dust_z = dust_r * np.sin(dust_phi)
    dust_pos_np = np.stack([dust_x, dust_y, dust_z], axis=-1).astype(np.float32)

    dust_c_np = np.random.uniform(0.12, 0.28, size=(NUM_DUST, 3)).astype(np.float32)
    dust_pos.from_numpy(dust_pos_np)
    dust_color.from_numpy(dust_c_np)

# -----------------------------------------------------------------------------
# MAIN SIMULATION LOOP & USER INTERFACE
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  COSMIC COLLISION: TIDAL TAIL GALAXY INTERACTION SIMULATION")
    print("  Symplectic Gravitational Dynamics (Toomre & Toomre 1972)")
    print("=" * 80)
    print("  PHYSICS & CONTROLS:")
    print("    [SPACE]        : Pause / Resume simulation")
    print("    [R]            : Reset with new randomized 3D collision trajectory")
    print("    [C]            : Cycle color modes:")
    print("                     0: Galactic Identity (Cyan vs Amber) + Kinetic Flare")
    print("                     1: Kinetic Temperature (Doppler Heatmap)")
    print("                     2: Tidal Disruption Strain (Hill Sphere Rupture)")
    print("    [T]            : Toggle Core Orbit Trails on/off")
    print("    [G]            : Toggle Galactic Reference Grid on/off")
    print("    [H]            : Toggle On-Screen HUD text display")
    print("    [UP / DOWN]    : Speed up / slow down simulation time")
    print("    [Left-Drag]    : Orbit 3D Camera around barycenter")
    print("    [Right-Drag]   : Pan / Zoom Camera")
    print("    [ESC]          : Exit simulation")
    print("=" * 80)

    res = (1440, 900)
    window = ti.ui.Window("Cosmic Collision: Tidal Tail Galaxy Interaction", res, vsync=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # Cinematic initial camera view
    cam_dist = 22.0
    cam_phi = 0.46
    cam_theta = 0.58
    camera.position(cam_dist * np.sin(cam_phi) * np.cos(cam_theta),
                    -cam_dist * np.sin(cam_phi) * np.sin(cam_theta),
                    cam_dist * np.cos(cam_phi))
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0.0, 0.0, 1.0)
    camera.fov(52)

    color_mode[None] = 0
    dt_scale[None] = 1.0
    dyn_friction[None] = 0.85

    setup_simulation(randomize=False)
    init_environment()

    is_paused = False
    show_trails = True
    show_grid = True
    show_gui = True
    sim_steps = 0

    mode_names = [
        "Galactic Identity (Cyan vs Amber) + Kinetic Flare",
        "Kinetic Temperature (Doppler Heatmap)",
        "Tidal Disruption Strain (Hill Sphere Rupture)"
    ]

    last_mouse_pos = window.get_cursor_pos()

    # Main interaction and rendering loop
    while window.running:
        # 1. Keyboard Event Handling
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.SPACE:
                is_paused = not is_paused
                print(f"[STATE] {'PAUSED' if is_paused else 'RESUMED'}")
            elif event.key == 'r' or event.key == 'R':
                setup_simulation(randomize=True)
            elif event.key == 'c' or event.key == 'C':
                color_mode[None] = (color_mode[None] + 1) % 3
                print(f"[COLOR MODE] {mode_names[color_mode[None]]}")
            elif event.key == 't' or event.key == 'T':
                show_trails = not show_trails
            elif event.key == 'g' or event.key == 'G':
                show_grid = not show_grid
            elif event.key == 'h' or event.key == 'H':
                show_gui = not show_gui
            elif event.key == ti.ui.UP:
                dt_scale[None] = min(dt_scale[None] * 1.25, 3.5)
                print(f"[TIME SCALE] {dt_scale[None]:.2f}x")
            elif event.key == ti.ui.DOWN:
                dt_scale[None] = max(dt_scale[None] / 1.25, 0.18)
                print(f"[TIME SCALE] {dt_scale[None]:.2f}x")
            elif event.key == ti.ui.ESCAPE:
                window.running = False

        # 2. Camera Controls (Trackball orbit & smooth zoom)
        curr_mouse = window.get_cursor_pos()
        if window.is_pressed(ti.ui.LMB):
            dx = curr_mouse[0] - last_mouse_pos[0]
            dy = curr_mouse[1] - last_mouse_pos[1]
            cam_theta -= dx * 3.5
            cam_phi = np.clip(cam_phi + dy * 3.0, 0.08, np.pi - 0.08)
        elif window.is_pressed(ti.ui.RMB):
            dy = curr_mouse[1] - last_mouse_pos[1]
            cam_dist = np.clip(cam_dist - dy * 30.0, 4.0, 60.0)

        last_mouse_pos = curr_mouse

        # Barycenter Camera Tracking
        bary = barycenter_out[None]
        cam_x = bary[0] + cam_dist * np.sin(cam_phi) * np.cos(cam_theta)
        cam_y = bary[1] + cam_dist * np.sin(cam_phi) * np.sin(cam_theta)
        cam_z = bary[2] + cam_dist * np.cos(cam_phi)
        camera.position(cam_x, cam_y, cam_z)
        camera.lookat(bary[0], bary[1], bary[2])
        camera.up(0.0, 0.0, 1.0)
        scene.set_camera(camera)

        # 3. Symplectic Velocity-Verlet Integration
        if not is_paused:
            sub_dt = (DT_DEFAULT * dt_scale[None]) / float(SUBSTEPS)
            for _ in range(SUBSTEPS):
                # Half-kick: v += 0.5 * a * dt
                kick_velocity(0.5 * sub_dt)
                # Full drift: x += v * dt
                drift_position(sub_dt)
                # Recompute accelerations at new positions
                compute_accelerations()
                # Second half-kick: v += 0.5 * a * dt
                kick_velocity(0.5 * sub_dt)

            compute_metrics()
            recycle_escaped_stars()
            update_core_halos()

            if sim_steps % 3 == 0:
                record_core_history()
            sim_steps += 1

        # 4. Color & Trail Updates
        update_colors()
        if show_trails:
            update_trail_vertices()

        # 5. Scene Lighting & Rendering (Self-luminous stellar emission on near-black cosmic background)
        scene.ambient_light((1.0, 1.0, 1.0))

        # Distant background celestial dust
        scene.particles(dust_pos, radius=0.007, per_vertex_color=dust_color)

        # Render 65,000 dense test-particle stars
        scene.particles(star_pos, radius=0.013, per_vertex_color=star_color)

        # Render Massive Galactic Cores (Supermassive nuclei)
        scene.particles(core_pos, radius=0.075, per_vertex_color=core_color)

        # Render Glowing Stellar Bulge Halo around each nucleus
        scene.particles(halo_pos, radius=0.024, per_vertex_color=halo_color)

        # Render Core Trajectory Orbit Trails
        if show_trails and trail_count[None] > 1:
            scene.lines(trail_verts, width=2.5, per_vertex_color=trail_colors)

        # Render Celestial Reference Grid on orbital plane
        if show_grid:
            scene.lines(grid_verts, width=1.0, per_vertex_color=grid_colors)

        canvas.scene(scene)

        # 6. Interactive On-Screen HUD Overlay
        if show_gui:
            gui = window.get_gui()
            metrics = core_metrics[None]
            r01_dist = metrics[0]
            v_rel_speed = metrics[1]

            phase_text = "Pre-Encounter (Approaching)"
            if r01_dist < 3.2:
                phase_text = "Pericenter Flyby (Maximum Tidal Disruption)"
            elif r01_dist > 5.0 and sim_steps > 300:
                phase_text = "Post-Encounter (Sweeping Tidal Tails & Bridges)"

            gui.text("COSMIC COLLISION SIMULATOR")
            gui.text(f"Stars: {NUM_STARS:,} | FPS: {int(window.get_fps())} | Phase: {phase_text}")
            gui.text(f"Core Separation: {r01_dist:.2f} LU (~{r01_dist * 16.3:.1f} kly) | V_rel: {v_rel_speed:.2f} (~{v_rel_speed * 97.8:.0f} km/s)")
            gui.text(f"Color Mode: {mode_names[color_mode[None]]}")

            if gui.button("Reset Collision [R]"):
                setup_simulation(randomize=True)
            if gui.button("Toggle Pause [SPACE]"):
                is_paused = not is_paused
            if gui.button("Cycle Color Mode [C]"):
                color_mode[None] = (color_mode[None] + 1) % 3
            if gui.button("Toggle Trails [T]"):
                show_trails = not show_trails

            dt_val = gui.slider_float("Time Dilation", dt_scale[None], 0.15, 3.0)
            dt_scale[None] = dt_val

        window.show()

if __name__ == "__main__":
    main()