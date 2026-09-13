import taichi as ti
import os
import math
import numpy as np

try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)
    print("Running on CPU, expect lower FPS.")

if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = os.path.abspath(".cache")

# ---------------------------------------------------------------------------
# ASTROPHYSICAL & SIMULATION CONSTANTS (High Density for Smooth Continuum)
# ---------------------------------------------------------------------------
G = 4.0 * (math.pi ** 2)
EPS_CORE_CORE = 0.28
EPS_CORE_STAR = 0.18
EPS_DIV_ZERO = 1e-3

NUM_STARS_A = 90000
NUM_STARS_B = 70000
NUM_STARS = NUM_STARS_A + NUM_STARS_B
NUM_CORES = 2

TRAIL_LEN = 240
NUM_TRAIL_VERTS = NUM_CORES * TRAIL_LEN * 2

SUBSTEPS = 3
DT_DEFAULT = 0.0022

# ---------------------------------------------------------------------------
# TAICHI FIELDS
# ---------------------------------------------------------------------------
star_pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_STARS)
star_vel = ti.Vector.field(3, dtype=ti.f32, shape=NUM_STARS)
star_acc = ti.Vector.field(3, dtype=ti.f32, shape=NUM_STARS)
star_color = ti.Vector.field(3, dtype=ti.f32, shape=NUM_STARS)
star_galaxy_id = ti.field(dtype=ti.i32, shape=NUM_STARS)
star_init_rad = ti.field(dtype=ti.f32, shape=NUM_STARS)

core_pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_CORES)
core_vel = ti.Vector.field(3, dtype=ti.f32, shape=NUM_CORES)
core_acc = ti.Vector.field(3, dtype=ti.f32, shape=NUM_CORES)
core_mass = ti.field(dtype=ti.f32, shape=NUM_CORES)
core_color = ti.Vector.field(3, dtype=ti.f32, shape=NUM_CORES)

NUM_HALO_PER_CORE = 300
NUM_HALO = NUM_CORES * NUM_HALO_PER_CORE
halo_pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_HALO)
halo_color = ti.Vector.field(3, dtype=ti.f32, shape=NUM_HALO)
halo_offset = ti.Vector.field(3, dtype=ti.f32, shape=NUM_HALO)

trail_verts = ti.Vector.field(3, dtype=ti.f32, shape=NUM_TRAIL_VERTS)
trail_colors = ti.Vector.field(3, dtype=ti.f32, shape=NUM_TRAIL_VERTS)
core_history = ti.Vector.field(3, dtype=ti.f32, shape=(NUM_CORES, TRAIL_LEN))

color_mode = ti.field(dtype=ti.i32, shape=())
trail_head = ti.field(dtype=ti.i32, shape=())
trail_count = ti.field(dtype=ti.i32, shape=())
dt_scale = ti.field(dtype=ti.f32, shape=())
dyn_friction = ti.field(dtype=ti.f32, shape=())
barycenter_out = ti.Vector.field(3, dtype=ti.f32, shape=())
core_metrics = ti.Vector.field(2, dtype=ti.f32, shape=())

# ---------------------------------------------------------------------------
# KERNELS
# ---------------------------------------------------------------------------
@ti.kernel
def kick_velocity(dt: ti.f32):
    for k in range(NUM_CORES):
        core_vel[k] += core_acc[k] * dt
    for i in range(NUM_STARS):
        star_vel[i] += star_acc[i] * dt

@ti.kernel
def drift_position(dt: ti.f32):
    for k in range(NUM_CORES):
        core_pos[k] += core_vel[k] * dt
    for i in range(NUM_STARS):
        star_pos[i] += star_vel[i] * dt

@ti.kernel
def compute_accelerations():
    r01 = core_pos[1] - core_pos[0]
    dist_sq_c = r01.dot(r01) + (EPS_CORE_CORE ** 2) + EPS_DIV_ZERO
    inv_dist_c = 1.0 / ti.sqrt(dist_sq_c)
    inv_dist3_c = inv_dist_c * inv_dist_c * inv_dist_c
    
    f_grav = G * inv_dist3_c
    a0 = f_grav * core_mass[1] * r01
    a1 = -f_grav * core_mass[0] * r01

    v_rel = core_vel[1] - core_vel[0]
    v_norm = v_rel.norm()
    dist_c = ti.sqrt(r01.dot(r01) + EPS_DIV_ZERO)
    halo_overlap = ti.exp(-dist_c / 2.2)
    df_mag = dyn_friction[None] * halo_overlap / (v_norm * v_norm * v_norm + 0.15 + EPS_DIV_ZERO)
    
    df_acc0 = df_mag * core_mass[1] * v_rel
    df_acc1 = -df_mag * core_mass[0] * v_rel

    core_acc[0] = a0 + df_acc0
    core_acc[1] = a1 + df_acc1

    for i in range(NUM_STARS):
        p = star_pos[i]
        acc = ti.Vector([0.0, 0.0, 0.0])
        
        d0 = p - core_pos[0]
        dist_sq_0 = d0.dot(d0) + (EPS_CORE_STAR ** 2) + EPS_DIV_ZERO
        inv_d0 = 1.0 / ti.sqrt(dist_sq_0)
        acc += (G * core_mass[0] * inv_d0 * inv_d0 * inv_d0) * d0

        d1 = p - core_pos[1]
        dist_sq_1 = d1.dot(d1) + (EPS_CORE_STAR ** 2) + EPS_DIV_ZERO
        inv_d1 = 1.0 / ti.sqrt(dist_sq_1)
        acc += (G * core_mass[1] * inv_d1 * inv_d1 * inv_d1) * d1

        star_acc[i] = -acc

@ti.kernel
def compute_metrics():
    total_m = core_mass[0] + core_mass[1]
    barycenter_out[None] = (core_pos[0] * core_mass[0] + core_pos[1] * core_mass[1]) / (total_m + EPS_DIV_ZERO)
    dist01 = (core_pos[1] - core_pos[0]).norm()
    v_rel = (core_vel[1] - core_vel[0]).norm()
    core_metrics[None] = ti.Vector([dist01, v_rel])

@ti.kernel
def recycle_escaped_stars():
    bary = barycenter_out[None]
    for i in range(NUM_STARS):
        dist_bary = (star_pos[i] - bary).norm()
        if dist_bary > 36.0:
            gid = star_galaxy_id[i]
            c_pos = core_pos[gid]
            c_vel = core_vel[gid]
            u = ti.random(ti.f32)
            theta = ti.random(ti.f32) * 2.0 * math.pi
            r_spawn = 0.6 + u * 2.2
            z_spawn = (ti.random(ti.f32) - 0.5) * 0.08
            dx = r_spawn * ti.cos(theta)
            dy = r_spawn * ti.sin(theta)
            star_pos[i] = c_pos + ti.Vector([dx, dy, z_spawn])
            dist_sq = r_spawn * r_spawn + (EPS_CORE_STAR ** 2) + EPS_DIV_ZERO
            v_circ = ti.sqrt(G * core_mass[gid] * r_spawn * r_spawn / (dist_sq * ti.sqrt(dist_sq)))
            star_vel[i] = c_vel + ti.Vector([-v_circ * ti.sin(theta), v_circ * ti.cos(theta), 0.01])

@ti.kernel
def update_core_halos():
    for k in range(NUM_CORES):
        c_pos = core_pos[k]
        for h in range(NUM_HALO_PER_CORE):
            idx = k * NUM_HALO_PER_CORE + h
            halo_pos[idx] = c_pos + halo_offset[idx]
            if k == 0:
                halo_color[idx] = ti.Vector([1.0, 0.95, 0.82])
            else:
                halo_color[idx] = ti.Vector([0.98, 0.92, 0.78])

@ti.kernel
def update_colors():
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
        
        F0 = core_mass[0] / (d0 * d0 + EPS_CORE_STAR ** 2)
        F1 = core_mass[1] / (d1 * d1 + EPS_CORE_STAR ** 2)
        tidal_ratio = F1 / (F0 + EPS_DIV_ZERO) if gid == 0 else F0 / (F1 + EPS_DIV_ZERO)
        
        col = ti.Vector([1.0, 1.0, 1.0])
        if mode == 0:
            # Elegant, smooth astronomical gradient (Deep space indigo/slate blue with soft warm golden cores & subtle dusty rose highlights)
            dist_parent = d0 if gid == 0 else d1
            r_norm = dist_parent / 3.0
            
            if gid == 0:
                core_c = ti.Vector([1.0, 0.96, 0.85])  # Warm golden core
                mid_c = ti.Vector([0.45, 0.60, 0.88])   # Rich dusty blue
                outer_c = ti.Vector([0.22, 0.35, 0.68]) # Deep celestial slate
                
                base = ti.Vector([0.0, 0.0, 0.0])
                if r_norm < 0.3:
                    t = r_norm / 0.3
                    base = (1.0 - t) * core_c + t * mid_c
                else:
                    t = ti.min((r_norm - 0.3) / 1.2, 1.0)
                    base = (1.0 - t) * mid_c + t * outer_c
                col = base
            else:
                core_c = ti.Vector([1.0, 0.96, 0.85])
                mid_c = ti.Vector([0.48, 0.63, 0.90])
                outer_c = ti.Vector([0.25, 0.38, 0.72])
                
                base = ti.Vector([0.0, 0.0, 0.0])
                if r_norm < 0.3:
                    t = r_norm / 0.3
                    base = (1.0 - t) * core_c + t * mid_c
                else:
                    t = ti.min((r_norm - 0.3) / 1.2, 1.0)
                    base = (1.0 - t) * mid_c + t * outer_c
                col = base
        elif mode == 1:
            s_norm = ti.min(speed / 9.0, 1.0)
            if s_norm < 0.33:
                col = ti.Vector([0.3, 0.5, 0.8])
            elif s_norm < 0.66:
                col = ti.Vector([0.7, 0.5, 0.8])
            else:
                col = ti.Vector([0.9, 0.7, 0.5])
        else:
            strain = ti.min(tidal_ratio / 1.2, 1.0)
            calm = ti.Vector([0.35, 0.55, 0.85]) if gid == 0 else ti.Vector([0.4, 0.6, 0.8])
            rip = ti.Vector([0.8, 0.35, 0.6])
            col = (1.0 - strain) * calm + strain * rip

        star_color[i] = col

@ti.kernel
def record_core_history():
    head = trail_head[None]
    for k in range(NUM_CORES):
        core_history[k, head] = core_pos[k]
    trail_head[None] = (head + 1) % TRAIL_LEN
    if trail_count[None] < TRAIL_LEN:
        trail_count[None] += 1

@ti.kernel
def update_trail_vertices():
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
                base_c = ti.Vector([0.4, 0.6, 0.8]) if k == 0 else ti.Vector([0.6, 0.5, 0.7])
                trail_colors[vert_idx] = base_c * (alpha * 0.4)
                trail_colors[vert_idx + 1] = base_c * (alpha * 0.5)
            else:
                trail_verts[vert_idx] = core_pos[k]
                trail_verts[vert_idx + 1] = core_pos[k]
                trail_colors[vert_idx] = ti.Vector([0.0, 0.0, 0.0])
                trail_colors[vert_idx + 1] = ti.Vector([0.0, 0.0, 0.0])

def build_disk(num_stars, core_m, r_scale, r_min, r_max, z_scale, normal_vec, spin_dir=1.0):
    u = np.random.uniform(0.0, 1.0, size=num_stars).astype(np.float32)
    e_min = np.exp(-r_min / r_scale)
    e_max = np.exp(-r_max / r_scale)
    r = -r_scale * np.log(e_min - u * (e_min - e_max))
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=num_stars).astype(np.float32)
    z = np.random.normal(0.0, z_scale, size=num_stars).astype(np.float32)
    x_local = r * np.cos(theta)
    y_local = r * np.sin(theta)
    pos_local = np.stack([x_local, y_local, z], axis=-1)
    
    r_sq = r * r
    dist_cube = (r_sq + 1.2 ** 2) ** 1.5
    v_sq_core = G * core_m * r_sq / dist_cube
    v_halo_sq = 0.35 * G * core_m
    v_sq_halo = v_halo_sq * r_sq / (r_sq + 1.2 ** 2)
    v_circ = np.sqrt(v_sq_core + v_sq_halo)
    
    vx_local = -spin_dir * v_circ * np.sin(theta)
    vy_local = spin_dir * v_circ * np.cos(theta)
    v_disp = 0.03 * v_circ
    vz_local = np.random.normal(0.0, v_disp, size=num_stars).astype(np.float32)
    vel_local = np.stack([vx_local, vy_local, vz_local], axis=-1)
    
    normal = normal_vec / (np.linalg.norm(normal_vec) + 1e-6)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if np.abs(np.dot(normal, ref)) > 0.92:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u_axis = np.cross(ref, normal)
    u_axis /= np.linalg.norm(u_axis)
    w_axis = np.cross(normal, u_axis)
    rot_matrix = np.stack([u_axis, w_axis, normal], axis=1)
    
    pos_3d = pos_local @ rot_matrix.T
    vel_3d = vel_local @ rot_matrix.T
    return pos_3d, vel_3d, r

def setup_simulation(randomize=False):
    m0 = 1.5
    m1 = 1.0 if not randomize else float(np.random.uniform(0.75, 1.35))
    D = 14.0
    q = 2.4 if not randomize else float(np.random.uniform(1.85, 3.2))
    orbit_inclination = 0.22 if not randomize else float(np.random.uniform(-0.6, 0.6))
    encounter_angle = 0.35 if not randomize else float(np.random.uniform(0.0, 2.0 * np.pi))
    
    total_m = m0 + m1
    L_orbit = np.sqrt(2.0 * G * total_m * q)
    v_r = -np.sqrt(np.maximum(2.0 * G * total_m * (1.0 / D - q / (D ** 2)), 0.01))
    v_theta = L_orbit / D
    
    cos_a, sin_a = np.cos(encounter_angle), np.sin(encounter_angle)
    r_rel_2d = np.array([D * cos_a, D * sin_a, 0.0], dtype=np.float32)
    v_rel_2d = np.array([v_r * cos_a - v_theta * sin_a, v_r * sin_a + v_theta * cos_a, 0.0], dtype=np.float32)
    
    cos_i, sin_i = np.cos(orbit_inclination), np.sin(orbit_inclination)
    rot_orbit = np.array([[1, 0, 0], [0, cos_i, sin_i], [0, -sin_i, cos_i]], dtype=np.float32)
    r_rel = rot_orbit @ r_rel_2d
    v_rel = rot_orbit @ v_rel_2d
    
    r0 = -(m1 / total_m) * r_rel
    r1 = (m0 / total_m) * r_rel
    v0 = -(m1 / total_m) * v_rel
    v1 = (m0 / total_m) * v_rel
    
    c_pos_np = np.stack([r0, r1], axis=0).astype(np.float32)
    c_vel_np = np.stack([v0, v1], axis=0).astype(np.float32)
    c_mass_np = np.array([m0, m1], dtype=np.float32)
    c_col_np = np.array([[1.0, 0.96, 0.85], [1.0, 0.94, 0.82]], dtype=np.float32)
    
    core_pos.from_numpy(c_pos_np)
    core_vel.from_numpy(c_vel_np)
    core_mass.from_numpy(c_mass_np)
    core_color.from_numpy(c_col_np)
    
    norm_A = np.array([0.15, 0.25, 1.0], dtype=np.float32)
    norm_B = np.array([-0.3, 0.4, 0.9], dtype=np.float32)
    norm_A /= np.linalg.norm(norm_A)
    norm_B /= np.linalg.norm(norm_B)
    
    pos_A, vel_A, r_A = build_disk(NUM_STARS_A, m0, r_scale=1.1, r_min=0.25, r_max=3.8, z_scale=0.035, normal_vec=norm_A, spin_dir=1.0)
    pos_B, vel_B, r_B = build_disk(NUM_STARS_B, m1, r_scale=0.9, r_min=0.22, r_max=3.0, z_scale=0.030, normal_vec=norm_B, spin_dir=1.0)
    
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
    
    halo_offset_np = np.random.normal(0.0, 0.04, size=(NUM_HALO, 3)).astype(np.float32)
    halo_offset.from_numpy(halo_offset_np)
    update_core_halos()
    
    trail_head[None] = 0
    trail_count[None] = 0
    hist_np = np.zeros((NUM_CORES, TRAIL_LEN, 3), dtype=np.float32)
    for k in range(NUM_CORES):
        hist_np[k, :] = c_pos_np[k]
    core_history.from_numpy(hist_np)
    
    compute_accelerations()
    compute_metrics()
    update_colors()

def main():
    res = (1440, 900)
    window = ti.ui.Window("Cosmic Collision: Cohesive Continuum", res, vsync=False)
    canvas = window.get_canvas()
    # Deep celestial background with subtle atmospheric depth
    canvas.set_background_color((0.02, 0.02, 0.05))
    scene = window.get_scene()
    camera = ti.ui.Camera()
    
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
    
    is_paused = False
    show_trails = True
    show_gui = True
    sim_steps = 0
    
    mode_names = [
        "Astronomical Continuous Gradient",
        "Kinetic Temperature (Doppler Heatmap)",
        "Tidal Disruption Strain"
    ]
    
    last_mouse_pos = window.get_cursor_pos()
    
    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.SPACE:
                is_paused = not is_paused
            elif event.key == 'r' or event.key == 'R':
                setup_simulation(randomize=True)
            elif event.key == 'c' or event.key == 'C':
                color_mode[None] = (color_mode[None] + 1) % 3
            elif event.key == 't' or event.key == 'T':
                show_trails = not show_trails
            elif event.key == 'h' or event.key == 'H':
                show_gui = not show_gui
            elif event.key == ti.ui.UP:
                dt_scale[None] = min(dt_scale[None] * 1.25, 3.5)
            elif event.key == ti.ui.DOWN:
                dt_scale[None] = max(dt_scale[None] / 1.25, 0.18)
            elif event.key == ti.ui.ESCAPE:
                window.running = False
                
        curr_mouse = window.get_cursor_pos()
        if window.is_pressed(ti.ui.LMB):
            dx = curr_mouse[0] - last_mouse_pos[0]
            dy = curr_mouse[1] - last_mouse_pos[1]
            cam_theta -= dx * 3.5
            cam_phi = np.clip(cam_phi - dy * 3.0, 0.08, np.pi - 0.08)
        elif window.is_pressed(ti.ui.RMB):
            dy = curr_mouse[1] - last_mouse_pos[1]
            cam_dist = np.clip(cam_dist + dy * 30.0, 4.0, 60.0)
        last_mouse_pos = curr_mouse
        
        bary = barycenter_out[None]
        cam_x = bary[0] + cam_dist * np.sin(cam_phi) * np.cos(cam_theta)
        cam_y = bary[1] + cam_dist * np.sin(cam_phi) * np.sin(cam_theta)
        cam_z = bary[2] + cam_dist * np.cos(cam_phi)
        camera.position(cam_x, cam_y, cam_z)
        camera.lookat(bary[0], bary[1], bary[2])
        camera.up(0.0, 0.0, 1.0)
        scene.set_camera(camera)
        
        if not is_paused:
            sub_dt = (DT_DEFAULT * dt_scale[None]) / float(SUBSTEPS)
            for _ in range(SUBSTEPS):
                kick_velocity(0.5 * sub_dt)
                drift_position(sub_dt)
                compute_accelerations()
                kick_velocity(0.5 * sub_dt)
            compute_metrics()
            recycle_escaped_stars()
            update_core_halos()
            if sim_steps % 3 == 0:
                record_core_history()
            sim_steps += 1
            
        update_colors()
        if show_trails:
            update_trail_vertices()
            
        scene.ambient_light((0.9, 0.9, 1.0))
        
        # Dense, micro-sized particles blended closely together to eliminate polka-dot artifacts and form a solid continuum
        scene.particles(star_pos, radius=0.011, per_vertex_color=star_color)
        scene.particles(core_pos, radius=0.07, per_vertex_color=core_color)
        scene.particles(halo_pos, radius=0.035, per_vertex_color=halo_color)
        
        if show_trails and trail_count[None] > 1:
            scene.lines(trail_verts, width=2.5, per_vertex_color=trail_colors)
            
        canvas.scene(scene)
        
        if show_gui:
            gui = window.get_gui()
            metrics = core_metrics[None]
            r01_dist = metrics[0]
            phase_text = "Pre-Encounter (Approaching)"
            if r01_dist < 3.2:
                phase_text = "Pericenter Flyby (Maximum Tidal Disruption)"
            elif r01_dist > 5.0 and sim_steps > 300:
                phase_text = "Post-Encounter (Sweeping Tidal Tails)"
                
            gui.text("COSMIC COLLISION SIMULATOR")
            gui.text(f"Stars: {NUM_STARS:,} | Phase: {phase_text}")
            gui.text(f"Core Separation: {r01_dist:.2f} LU (~{r01_dist * 16.3:.1f} kly)")
            gui.text(f"Color Mode: {mode_names[color_mode[None]]}")
            if gui.button("Reset Collision [R]"):
                setup_simulation(randomize=True)
            if gui.button("Toggle Pause [SPACE]"):
                is_paused = not is_paused
            if gui.button("Cycle Color Mode [C]"):
                color_mode[None] = (color_mode[None] + 1) % 3
            dt_val = gui.slider_float("Time Dilation", dt_scale[None], 0.15, 3.0)
            dt_scale[None] = dt_val
            
        window.show()

if __name__ == '__main__':
    main()