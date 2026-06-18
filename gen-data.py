import rebound
from rebound import Orbit
import numpy as np
from tqdm import tqdm
import os
import uuid
import pandas as pd
import json
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

n_packs = 1            
n_sim_per_pack = 10_000
output_file = "data_test.csv"
G = 4*np.pi**2 # gravitational constant in au M_sun year units (au^3 year^-2 M_sun^-1)
N = 3 # particle number
A = 10 # A = d_min / r
R = 5e+4 # the maximum volume containing N stars
H = [0.0, 1.0] # hardness range; PE / KE
mass = [0.08, 150] # stars mass in M_sun
velocity = [0.01, 100.0] # stars initial velocity range in au/year
enc_r = [0.01, 10000] # encounter radius range in au
eps = 1e-12
mode = 1 # = 0 blank, = 1 test run, = 2 data production

def _norm(x):
    return np.linalg.norm(x)

def _cross(vec1, vec2):
    return np.cross(vec1, vec2)

def _mu(m1, m2):
    return m1*m2/(m1+m2)

def rand_perp(vec):
    # Random perpendicular vector
    perp = np.array([0,0,0])
    while _norm(perp) < eps:
        perp = np.random.randn(3)
        perp -= np.dot(perp, vec) * vec  
    perp /= _norm(perp)
    perp_dot_vec = np.dot(perp, vec)
    if perp_dot_vec > eps:
        print(f"WARNING: b not perpendicular to v: {perp_dot_vec}")    
    return perp

def _dir(theta, phi):
    dir_vec = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    dir_vec /= _norm(dir_vec)
    return dir_vec

def angle_between(v1, v2):
    cos_theta = np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def stellar_radius(mass_msun, object_type=None):    
    # Constants
    R_sun_AU = 0.00465
    AU_km = 149597870.7    
    if object_type == "white_dwarf":
        radius_au = 0.0000465 * (0.6 / max(mass_msun, 0.2)) ** (1.0/3.0)        
    elif object_type == "neutron_star":
        radius_km = 12.0 * (1.4 / max(mass_msun, 0.8)) ** 0.3
        radius_au = radius_km / AU_km        
    elif object_type == "black_hole":
        radius_km = 2.95 * mass_msun
        radius_au = radius_km / AU_km        
    elif object_type == "main_sequence" or object_type is None:
        if mass_msun < 1.0:
            radius_au = R_sun_AU * (mass_msun ** 0.9)
        else:
            radius_au = R_sun_AU * (mass_msun ** 0.6)
        radius_au = max(radius_au, 0.0001)
        radius_au = min(radius_au, 0.0465)        
    else:
        raise ValueError(f"Unknown object_type: {object_type}")    
    return radius_au

def gr_radius(masses_msun):
    # Constants
    G_c2_per_msun_au = 1.4766e-8   # GM/c² in AU for 1 M☉ (≈ 2.95 km in AU)    
    # Find the maximum total mass of any pair
    max_total_mass = 0.0
    for i in range(N):
        for j in range(i+1, N):
            total_mass = masses_msun[i] + masses_msun[j]
            if total_mass > max_total_mass:
                max_total_mass = total_mass    
    # Schwarzschild radius for this pair
    r_sch_au = 2.0 * G_c2_per_msun_au * max_total_mass    
    # GR becomes noticeable when v^2/c^2 ≈ 0.01 → r ≈ 100 × (GM/c²)
    # This is 50 × Schwarzschild radius
    r_gr_au = 50.0 * r_sch_au    
    # For main sequence stars, stellar radius is larger than r_gr
    # So Newtonian is fine until physical collision
    return r_gr_au

def _orbit(r_vec, v_vec, m1, m2):
    r = _norm(r_vec)
    v = _norm(v_vec)
    se = 0.5 * v*v - G * (m1 + m2) / r
    be = _mu(m1, m2) * se    
    GM = G*(m1 + m2)
    # Semi-major Axis
    if abs(se) < eps:
        a = np.inf
    else:
        a = -GM / (2 * se)    
    # Orbital Period 
    if se < 0:
        P = 2 * np.pi * np.sqrt(a**3 / GM)
    else:
        P = np.inf    
    # Specific Angular Momentum
    h_vec = _cross(r_vec, v_vec)
    # Eccentricity Vector
    e_vec = (_cross(v_vec, h_vec) / GM) - (r_vec / r)
    e = _norm(e_vec)
    inc = angle_between(h_vec, np.array([0,0,1]))
    return {"a": a, "e": e, "p": P, "h_vec": h_vec, "inc": inc, "E": be}

def _check(sim):    
    particles = list(sim.particles)
    m = [p.m for p in particles]
    r = [np.array([p.x, p.y, p.z]) for p in particles]
    v = [np.array([p.vx, p.vy, p.vz]) for p in particles]    
    # Find bound pairs (negative energy)
    bound_pairs = []
    pair_orbits = {}    
    for i in range(N):
        for j in range(i+1, N):
            dr_vec = r[i] - r[j]
            dv_vec = v[i] - v[j]
            orbit = _orbit(dr_vec, dv_vec, m[i], m[j])
            pair_orbits[(i,j)] = orbit            
            if orbit['E'] < 0:
                bound_pairs.append((i, j))    
    # No binary formed
    if len(bound_pairs) == 0:
        return {
            "bin": False,
            "a": np.nan,
            "e": np.nan,
            "p": np.nan,
            "E": np.nan,
            "hx": np.nan,
            "hy": np.nan,
            "hz": np.nan,
            "inc": np.nan,
            "bin_comp1": np.nan,
            "bin_comp2": np.nan,
            "escapee": np.nan
        }    
    # Take most bound pair (lowest energy)
    bound_pairs.sort(key=lambda p: pair_orbits[p]['E'])
    i, j = bound_pairs[0]
    orbit = pair_orbits[(i, j)]  # FIXED: get orbit dict from pair_orbits    
    # Identify escaped star (the one not in binary)
    escaped = [k for k in range(N) if k not in [i, j]][0]    
    a = orbit['a']
    e = orbit['e']
    p = orbit['p']
    E = orbit['E']
    inc = orbit['inc']
    h_vec = orbit['h_vec']
    # Calculate binary center of mass position and velocity
    M_bin = m[i] + m[j]
    r_cm_bin = (m[i] * r[i] + m[j] * r[j]) / M_bin
    v_cm_bin = (m[i] * v[i] + m[j] * v[j]) / M_bin    
    # Relative position and velocity of escapee relative to binary CM
    r_rel = r[escaped] - r_cm_bin
    v_rel = v[escaped] - v_cm_bin    
    # Binding energy between escapee and binary CM
    mu = M_bin * m[escaped] / (M_bin + m[escaped])
    ke_rel = 0.5 * mu * np.dot(v_rel, v_rel)
    pe_rel = -sim.G * M_bin * m[escaped] / np.linalg.norm(r_rel)
    E_bind_escape = ke_rel + pe_rel
    is_escaped = E_bind_escape > 0     
    return {
        "bin": True if is_escaped else False,
        "a": a,
        "e": e,
        "p": p,
        "E": E,
        "hx": h_vec[0],
        "hy": h_vec[1],
        "hz": h_vec[2],
        "inc": inc,
        "bin_comp1": i,
        "bin_comp2": j,
        "escapee": escaped
    }             

def run_single_simulation(param):
    sim = initial_config(param)
    param['sim_status'] = 'valid'
    try:
        # Attempt the integration
        sim.integrate(sim.exit_max_time)
    except rebound.Encounter as error:
        #print(f"Exception: relativistic treatment required!")
        param['sim_status'] = 'invalid: post-newtonian'
    except rebound.Collision as error:
        #print(f"Collision exception: {error}")
        param['sim_status'] = 'invalid: collision'
    except Exception as e:
        #print(f"Unexpected error: {e}")
        param['sim_status'] = 'invalid: integration_error'    
    return param, sim

def save_simulation_result(param, sim):
    if param['sim_status'] != 'valid':
        return
    E_f = sim.energy()
    E_rel_error = abs((E_f - param['E0']) / param['E0'])
    result = {
        'id': param['id'], 
        'sim_e_err': E_rel_error, 
        'sim_time': sim.t,
        't_enc': param['t_enc'], 
        'r_enc': param['r_enc'], 
        'H': param['H']
    }
    for i in range(N):
        result[f'm{i+1}'] = param['m'][i]
        result[f'xi{i+1}'] = param['r'][i][0]
        result[f'yi{i+1}'] = param['r'][i][1]
        result[f'zi{i+1}'] = param['r'][i][2]
        result[f'bx{i+1}'] = param['b'][i][0]
        result[f'by{i+1}'] = param['b'][i][1]
        result[f'bz{i+1}'] = param['b'][i][2]
        result[f'vxi{i+1}'] = param['v'][i][0]
        result[f'vyi{i+1}'] = param['v'][i][1]
        result[f'vzi{i+1}'] = param['v'][i][2]
        result[f'theta{i+1}'] = param['theta'][i]
        result[f'phi{i+1}'] = param['phi'][i]    
    for i, p in enumerate(sim.particles):
        result[f'xf{i+1}'] = p.x
        result[f'yf{i+1}'] = p.y
        result[f'zf{i+1}'] = p.z
        result[f'vxf{i+1}'] = p.vx
        result[f'vyf{i+1}'] = p.vy
        result[f'vzf{i+1}'] = p.vz
    _state = _check(sim)
    for key, value in _state.items():
        result[key] = value
    df = pd.DataFrame([result])
    df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

def is_valid(m, r, v):
    # check system to be unbound and in-range
    KE = 0.0
    PE = 0.0
    for i in range(N):
        KE += 0.5 * m[i] * _norm(v[i])**2
        for j in range(i+1, N):
            dr = _norm(r[i] - r[j])
            dv = _norm(v[i] - v[j])
            mu = m[i] * m[j] / (m[i] + m[j])
            ke = 0.5 * mu * dv*dv
            pe = -G * (m[i] * m[j]) / dr
            if ke + pe < 0.0:
                return False, np.nan
            PE += pe                    
    hardness = abs(PE/KE)
    if hardness > H[0] and hardness < H[1]:
        return True, hardness
    return False, np.nan

def generate_random_parameters(n_simulations):
    pbar = tqdm(total=n_simulations, desc="Generating parameters")
    parameters = []
    log_min_m = np.log10(mass[0])
    log_max_m = np.log10(mass[1])
    log_min_v = np.log10(velocity[0])
    log_max_v = np.log10(velocity[1])
    log_min_r = np.log10(enc_r[0])
    log_max_r = np.log10(enc_r[1])
    i = 0
    while i < n_simulations:
        # ********************************************************************************* RANDOM GENERATION
        log_m = np.random.uniform(log_min_m, log_max_m, N)
        m = 10**log_m
        log_v = np.random.uniform(log_min_v, log_max_v, N)
        v = 10**log_v
        log_r = np.random.uniform(log_min_r, log_max_r)
        r = 10**log_r
        b = r * np.random.uniform(0.0, 1.0, N)
        cos_theta = np.random.uniform(-1.0, 1.0, N)
        theta = np.arccos(cos_theta)  
        phi = np.random.uniform(0.0, 2*np.pi, N)
        # ********************************************************************************* CHECK VALUES
        index_v_min = np.argmin(v)
        v_min = v[index_v_min]
        d_min = min(A * r, R)
        t_enc = d_min / v_min
        b_vec = []
        d_vec = []        
        v_vec = []
        for j in range(N):
            motion_dir = _dir(theta[j], phi[j])
            perp_vec = rand_perp(motion_dir)
            bj = b[j] * perp_vec
            dj = v[j] * t_enc * motion_dir + bj
            vj = -v[j] * motion_dir
            b_vec.append(bj)
            d_vec.append(dj) 
            v_vec.append(vj)
        sys_is_valid, hardness = is_valid(m, d_vec, v_vec)
        if sys_is_valid:
            i += 1
            pbar.update(1)
            parameters.append({
                'id':str(uuid.uuid4()), 'm': m.copy(), 'r': d_vec.copy(), 'b': b_vec.copy(),
                'v': v_vec.copy(), 'theta': theta.copy(), 'phi': phi.copy(), 
                'r_enc': r, 't_enc': t_enc, 'H': hardness
            })
        else:
            continue  
    pbar.close()      
    return parameters

def generate_single_parameter(args):
    mass_range, vel_range, enc_r_range, A, R, N, eps = args    
    while True:
        # Random generation
        log_m = np.random.uniform(mass_range[0], mass_range[1], N)
        m = 10**log_m
        log_v = np.random.uniform(vel_range[0], vel_range[1], N)
        v = 10**log_v
        log_r = np.random.uniform(enc_r_range[0], enc_r_range[1])
        r = 10**log_r
        b = r * np.random.uniform(0.0, 1.0, N)
        cos_theta = np.random.uniform(-1.0, 1.0, N)
        theta = np.arccos(cos_theta)
        phi = np.random.uniform(0.0, 2*np.pi, N)        
        # Check values
        index_v_min = np.argmin(v)
        v_min = v[index_v_min]
        d_min = min(A * r, R)
        t_enc = d_min / v_min        
        b_vec = []
        d_vec = []
        v_vec = []
        for j in range(N):
            motion_dir = _dir(theta[j], phi[j])
            perp_vec = rand_perp(motion_dir)
            bj = b[j] * perp_vec
            dj = v[j] * t_enc * motion_dir + bj
            vj = -v[j] * motion_dir
            b_vec.append(bj)
            d_vec.append(dj)
            v_vec.append(vj)

        sys_is_valid, hardness = is_valid(m, d_vec, v_vec)       
        if sys_is_valid:
            return {
                'id': str(uuid.uuid4()),
                'm': m.copy(),
                'r': d_vec.copy(),
                'b': b_vec.copy(),
                'v': v_vec.copy(),
                'theta': theta.copy(),
                'phi': phi.copy(),
                'r_enc': r,
                't_enc': t_enc,
                'H': hardness
            }

def generate_random_parameters_parallel(n_simulations):
    # Precompute ranges
    log_min_m = np.log10(mass[0])
    log_max_m = np.log10(mass[1])
    log_min_v = np.log10(velocity[0])
    log_max_v = np.log10(velocity[1])
    log_min_r = np.log10(enc_r[0])
    log_max_r = np.log10(enc_r[1])    
    # Arguments for each worker
    args = [(log_min_m, log_max_m), (log_min_v, log_max_v), (log_min_r, log_max_r), 
            A, R, N, eps]    
    # Create pool
    with Pool(processes=cpu_count()) as pool:
        # Map function to generate n_simulations parameter sets
        results = list(tqdm(
            pool.imap(generate_single_parameter, [args] * n_simulations),
            total=n_simulations,
            desc="Generating parameters"
        ))    
    return results

def initial_config(param):
    m = param['m'] 
    v = param['v'] 
    d = param['r'] 
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.units = ("yr", "AU", "Msun")
    sim.G = G
    sim.exit_min_distance = gr_radius(m)
    sim.exit_max_distance = 1e+7 #200*R
    sim.track_energy_offset = 1
    sim.dt = 0.01
    #sim.ri_ias15.adaptive_mode = 2
    #sim.collision = 'direct'
    #sim.collision_resolve = 'halt'
    #sim.minimum_collision_velocity = 1e-5 
    for i in range(N):
        sim.add(m=m[i], #r=stellar_radius(mass[i]), 
                x=d[i][0], y=d[i][1], z=d[i][2],
                vx=v[i][0], vy=v[i][1], vz=v[i][2])    
    sim.move_to_com()
    sim.exit_max_time = N * param['t_enc']
    param['E0'] = sim.energy()
    return sim

def main():
    global output_file
    for pack in range(n_packs):
        # Before writing, find an available index
        file_name = "test"
        if mode == 2:
            file_name = "dataset"
        file_index = 1
        base_pattern = f"data/"
        output_file = f"{base_pattern}{file_name}_{file_index:03d}.csv"
        while os.path.exists(output_file):
            file_index += 1
            output_file = f"{base_pattern}{file_name}_{file_index:03d}.csv"
        #params = generate_random_parameters(n_sim_per_pack)
        params = generate_random_parameters_parallel(n_sim_per_pack)
        BATCH_SIZE = 1000
        total_batches = (len(params) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Running {len(params)} simulations on {cpu_count()} cores")
        print(f"Total batches: {total_batches}")
        with Pool(processes=cpu_count()) as pool:
            for batch_num, i in enumerate(
                tqdm(range(0, len(params), BATCH_SIZE), desc="Overall Progress", total=total_batches)
            ):
                batch = params[i:i+BATCH_SIZE]
                batch_results = list(tqdm(
                    pool.imap(run_single_simulation, batch),
                    total=len(batch),
                    desc=f"Batch {batch_num + 1}",
                    leave=False
                ))
                for result in batch_results:
                    param, sim = result
                    save_simulation_result(param, sim)
                print(f"✅ Batch {batch_num + 1}/{total_batches} completed")
        print(f"\n🎉 All {len(params)} simulations completed!")

if __name__ == "__main__":    
    mp.freeze_support()  
    if mode in [1,2]:
        main()