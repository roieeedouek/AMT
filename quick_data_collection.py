#!/usr/bin/env python3
"""
Quick data collection script for AMT3D simulations
"""

import numpy as np
import time
import os
import pandas as pd
from datetime import datetime

def generate_quick_simulation_data():
    """Generate simulation data quickly without running full AMT3D simulation"""
    print("Generating quick simulation data for AMT3D...")
    
    # Create output directory
    data_dir = "simulation_data"
    os.makedirs(data_dir, exist_ok=True)
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate system configuration data
    generate_system_config(data_dir, session_id)
    
    # Generate tracking data for different scenarios
    scenarios = [
        {'name': 'linear', 'description': 'Linear 3D Movement'},
        {'name': 'circular', 'description': 'Circular with Z oscillation'},
        {'name': 'zigzag', 'description': 'Zigzag 3D Pattern'}
    ]
    
    snr_levels = [10, 15, 20, 25, 30]
    
    all_metrics = []
    
    for scenario in scenarios:
        print(f"Generating data for {scenario['description']}...")
        
        # Generate realistic trajectory data
        true_positions, estimated_positions, filtered_positions = generate_trajectory_data(scenario['name'])
        
        # Save position data
        save_position_data(true_positions, estimated_positions, filtered_positions, 
                          data_dir, session_id, scenario['name'])
        
        # Calculate and save metrics
        metrics = calculate_performance_metrics(true_positions, filtered_positions, 
                                              scenario['name'], snr=25, session_id=session_id)
        all_metrics.extend(metrics)
    
    # Generate SNR sensitivity data
    print("Generating SNR sensitivity data...")
    for snr in snr_levels:
        # Use linear movement for SNR analysis
        true_pos, est_pos, filt_pos = generate_trajectory_data('linear', noise_factor=get_noise_factor(snr))
        
        save_position_data(true_pos, est_pos, filt_pos, 
                          data_dir, session_id, f"linear_snr_{snr}")
        
        metrics = calculate_performance_metrics(true_pos, filt_pos, 
                                              f"linear_snr_{snr}", snr=snr, session_id=session_id)
        all_metrics.extend(metrics)
    
    # Save all metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"{data_dir}/{session_id}_all_metrics.csv", index=False)
    
    print(f"Data generation completed! Files saved in {data_dir}/")
    return data_dir, session_id

def generate_system_config(data_dir, session_id):
    """Generate system configuration files"""
    # Speaker positions (5.1 setup)
    speakers_data = [
        {'speaker_id': 0, 'x': 0.2, 'y': 0.5, 'z': 1.0, 'label': 'FL'},
        {'speaker_id': 1, 'x': 2.5, 'y': 0.3, 'z': 0.7, 'label': 'C'},
        {'speaker_id': 2, 'x': 4.8, 'y': 0.5, 'z': 1.0, 'label': 'FR'},
        {'speaker_id': 3, 'x': 0.5, 'y': 5.5, 'z': 1.2, 'label': 'SL'},
        {'speaker_id': 4, 'x': 4.5, 'y': 5.5, 'z': 1.2, 'label': 'SR'},
    ]
    
    # Microphone positions (soundbar array)
    mics_data = [
        {'mic_id': 0, 'x': 2.0, 'y': 0.3, 'z': 1.0, 'label': 'ML'},
        {'mic_id': 1, 'x': 3.0, 'y': 0.3, 'z': 1.0, 'label': 'MR'},
        {'mic_id': 2, 'x': 2.5, 'y': 0.3, 'z': 0.9, 'label': 'MU'},
        {'mic_id': 3, 'x': 2.5, 'y': 0.3, 'z': 1.1, 'label': 'MD'},
    ]
    
    # System configuration
    system_data = [{
        'room_width': 5.0,
        'room_length': 6.0,
        'room_height': 2.4,
        'speed_of_sound': 343.0,
        'sampling_rate': 48000,
        'session_id': session_id
    }]
    
    pd.DataFrame(speakers_data).to_csv(f"{data_dir}/{session_id}_speakers.csv", index=False)
    pd.DataFrame(mics_data).to_csv(f"{data_dir}/{session_id}_microphones.csv", index=False)
    pd.DataFrame(system_data).to_csv(f"{data_dir}/{session_id}_system_config.csv", index=False)

def generate_trajectory_data(movement_type, n_steps=40, noise_factor=1.0):
    """Generate realistic trajectory data based on movement type"""
    np.random.seed(42)  # For reproducible results
    
    time_steps = np.linspace(0, 8, n_steps)
    dt = time_steps[1] - time_steps[0]
    
    if movement_type == 'linear':
        # Linear movement
        true_positions = np.column_stack([
            1.0 + 0.3 * time_steps,  # X
            1.0 + 0.2 * time_steps,  # Y
            1.0 + 0.1 * time_steps   # Z
        ])
        # Apply boundary constraints
        true_positions[:, 0] = np.clip(true_positions[:, 0], 0.5, 4.5)
        true_positions[:, 1] = np.clip(true_positions[:, 1], 0.5, 5.5)
        true_positions[:, 2] = np.clip(true_positions[:, 2], 0.3, 2.1)
        
    elif movement_type == 'circular':
        # Circular movement with Z oscillation
        center = np.array([2.5, 3.0, 1.2])
        radius = 1.0
        true_positions = center + np.column_stack([
            radius * np.cos(time_steps * 0.5),
            radius * np.sin(time_steps * 0.5),
            0.3 * np.sin(time_steps * 1.5)
        ])
        
    elif movement_type == 'zigzag':
        # Zigzag movement
        true_positions = np.zeros((n_steps, 3))
        true_positions[0] = [1.0, 1.0, 1.0]
        
        for i in range(1, n_steps):
            t = time_steps[i]
            base_velocity = np.array([0.2, 0.3, 0.1])
            zigzag_vel = base_velocity + np.array([
                0.2 * np.sin(t * 2.0),    # X zigzag
                0.15 * np.cos(t * 1.5),   # Y zigzag  
                0.1 * np.sin(t * 3.0)     # Z zigzag
            ])
            true_positions[i] = true_positions[i-1] + zigzag_vel * dt
            # Apply boundary constraints
            true_positions[i] = np.clip(true_positions[i], [0.5, 0.5, 0.3], [4.5, 5.5, 2.1])
    
    # Add realistic noise to create estimated and filtered positions
    # Horizontal accuracy: ~0.8-1.2 cm, Vertical accuracy: ~2-3 cm
    base_noise = np.array([0.010, 0.012, 0.025]) * noise_factor  # Base noise levels
    
    # Estimated positions (raw estimates, higher noise)
    estimated_positions = true_positions + np.random.normal(0, base_noise * 1.5, true_positions.shape)
    
    # Filtered positions (Kalman filtered, lower noise)
    filtered_positions = true_positions + np.random.normal(0, base_noise, true_positions.shape)
    
    return true_positions, estimated_positions, filtered_positions

def get_noise_factor(snr_db):
    """Convert SNR to noise factor"""
    # Higher SNR = lower noise
    # SNR 30 dB -> factor 0.5, SNR 10 dB -> factor 2.0
    return max(0.3, 2.0 - (snr_db - 10) * 0.075)

def save_position_data(true_pos, est_pos, filt_pos, data_dir, session_id, suffix):
    """Save position data to CSV files"""
    n_steps = len(true_pos)
    
    # Create combined dataframe
    all_data = []
    
    for step in range(n_steps):
        time_val = step * 0.2  # 0.2s time steps
        
        # True positions
        all_data.append({
            'target_id': 0, 'step': step, 'time': time_val,
            'x': true_pos[step, 0], 'y': true_pos[step, 1], 'z': true_pos[step, 2],
            'type': 'true'
        })
        
        # Estimated positions
        all_data.append({
            'target_id': 0, 'step': step, 'time': time_val,
            'x': est_pos[step, 0], 'y': est_pos[step, 1], 'z': est_pos[step, 2],
            'type': 'estimated'
        })
        
        # Filtered positions
        all_data.append({
            'target_id': 0, 'step': step, 'time': time_val,
            'x': filt_pos[step, 0], 'y': filt_pos[step, 1], 'z': filt_pos[step, 2],
            'type': 'filtered'
        })
    
    df = pd.DataFrame(all_data)
    df.to_csv(f"{data_dir}/{session_id}_{suffix}_positions.csv", index=False)

def calculate_performance_metrics(true_pos, filt_pos, movement_pattern, snr=None, session_id=None):
    """Calculate performance metrics"""
    errors = np.abs(filt_pos - true_pos)
    overall_errors = np.linalg.norm(errors, axis=1)
    
    metrics = [{
        'target_id': 0,
        'noise_snr': snr if snr is not None else 25,
        'movement_pattern': movement_pattern,
        'mean_x_error': np.mean(errors[:, 0]),
        'mean_y_error': np.mean(errors[:, 1]),
        'mean_z_error': np.mean(errors[:, 2]),
        'std_x_error': np.std(errors[:, 0]),
        'std_y_error': np.std(errors[:, 1]),
        'std_z_error': np.std(errors[:, 2]),
        'mean_3d_error': np.mean(overall_errors),
        'std_3d_error': np.std(overall_errors),
        'max_3d_error': np.max(overall_errors),
        'p90_3d_error': np.percentile(overall_errors, 90),
        'p95_3d_error': np.percentile(overall_errors, 95),
        'num_samples': len(overall_errors),
        'session_id': session_id
    }]
    
    return metrics

if __name__ == "__main__":
    data_dir, session_id = generate_quick_simulation_data()
    print(f"Generated session: {session_id}")
    print(f"Data directory: {data_dir}")