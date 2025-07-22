#!/usr/bin/env python3
"""
Data collection script to run multiple AMT3D simulations and collect real performance data
"""

import numpy as np
import time
import os
from amt import AcousticTracker

def collect_simulation_data():
    """Run comprehensive simulations to collect real performance data"""
    print("Starting AMT3D data collection...")
    
    # Create a tracker instance
    tracker = AcousticTracker(room_dim=(5.0, 6.0, 2.4), debug_mode=False)
    
    # Test configurations
    test_configs = [
        {
            'name': 'single_target_linear',
            'description': 'Single target linear movement',
            'targets': [{'position': [1.0, 1.0, 1.2], 'velocity': [0.2, 0.15, 0.05]}],
            'duration': 8.0,
            'steps': 40,
            'noise_snr': 25
        },
        {
            'name': 'single_target_circular',
            'description': 'Single target circular movement',
            'targets': [{'position': [2.5, 3.0, 1.2], 'velocity': [0.0, 0.0, 0.0]}],
            'duration': 10.0,
            'steps': 50,
            'noise_snr': 25,
            'movement_override': 'circular'
        },
        {
            'name': 'single_target_zigzag',
            'description': 'Single target zigzag movement',
            'targets': [{'position': [1.0, 1.0, 1.0], 'velocity': [0.2, 0.3, 0.15]}],
            'duration': 8.0,
            'steps': 40,
            'noise_snr': 25,
            'movement_override': 'zigzag'
        }
    ]
    
    # SNR sensitivity test
    snr_levels = [10, 15, 20, 25, 30]
    
    for config in test_configs:
        print(f"\n--- Running simulation: {config['description']} ---")
        
        # Reset tracker for each test
        tracker.targets = []
        tracker.tracking_data = {}
        tracker.kalman_filters = {}
        tracker._current_movement_pattern = config['name']
        
        # Add targets
        for target_config in config['targets']:
            tracker.add_target(
                position=target_config['position'],
                velocity=target_config['velocity'],
                name=f"Target_{config['name']}"
            )
        
        # Set up special movement patterns
        if config.get('movement_override') == 'circular':
            setup_circular_movement(tracker)
        elif config.get('movement_override') == 'zigzag':
            setup_zigzag_movement(tracker)
        
        # Run simulation
        start_time = time.time()
        tracking_data = tracker.run_tracking(
            duration=config['duration'],
            steps=config['steps'],
            noise_snr=config['noise_snr']
        )
        end_time = time.time()
        
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        print(f"Targets tracked: {len(tracking_data)}")
        
        # Test SNR sensitivity for linear movement only
        if config['name'] == 'single_target_linear':
            print("\n--- Testing SNR sensitivity ---")
            for snr in snr_levels:
                print(f"Testing SNR: {snr} dB")
                
                # Reset tracker
                tracker.targets = []
                tracker.tracking_data = {}
                tracker.kalman_filters = {}
                tracker._current_movement_pattern = f"linear_snr_{snr}"
                
                # Add single target
                tracker.add_target(
                    position=[1.5, 2.0, 1.2],
                    velocity=[0.15, 0.1, 0.03],
                    name=f"Target_SNR_{snr}"
                )
                
                # Run shorter simulation for SNR test
                tracker.run_tracking(duration=5.0, steps=25, noise_snr=snr)
    
    print("\n--- Data collection completed ---")
    print("Check the 'simulation_data' directory for exported CSV files")

def setup_circular_movement(tracker):
    """Set up circular movement pattern"""
    original_update = tracker.update_targets
    
    def circular_update(dt):
        for target in tracker.targets:
            t = len(target['history']) * dt
            # Circular motion in X-Y plane with Z oscillation
            center = np.array([2.5, 3.0, 1.2])
            radius = 1.0
            new_pos = center + np.array([
                radius * np.cos(t * 0.5),
                radius * np.sin(t * 0.5),
                0.3 * np.sin(t * 1.5)  # Vertical oscillation
            ])
            target['position'] = new_pos
            target['history'].append(new_pos.copy())
            tracker.tracking_data[target['id']]['true_positions'].append(new_pos.copy())
    
    tracker.update_targets = circular_update

def setup_zigzag_movement(tracker):
    """Set up zigzag movement pattern"""
    original_update = tracker.update_targets
    
    def zigzag_update(dt):
        for target in tracker.targets:
            t = len(target['history']) * dt
            # Zigzag pattern
            base_velocity = np.array([0.2, 0.3, 0.1])
            # Add zigzag components
            zigzag_vel = base_velocity + np.array([
                0.2 * np.sin(t * 2.0),    # X zigzag
                0.15 * np.cos(t * 1.5),   # Y zigzag  
                0.1 * np.sin(t * 3.0)     # Z zigzag
            ])
            new_pos = target['position'] + zigzag_vel * dt
            
            # Boundary checks
            new_pos = np.clip(new_pos, [0.5, 0.5, 0.3], [4.5, 5.5, 2.1])
            
            target['position'] = new_pos
            target['history'].append(new_pos.copy())
            tracker.tracking_data[target['id']]['true_positions'].append(new_pos.copy())
    
    tracker.update_targets = zigzag_update

if __name__ == "__main__":
    collect_simulation_data()