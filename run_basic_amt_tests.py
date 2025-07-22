#!/usr/bin/env python3
"""
Basic AMT3D tests to generate real simulation data quickly
"""

from amt import AcousticTracker
import time

def run_basic_tests():
    print("Running basic AMT3D tests...")
    
    tracker = AcousticTracker()
    print(f"Session ID: {tracker.session_id}")
    
    # Test 1: Linear movement with different SNR levels
    snr_levels = [10, 20, 30]
    
    for snr in snr_levels:
        print(f"\nTesting SNR {snr} dB...")
        
        # Clear previous targets
        tracker.targets = []
        tracker.tracking_data = {}
        tracker.kalman_filters = {}
        
        # Add a target for linear movement
        tracker.add_target(position=[2.5, 2.0, 1.2], velocity=[0.0, 0.4, 0.0], name=f"SNR_{snr}_Target")
        
        # Run a short test
        try:
            result = tracker._run_single_test(
                steps=10, 
                duration=2.0, 
                noise_snr=snr, 
                test_name=f"linear_snr_{snr}"
            )
            print(f"  Completed SNR {snr} test: {result['is_valid']}")
        except Exception as e:
            print(f"  Error in SNR {snr} test: {e}")
    
    # Test 2: Different movement patterns at fixed SNR
    patterns = [
        {'name': 'linear_x', 'pos': [1.0, 3.0, 1.2], 'vel': [0.3, 0.0, 0.0]},
        {'name': 'linear_y', 'pos': [2.5, 1.0, 1.2], 'vel': [0.0, 0.3, 0.0]},
        {'name': 'linear_z', 'pos': [2.5, 3.0, 0.8], 'vel': [0.0, 0.0, 0.2]}
    ]
    
    for pattern in patterns:
        print(f"\nTesting {pattern['name']} movement...")
        
        # Clear previous targets
        tracker.targets = []
        tracker.tracking_data = {}
        tracker.kalman_filters = {}
        
        # Add target with specific movement
        tracker.add_target(position=pattern['pos'], velocity=pattern['vel'], name=f"{pattern['name']}_Target")
        
        try:
            result = tracker._run_single_test(
                steps=10,
                duration=2.0,
                noise_snr=15,
                test_name=f"movement_{pattern['name']}"
            )
            print(f"  Completed {pattern['name']} test: {result['is_valid']}")
        except Exception as e:
            print(f"  Error in {pattern['name']} test: {e}")
    
    print(f"\nBasic tests completed! Data exported to simulation_data/{tracker.session_id}_*")
    return tracker.session_id

if __name__ == "__main__":
    session_id = run_basic_tests()