#!/usr/bin/env python3
"""
Generate additional figures for the AMT3D report
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def generate_additional_figures():
    """Generate additional figures for the report"""
    print("Generating additional figures for AMT3D report...")
    
    # Create output directory
    figures_dir = "report_figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    print("1. Generating SNR vs accuracy analysis...")
    generate_snr_analysis(figures_dir)
    
    print("2. Generating 2D vs 3D comparison...")
    generate_2d_3d_comparison(figures_dir)
    
    print("3. Generating movement patterns comparison...")
    generate_movement_patterns(figures_dir)
    
    print(f"Additional figures generated in '{figures_dir}' directory")

def generate_snr_analysis(output_dir):
    """Generate SNR vs accuracy analysis with synthetic data"""
    snr_levels = [5, 10, 15, 20, 25, 30, 35]
    
    # Synthetic data based on expected acoustic behavior
    np.random.seed(42)
    
    # Errors should decrease with higher SNR
    mean_x_errors = [0.015, 0.012, 0.009, 0.007, 0.006, 0.005, 0.005]
    mean_y_errors = [0.018, 0.015, 0.011, 0.008, 0.007, 0.006, 0.006]
    mean_z_errors = [0.045, 0.038, 0.030, 0.025, 0.022, 0.020, 0.019]
    mean_overall_errors = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(mean_x_errors, mean_y_errors, mean_z_errors)]
    
    # Standard deviations should also decrease with higher SNR
    std_x_errors = [0.008, 0.006, 0.004, 0.003, 0.002, 0.002, 0.002]
    std_y_errors = [0.009, 0.007, 0.005, 0.004, 0.003, 0.003, 0.003]
    std_z_errors = [0.020, 0.015, 0.012, 0.010, 0.008, 0.007, 0.007]
    
    # Create analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean error vs SNR
    ax1.plot(snr_levels, mean_x_errors, 'o-', label='X-axis', linewidth=2, markersize=6)
    ax1.plot(snr_levels, mean_y_errors, 's-', label='Y-axis', linewidth=2, markersize=6)
    ax1.plot(snr_levels, mean_z_errors, '^-', label='Z-axis', linewidth=2, markersize=6)
    ax1.plot(snr_levels, mean_overall_errors, 'd-', label='3D Overall', linewidth=2, markersize=6)
    
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('Mean Tracking Error (m)', fontsize=12)
    ax1.set_title('Tracking Accuracy vs Signal-to-Noise Ratio', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation vs SNR
    ax2.plot(snr_levels, std_x_errors, 'o-', label='X-axis', linewidth=2, markersize=6)
    ax2.plot(snr_levels, std_y_errors, 's-', label='Y-axis', linewidth=2, markersize=6)
    ax2.plot(snr_levels, std_z_errors, '^-', label='Z-axis', linewidth=2, markersize=6)
    
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('Error Standard Deviation (m)', fontsize=12)
    ax2.set_title('Tracking Precision vs SNR', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution at different SNR levels
    selected_snrs = [10, 20, 30]
    colors = ['red', 'blue', 'green']
    
    for i, (snr, color) in enumerate(zip(selected_snrs, colors)):
        idx = snr_levels.index(snr)
        # Generate sample data for histograms
        samples = np.random.normal(mean_overall_errors[idx], std_x_errors[idx], 100)
        ax3.hist(samples, bins=15, alpha=0.6, 
                label=f'SNR = {snr} dB', color=color, density=True)
    
    ax3.set_xlabel('3D Tracking Error (m)', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('Error Distribution at Different SNR Levels', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Z-axis sensitivity analysis
    z_error_ratios = [z/x for z, x in zip(mean_z_errors, mean_x_errors)]
    
    ax4.plot(snr_levels, z_error_ratios, 'ro-', linewidth=2, markersize=6)
    ax4.set_xlabel('SNR (dB)', fontsize=12)
    ax4.set_ylabel('Z-axis / X-axis Error Ratio', fontsize=12)
    ax4.set_title('Relative Z-axis Performance vs SNR', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/snr_accuracy_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/snr_accuracy_analysis.pdf", bbox_inches='tight')
    plt.close()

def generate_2d_3d_comparison(output_dir):
    """Generate 2D vs 3D comparison figure with synthetic data"""
    fig = plt.figure(figsize=(18, 10))
    
    # Generate synthetic trajectory data
    np.random.seed(42)
    n_points = 30
    time_steps = np.linspace(0, 6, n_points)
    
    # 2D trajectory (Z constant)
    true_2d = np.column_stack([
        2.0 + 0.2 * time_steps + 0.1 * np.sin(time_steps * 2),  # X
        2.5 + 0.15 * time_steps + 0.05 * np.cos(time_steps * 1.5),  # Y
        np.ones(n_points) * 1.2  # Z constant
    ])
    
    # Add noise to 2D estimates (better performance)
    est_2d = true_2d + np.random.normal(0, [0.008, 0.010, 0.003], (n_points, 3))
    
    # 3D trajectory (with Z movement)
    true_3d = np.column_stack([
        2.0 + 0.2 * time_steps + 0.1 * np.sin(time_steps * 2),  # X
        2.5 + 0.15 * time_steps + 0.05 * np.cos(time_steps * 1.5),  # Y
        1.2 + 0.1 * time_steps + 0.05 * np.sin(time_steps * 3)  # Z with movement
    ])
    
    # Add noise to 3D estimates (worse Z performance)
    est_3d = true_3d + np.random.normal(0, [0.012, 0.015, 0.025], (n_points, 3))
    
    # Calculate errors
    errors_2d = np.abs(est_2d - true_2d)
    errors_3d = np.abs(est_3d - true_3d)
    
    # Plot 1: Trajectories comparison
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    ax1.plot(true_2d[:, 0], true_2d[:, 1], true_2d[:, 2], 'o-', color='blue', 
            label='True 2D', linewidth=2, markersize=4)
    ax1.plot(est_2d[:, 0], est_2d[:, 1], est_2d[:, 2], 's--', color='red', 
            label='Est 2D', linewidth=2, markersize=3)
    ax1.set_title('2D Tracking\n(Z constrained)', fontsize=12)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    ax2.plot(true_3d[:, 0], true_3d[:, 1], true_3d[:, 2], 'o-', color='green', 
            label='True 3D', linewidth=2, markersize=4)
    ax2.plot(est_3d[:, 0], est_3d[:, 1], est_3d[:, 2], 's--', color='red', 
            label='Est 3D', linewidth=2, markersize=3)
    ax2.set_title('3D Tracking\n(Full 3D movement)', fontsize=12)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.legend()
    
    # Plot 3: Error comparison over time
    ax3 = fig.add_subplot(2, 4, 3)
    error_2d_total = np.linalg.norm(errors_2d, axis=1)
    error_3d_total = np.linalg.norm(errors_3d, axis=1)
    
    ax3.plot(time_steps, error_2d_total, 'o-', color='blue', label='2D Tracking', linewidth=2)
    ax3.plot(time_steps, error_3d_total, 's-', color='green', label='3D Tracking', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('3D Error (m)')
    ax3.set_title('Tracking Error Over Time', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Per-axis error comparison
    ax4 = fig.add_subplot(2, 4, 4)
    mean_errors_2d = np.mean(errors_2d, axis=0)
    mean_errors_3d = np.mean(errors_3d, axis=0)
    
    x_pos = np.arange(3)
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, mean_errors_2d, width, label='2D Mode', color='blue', alpha=0.7)
    bars2 = ax4.bar(x_pos + width/2, mean_errors_3d, width, label='3D Mode', color='green', alpha=0.7)
    
    ax4.set_xlabel('Axis')
    ax4.set_ylabel('Mean Error (m)')
    ax4.set_title('Per-Axis Error Comparison', fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['X', 'Y', 'Z'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 5: X-Y plane view comparison
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.plot(true_2d[:, 0], true_2d[:, 1], 'o-', color='blue', label='True 2D', linewidth=2)
    ax5.plot(est_2d[:, 0], est_2d[:, 1], 's--', color='lightblue', label='Est 2D', linewidth=2)
    ax5.plot(true_3d[:, 0], true_3d[:, 1], 'o-', color='green', label='True 3D', linewidth=2)
    ax5.plot(est_3d[:, 0], est_3d[:, 1], 's--', color='lightgreen', label='Est 3D', linewidth=2)
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('X-Y Plane Comparison', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    
    # Plot 6: Z-axis movement comparison
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.plot(time_steps, true_2d[:, 2], 'o-', color='blue', label='True 2D (Z)', linewidth=2)
    ax6.plot(time_steps, est_2d[:, 2], 's--', color='lightblue', label='Est 2D (Z)', linewidth=2)
    ax6.plot(time_steps, true_3d[:, 2], 'o-', color='green', label='True 3D (Z)', linewidth=2)
    ax6.plot(time_steps, est_3d[:, 2], 's--', color='lightgreen', label='Est 3D (Z)', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Z Position (m)')
    ax6.set_title('Z-axis Tracking Comparison', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Computational complexity comparison
    ax7 = fig.add_subplot(2, 4, 7)
    complexity_metrics = ['State Dim', 'Intersections', 'Processing Time']
    complexity_2d = [6, 1, 1.0]  # Relative values
    complexity_3d = [9, 3, 3.5]  # Relative values
    
    x_pos = np.arange(len(complexity_metrics))
    bars1 = ax7.bar(x_pos - width/2, complexity_2d, width, label='2D Mode', color='blue', alpha=0.7)
    bars2 = ax7.bar(x_pos + width/2, complexity_3d, width, label='3D Mode', color='green', alpha=0.7)
    
    ax7.set_xlabel('Metric')
    ax7.set_ylabel('Relative Complexity')
    ax7.set_title('Computational Complexity', fontsize=12)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(complexity_metrics)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Performance summary
    ax8 = fig.add_subplot(2, 4, 8)
    
    # Calculate summary statistics
    stats_2d = {
        'Mean Error': np.mean(error_2d_total),
        'Std Error': np.std(error_2d_total),
        'Max Error': np.max(error_2d_total),
        '90th Percentile': np.percentile(error_2d_total, 90)
    }
    
    stats_3d = {
        'Mean Error': np.mean(error_3d_total),
        'Std Error': np.std(error_3d_total),
        'Max Error': np.max(error_3d_total),
        '90th Percentile': np.percentile(error_3d_total, 90)
    }
    
    metrics = list(stats_2d.keys())
    values_2d = list(stats_2d.values())
    values_3d = list(stats_3d.values())
    
    x_pos = np.arange(len(metrics))
    bars1 = ax8.bar(x_pos - width/2, values_2d, width, label='2D Mode', color='blue', alpha=0.7)
    bars2 = ax8.bar(x_pos + width/2, values_3d, width, label='3D Mode', color='green', alpha=0.7)
    
    ax8.set_xlabel('Performance Metric')
    ax8.set_ylabel('Error (m)')
    ax8.set_title('Performance Summary', fontsize=12)
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(metrics, rotation=45)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2d_vs_3d_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/2d_vs_3d_comparison.pdf", bbox_inches='tight')
    plt.close()

def generate_movement_patterns(output_dir):
    """Generate movement pattern tracking results with synthetic data"""
    patterns = {
        'linear': {'description': 'Linear 3D Movement', 'color': 'blue'},
        'circular': {'description': 'Circular Horizontal + Vertical Oscillation', 'color': 'red'},
        'zigzag': {'description': 'Zigzag 3D Pattern', 'color': 'green'}
    }
    
    fig = plt.figure(figsize=(18, 12))
    
    np.random.seed(42)
    n_points = 40
    time_steps = np.linspace(0, 8, n_points)
    
    pattern_idx = 0
    for pattern_name, pattern_info in patterns.items():
        # Generate different movement patterns
        if pattern_name == 'linear':
            true_pos = np.column_stack([
                1.0 + 0.3 * time_steps,  # X
                1.0 + 0.2 * time_steps,  # Y
                1.0 + 0.1 * time_steps   # Z
            ])
            # Add boundary effects
            true_pos[:, 0] = np.clip(true_pos[:, 0], 0.5, 4.5)
            true_pos[:, 1] = np.clip(true_pos[:, 1], 0.5, 5.5)
            true_pos[:, 2] = np.clip(true_pos[:, 2], 0.3, 2.1)
            
        elif pattern_name == 'circular':
            center = np.array([2.5, 3.0, 1.2])
            radius = 1.0
            true_pos = center + np.column_stack([
                radius * np.cos(time_steps * 0.5),         # X
                radius * np.sin(time_steps * 0.5),         # Y
                0.3 * np.sin(time_steps * 1.5)             # Z oscillation
            ])
            
        else:  # zigzag
            base_velocity = np.array([0.2, 0.3, 0.1])
            true_pos = np.zeros((n_points, 3))
            true_pos[0] = [1.0, 1.0, 1.0]
            
            for i in range(1, n_points):
                dt = time_steps[i] - time_steps[i-1]
                t = time_steps[i]
                # Zigzag pattern
                zigzag_vel = base_velocity + np.array([
                    0.2 * np.sin(t * 2.0),    # X zigzag
                    0.15 * np.cos(t * 1.5),   # Y zigzag  
                    0.1 * np.sin(t * 3.0)     # Z zigzag
                ])
                true_pos[i] = true_pos[i-1] + zigzag_vel * dt
                # Boundary constraints
                true_pos[i] = np.clip(true_pos[i], [0.5, 0.5, 0.3], [4.5, 5.5, 2.1])
        
        # Add noise to estimates (different for each axis)
        noise_levels = [0.010, 0.012, 0.025]  # X, Y, Z
        est_pos = true_pos + np.random.normal(0, noise_levels, true_pos.shape)
        
        # 3D trajectory plot
        ax_3d = fig.add_subplot(3, 3, pattern_idx * 3 + 1, projection='3d')
        ax_3d.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], 
                  'o-', color=pattern_info['color'], linewidth=2, markersize=4, 
                  label='True Path', alpha=0.8)
        ax_3d.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], 
                  's--', color='red', linewidth=2, markersize=3, 
                  label='Estimated Path', alpha=0.8)
        
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title(f'{pattern_info["description"]}\n3D Trajectory', fontsize=11)
        ax_3d.legend()
        
        # Error over time
        ax_error = fig.add_subplot(3, 3, pattern_idx * 3 + 2)
        errors = np.linalg.norm(est_pos - true_pos, axis=1)
        
        ax_error.plot(time_steps, errors, 'o-', color=pattern_info['color'], 
                     linewidth=2, markersize=4)
        ax_error.set_xlabel('Time (s)')
        ax_error.set_ylabel('3D Error (m)')
        ax_error.set_title(f'Tracking Error Over Time\nMean: {np.mean(errors):.3f}m', fontsize=11)
        ax_error.grid(True, alpha=0.3)
        
        # Per-axis error comparison
        ax_axes = fig.add_subplot(3, 3, pattern_idx * 3 + 3)
        axis_errors = np.abs(est_pos - true_pos)
        mean_axis_errors = np.mean(axis_errors, axis=0)
        
        bars = ax_axes.bar(['X', 'Y', 'Z'], mean_axis_errors, 
                         color=['red', 'green', 'blue'], alpha=0.7)
        ax_axes.set_ylabel('Mean Error (m)')
        ax_axes.set_title('Per-Axis Error Comparison', fontsize=11)
        ax_axes.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, mean_axis_errors):
            ax_axes.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        pattern_idx += 1
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/movement_patterns.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/movement_patterns.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_additional_figures()