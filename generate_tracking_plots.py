#!/usr/bin/env python3
"""
Generate comprehensive tracking plots from real simulation data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from mpl_toolkits.mplot3d import Axes3D

def generate_tracking_plots():
    """Generate comprehensive tracking visualization plots from CSV data"""
    print("Generating tracking plots from real simulation data...")
    
    # Find the latest session data
    data_dir = "simulation_data"
    if not os.path.exists(data_dir):
        print("No simulation data found. Please run data collection first.")
        return
    
    # Get latest session
    session_files = glob.glob(f"{data_dir}/*_system_config.csv")
    if not session_files:
        print("No session data found.")
        return
    
    latest_session = sorted(session_files)[-1]
    session_id = os.path.basename(latest_session).split('_system_config.csv')[0]
    print(f"Using session: {session_id}")
    
    # Create output directory
    plots_dir = "tracking_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate different types of tracking plots
    print("1. Generating 3D trajectory comparison plots...")
    generate_3d_trajectory_plots(data_dir, session_id, plots_dir)
    
    print("2. Generating tracking error analysis plots...")
    generate_error_analysis_plots(data_dir, session_id, plots_dir)
    
    print("3. Generating time-series tracking plots...")
    generate_time_series_plots(data_dir, session_id, plots_dir)
    
    print("4. Generating movement pattern comparison...")
    generate_pattern_comparison_plots(data_dir, session_id, plots_dir)
    
    print("5. Generating SNR performance plots...")
    generate_snr_performance_plots(data_dir, session_id, plots_dir)
    
    print(f"All tracking plots generated in '{plots_dir}' directory")

def load_position_data(data_dir, session_id, pattern):
    """Load position data for a specific pattern"""
    pos_file = f"{data_dir}/{session_id}_{pattern}_positions.csv"
    if not os.path.exists(pos_file):
        print(f"Position file not found: {pos_file}")
        return None
    
    df = pd.read_csv(pos_file)
    
    # Separate different types of positions
    true_data = df[df['type'] == 'true'].sort_values('step')
    estimated_data = df[df['type'] == 'estimated'].sort_values('step')
    filtered_data = df[df['type'] == 'filtered'].sort_values('step')
    
    result = {}
    if len(true_data) > 0:
        result['true'] = {
            'positions': true_data[['x', 'y', 'z']].values,
            'time': true_data['time'].values
        }
    
    if len(estimated_data) > 0:
        result['estimated'] = {
            'positions': estimated_data[['x', 'y', 'z']].values,
            'time': estimated_data['time'].values
        }
    
    if len(filtered_data) > 0:
        result['filtered'] = {
            'positions': filtered_data[['x', 'y', 'z']].values,
            'time': filtered_data['time'].values
        }
    
    return result

def generate_3d_trajectory_plots(data_dir, session_id, output_dir):
    """Generate 3D trajectory comparison plots"""
    patterns = ['linear', 'circular', 'zigzag']
    pattern_names = ['Linear Movement', 'Circular Movement', 'Zigzag Movement']
    colors = ['blue', 'red', 'green']
    
    fig = plt.figure(figsize=(18, 6))
    
    for i, (pattern, name, color) in enumerate(zip(patterns, pattern_names, colors)):
        data = load_position_data(data_dir, session_id, pattern)
        if data is None or 'true' not in data or 'filtered' not in data:
            continue
        
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        true_pos = data['true']['positions']
        filtered_pos = data['filtered']['positions']
        
        # Plot true trajectory
        ax.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], 
                'o-', color=color, linewidth=3, markersize=6, 
                label='True Trajectory', alpha=0.8)
        
        # Plot filtered trajectory
        ax.plot(filtered_pos[:, 0], filtered_pos[:, 1], filtered_pos[:, 2], 
                's--', color='red', linewidth=2, markersize=4, 
                label='Tracked Trajectory', alpha=0.8)
        
        # Mark start and end points
        ax.scatter(*true_pos[0], color='green', s=200, marker='o', 
                  label='Start', alpha=1.0, edgecolors='black', linewidth=2)
        ax.scatter(*true_pos[-1], color='orange', s=200, marker='s', 
                  label='End', alpha=1.0, edgecolors='black', linewidth=2)
        
        # Calculate and display tracking error statistics
        errors = np.linalg.norm(filtered_pos - true_pos, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(f'{name}\nMean Error: {mean_error:.2f}cm, Max: {max_error:.2f}cm', 
                    fontsize=14)
        ax.legend(loc='upper right')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,0.5])  # Make Z-axis shorter for better visualization
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3d_trajectory_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/3d_trajectory_comparison.pdf", bbox_inches='tight')
    plt.close()

def generate_error_analysis_plots(data_dir, session_id, output_dir):
    """Generate detailed error analysis plots"""
    patterns = ['linear', 'circular', 'zigzag']
    pattern_names = ['Linear', 'Circular', 'Zigzag']
    colors = ['blue', 'red', 'green']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    all_errors_x = []
    all_errors_y = []
    all_errors_z = []
    all_errors_3d = []
    pattern_labels = []
    
    for pattern, name, color in zip(patterns, pattern_names, colors):
        data = load_position_data(data_dir, session_id, pattern)
        if data is None or 'true' not in data or 'filtered' not in data:
            continue
        
        true_pos = data['true']['positions']
        filtered_pos = data['filtered']['positions']
        time_steps = data['true']['time']
        
        # Calculate errors
        errors = np.abs(filtered_pos - true_pos)
        errors_3d = np.linalg.norm(errors, axis=1)
        
        # Store for aggregated analysis
        all_errors_x.extend(errors[:, 0])
        all_errors_y.extend(errors[:, 1])
        all_errors_z.extend(errors[:, 2])
        all_errors_3d.extend(errors_3d)
        pattern_labels.extend([name] * len(errors_3d))
        
        # Plot 1: Error evolution over time
        ax1.plot(time_steps, errors_3d * 100, 'o-', color=color, 
                linewidth=2, markersize=4, label=f'{name} (μ={np.mean(errors_3d)*100:.1f}cm)')
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('3D Tracking Error (cm)', fontsize=12)
    ax1.set_title('Tracking Error Evolution Over Time', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution by axis
    error_data = [np.array(all_errors_x) * 100, 
                  np.array(all_errors_y) * 100, 
                  np.array(all_errors_z) * 100]
    
    bp = ax2.boxplot(error_data, tick_labels=['X-axis', 'Y-axis', 'Z-axis'], 
                     patch_artist=True, notch=True)
    
    colors_box = ['lightcoral', 'lightgreen', 'lightblue']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Tracking Error (cm)', fontsize=12)
    ax2.set_title('Error Distribution by Axis', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add mean values as text
    means = [np.mean(data) * 100 for data in error_data]
    for i, mean_val in enumerate(means):
        ax2.text(i+1, mean_val + 0.5, f'μ={mean_val:.1f}cm', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Error histogram
    ax3.hist(np.array(all_errors_3d) * 100, bins=25, alpha=0.7, color='purple', 
             edgecolor='black', density=True)
    ax3.set_xlabel('3D Tracking Error (cm)', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('Overall Error Distribution', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    mean_3d = np.mean(all_errors_3d) * 100
    std_3d = np.std(all_errors_3d) * 100
    p90_3d = np.percentile(all_errors_3d, 90) * 100
    p95_3d = np.percentile(all_errors_3d, 95) * 100
    
    ax3.axvline(mean_3d, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_3d:.1f}cm')
    ax3.axvline(p90_3d, color='orange', linestyle='--', linewidth=2, label=f'90th: {p90_3d:.1f}cm')
    ax3.axvline(p95_3d, color='darkred', linestyle='--', linewidth=2, label=f'95th: {p95_3d:.1f}cm')
    ax3.legend()
    
    # Plot 4: Accuracy comparison by movement pattern
    pattern_means = []
    pattern_stds = []
    unique_patterns = ['Linear', 'Circular', 'Zigzag']
    
    for pattern in unique_patterns:
        pattern_errors = [err for err, label in zip(all_errors_3d, pattern_labels) if label == pattern]
        if pattern_errors:
            pattern_means.append(np.mean(pattern_errors) * 100)
            pattern_stds.append(np.std(pattern_errors) * 100)
        else:
            pattern_means.append(0)
            pattern_stds.append(0)
    
    x_pos = np.arange(len(unique_patterns))
    bars = ax4.bar(x_pos, pattern_means, yerr=pattern_stds, capsize=5, 
                   alpha=0.7, color=['blue', 'red', 'green'])
    
    ax4.set_xlabel('Movement Pattern', fontsize=12)
    ax4.set_ylabel('Mean 3D Error (cm)', fontsize=12)
    ax4.set_title('Tracking Accuracy by Movement Pattern', fontsize=14)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(unique_patterns)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, pattern_means, pattern_stds):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_val + 0.1,
                f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/error_analysis.pdf", bbox_inches='tight')
    plt.close()

def generate_time_series_plots(data_dir, session_id, output_dir):
    """Generate time-series tracking plots showing position and error evolution"""
    # Use linear movement as representative example
    data = load_position_data(data_dir, session_id, 'linear')
    if data is None or 'true' not in data or 'filtered' not in data:
        print("No linear movement data found for time series plot")
        return
    
    true_pos = data['true']['positions']
    filtered_pos = data['filtered']['positions']
    time_steps = data['true']['time']
    
    # Calculate errors
    errors = np.abs(filtered_pos - true_pos)
    errors_3d = np.linalg.norm(errors, axis=1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Position evolution over time (all axes)
    ax1.plot(time_steps, true_pos[:, 0], 'b-', linewidth=3, label='True X', alpha=0.8)
    ax1.plot(time_steps, filtered_pos[:, 0], 'b--', linewidth=2, label='Tracked X', alpha=0.8)
    ax1.plot(time_steps, true_pos[:, 1], 'g-', linewidth=3, label='True Y', alpha=0.8)
    ax1.plot(time_steps, filtered_pos[:, 1], 'g--', linewidth=2, label='Tracked Y', alpha=0.8)
    ax1.plot(time_steps, true_pos[:, 2], 'r-', linewidth=3, label='True Z', alpha=0.8)
    ax1.plot(time_steps, filtered_pos[:, 2], 'r--', linewidth=2, label='Tracked Z', alpha=0.8)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Position (m)', fontsize=12)
    ax1.set_title('Position Tracking Over Time (Linear Movement)', fontsize=14)
    ax1.legend(ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual axis errors over time
    ax2.plot(time_steps, errors[:, 0] * 100, 'b-', linewidth=2, label='X Error', marker='o', markersize=4)
    ax2.plot(time_steps, errors[:, 1] * 100, 'g-', linewidth=2, label='Y Error', marker='s', markersize=4)
    ax2.plot(time_steps, errors[:, 2] * 100, 'r-', linewidth=2, label='Z Error', marker='^', markersize=4)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Tracking Error (cm)', fontsize=12)
    ax2.set_title('Per-Axis Tracking Error Evolution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 3D error with moving average
    window_size = 5
    if len(errors_3d) >= window_size:
        moving_avg = np.convolve(errors_3d, np.ones(window_size)/window_size, mode='valid')
        moving_time = time_steps[window_size-1:]
        
        ax3.plot(time_steps, errors_3d * 100, 'o-', color='purple', linewidth=2, 
                markersize=4, label='Instantaneous Error', alpha=0.7)
        ax3.plot(moving_time, moving_avg * 100, '-', color='darkred', linewidth=3, 
                label=f'Moving Average (n={window_size})')
    else:
        ax3.plot(time_steps, errors_3d * 100, 'o-', color='purple', linewidth=2, 
                markersize=4, label='3D Error')
    
    # Add error statistics as horizontal lines
    mean_error = np.mean(errors_3d) * 100
    ax3.axhline(mean_error, color='red', linestyle='--', alpha=0.8, 
               label=f'Mean: {mean_error:.1f}cm')
    ax3.axhline(np.percentile(errors_3d, 90) * 100, color='orange', linestyle='--', alpha=0.8,
               label=f'90th percentile: {np.percentile(errors_3d, 90)*100:.1f}cm')
    
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('3D Tracking Error (cm)', fontsize=12)
    ax3.set_title('Overall 3D Tracking Error Evolution', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Velocity estimation (if we have enough points)
    if len(time_steps) > 1:
        dt = np.diff(time_steps)
        true_velocity = np.diff(true_pos, axis=0) / dt[:, np.newaxis]
        filtered_velocity = np.diff(filtered_pos, axis=0) / dt[:, np.newaxis]
        
        # Calculate speed (magnitude of velocity)
        true_speed = np.linalg.norm(true_velocity, axis=1)
        filtered_speed = np.linalg.norm(filtered_velocity, axis=1)
        
        mid_time = (time_steps[1:] + time_steps[:-1]) / 2
        
        ax4.plot(mid_time, true_speed, 'b-', linewidth=3, label='True Speed', alpha=0.8)
        ax4.plot(mid_time, filtered_speed, 'r--', linewidth=2, label='Estimated Speed', alpha=0.8)
        
        ax4.set_xlabel('Time (s)', fontsize=12)
        ax4.set_ylabel('Speed (m/s)', fontsize=12)
        ax4.set_title('Target Speed Estimation', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_series_tracking.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/time_series_tracking.pdf", bbox_inches='tight')
    plt.close()

def generate_pattern_comparison_plots(data_dir, session_id, output_dir):
    """Generate comparative analysis of different movement patterns"""
    patterns = ['linear', 'circular', 'zigzag']
    pattern_names = ['Linear Movement', 'Circular Movement', 'Zigzag Movement']
    colors = ['blue', 'red', 'green']
    
    fig = plt.figure(figsize=(18, 12))
    
    pattern_data = {}
    
    # Load all pattern data
    for pattern in patterns:
        data = load_position_data(data_dir, session_id, pattern)
        if data and 'true' in data and 'filtered' in data:
            pattern_data[pattern] = data
    
    # Create 2x3 grid for different views
    for i, (pattern, name, color) in enumerate(zip(patterns, pattern_names, colors)):
        if pattern not in pattern_data:
            continue
        
        data = pattern_data[pattern]
        true_pos = data['true']['positions']
        filtered_pos = data['filtered']['positions']
        time_steps = data['true']['time']
        
        # Top row: X-Y projection
        ax_xy = fig.add_subplot(2, 3, i+1)
        ax_xy.plot(true_pos[:, 0], true_pos[:, 1], 'o-', color=color, 
                  linewidth=3, markersize=6, label='True Path', alpha=0.8)
        ax_xy.plot(filtered_pos[:, 0], filtered_pos[:, 1], 's--', color='red', 
                  linewidth=2, markersize=4, label='Tracked Path', alpha=0.8)
        
        # Mark start and end
        ax_xy.scatter(true_pos[0, 0], true_pos[0, 1], color='green', s=150, 
                     marker='o', label='Start', edgecolors='black', linewidth=2)
        ax_xy.scatter(true_pos[-1, 0], true_pos[-1, 1], color='orange', s=150, 
                     marker='s', label='End', edgecolors='black', linewidth=2)
        
        ax_xy.set_xlabel('X Position (m)', fontsize=12)
        ax_xy.set_ylabel('Y Position (m)', fontsize=12)
        ax_xy.set_title(f'{name}\nX-Y Projection', fontsize=14)
        ax_xy.legend()
        ax_xy.grid(True, alpha=0.3)
        ax_xy.axis('equal')
        
        # Bottom row: Z-axis behavior over time
        ax_z = fig.add_subplot(2, 3, i+4)
        ax_z.plot(time_steps, true_pos[:, 2], 'o-', color=color, 
                 linewidth=3, markersize=6, label='True Z', alpha=0.8)
        ax_z.plot(time_steps, filtered_pos[:, 2], 's--', color='red', 
                 linewidth=2, markersize=4, label='Tracked Z', alpha=0.8)
        
        # Calculate and show Z-axis error statistics
        z_errors = np.abs(filtered_pos[:, 2] - true_pos[:, 2])
        mean_z_error = np.mean(z_errors) * 100
        
        ax_z.fill_between(time_steps, 
                         true_pos[:, 2] - z_errors, 
                         true_pos[:, 2] + z_errors, 
                         alpha=0.3, color='red', label=f'Error Band (μ={mean_z_error:.1f}cm)')
        
        ax_z.set_xlabel('Time (s)', fontsize=12)
        ax_z.set_ylabel('Z Position (m)', fontsize=12)
        ax_z.set_title(f'Z-axis Tracking\nMean Error: {mean_z_error:.1f}cm', fontsize=14)
        ax_z.legend()
        ax_z.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pattern_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/pattern_comparison.pdf", bbox_inches='tight')
    plt.close()

def generate_snr_performance_plots(data_dir, session_id, output_dir):
    """Generate SNR performance analysis plots"""
    # Load metrics data
    metrics_file = f"{data_dir}/{session_id}_all_metrics.csv"
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return
    
    metrics_df = pd.read_csv(metrics_file)
    
    # Filter for SNR test data
    snr_data = metrics_df[metrics_df['movement_pattern'].str.contains('snr')]
    
    if len(snr_data) == 0:
        print("No SNR data found")
        return
    
    # Extract SNR levels and metrics
    snr_levels = sorted(snr_data['noise_snr'].unique())
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract metrics by SNR
    metrics_by_snr = {}
    for snr in snr_levels:
        row = snr_data[snr_data['noise_snr'] == snr].iloc[0]
        metrics_by_snr[snr] = {
            'x_error': row['mean_x_error'] * 100,  # Convert to cm
            'y_error': row['mean_y_error'] * 100,
            'z_error': row['mean_z_error'] * 100,
            '3d_error': row['mean_3d_error'] * 100,
            'x_std': row['std_x_error'] * 100,
            'y_std': row['std_y_error'] * 100,
            'z_std': row['std_z_error'] * 100,
            '3d_std': row['std_3d_error'] * 100
        }
    
    # Plot 1: Mean error vs SNR with error bars
    x_errors = [metrics_by_snr[snr]['x_error'] for snr in snr_levels]
    y_errors = [metrics_by_snr[snr]['y_error'] for snr in snr_levels]
    z_errors = [metrics_by_snr[snr]['z_error'] for snr in snr_levels]
    overall_errors = [metrics_by_snr[snr]['3d_error'] for snr in snr_levels]
    
    x_stds = [metrics_by_snr[snr]['x_std'] for snr in snr_levels]
    y_stds = [metrics_by_snr[snr]['y_std'] for snr in snr_levels]
    z_stds = [metrics_by_snr[snr]['z_std'] for snr in snr_levels]
    
    ax1.errorbar(snr_levels, x_errors, yerr=x_stds, fmt='o-', label='X-axis', 
                linewidth=2, markersize=8, capsize=5)
    ax1.errorbar(snr_levels, y_errors, yerr=y_stds, fmt='s-', label='Y-axis', 
                linewidth=2, markersize=8, capsize=5)
    ax1.errorbar(snr_levels, z_errors, yerr=z_stds, fmt='^-', label='Z-axis', 
                linewidth=2, markersize=8, capsize=5)
    ax1.errorbar(snr_levels, overall_errors, fmt='d-', label='3D Overall', 
                linewidth=2, markersize=8, capsize=5)
    
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('Mean Tracking Error (cm)', fontsize=12)
    ax1.set_title('Tracking Accuracy vs Signal-to-Noise Ratio', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visibility
    
    # Plot 2: Z-axis sensitivity (Z/X ratio)
    z_to_x_ratios = [z/x for z, x in zip(z_errors, x_errors)]
    
    ax2.plot(snr_levels, z_to_x_ratios, 'ro-', linewidth=3, markersize=10)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, 
               label='Equal Performance')
    
    # Add ratio values as text
    for snr, ratio in zip(snr_levels, z_to_x_ratios):
        ax2.text(snr, ratio + 0.05, f'{ratio:.1f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('Z-axis / X-axis Error Ratio', fontsize=12)
    ax2.set_title('Relative Z-axis Performance vs SNR', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Load actual position data for SNR comparison
    snr_positions_data = {}
    for snr in [10, 20, 30]:  # Select representative SNR levels
        data = load_position_data(data_dir, session_id, f'linear_snr_{snr}')
        if data and 'true' in data and 'filtered' in data:
            snr_positions_data[snr] = data
    
    if snr_positions_data:
        for snr, color in zip([10, 20, 30], ['red', 'blue', 'green']):
            if snr in snr_positions_data:
                data = snr_positions_data[snr]
                true_pos = data['true']['positions']
                filtered_pos = data['filtered']['positions']
                errors_3d = np.linalg.norm(filtered_pos - true_pos, axis=1) * 100
                
                ax3.hist(errors_3d, bins=15, alpha=0.6, 
                        label=f'SNR = {snr} dB (μ={np.mean(errors_3d):.1f}cm)', 
                        color=color, density=True)
        
        ax3.set_xlabel('3D Tracking Error (cm)', fontsize=12)
        ax3.set_ylabel('Probability Density', fontsize=12)
        ax3.set_title('Error Distribution at Different SNR Levels', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance improvement with SNR
    if len(snr_levels) > 1:
        # Calculate improvement relative to worst SNR
        worst_snr_error = max(overall_errors)
        improvements = [(worst_snr_error - err) / worst_snr_error * 100 for err in overall_errors]
        
        bars = ax4.bar(snr_levels, improvements, alpha=0.7, color='skyblue', 
                      edgecolor='navy', linewidth=2)
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{improvement:.0f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        ax4.set_xlabel('SNR (dB)', fontsize=12)
        ax4.set_ylabel('Error Reduction (%)', fontsize=12)
        ax4.set_title(f'Performance Improvement vs {min(snr_levels)} dB Baseline', fontsize=14)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/snr_performance.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/snr_performance.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_tracking_plots()