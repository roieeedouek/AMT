#!/usr/bin/env python3
"""
Generate comprehensive tracking plots from real AMT3D simulation data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from mpl_toolkits.mplot3d import Axes3D

def generate_real_amt_tracking_plots():
    """Generate comprehensive tracking visualization plots from real AMT3D CSV data"""
    print("Generating tracking plots from real AMT3D simulation data...")
    
    # Find all simulation sessions
    data_dir = "simulation_data"
    if not os.path.exists(data_dir):
        print("No simulation data found.")
        return
    
    # Get all session IDs
    session_files = glob.glob(f"{data_dir}/*_system_config.csv")
    if not session_files:
        print("No session data found.")
        return
    
    sessions = []
    for file in session_files:
        session_id = os.path.basename(file).replace('_system_config.csv', '')
        sessions.append(session_id)
    
    print(f"Found {len(sessions)} simulation sessions: {sessions}")
    
    # Use the most recent session (latest)
    latest_session = sorted(sessions)[-1]
    print(f"Using latest session: {latest_session}")
    
    # Create output directory
    plots_dir = "real_amt_tracking_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load all tracking data for this session
    tracking_data = load_session_tracking_data(data_dir, latest_session)
    
    # Generate different types of tracking plots
    print("1. Generating 3D trajectory comparison plots...")
    generate_3d_trajectory_plots_real(tracking_data, plots_dir)
    
    print("2. Generating tracking error analysis plots...")
    generate_error_analysis_plots_real(tracking_data, plots_dir)
    
    print("3. Generating time-series tracking plots...")
    generate_time_series_plots_real(tracking_data, plots_dir)
    
    print("4. Generating consolidated performance analysis...")
    generate_consolidated_performance_plots(data_dir, sessions, plots_dir)
    
    print(f"All real AMT3D tracking plots generated in '{plots_dir}' directory")

def load_session_tracking_data(data_dir, session_id):
    """Load all tracking data for a specific session"""
    tracking_data = {}
    
    # Find all tracking files for this session
    position_files = glob.glob(f"{data_dir}/{session_id}_*_true_positions.csv")
    
    for pos_file in position_files:
        # Extract test identifier from filename
        base_name = os.path.basename(pos_file).replace('_true_positions.csv', '')
        test_id = base_name.replace(f'{session_id}_', '')
        
        # Load true, estimated, and filtered positions
        true_file = f"{data_dir}/{session_id}_{test_id}_true_positions.csv"
        filtered_file = f"{data_dir}/{session_id}_{test_id}_filtered_positions.csv"
        estimated_file = f"{data_dir}/{session_id}_{test_id}_estimated_positions.csv"
        metrics_file = f"{data_dir}/{session_id}_{test_id}_metrics.csv"
        
        test_data = {}
        
        if os.path.exists(true_file):
            test_data['true'] = pd.read_csv(true_file)
        
        if os.path.exists(filtered_file):
            test_data['filtered'] = pd.read_csv(filtered_file)
            
        if os.path.exists(estimated_file):
            test_data['estimated'] = pd.read_csv(estimated_file)
            
        if os.path.exists(metrics_file):
            test_data['metrics'] = pd.read_csv(metrics_file)
        
        if len(test_data) > 0:
            tracking_data[test_id] = test_data
    
    print(f"Loaded {len(tracking_data)} test runs: {list(tracking_data.keys())}")
    return tracking_data

def generate_3d_trajectory_plots_real(tracking_data, output_dir):
    """Generate 3D trajectory comparison plots from real data with full room context"""
    # Load room configuration
    data_dir = "simulation_data"
    session_id = "20250712_193646"  # Use latest session
    
    # Load room dimensions
    room_config = pd.read_csv(f"{data_dir}/{session_id}_system_config.csv")
    room_width = room_config['room_width'].iloc[0]
    room_length = room_config['room_length'].iloc[0] 
    room_height = room_config['room_height'].iloc[0]
    
    # Load speaker and microphone positions
    speakers_df = pd.read_csv(f"{data_dir}/{session_id}_speakers.csv")
    mics_df = pd.read_csv(f"{data_dir}/{session_id}_microphones.csv")
    
    # Select a few representative tests
    test_names = list(tracking_data.keys())[:3]  # Take first 3 tests
    
    fig = plt.figure(figsize=(20, 7))
    
    for i, test_name in enumerate(test_names):
        if i >= 3:  # Limit to 3 plots
            break
            
        data = tracking_data[test_name]
        if 'true' not in data or 'filtered' not in data:
            continue
        
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Draw room boundaries first
        draw_room_boundaries(ax, room_width, room_length, room_height)
        
        # Plot speakers
        ax.scatter(speakers_df['x'], speakers_df['y'], speakers_df['z'], 
                  c='purple', s=150, marker='^', alpha=0.8, 
                  label='Speakers', edgecolors='black', linewidth=1)
        
        # Plot microphones  
        ax.scatter(mics_df['x'], mics_df['y'], mics_df['z'],
                  c='cyan', s=120, marker='s', alpha=0.8,
                  label='Microphones', edgecolors='black', linewidth=1)
        
        # Plot trajectories
        true_pos = data['true'][['x', 'y', 'z']].values
        filtered_pos = data['filtered'][['x', 'y', 'z']].values
        
        # Plot true trajectory
        ax.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], 
                'o-', color='blue', linewidth=4, markersize=8, 
                label='True Trajectory', alpha=0.9, zorder=10)
        
        # Plot filtered trajectory
        ax.plot(filtered_pos[:, 0], filtered_pos[:, 1], filtered_pos[:, 2], 
                's--', color='red', linewidth=3, markersize=6, 
                label='AMT3D Tracked', alpha=0.9, zorder=10)
        
        # Mark start and end points
        ax.scatter(*true_pos[0], color='green', s=250, marker='o', 
                  label='Start', alpha=1.0, edgecolors='black', linewidth=2, zorder=15)
        ax.scatter(*true_pos[-1], color='orange', s=250, marker='s', 
                  label='End', alpha=1.0, edgecolors='black', linewidth=2, zorder=15)
        
        # Calculate and display tracking error statistics
        errors = np.linalg.norm(filtered_pos - true_pos, axis=1)
        mean_error = np.mean(errors) * 100  # Convert to cm
        max_error = np.max(errors) * 100
        
        # Set room-relative axes
        ax.set_xlim(0, room_width)
        ax.set_ylim(0, room_length) 
        ax.set_zlim(0, room_height)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(f'AMT3D Room Context: {test_name}\\nRoom: {room_width}×{room_length}×{room_height}m\\nError: {mean_error:.1f}cm±{max_error:.1f}cm', 
                    fontsize=11)
        ax.legend(loc='upper left', fontsize=9)
        
        # Set aspect ratio to show room proportions
        ax.set_box_aspect([room_width, room_length, room_height/2])
        
        # Improve viewing angle
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/real_3d_trajectory_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/real_3d_trajectory_comparison.pdf", bbox_inches='tight')
    plt.close()

def draw_room_boundaries(ax, width, length, height):
    """Draw room boundaries as wireframe"""
    # Room corners
    corners = np.array([
        [0, 0, 0], [width, 0, 0], [width, length, 0], [0, length, 0],  # Floor
        [0, 0, height], [width, 0, height], [width, length, height], [0, length, height]  # Ceiling
    ])
    
    # Floor edges
    floor_edges = [[0,1], [1,2], [2,3], [3,0]]
    # Ceiling edges  
    ceiling_edges = [[4,5], [5,6], [6,7], [7,4]]
    # Vertical edges
    vertical_edges = [[0,4], [1,5], [2,6], [3,7]]
    
    # Draw all edges
    all_edges = floor_edges + ceiling_edges + vertical_edges
    
    for edge in all_edges:
        points = corners[edge]
        ax.plot3D(*points.T, 'k-', alpha=0.3, linewidth=1)
    
    # Add floor grid for better spatial reference
    x_grid = np.linspace(0, width, 6)
    y_grid = np.linspace(0, length, 7)
    
    # Grid lines parallel to X-axis
    for y in y_grid[1:-1]:  # Skip edges
        ax.plot3D([0, width], [y, y], [0, 0], 'k--', alpha=0.2, linewidth=0.5)
    
    # Grid lines parallel to Y-axis  
    for x in x_grid[1:-1]:  # Skip edges
        ax.plot3D([x, x], [0, length], [0, 0], 'k--', alpha=0.2, linewidth=0.5)

def generate_error_analysis_plots_real(tracking_data, output_dir):
    """Generate detailed error analysis plots from real data"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    all_errors_x = []
    all_errors_y = []
    all_errors_z = []
    all_errors_3d = []
    test_labels = []
    
    # Collect error data from all tests
    for test_name, data in tracking_data.items():
        if 'true' not in data or 'filtered' not in data:
            continue
        
        true_pos = data['true'][['x', 'y', 'z']].values
        filtered_pos = data['filtered'][['x', 'y', 'z']].values
        time_steps = data['true']['time'].values
        
        # Calculate errors
        errors = np.abs(filtered_pos - true_pos)
        errors_3d = np.linalg.norm(errors, axis=1)
        
        # Store for aggregated analysis
        all_errors_x.extend(errors[:, 0])
        all_errors_y.extend(errors[:, 1])
        all_errors_z.extend(errors[:, 2])
        all_errors_3d.extend(errors_3d)
        test_labels.extend([test_name] * len(errors_3d))
        
        # Plot error evolution over time for first few tests
        if len(ax1.lines) < 3:  # Limit to 3 tests for clarity
            color = ['blue', 'red', 'green'][len(ax1.lines)]
            ax1.plot(time_steps, errors_3d * 100, 'o-', color=color, 
                    linewidth=2, markersize=4, 
                    label=f'{test_name} (μ={np.mean(errors_3d)*100:.1f}cm)', alpha=0.7)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('3D Tracking Error (cm)', fontsize=12)
    ax1.set_title('Real AMT3D Tracking Error Evolution', fontsize=14)
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
    ax2.set_title('Real AMT3D Error Distribution by Axis', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add mean values as text
    means = [np.mean(data) for data in error_data]
    for i, mean_val in enumerate(means):
        ax2.text(i+1, mean_val + 0.5, f'μ={mean_val:.1f}cm', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Overall error histogram
    ax3.hist(np.array(all_errors_3d) * 100, bins=25, alpha=0.7, color='purple', 
             edgecolor='black', density=True)
    ax3.set_xlabel('3D Tracking Error (cm)', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('Real AMT3D Overall Error Distribution', fontsize=14)
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
    
    # Plot 4: Z-axis degradation analysis
    x_errors_cm = np.array(all_errors_x) * 100
    z_errors_cm = np.array(all_errors_z) * 100
    
    # Create scatter plot of Z vs X errors
    ax4.scatter(x_errors_cm, z_errors_cm, alpha=0.6, s=20, color='blue')
    
    # Add trend line
    z = np.polyfit(x_errors_cm, z_errors_cm, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(x_errors_cm), max(x_errors_cm), 100)
    ax4.plot(x_trend, p(x_trend), "r--", linewidth=2, 
             label=f'Trend: Z = {z[0]:.1f}×X + {z[1]:.1f}')
    
    # Add diagonal line for reference
    max_val = max(max(x_errors_cm), max(z_errors_cm))
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Z = X')
    
    ax4.set_xlabel('X-axis Error (cm)', fontsize=12)
    ax4.set_ylabel('Z-axis Error (cm)', fontsize=12)
    ax4.set_title('Real AMT3D Z-axis vs X-axis Error Correlation', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(x_errors_cm, z_errors_cm)[0, 1]
    z_to_x_ratio = np.mean(z_errors_cm) / np.mean(x_errors_cm)
    ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}\\nZ/X Ratio: {z_to_x_ratio:.1f}', 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/real_error_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/real_error_analysis.pdf", bbox_inches='tight')
    plt.close()

def generate_time_series_plots_real(tracking_data, output_dir):
    """Generate time-series tracking plots from real data"""
    # Use the first test with good data
    test_name = None
    for name, data in tracking_data.items():
        if 'true' in data and 'filtered' in data and len(data['true']) > 5:
            test_name = name
            break
    
    if not test_name:
        print("No suitable test data found for time series plot")
        return
    
    data = tracking_data[test_name]
    true_pos = data['true'][['x', 'y', 'z']].values
    filtered_pos = data['filtered'][['x', 'y', 'z']].values
    time_steps = data['true']['time'].values
    
    # Calculate errors
    errors = np.abs(filtered_pos - true_pos)
    errors_3d = np.linalg.norm(errors, axis=1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Position evolution over time (all axes)
    ax1.plot(time_steps, true_pos[:, 0], 'b-', linewidth=3, label='True X', alpha=0.8)
    ax1.plot(time_steps, filtered_pos[:, 0], 'b--', linewidth=2, label='AMT3D X', alpha=0.8)
    ax1.plot(time_steps, true_pos[:, 1], 'g-', linewidth=3, label='True Y', alpha=0.8)
    ax1.plot(time_steps, filtered_pos[:, 1], 'g--', linewidth=2, label='AMT3D Y', alpha=0.8)
    ax1.plot(time_steps, true_pos[:, 2], 'r-', linewidth=3, label='True Z', alpha=0.8)
    ax1.plot(time_steps, filtered_pos[:, 2], 'r--', linewidth=2, label='AMT3D Z', alpha=0.8)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Position (m)', fontsize=12)
    ax1.set_title(f'Real AMT3D Position Tracking Over Time\\nTest: {test_name}', fontsize=14)
    ax1.legend(ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual axis errors over time
    ax2.plot(time_steps, errors[:, 0] * 100, 'b-', linewidth=2, label='X Error', marker='o', markersize=4)
    ax2.plot(time_steps, errors[:, 1] * 100, 'g-', linewidth=2, label='Y Error', marker='s', markersize=4)
    ax2.plot(time_steps, errors[:, 2] * 100, 'r-', linewidth=2, label='Z Error', marker='^', markersize=4)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Tracking Error (cm)', fontsize=12)
    ax2.set_title('Real AMT3D Per-Axis Error Evolution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 3D error with statistics
    ax3.plot(time_steps, errors_3d * 100, 'o-', color='purple', linewidth=2, 
            markersize=4, label='3D Error', alpha=0.7)
    
    # Add error statistics as horizontal lines
    mean_error = np.mean(errors_3d) * 100
    ax3.axhline(mean_error, color='red', linestyle='--', alpha=0.8, 
               label=f'Mean: {mean_error:.1f}cm')
    ax3.axhline(np.percentile(errors_3d, 90) * 100, color='orange', linestyle='--', alpha=0.8,
               label=f'90th percentile: {np.percentile(errors_3d, 90)*100:.1f}cm')
    
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('3D Tracking Error (cm)', fontsize=12)
    ax3.set_title('Real AMT3D Overall Error Evolution', fontsize=14)
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
        ax4.plot(mid_time, filtered_speed, 'r--', linewidth=2, label='AMT3D Speed', alpha=0.8)
        
        ax4.set_xlabel('Time (s)', fontsize=12)
        ax4.set_ylabel('Speed (m/s)', fontsize=12)
        ax4.set_title('Real AMT3D Target Speed Estimation', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/real_time_series_tracking.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/real_time_series_tracking.pdf", bbox_inches='tight')
    plt.close()

def generate_consolidated_performance_plots(data_dir, sessions, output_dir):
    """Generate consolidated performance analysis across all sessions"""
    # Load all metrics files
    all_metrics = []
    
    for session in sessions:
        metrics_files = glob.glob(f"{data_dir}/{session}_*_metrics.csv")
        for metrics_file in metrics_files:
            try:
                df = pd.read_csv(metrics_file)
                df['session'] = session
                df['test_file'] = os.path.basename(metrics_file)
                all_metrics.append(df)
            except Exception as e:
                print(f"Error loading {metrics_file}: {e}")
    
    if not all_metrics:
        print("No metrics data found")
        return
    
    # Combine all metrics
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Error comparison across axes
    x_errors = combined_metrics['mean_x_error'].values * 100
    y_errors = combined_metrics['mean_y_error'].values * 100  
    z_errors = combined_metrics['mean_z_error'].values * 100
    
    error_data = [x_errors, y_errors, z_errors]
    bp = ax1.boxplot(error_data, tick_labels=['X-axis', 'Y-axis', 'Z-axis'], 
                     patch_artist=True, notch=True)
    
    colors = ['lightcoral', 'lightgreen', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Mean Tracking Error (cm)', fontsize=12)
    ax1.set_title('Real AMT3D Error Distribution Across All Tests', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    for i, data in enumerate(error_data):
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax1.text(i+1, mean_val + std_val + 0.5, f'μ={mean_val:.1f}±{std_val:.1f}cm', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: 3D error distribution
    overall_3d_errors = combined_metrics['mean_3d_error'].values * 100
    
    ax2.hist(overall_3d_errors, bins=20, alpha=0.7, color='purple', 
             edgecolor='black', density=True)
    ax2.set_xlabel('Mean 3D Error (cm)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Real AMT3D Overall Performance Distribution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    mean_overall = np.mean(overall_3d_errors)
    std_overall = np.std(overall_3d_errors)
    p90_overall = np.percentile(overall_3d_errors, 90)
    
    ax2.axvline(mean_overall, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_overall:.1f}cm')
    ax2.axvline(p90_overall, color='orange', linestyle='--', linewidth=2, 
               label=f'90th: {p90_overall:.1f}cm')
    ax2.legend()
    
    # Plot 3: Z-axis degradation analysis
    z_to_x_ratios = z_errors / x_errors
    valid_ratios = z_to_x_ratios[np.isfinite(z_to_x_ratios)]
    
    ax3.hist(valid_ratios, bins=15, alpha=0.7, color='orange', 
             edgecolor='black', density=True)
    ax3.set_xlabel('Z-axis / X-axis Error Ratio', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('Real AMT3D Z-axis Degradation Factor', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    mean_ratio = np.mean(valid_ratios)
    ax3.axvline(mean_ratio, color='red', linestyle='--', linewidth=2, 
               label=f'Mean Ratio: {mean_ratio:.1f}')
    ax3.axvline(1.0, color='black', linestyle='-', alpha=0.5, 
               label='Equal Performance')
    ax3.legend()
    
    # Plot 4: Performance summary
    summary_stats = {
        'X-axis (cm)': [np.mean(x_errors), np.std(x_errors)],
        'Y-axis (cm)': [np.mean(y_errors), np.std(y_errors)],
        'Z-axis (cm)': [np.mean(z_errors), np.std(z_errors)],
        '3D Overall (cm)': [np.mean(overall_3d_errors), np.std(overall_3d_errors)]
    }
    
    categories = list(summary_stats.keys())
    means = [summary_stats[cat][0] for cat in categories]
    stds = [summary_stats[cat][1] for cat in categories]
    
    bars = ax4.bar(categories, means, yerr=stds, capsize=5, 
                   alpha=0.7, color=['red', 'green', 'blue', 'purple'])
    
    ax4.set_ylabel('Mean Error (cm)', fontsize=12)
    ax4.set_title('Real AMT3D Performance Summary', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, means, stds):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_val + 0.1,
                f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/real_consolidated_performance.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/real_consolidated_performance.pdf", bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\n=== REAL AMT3D PERFORMANCE SUMMARY ===")
    print(f"Total test runs analyzed: {len(combined_metrics)}")
    print(f"X-axis accuracy: {np.mean(x_errors):.1f} ± {np.std(x_errors):.1f} cm")
    print(f"Y-axis accuracy: {np.mean(y_errors):.1f} ± {np.std(y_errors):.1f} cm")
    print(f"Z-axis accuracy: {np.mean(z_errors):.1f} ± {np.std(z_errors):.1f} cm")
    print(f"Overall 3D accuracy: {np.mean(overall_3d_errors):.1f} ± {np.std(overall_3d_errors):.1f} cm")
    print(f"Z-axis degradation factor: {np.mean(valid_ratios):.1f}x")
    print(f"90th percentile error: {p90_overall:.1f} cm")

if __name__ == "__main__":
    generate_real_amt_tracking_plots()