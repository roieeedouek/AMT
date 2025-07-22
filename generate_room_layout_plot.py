#!/usr/bin/env python3
"""
Generate detailed room layout visualization for AMT3D system
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def generate_room_layout_plot():
    """Generate comprehensive room layout visualization"""
    print("Generating AMT3D room layout visualization...")
    
    # Load room configuration
    data_dir = "simulation_data"
    session_id = "20250712_193646"
    
    # Load room dimensions
    room_config = pd.read_csv(f"{data_dir}/{session_id}_system_config.csv")
    room_width = room_config['room_width'].iloc[0]
    room_length = room_config['room_length'].iloc[0] 
    room_height = room_config['room_height'].iloc[0]
    
    # Load speaker and microphone positions
    speakers_df = pd.read_csv(f"{data_dir}/{session_id}_speakers.csv")
    mics_df = pd.read_csv(f"{data_dir}/{session_id}_microphones.csv")
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 12))
    
    # 3D view
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    draw_detailed_room_3d(ax1, room_width, room_length, room_height, speakers_df, mics_df)
    
    # Top view (X-Y plane)
    ax2 = fig.add_subplot(2, 3, 2)
    draw_room_top_view(ax2, room_width, room_length, speakers_df, mics_df)
    
    # Side view (X-Z plane)
    ax3 = fig.add_subplot(2, 3, 3)
    draw_room_side_view_xz(ax3, room_width, room_height, speakers_df, mics_df)
    
    # Front view (Y-Z plane)  
    ax4 = fig.add_subplot(2, 3, 4)
    draw_room_side_view_yz(ax4, room_length, room_height, speakers_df, mics_df)
    
    # Speaker-microphone connectivity
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    draw_acoustic_paths(ax5, room_width, room_length, room_height, speakers_df, mics_df)
    
    # Geometric quality analysis
    ax6 = fig.add_subplot(2, 3, 6)
    draw_geometric_quality(ax6, speakers_df, mics_df)
    
    plt.tight_layout()
    plt.savefig("real_amt_tracking_plots/amt3d_room_layout_detailed.png", dpi=300, bbox_inches='tight')
    plt.savefig("real_amt_tracking_plots/amt3d_room_layout_detailed.pdf", bbox_inches='tight')
    plt.close()
    
    print("Room layout visualization saved to real_amt_tracking_plots/amt3d_room_layout_detailed.png")

def draw_detailed_room_3d(ax, width, length, height, speakers_df, mics_df):
    """Draw detailed 3D room with equipment"""
    # Draw room boundaries
    draw_room_wireframe(ax, width, length, height)
    
    # Plot speakers with labels
    colors_speakers = ['red', 'orange', 'red', 'green', 'green']
    for i, row in speakers_df.iterrows():
        ax.scatter(row['x'], row['y'], row['z'], 
                  c=colors_speakers[i], s=200, marker='^', alpha=0.9,
                  edgecolors='black', linewidth=2)
        ax.text(row['x'], row['y'], row['z'] + 0.1, row['label'], 
               fontsize=10, ha='center', weight='bold')
    
    # Plot microphones with labels
    for i, row in mics_df.iterrows():
        ax.scatter(row['x'], row['y'], row['z'],
                  c='cyan', s=150, marker='s', alpha=0.9,
                  edgecolors='black', linewidth=2)
        ax.text(row['x'], row['y'], row['z'] + 0.1, row['label'], 
               fontsize=10, ha='center', weight='bold')
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, length)
    ax.set_zlim(0, height)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('AMT3D System Layout\n5.1 Surround + 4-Mic Array', fontsize=14)
    ax.view_init(elev=20, azim=45)

def draw_room_top_view(ax, width, length, speakers_df, mics_df):
    """Draw top-down view (X-Y plane)"""
    # Room outline
    ax.plot([0, width, width, 0, 0], [0, 0, length, length, 0], 'k-', linewidth=2)
    
    # Grid
    for x in np.linspace(0, width, 6):
        ax.plot([x, x], [0, length], 'k--', alpha=0.3, linewidth=0.5)
    for y in np.linspace(0, length, 7):
        ax.plot([0, width], [y, y], 'k--', alpha=0.3, linewidth=0.5)
    
    # Speakers
    colors_speakers = ['red', 'orange', 'red', 'green', 'green']
    for i, row in speakers_df.iterrows():
        ax.scatter(row['x'], row['y'], c=colors_speakers[i], s=200, marker='^', 
                  alpha=0.9, edgecolors='black', linewidth=2)
        ax.annotate(f"{row['label']}\n(z={row['z']:.1f})", 
                   (row['x'], row['y']), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9, weight='bold')
    
    # Microphones
    for i, row in mics_df.iterrows():
        ax.scatter(row['x'], row['y'], c='cyan', s=150, marker='s', 
                  alpha=0.9, edgecolors='black', linewidth=2)
        ax.annotate(f"{row['label']}\n(z={row['z']:.1f})", 
                   (row['x'], row['y']), xytext=(5, -15), 
                   textcoords='offset points', fontsize=9, weight='bold')
    
    ax.set_xlim(-0.2, width + 0.2)
    ax.set_ylim(-0.2, length + 0.2)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Top View (X-Y Plane)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

def draw_room_side_view_xz(ax, width, height, speakers_df, mics_df):
    """Draw side view (X-Z plane)"""
    # Room outline
    ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'k-', linewidth=2)
    
    # Grid
    for x in np.linspace(0, width, 6):
        ax.plot([x, x], [0, height], 'k--', alpha=0.3, linewidth=0.5)
    for z in np.linspace(0, height, 4):
        ax.plot([0, width], [z, z], 'k--', alpha=0.3, linewidth=0.5)
    
    # Speakers
    colors_speakers = ['red', 'orange', 'red', 'green', 'green']
    for i, row in speakers_df.iterrows():
        ax.scatter(row['x'], row['z'], c=colors_speakers[i], s=200, marker='^', 
                  alpha=0.9, edgecolors='black', linewidth=2)
        ax.annotate(f"{row['label']}\n(y={row['y']:.1f})", 
                   (row['x'], row['z']), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9, weight='bold')
    
    # Microphones
    for i, row in mics_df.iterrows():
        ax.scatter(row['x'], row['z'], c='cyan', s=150, marker='s', 
                  alpha=0.9, edgecolors='black', linewidth=2)
        ax.annotate(f"{row['label']}\n(y={row['y']:.1f})", 
                   (row['x'], row['z']), xytext=(5, -15), 
                   textcoords='offset points', fontsize=9, weight='bold')
    
    ax.set_xlim(-0.2, width + 0.2)
    ax.set_ylim(-0.1, height + 0.1)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title('Side View (X-Z Plane)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

def draw_room_side_view_yz(ax, length, height, speakers_df, mics_df):
    """Draw front view (Y-Z plane)"""
    # Room outline
    ax.plot([0, length, length, 0, 0], [0, 0, height, height, 0], 'k-', linewidth=2)
    
    # Grid
    for y in np.linspace(0, length, 7):
        ax.plot([y, y], [0, height], 'k--', alpha=0.3, linewidth=0.5)
    for z in np.linspace(0, height, 4):
        ax.plot([0, length], [z, z], 'k--', alpha=0.3, linewidth=0.5)
    
    # Speakers
    colors_speakers = ['red', 'orange', 'red', 'green', 'green']
    for i, row in speakers_df.iterrows():
        ax.scatter(row['y'], row['z'], c=colors_speakers[i], s=200, marker='^', 
                  alpha=0.9, edgecolors='black', linewidth=2)
        ax.annotate(f"{row['label']}\n(x={row['x']:.1f})", 
                   (row['y'], row['z']), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9, weight='bold')
    
    # Microphones
    for i, row in mics_df.iterrows():
        ax.scatter(row['y'], row['z'], c='cyan', s=150, marker='s', 
                  alpha=0.9, edgecolors='black', linewidth=2)
        ax.annotate(f"{row['label']}\n(x={row['x']:.1f})", 
                   (row['y'], row['z']), xytext=(5, -15), 
                   textcoords='offset points', fontsize=9, weight='bold')
    
    ax.set_xlim(-0.2, length + 0.2)
    ax.set_ylim(-0.1, height + 0.1)
    ax.set_xlabel('Y (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title('Front View (Y-Z Plane)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

def draw_acoustic_paths(ax, width, length, height, speakers_df, mics_df):
    """Draw acoustic paths between speakers and microphones"""
    # Draw room wireframe
    draw_room_wireframe(ax, width, length, height)
    
    # Plot equipment
    ax.scatter(speakers_df['x'], speakers_df['y'], speakers_df['z'], 
              c='red', s=200, marker='^', alpha=0.9, label='Speakers')
    ax.scatter(mics_df['x'], mics_df['y'], mics_df['z'],
              c='cyan', s=150, marker='s', alpha=0.9, label='Microphones')
    
    # Draw acoustic paths
    path_count = 0
    for _, speaker in speakers_df.iterrows():
        for _, mic in mics_df.iterrows():
            # Calculate path distance
            dist = np.sqrt((speaker['x'] - mic['x'])**2 + 
                          (speaker['y'] - mic['y'])**2 + 
                          (speaker['z'] - mic['z'])**2)
            
            # Draw path with alpha based on distance (closer = more opaque)
            alpha = max(0.1, 1.0 - (dist - 1.0) / 5.0)
            ax.plot([speaker['x'], mic['x']], 
                   [speaker['y'], mic['y']], 
                   [speaker['z'], mic['z']], 
                   'g--', alpha=alpha, linewidth=1)
            path_count += 1
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, length)
    ax.set_zlim(0, height)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'Acoustic Paths\n{path_count} Speaker-Mic Pairs', fontsize=14)
    ax.legend()
    ax.view_init(elev=15, azim=60)

def draw_geometric_quality(ax, speakers_df, mics_df):
    """Draw geometric quality analysis"""
    # Calculate elevation angles for all speaker-mic pairs
    elevation_angles = []
    distances = []
    
    for _, speaker in speakers_df.iterrows():
        for _, mic in mics_df.iterrows():
            # Calculate distance and elevation angle
            dx = speaker['x'] - mic['x']
            dy = speaker['y'] - mic['y']
            dz = speaker['z'] - mic['z']
            
            horizontal_dist = np.sqrt(dx**2 + dy**2)
            total_dist = np.sqrt(dx**2 + dy**2 + dz**2)
            
            if horizontal_dist > 0:
                elevation = np.degrees(np.arctan(dz / horizontal_dist))
                elevation_angles.append(elevation)
                distances.append(total_dist)
    
    # Create scatter plot
    ax.scatter(distances, elevation_angles, s=100, alpha=0.7, c='blue')
    
    # Add statistics
    mean_elevation = np.mean(elevation_angles)
    std_elevation = np.std(elevation_angles)
    geometric_quality = std_elevation * 10  # Same metric as in AMT3D
    
    ax.axhline(mean_elevation, color='red', linestyle='--', 
              label=f'Mean: {mean_elevation:.1f}°')
    ax.axhline(mean_elevation + std_elevation, color='orange', linestyle='--', 
              label=f'+1σ: {mean_elevation + std_elevation:.1f}°')
    ax.axhline(mean_elevation - std_elevation, color='orange', linestyle='--', 
              label=f'-1σ: {mean_elevation - std_elevation:.1f}°')
    
    ax.set_xlabel('Speaker-Mic Distance (m)', fontsize=12)
    ax.set_ylabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_title(f'Geometric Quality Analysis\nZ-Quality: {geometric_quality:.1f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text box with analysis
    textstr = f'''Elevation Statistics:
Mean: {mean_elevation:.1f}°
Std: {std_elevation:.1f}°
Range: {min(elevation_angles):.1f}° to {max(elevation_angles):.1f}°
Z-axis Quality: {geometric_quality:.1f}
Total Paths: {len(elevation_angles)}'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)

def draw_room_wireframe(ax, width, length, height):
    """Draw room wireframe"""
    # Room corners
    corners = np.array([
        [0, 0, 0], [width, 0, 0], [width, length, 0], [0, length, 0],
        [0, 0, height], [width, 0, height], [width, length, height], [0, length, height]
    ])
    
    # Edges
    edges = [[0,1], [1,2], [2,3], [3,0],  # Floor
             [4,5], [5,6], [6,7], [7,4],  # Ceiling
             [0,4], [1,5], [2,6], [3,7]]  # Vertical
    
    for edge in edges:
        points = corners[edge]
        ax.plot3D(*points.T, 'k-', alpha=0.3, linewidth=1)

if __name__ == "__main__":
    generate_room_layout_plot()