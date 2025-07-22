import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_config_data_and_metrics():
    """Load microphone configurations and performance metrics"""
    data_dir = Path("simulation_data")
    
    config_data = {}
    mic_configs = {}
    
    for config in ['tvplus', 'tvx']:
        # Load microphone configuration from any SNR file
        for snr in [0, 5, 10, 20]:
            mic_pattern = f"{config}_SNR{snr}_*microphones.csv"
            mic_files = list(data_dir.glob(mic_pattern))
            if mic_files:
                mic_configs[config] = pd.read_csv(mic_files[0])
                break
        
        # Load all metrics for this configuration
        all_metrics = []
        for snr in [0, 5, 10, 20]:
            metrics_pattern = f"{config}_SNR{snr}_*_metrics.csv"
            metrics_files = list(data_dir.glob(metrics_pattern))
            if metrics_files:
                metrics = pd.read_csv(metrics_files[0])
                metrics['snr'] = snr
                all_metrics.append(metrics)
        
        if all_metrics:
            config_data[config] = pd.concat(all_metrics, ignore_index=True)
    
    return config_data, mic_configs

def calculate_geometric_properties(mic_data):
    """Calculate geometric properties of microphone array"""
    positions = mic_data[['x', 'y', 'z']].values
    
    # Calculate all pairwise distances
    distances = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances.append(dist)
    
    # Calculate centroid and spread
    centroid = np.mean(positions, axis=0)
    spread = np.mean([np.linalg.norm(pos - centroid) for pos in positions])
    
    # Calculate bounding box dimensions
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    dimensions = max_coords - min_coords
    
    return {
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'spread': spread,
        'dimensions': dimensions,
        'centroid': centroid,
        'positions': positions
    }

def create_side_by_side_comparison(config_data, mic_configs):
    """Create comprehensive side-by-side comparison"""
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 14,
        'figure.figsize': (20, 12)
    })
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1, 1])
    
    fig.suptitle('TV Plus vs TV X Configuration Comparison', fontsize=20, fontweight='bold', y=0.95)
    
    # Colors and labels
    colors = {'tvplus': '#2E86AB', 'tvx': '#A23B72'}
    config_labels = {'tvplus': 'TV Plus', 'tvx': 'TV X'}
    
    # Calculate performance metrics
    performance_data = {}
    for config, data in config_data.items():
        performance_data[config] = {
            'avg_error': data['mean_3d_error'].mean(),
            'avg_std': data['std_3d_error'].mean(),
            'avg_max': data['max_3d_error'].mean(),
            'best_error': data[data['snr'] == 20]['mean_3d_error'].mean() if len(data[data['snr'] == 20]) > 0 else np.nan,
            'best_std': data[data['snr'] == 20]['std_3d_error'].mean() if len(data[data['snr'] == 20]) > 0 else np.nan,
            'best_max': data[data['snr'] == 20]['max_3d_error'].mean() if len(data[data['snr'] == 20]) > 0 else np.nan
        }
    
    # Calculate geometric properties
    geometry_data = {}
    for config, mic_data in mic_configs.items():
        geometry_data[config] = calculate_geometric_properties(mic_data)
    
    # Row 1: Microphone Layout Comparison (3D view)
    for i, (config, mic_data) in enumerate(mic_configs.items()):
        ax = fig.add_subplot(gs[0, i*2:(i*2)+2], projection='3d')
        
        # Plot microphones
        scatter = ax.scatter(mic_data['x'], mic_data['y'], mic_data['z'], 
                           s=500, c=colors[config], alpha=0.8, edgecolors='black', linewidth=3)
        
        # Add microphone labels
        for _, row in mic_data.iterrows():
            ax.text(row['x'], row['y'], row['z'], f"  {row['mic_id']}", 
                   fontsize=12, fontweight='bold')
        
        # Add connecting lines to show geometry
        positions = mic_data[['x', 'y', 'z']].values
        for j in range(len(positions)):
            for k in range(j+1, len(positions)):
                ax.plot([positions[j,0], positions[k,0]], 
                       [positions[j,1], positions[k,1]], 
                       [positions[j,2], positions[k,2]], 
                       'k--', alpha=0.4, linewidth=2)
        
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
        ax.set_title(f'{config_labels[config]} - 3D Layout', fontsize=16, fontweight='bold')
        
        # Set equal aspect ratio
        max_range = np.array([mic_data['x'].max()-mic_data['x'].min(),
                             mic_data['y'].max()-mic_data['y'].min(),
                             mic_data['z'].max()-mic_data['z'].min()]).max() / 2.0
        mid_x = (mic_data['x'].max()+mic_data['x'].min()) * 0.5
        mid_y = (mic_data['y'].max()+mic_data['y'].min()) * 0.5
        mid_z = (mic_data['z'].max()+mic_data['z'].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Row 2: Top-down view with detailed annotations
    for i, (config, mic_data) in enumerate(mic_configs.items()):
        ax = fig.add_subplot(gs[1, i*2:(i*2)+2])
        
        # Plot microphones
        scatter = ax.scatter(mic_data['x'], mic_data['y'], 
                           s=600, c=colors[config], alpha=0.8, edgecolors='black', linewidth=3)
        
        # Add microphone labels with better positioning
        for _, row in mic_data.iterrows():
            ax.annotate(row['mic_id'], (row['x'], row['y']), 
                       xytext=(0, 0), textcoords='offset points',
                       fontsize=14, fontweight='bold', ha='center', va='center',
                       color='white')
        
        # Add connecting lines
        positions = mic_data[['x', 'y']].values
        for j in range(len(positions)):
            for k in range(j+1, len(positions)):
                ax.plot([positions[j,0], positions[k,0]], 
                       [positions[j,1], positions[k,1]], 
                       'k--', alpha=0.4, linewidth=2)
                
                # Add distance labels on lines
                mid_x = (positions[j,0] + positions[k,0]) / 2
                mid_y = (positions[j,1] + positions[k,1]) / 2
                distance = np.linalg.norm(positions[j] - positions[k])
                ax.text(mid_x, mid_y, f'{distance:.2f}m', 
                       fontsize=10, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Add geometric info
        geom = geometry_data[config]
        info_text = f"Spread: {geom['spread']:.3f} m\n"
        info_text += f"Max Distance: {geom['max_distance']:.3f} m\n"
        info_text += f"Mean Distance: {geom['mean_distance']:.3f} m"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=11, va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[config], alpha=0.2))
        
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_title(f'{config_labels[config]} - Top View', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Fix Y-axis range if all mics are at same Y coordinate
        y_range = mic_data['y'].max() - mic_data['y'].min()
        if y_range < 0.1:  # If Y range is very small
            y_center = mic_data['y'].mean()
            x_range = mic_data['x'].max() - mic_data['x'].min()
            y_padding = max(0.3, x_range * 0.4)  # Use 40% of X range or minimum 0.3m
            ax.set_ylim(y_center - y_padding, y_center + y_padding)
        
        ax.set_aspect('equal')
    
    # Row 3: Performance comparison tables
    # Left side: Average performance
    ax_left = fig.add_subplot(gs[2, 0:2])
    ax_left.axis('off')
    
    # Create performance comparison table
    table_data = []
    metrics = [
        ('Mean Error', 'avg_error', 'm'),
        ('Std Error', 'avg_std', 'm'),
        ('Max Error', 'avg_max', 'm')
    ]
    
    for metric_name, metric_key, unit in metrics:
        tvplus_val = performance_data['tvplus'][metric_key]
        tvx_val = performance_data['tvx'][metric_key]
        winner = 'TV Plus' if tvplus_val < tvx_val else 'TV X'
        diff_pct = abs((tvx_val - tvplus_val) / min(tvplus_val, tvx_val)) * 100
        table_data.append([metric_name, f'{tvplus_val:.4f} {unit}', f'{tvx_val:.4f} {unit}', 
                          f'{winner}\n({diff_pct:.1f}% better)'])
    
    table = ax_left.table(cellText=table_data,
                         colLabels=['Metric', 'TV Plus', 'TV X', 'Winner'],
                         cellLoc='center',
                         loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#E6E6E6')
                cell.set_text_props(weight='bold')
            else:
                if j == 1:  # TV Plus column
                    cell.set_facecolor('#E3F2FD')
                elif j == 2:  # TV X column
                    cell.set_facecolor('#FCE4EC')
    
    ax_left.set_title('Average Performance (All SNR Levels)', fontsize=14, fontweight='bold', pad=20)
    
    # Right side: Best performance
    ax_right = fig.add_subplot(gs[2, 2:4])
    ax_right.axis('off')
    
    table_data_best = []
    metrics_best = [
        ('Mean Error', 'best_error', 'm'),
        ('Std Error', 'best_std', 'm'),
        ('Max Error', 'best_max', 'm')
    ]
    
    for metric_name, metric_key, unit in metrics_best:
        tvplus_val = performance_data['tvplus'][metric_key]
        tvx_val = performance_data['tvx'][metric_key]
        winner = 'TV Plus' if tvplus_val < tvx_val else 'TV X'
        diff_pct = abs((tvx_val - tvplus_val) / min(tvplus_val, tvx_val)) * 100
        table_data_best.append([metric_name, f'{tvplus_val:.4f} {unit}', f'{tvx_val:.4f} {unit}', 
                               f'{winner}\n({diff_pct:.1f}% better)'])
    
    table_best = ax_right.table(cellText=table_data_best,
                               colLabels=['Metric', 'TV Plus', 'TV X', 'Winner'],
                               cellLoc='center',
                               loc='center')
    table_best.auto_set_font_size(False)
    table_best.set_fontsize(12)
    table_best.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data_best) + 1):
        for j in range(4):
            cell = table_best[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#E6E6E6')
                cell.set_text_props(weight='bold')
            else:
                if j == 1:  # TV Plus column
                    cell.set_facecolor('#E3F2FD')
                elif j == 2:  # TV X column
                    cell.set_facecolor('#FCE4EC')
    
    ax_right.set_title('Best Performance (SNR 20 dB)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('report_figures/side_by_side_config_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/side_by_side_config_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_simple_side_by_side():
    """Create a simpler side-by-side view focused on layout only"""
    
    config_data, mic_configs = load_config_data_and_metrics()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Microphone Configuration Comparison', fontsize=18, fontweight='bold')
    
    colors = {'tvplus': '#2E86AB', 'tvx': '#A23B72'}
    config_labels = {'tvplus': 'TV Plus (Compact)', 'tvx': 'TV X (Distributed)'}
    
    for i, (config, mic_data) in enumerate(mic_configs.items()):
        ax = axes[i]
        
        # Plot microphones
        scatter = ax.scatter(mic_data['x'], mic_data['y'], 
                           s=800, c=colors[config], alpha=0.8, 
                           edgecolors='black', linewidth=4)
        
        # Add microphone labels
        for _, row in mic_data.iterrows():
            ax.annotate(row['mic_id'], (row['x'], row['y']), 
                       xytext=(0, 0), textcoords='offset points',
                       fontsize=16, fontweight='bold', ha='center', va='center',
                       color='white')
        
        # Add connecting lines
        positions = mic_data[['x', 'y']].values
        for j in range(len(positions)):
            for k in range(j+1, len(positions)):
                ax.plot([positions[j,0], positions[k,0]], 
                       [positions[j,1], positions[k,1]], 
                       'k--', alpha=0.5, linewidth=2)
        
        # Calculate and display key metrics
        geometry = calculate_geometric_properties(mic_data)
        
        # Performance data
        data = config_data[config]
        avg_error = data['mean_3d_error'].mean()
        best_error = data[data['snr'] == 20]['mean_3d_error'].mean() if len(data[data['snr'] == 20]) > 0 else np.nan
        
        # Add info box
        info_text = f"Spread: {geometry['spread']:.3f} m\n"
        info_text += f"Max Distance: {geometry['max_distance']:.3f} m\n"
        info_text += f"Avg Error: {avg_error:.4f} m\n"
        info_text += f"Best Error: {best_error:.4f} m"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=12, va='top', ha='left', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[config], alpha=0.2))
        
        ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
        ax.set_title(config_labels[config], fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Fix Y-axis range if all mics are at same Y coordinate
        y_range = mic_data['y'].max() - mic_data['y'].min()
        if y_range < 0.1:  # If Y range is very small
            y_center = mic_data['y'].mean()
            x_range = mic_data['x'].max() - mic_data['x'].min()
            y_padding = max(0.2, x_range * 0.3)  # Use 30% of X range or minimum 0.2m
            ax.set_ylim(y_center - y_padding, y_center + y_padding)
        
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('report_figures/simple_side_by_side_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/simple_side_by_side_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    print("Loading configuration data...")
    config_data, mic_configs = load_config_data_and_metrics()
    
    print("Creating detailed side-by-side comparison...")
    create_side_by_side_comparison(config_data, mic_configs)
    
    print("Creating simple side-by-side comparison...")
    create_simple_side_by_side()
    
    print("\nSide-by-side comparison complete!")
    print("Generated files:")
    print("- report_figures/side_by_side_config_comparison.pdf")
    print("- report_figures/side_by_side_config_comparison.png")
    print("- report_figures/simple_side_by_side_comparison.pdf")
    print("- report_figures/simple_side_by_side_comparison.png")

if __name__ == "__main__":
    main()