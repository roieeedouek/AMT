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
    
    return {
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'spread': spread,
        'centroid': centroid,
        'positions': positions
    }

def create_clear_side_by_side_comparison():
    """Create a clear side-by-side comparison using both X-Y and X-Z views"""
    
    config_data, mic_configs = load_config_data_and_metrics()
    
    # Calculate performance metrics
    performance_data = {}
    for config, data in config_data.items():
        performance_data[config] = {
            'avg_error': data['mean_3d_error'].mean(),
            'best_error': data[data['snr'] == 20]['mean_3d_error'].mean() if len(data[data['snr'] == 20]) > 0 else np.nan,
        }
    
    # Calculate geometric properties
    geometry_data = {}
    for config, mic_data in mic_configs.items():
        geometry_data[config] = calculate_geometric_properties(mic_data)
    
    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Microphone Configuration Comparison: TV Plus vs TV X', fontsize=18, fontweight='bold')
    
    colors = {'tvplus': '#2E86AB', 'tvx': '#A23B72'}
    config_labels = {'tvplus': 'TV Plus (Compact)', 'tvx': 'TV X (Distributed)'}
    
    # Row 1: X-Y view (top down) - even though Y is constant, show it for completeness
    for i, (config, mic_data) in enumerate(mic_configs.items()):
        ax = axes[0, i]
        
        # Plot microphones
        scatter = ax.scatter(mic_data['x'], mic_data['y'], 
                           s=800, c=colors[config], alpha=0.8, 
                           edgecolors='black', linewidth=4)
        
        # Add microphone labels
        for _, row in mic_data.iterrows():
            ax.annotate(row['label'], (row['x'], row['y']), 
                       xytext=(0, 0), textcoords='offset points',
                       fontsize=14, fontweight='bold', ha='center', va='center',
                       color='white')
        
        # Add connecting lines
        positions = mic_data[['x', 'y']].values
        for j in range(len(positions)):
            for k in range(j+1, len(positions)):
                ax.plot([positions[j,0], positions[k,0]], 
                       [positions[j,1], positions[k,1]], 
                       'k--', alpha=0.5, linewidth=2)
        
        # Set proper axis limits with padding
        x_center = mic_data['x'].mean()
        x_range = mic_data['x'].max() - mic_data['x'].min()
        x_padding = max(0.3, x_range * 0.2)
        
        y_center = mic_data['y'].mean()
        y_padding = max(0.3, x_range * 0.3)  # Use X range for Y padding since Y is constant
        
        ax.set_xlim(x_center - x_range/2 - x_padding, x_center + x_range/2 + x_padding)
        ax.set_ylim(y_center - y_padding, y_center + y_padding)
        
        ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        ax.set_title(f'{config_labels[config]} - Top View (X-Y)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add info text
        geom = geometry_data[config]
        perf = performance_data[config]
        info_text = f"Max Dist: {geom['max_distance']:.2f} m\n"
        info_text += f"Avg Error: {perf['avg_error']:.3f} m"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=11, va='top', ha='left', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[config], alpha=0.2))
    
    # Row 2: X-Z view (side view) - this will show the height differences better
    for i, (config, mic_data) in enumerate(mic_configs.items()):
        ax = axes[1, i]
        
        # Plot microphones in X-Z plane
        scatter = ax.scatter(mic_data['x'], mic_data['z'], 
                           s=800, c=colors[config], alpha=0.8, 
                           edgecolors='black', linewidth=4)
        
        # Add microphone labels
        for _, row in mic_data.iterrows():
            ax.annotate(row['label'], (row['x'], row['z']), 
                       xytext=(0, 0), textcoords='offset points',
                       fontsize=14, fontweight='bold', ha='center', va='center',
                       color='white')
        
        # Add connecting lines
        positions = mic_data[['x', 'z']].values
        for j in range(len(positions)):
            for k in range(j+1, len(positions)):
                ax.plot([positions[j,0], positions[k,0]], 
                       [positions[j,1], positions[k,1]], 
                       'k--', alpha=0.5, linewidth=2)
                
                # Add distance labels on some key lines
                if j == 0 and k == 1:  # Distance between left and right mics
                    mid_x = (positions[j,0] + positions[k,0]) / 2
                    mid_z = (positions[j,1] + positions[k,1]) / 2
                    distance = np.linalg.norm(positions[j] - positions[k])
                    ax.text(mid_x, mid_z + 0.05, f'{distance:.2f}m', 
                           fontsize=10, ha='center', va='bottom', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Set proper axis limits
        x_center = mic_data['x'].mean()
        x_range = mic_data['x'].max() - mic_data['x'].min()
        x_padding = max(0.3, x_range * 0.2)
        
        z_center = mic_data['z'].mean()
        z_range = mic_data['z'].max() - mic_data['z'].min()
        z_padding = max(0.2, z_range * 0.5)
        
        ax.set_xlim(x_center - x_range/2 - x_padding, x_center + x_range/2 + x_padding)
        ax.set_ylim(z_center - z_range/2 - z_padding, z_center + z_range/2 + z_padding)
        
        ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z Position (Height, m)', fontsize=12, fontweight='bold')
        ax.set_title(f'{config_labels[config]} - Side View (X-Z)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add coordinate info
        coord_text = "Microphone Coordinates:\n"
        for _, row in mic_data.iterrows():
            coord_text += f"{row['label']}: ({row['x']:.1f}, {row['y']:.1f}, {row['z']:.1f})\n"
        
        ax.text(0.98, 0.02, coord_text.strip(), transform=ax.transAxes, 
               fontsize=9, va='bottom', ha='right',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('report_figures/fixed_side_by_side_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/fixed_side_by_side_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_comparison_table():
    """Create a clean summary table comparing the configurations"""
    
    config_data, mic_configs = load_config_data_and_metrics()
    
    # Calculate metrics
    comparison_data = {}
    for config, data in config_data.items():
        mic_data = mic_configs[config]
        geom = calculate_geometric_properties(mic_data)
        
        comparison_data[config] = {
            'config_name': config.upper(),
            'layout_type': 'Compact' if config == 'tvplus' else 'Distributed',
            'max_distance': geom['max_distance'],
            'spread': geom['spread'],
            'avg_error': data['mean_3d_error'].mean(),
            'best_error': data[data['snr'] == 20]['mean_3d_error'].mean() if len(data[data['snr'] == 20]) > 0 else np.nan,
            'x_range': mic_data['x'].max() - mic_data['x'].min(),
            'z_range': mic_data['z'].max() - mic_data['z'].min()
        }
    
    # Create comparison figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('off')
    
    # Prepare table data
    table_data = [
        ['Configuration', 'TV Plus', 'TV X', 'Difference'],
        ['Layout Type', 'Compact', 'Distributed', '-'],
        ['Max Distance (m)', f"{comparison_data['tvplus']['max_distance']:.3f}", 
         f"{comparison_data['tvx']['max_distance']:.3f}", 
         f"{comparison_data['tvx']['max_distance'] - comparison_data['tvplus']['max_distance']:+.3f}"],
        ['Array Spread (m)', f"{comparison_data['tvplus']['spread']:.3f}", 
         f"{comparison_data['tvx']['spread']:.3f}", 
         f"{comparison_data['tvx']['spread'] - comparison_data['tvplus']['spread']:+.3f}"],
        ['X Range (m)', f"{comparison_data['tvplus']['x_range']:.1f}", 
         f"{comparison_data['tvx']['x_range']:.1f}", 
         f"{comparison_data['tvx']['x_range'] - comparison_data['tvplus']['x_range']:+.1f}"],
        ['Z Range (m)', f"{comparison_data['tvplus']['z_range']:.1f}", 
         f"{comparison_data['tvx']['z_range']:.1f}", 
         f"{comparison_data['tvx']['z_range'] - comparison_data['tvplus']['z_range']:+.1f}"],
        ['Avg Error (m)', f"{comparison_data['tvplus']['avg_error']:.4f}", 
         f"{comparison_data['tvx']['avg_error']:.4f}", 
         f"{comparison_data['tvx']['avg_error'] - comparison_data['tvplus']['avg_error']:+.4f}"],
        ['Best Error (m)', f"{comparison_data['tvplus']['best_error']:.4f}", 
         f"{comparison_data['tvx']['best_error']:.4f}", 
         f"{comparison_data['tvx']['best_error'] - comparison_data['tvplus']['best_error']:+.4f}"]
    ]
    
    # Create table
    table = ax.table(cellText=table_data[1:],
                    colLabels=table_data[0],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.5)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 0:  # First column
                    cell.set_facecolor('#E7E6E6')
                    cell.set_text_props(weight='bold')
                elif j == 1:  # TV Plus column
                    cell.set_facecolor('#E3F2FD')
                elif j == 2:  # TV X column
                    cell.set_facecolor('#FCE4EC')
                elif j == 3:  # Difference column
                    cell.set_facecolor('#F0F0F0')
                    # Color code differences
                    cell_text = table_data[i][j]
                    if '+' in cell_text and 'Error' in table_data[i][0]:
                        cell.set_facecolor('#FFEBEE')  # Light red for worse performance
                    elif '-' in cell_text and 'Error' in table_data[i][0]:
                        cell.set_facecolor('#E8F5E8')  # Light green for better performance
    
    ax.set_title('Configuration Comparison Summary', fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('report_figures/configuration_comparison_table.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/configuration_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_data

def main():
    """Main function"""
    print("Loading configuration data...")
    
    print("Creating fixed side-by-side comparison...")
    create_clear_side_by_side_comparison()
    
    print("Creating summary comparison table...")
    comparison_data = create_summary_comparison_table()
    
    print("\nFixed side-by-side comparison complete!")
    print("Generated files:")
    print("- report_figures/fixed_side_by_side_comparison.pdf")
    print("- report_figures/fixed_side_by_side_comparison.png")
    print("- report_figures/configuration_comparison_table.pdf")
    print("- report_figures/configuration_comparison_table.png")
    
    print("\nKey findings:")
    print(f"TV Plus max distance: {comparison_data['tvplus']['max_distance']:.3f} m")
    print(f"TV X max distance: {comparison_data['tvx']['max_distance']:.3f} m")
    print(f"TV X has {comparison_data['tvx']['max_distance']/comparison_data['tvplus']['max_distance']:.1f}x larger baseline")

if __name__ == "__main__":
    main()