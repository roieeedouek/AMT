import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_all_configuration_data():
    """Load microphone configurations and performance metrics for all three configs"""
    data_dir = Path("simulation_data")
    
    config_data = {}
    mic_configs = {}
    
    # Define all configurations
    configs = ['tvplus', 'tvx', 'soundbar3']
    config_labels = {
        'tvplus': 'TV Plus',
        'tvx': 'TV X', 
        'soundbar3': 'Soundbar3'
    }
    
    for config in configs:
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
    
    # Calculate array area/geometry metric
    if len(positions) == 3:
        # Triangle area for 3 mics
        v1 = positions[1] - positions[0]
        v2 = positions[2] - positions[0]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        geometry_metric = area
    else:
        # Quadrilateral area for 4 mics (approximate)
        geometry_metric = dimensions[0] * dimensions[1]  # X-Y area
    
    return {
        'num_mics': len(positions),
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'spread': spread,
        'geometry_metric': geometry_metric,
        'dimensions': dimensions,
        'centroid': centroid,
        'positions': positions
    }

def create_updated_configuration_comparison_table():
    """Create comprehensive comparison table for all three configurations"""
    
    config_data, mic_configs = load_all_configuration_data()
    
    # Calculate metrics for each configuration
    comparison_data = {}
    for config, data in config_data.items():
        mic_data = mic_configs[config]
        geom = calculate_geometric_properties(mic_data)
        
        comparison_data[config] = {
            'config_name': config.upper(),
            'layout_type': 'Linear' if config == 'soundbar3' else ('Compact' if config == 'tvplus' else 'Distributed'),
            'mic_count': geom['num_mics'],
            'max_distance': geom['max_distance'],
            'mean_distance': geom['mean_distance'],
            'spread': geom['spread'],
            'avg_error': data['mean_3d_error'].mean(),
            'best_error': data[data['snr'] == 20]['mean_3d_error'].mean() if len(data[data['snr'] == 20]) > 0 else np.nan,
            'worst_error': data[data['snr'] == 0]['mean_3d_error'].mean() if len(data[data['snr'] == 0]) > 0 else np.nan,
            'x_range': geom['dimensions'][0],
            'z_range': geom['dimensions'][2],
            'geometry_metric': geom['geometry_metric']
        }
    
    # Create comparison figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.axis('off')
    
    # Prepare comprehensive table data
    table_data = [
        ['Metric', 'TV Plus', 'TV X', 'Soundbar3', 'Best Config']
    ]
    
    # Configuration overview
    table_data.extend([
        ['Configuration Type', 'Compact', 'Distributed', 'Linear', '-'],
        ['Microphone Count', '4', '4', '3', 'TV Plus/TV X'],
        ['', '', '', '', ''],  # Spacer row
        ['GEOMETRIC PROPERTIES', '', '', '', ''],
    ])
    
    # Geometric comparisons
    metrics_to_compare = [
        ('Max Distance (m)', 'max_distance'),
        ('Mean Distance (m)', 'mean_distance'),
        ('Array Spread (m)', 'spread'),
        ('X Range (m)', 'x_range'),
        ('Z Range (m)', 'z_range'),
    ]
    
    for metric_name, metric_key in metrics_to_compare:
        tvplus_val = comparison_data['tvplus'][metric_key]
        tvx_val = comparison_data['tvx'][metric_key]
        soundbar3_val = comparison_data['soundbar3'][metric_key]
        
        # Determine best (depends on metric - for distances, larger might be better for triangulation)
        if metric_key in ['max_distance', 'mean_distance', 'spread']:
            best_config = 'TV X' if tvx_val == max(tvplus_val, tvx_val, soundbar3_val) else \
                         ('Soundbar3' if soundbar3_val == max(tvplus_val, tvx_val, soundbar3_val) else 'TV Plus')
        else:
            # For ranges, it depends on application
            best_config = 'Varies'
        
        table_data.append([
            metric_name,
            f'{tvplus_val:.3f}',
            f'{tvx_val:.3f}',
            f'{soundbar3_val:.3f}',
            best_config
        ])
    
    # Performance comparisons
    table_data.extend([
        ['', '', '', '', ''],  # Spacer
        ['PERFORMANCE METRICS', '', '', '', ''],
    ])
    
    performance_metrics = [
        ('Average Error (m)', 'avg_error'),
        ('Best Error (SNR 20dB)', 'best_error'),
        ('Worst Error (SNR 0dB)', 'worst_error'),
    ]
    
    for metric_name, metric_key in performance_metrics:
        tvplus_val = comparison_data['tvplus'][metric_key]
        tvx_val = comparison_data['tvx'][metric_key]
        soundbar3_val = comparison_data['soundbar3'][metric_key]
        
        # For performance metrics, lower is better
        values = [tvplus_val, tvx_val, soundbar3_val]
        min_val = min(values)
        if tvplus_val == min_val:
            best_config = 'TV Plus'
        elif tvx_val == min_val:
            best_config = 'TV X'
        else:
            best_config = 'Soundbar3'
        
        table_data.append([
            metric_name,
            f'{tvplus_val:.4f}' if not np.isnan(tvplus_val) else 'N/A',
            f'{tvx_val:.4f}' if not np.isnan(tvx_val) else 'N/A',
            f'{soundbar3_val:.4f}' if not np.isnan(soundbar3_val) else 'N/A',
            best_config
        ])
    
    # Performance improvements
    table_data.extend([
        ['', '', '', '', ''],  # Spacer
        ['SNR IMPROVEMENT POTENTIAL', '', '', '', ''],
    ])
    
    for config in ['tvplus', 'tvx', 'soundbar3']:
        data = config_data[config]
        snr_0_error = data[data['snr'] == 0]['mean_3d_error'].mean()
        snr_20_data = data[data['snr'] == 20]
        if len(snr_20_data) > 0:
            snr_20_error = snr_20_data['mean_3d_error'].mean()
            improvement = ((snr_0_error - snr_20_error) / snr_0_error) * 100
            comparison_data[config]['snr_improvement'] = improvement
    
    # Find best improvement
    improvements = [comparison_data[config]['snr_improvement'] for config in ['tvplus', 'tvx', 'soundbar3']]
    max_improvement = max(improvements)
    best_improvement_config = ['TV Plus', 'TV X', 'Soundbar3'][improvements.index(max_improvement)]
    
    table_data.append([
        'SNR 0→20dB Improvement (%)',
        f'{comparison_data["tvplus"]["snr_improvement"]:.1f}%',
        f'{comparison_data["tvx"]["snr_improvement"]:.1f}%',
        f'{comparison_data["soundbar3"]["snr_improvement"]:.1f}%',
        best_improvement_config
    ])
    
    # Deployment recommendations
    table_data.extend([
        ['', '', '', '', ''],  # Spacer
        ['DEPLOYMENT RECOMMENDATIONS', '', '', '', ''],
        ['Best for Average Conditions', 'Excellent', 'Good', 'Fair', 'TV Plus'],
        ['Best for High SNR', 'Good', 'Excellent', 'Fair', 'TV X'],
        ['Simplest Hardware', 'No', 'No', 'Yes', 'Soundbar3'],
        ['Most Consistent', 'Excellent', 'Good', 'Good', 'TV Plus'],
        ['Widest Coverage', 'Fair', 'Excellent', 'Good', 'TV X'],
    ])
    
    # Create the table
    table = ax.table(cellText=table_data[1:],
                    colLabels=table_data[0],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#2E5984')
                cell.set_text_props(weight='bold', color='white')
            elif j == 0 and any(keyword in table_data[i][0] for keyword in ['GEOMETRIC', 'PERFORMANCE', 'SNR IMPROVEMENT', 'DEPLOYMENT']):
                # Section headers
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            elif table_data[i][0] == '':  # Spacer rows
                cell.set_facecolor('#FFFFFF')
            elif j == 0:  # First column (metric names)
                cell.set_facecolor('#E7E6E6')
                cell.set_text_props(weight='bold')
            elif j == 1:  # TV Plus column
                cell.set_facecolor('#E3F2FD')
            elif j == 2:  # TV X column
                cell.set_facecolor('#FCE4EC')
            elif j == 3:  # Soundbar3 column
                cell.set_facecolor('#FFF3E0')
            elif j == 4:  # Best config column
                cell.set_facecolor('#E8F5E8')
                cell.set_text_props(weight='bold')
                
                # Color code the best config cells
                if 'TV Plus' in table_data[i][j]:
                    cell.set_facecolor('#C8E6C9')
                elif 'TV X' in table_data[i][j]:
                    cell.set_facecolor('#F8BBD9')
                elif 'Soundbar3' in table_data[i][j]:
                    cell.set_facecolor('#FFE0B2')
    
    ax.set_title('Comprehensive Configuration Comparison Table\nTV Plus vs TV X vs Soundbar3', 
                fontsize=16, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig('report_figures/updated_configuration_comparison_table.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/updated_configuration_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_data

def create_configuration_summary_stats():
    """Create a summary statistics comparison"""
    
    config_data, mic_configs = load_all_configuration_data()
    
    print("\n" + "="*100)
    print("UPDATED CONFIGURATION COMPARISON SUMMARY")
    print("="*100)
    
    print("\n1. CONFIGURATION OVERVIEW:")
    print("-" * 70)
    print(f"{'Configuration':<15} {'Type':<12} {'Mics':<5} {'Max Dist':<10} {'Avg Error':<12} {'Best Error':<12}")
    print("-" * 70)
    
    config_labels = {
        'tvplus': 'TV Plus',
        'tvx': 'TV X',
        'soundbar3': 'Soundbar3'
    }
    
    config_types = {
        'tvplus': 'Compact',
        'tvx': 'Distributed', 
        'soundbar3': 'Linear'
    }
    
    for config, data in config_data.items():
        mic_data = mic_configs[config]
        geom = calculate_geometric_properties(mic_data)
        
        avg_error = data['mean_3d_error'].mean()
        best_error = data[data['snr'] == 20]['mean_3d_error'].mean() if len(data[data['snr'] == 20]) > 0 else np.nan
        
        print(f"{config_labels[config]:<15} {config_types[config]:<12} {geom['num_mics']:<5} {geom['max_distance']:<10.3f} {avg_error:<12.4f} {best_error:<12.4f}")
    
    print("\n2. RANKING BY PERFORMANCE:")
    print("-" * 50)
    
    # Average performance ranking
    avg_performance = []
    for config, data in config_data.items():
        avg_error = data['mean_3d_error'].mean()
        avg_performance.append((config_labels[config], avg_error))
    
    avg_performance.sort(key=lambda x: x[1])
    
    print("Average Performance Ranking:")
    for i, (config_name, error) in enumerate(avg_performance, 1):
        print(f"{i}. {config_name}: {error:.4f} m")
    
    # Best performance ranking
    best_performance = []
    for config, data in config_data.items():
        best_data = data[data['snr'] == 20]
        if len(best_data) > 0:
            best_error = best_data['mean_3d_error'].mean()
            best_performance.append((config_labels[config], best_error))
    
    best_performance.sort(key=lambda x: x[1])
    
    print("\nBest Performance Ranking (SNR 20 dB):")
    for i, (config_name, error) in enumerate(best_performance, 1):
        print(f"{i}. {config_name}: {error:.4f} m")
    
    print("\n3. KEY INSIGHTS:")
    print("-" * 50)
    
    # 3-mic vs 4-mic analysis
    soundbar3_avg = config_data['soundbar3']['mean_3d_error'].mean()
    tvplus_avg = config_data['tvplus']['mean_3d_error'].mean()
    tvx_avg = config_data['tvx']['mean_3d_error'].mean()
    
    print(f"• Microphone Count Impact:")
    print(f"  - 3-mic (Soundbar3): {soundbar3_avg:.4f} m")
    print(f"  - 4-mic Average: {(tvplus_avg + tvx_avg)/2:.4f} m")
    
    if soundbar3_avg > (tvplus_avg + tvx_avg)/2:
        diff_pct = ((soundbar3_avg - (tvplus_avg + tvx_avg)/2) / (tvplus_avg + tvx_avg)/2) * 100
        print(f"  -> 4-mic configurations outperform 3-mic by {diff_pct:.1f}%")
    else:
        diff_pct = (((tvplus_avg + tvx_avg)/2 - soundbar3_avg) / soundbar3_avg) * 100
        print(f"  -> 3-mic configuration competitive with 4-mic (within {diff_pct:.1f}%)")
    
    # Geometry vs performance
    print(f"\n• Geometry vs Performance:")
    for config, data in config_data.items():
        mic_data = mic_configs[config]
        geom = calculate_geometric_properties(mic_data)
        avg_error = data['mean_3d_error'].mean()
        
        print(f"  - {config_labels[config]}: {geom['max_distance']:.3f}m baseline -> {avg_error:.4f}m error")
    
    print("="*100)

def main():
    """Main function"""
    print("Creating updated configuration comparison table...")
    comparison_data = create_updated_configuration_comparison_table()
    
    print("Generating configuration summary statistics...")
    create_configuration_summary_stats()
    
    print("\nUpdated configuration comparison complete!")
    print("Generated files:")
    print("- report_figures/updated_configuration_comparison_table.pdf")
    print("- report_figures/updated_configuration_comparison_table.png")

if __name__ == "__main__":
    main()