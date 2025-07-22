import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def load_all_configurations():
    """Load simulation data for all three configurations: tvplus, tvx, soundbar3"""
    data_dir = Path("simulation_data")
    snr_levels = [0, 5, 10, 20]
    
    all_config_data = {}
    mic_configs = {}
    
    # Define all configurations
    configs = ['tvplus', 'tvx', 'soundbar3']
    config_labels = {
        'tvplus': 'TV Plus (4-mic Compact)',
        'tvx': 'TV X (4-mic Distributed)', 
        'soundbar3': 'Soundbar (3-mic Linear)'
    }
    
    for config in configs:
        config_metrics = []
        config_tracking = {}
        
        for snr in snr_levels:
            try:
                # Find metrics file
                metrics_pattern = f"{config}_SNR{snr}_*_metrics.csv"
                metrics_files = list(data_dir.glob(metrics_pattern))
                
                if not metrics_files:
                    print(f"Warning: No metrics file found for {config} SNR{snr}")
                    continue
                
                metrics_file = metrics_files[0]
                filename_parts = metrics_file.stem.split('_')
                
                # Handle different naming patterns
                if config == 'soundbar3':
                    session_start = filename_parts[2] + '_' + filename_parts[3]
                else:
                    session_start = filename_parts[2] + '_' + filename_parts[3]
                
                # Load metrics
                metrics = pd.read_csv(metrics_file)
                metrics['snr'] = snr
                metrics['config'] = config
                metrics['config_label'] = config_labels[config]
                metrics['mic_count'] = 3 if config == 'soundbar3' else 4
                config_metrics.append(metrics)
                
                # Load tracking data
                true_pos_pattern = f"{config}_SNR{snr}_{session_start}_*_tracking_*_true_positions.csv"
                est_pos_pattern = f"{config}_SNR{snr}_{session_start}_*_tracking_*_estimated_positions.csv"
                filtered_pos_pattern = f"{config}_SNR{snr}_{session_start}_*_tracking_*_filtered_positions.csv"
                
                true_pos_files = list(data_dir.glob(true_pos_pattern))
                est_pos_files = list(data_dir.glob(est_pos_pattern))
                filtered_pos_files = list(data_dir.glob(filtered_pos_pattern))
                
                if all([true_pos_files, est_pos_files, filtered_pos_files]):
                    config_tracking[snr] = {
                        'true': pd.read_csv(true_pos_files[0]),
                        'estimated': pd.read_csv(est_pos_files[0]),
                        'filtered': pd.read_csv(filtered_pos_files[0])
                    }
                
                # Load microphone configuration (once per config)
                if config not in mic_configs:
                    mic_pattern = f"{config}_SNR{snr}_{session_start}_*microphones.csv"
                    mic_files = list(data_dir.glob(mic_pattern))
                    if mic_files:
                        mic_data = pd.read_csv(mic_files[0])
                        mic_data['config'] = config
                        mic_data['config_label'] = config_labels[config]
                        mic_configs[config] = mic_data
                        
            except Exception as e:
                print(f"Warning: Could not load data for {config} SNR{snr}: {e}")
                continue
        
        if config_metrics:
            all_config_data[config] = {
                'metrics': pd.concat(config_metrics, ignore_index=True),
                'tracking': config_tracking
            }
    
    return all_config_data, mic_configs

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
    
    # Calculate array area/volume
    if len(positions) == 3:
        # Triangle area for 3 mics
        v1 = positions[1] - positions[0]
        v2 = positions[2] - positions[0]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        geometry_metric = area
    else:
        # Quadrilateral area for 4 mics (approximate)
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        dims = max_coords - min_coords
        geometry_metric = dims[0] * dims[1]  # X-Y area
    
    return {
        'num_mics': len(positions),
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'spread': spread,
        'geometry_metric': geometry_metric,
        'centroid': centroid,
        'positions': positions
    }

def create_comprehensive_analysis_plots(all_config_data, mic_configs):
    """Create comprehensive analysis comparing all three configurations"""
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (20, 16)
    })
    
    # Create main comparison figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Comprehensive Analysis: 3-Mic vs 4-Mic Configuration Comparison', fontsize=18, fontweight='bold')
    
    snr_levels = [0, 5, 10, 20]
    colors = {'tvplus': '#2E86AB', 'tvx': '#A23B72', 'soundbar3': '#F18F01'}
    markers = {'tvplus': 'o', 'tvx': 's', 'soundbar3': '^'}
    
    # Plot 1: SNR Performance Comparison
    for config, data in all_config_data.items():
        metrics = data['metrics']
        snr_stats = metrics.groupby('snr')['mean_3d_error'].agg(['mean', 'std'])
        
        mean_errors = [snr_stats.loc[snr, 'mean'] if snr in snr_stats.index else np.nan for snr in snr_levels]
        error_stds = [snr_stats.loc[snr, 'std'] if snr in snr_stats.index else 0 for snr in snr_levels]
        
        valid_idx = ~np.isnan(mean_errors)
        valid_snr = np.array(snr_levels)[valid_idx]
        valid_means = np.array(mean_errors)[valid_idx]
        valid_stds = np.array(error_stds)[valid_idx]
        
        config_label = metrics['config_label'].iloc[0]
        axes[0,0].errorbar(valid_snr, valid_means, yerr=valid_stds, 
                          marker=markers[config], linewidth=3, markersize=10, capsize=5,
                          label=config_label, color=colors[config])
    
    axes[0,0].set_xlabel('SNR (dB)')
    axes[0,0].set_ylabel('Mean 3D Error (m)')
    axes[0,0].set_title('Mean 3D Error vs SNR by Configuration')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Error Variability Comparison
    for config, data in all_config_data.items():
        metrics = data['metrics']
        snr_stats = metrics.groupby('snr')['std_3d_error'].agg(['mean'])
        
        std_errors = [snr_stats.loc[snr, 'mean'] if snr in snr_stats.index else np.nan for snr in snr_levels]
        
        valid_idx = ~np.isnan(std_errors)
        valid_snr = np.array(snr_levels)[valid_idx]
        valid_stds = np.array(std_errors)[valid_idx]
        
        config_label = metrics['config_label'].iloc[0]
        axes[0,1].plot(valid_snr, valid_stds, 
                      marker=markers[config], linewidth=3, markersize=10,
                      label=config_label, color=colors[config])
    
    axes[0,1].set_xlabel('SNR (dB)')
    axes[0,1].set_ylabel('Error Standard Deviation (m)')
    axes[0,1].set_title('Error Variability by Configuration')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Microphone Configurations Visualization
    for i, (config, mic_data) in enumerate(mic_configs.items()):
        if i < 3:  # Only plot first 3 configs
            ax = axes[0, 2] if i == 0 else (axes[1, 2] if i == 1 else axes[2, 2])
            
            # Plot microphones
            scatter = ax.scatter(mic_data['x'], mic_data['y'], 
                               s=500, c=colors[config], alpha=0.8,
                               edgecolors='black', linewidth=3, marker=markers[config])
            
            # Add microphone labels
            for _, row in mic_data.iterrows():
                ax.annotate(row['label'], (row['x'], row['y']), 
                           xytext=(0, 0), textcoords='offset points',
                           fontsize=12, fontweight='bold', ha='center', va='center',
                           color='white')
            
            # Add connecting lines
            positions = mic_data[['x', 'y']].values
            for j in range(len(positions)):
                for k in range(j+1, len(positions)):
                    ax.plot([positions[j,0], positions[k,0]], 
                           [positions[j,1], positions[k,1]], 
                           'k--', alpha=0.4, linewidth=2)
            
            # Calculate and display geometry info
            geom = calculate_geometric_properties(mic_data)
            info_text = f"Mics: {geom['num_mics']}\n"
            info_text += f"Max Dist: {geom['max_distance']:.2f}m\n"
            info_text += f"Spread: {geom['spread']:.3f}m"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=11, va='top', ha='left', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[config], alpha=0.2))
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{mic_data["config_label"].iloc[0]}')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Fix axis limits
            x_center = mic_data['x'].mean()
            y_center = mic_data['y'].mean()
            x_range = mic_data['x'].max() - mic_data['x'].min()
            padding = max(0.3, x_range * 0.3)
            
            ax.set_xlim(x_center - x_range/2 - padding, x_center + x_range/2 + padding)
            ax.set_ylim(y_center - padding, y_center + padding)
    
    # Plot 4: Performance Summary Bar Chart
    summary_data = []
    for config, data in all_config_data.items():
        metrics = data['metrics']
        config_label = metrics['config_label'].iloc[0]
        
        avg_error = metrics['mean_3d_error'].mean()
        best_error = metrics[metrics['snr'] == 20]['mean_3d_error'].mean() if len(metrics[metrics['snr'] == 20]) > 0 else np.nan
        
        summary_data.append({
            'config': config,
            'config_label': config_label,
            'avg_error': avg_error,
            'best_error': best_error
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    x_pos = np.arange(len(summary_df))
    width = 0.35
    
    bars1 = axes[1,0].bar(x_pos - width/2, summary_df['avg_error'], width, 
                         label='Average (All SNR)', alpha=0.8)
    bars2 = axes[1,0].bar(x_pos + width/2, summary_df['best_error'], width, 
                         label='Best (SNR 20 dB)', alpha=0.8)
    
    # Color bars according to configuration
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        config = summary_df.iloc[i]['config']
        bar1.set_color(colors[config])
        bar2.set_color(colors[config])
        bar2.set_alpha(0.6)
    
    axes[1,0].set_xlabel('Configuration')
    axes[1,0].set_ylabel('Mean 3D Error (m)')
    axes[1,0].set_title('Performance Summary')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels([label.replace(' ', '\\n') for label in summary_df['config_label']], fontsize=10)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: SNR Improvement Analysis
    for config, data in all_config_data.items():
        metrics = data['metrics']
        config_label = metrics['config_label'].iloc[0]
        
        snr_0_error = metrics[metrics['snr'] == 0]['mean_3d_error'].mean()
        improvements = []
        
        for snr in [5, 10, 20]:
            snr_data = metrics[metrics['snr'] == snr]
            if len(snr_data) > 0:
                snr_error = snr_data['mean_3d_error'].mean()
                improvement = ((snr_0_error - snr_error) / snr_0_error) * 100
                improvements.append(improvement)
            else:
                improvements.append(0)
        
        axes[1,1].plot([5, 10, 20], improvements, 
                      marker=markers[config], linewidth=3, markersize=10,
                      label=config_label, color=colors[config])
    
    axes[1,1].set_xlabel('SNR (dB)')
    axes[1,1].set_ylabel('Improvement over SNR 0 dB (%)')
    axes[1,1].set_title('SNR Improvement Potential')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Configuration Comparison Table (as text plot)
    axes[2,0].axis('off')
    axes[2,1].axis('off')
    
    # Create comparison table data
    table_data = [['Configuration', 'Mic Count', 'Avg Error (m)', 'Best Error (m)', 'Max Improvement']]
    
    for config, data in all_config_data.items():
        metrics = data['metrics']
        mic_data = mic_configs[config]
        config_label = metrics['config_label'].iloc[0]
        
        mic_count = len(mic_data)
        avg_error = metrics['mean_3d_error'].mean()
        best_error = metrics[metrics['snr'] == 20]['mean_3d_error'].mean() if len(metrics[metrics['snr'] == 20]) > 0 else np.nan
        
        snr_0_error = metrics[metrics['snr'] == 0]['mean_3d_error'].mean()
        max_improvement = ((snr_0_error - best_error) / snr_0_error) * 100 if not np.isnan(best_error) else 0
        
        table_data.append([
            config_label,
            str(mic_count),
            f'{avg_error:.4f}',
            f'{best_error:.4f}' if not np.isnan(best_error) else 'N/A',
            f'{max_improvement:.1f}%'
        ])
    
    # Display table
    table = axes[2,0].table(cellText=table_data[1:],
                           colLabels=table_data[0],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    axes[2,0].set_title('Configuration Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('report_figures/comprehensive_three_config_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/comprehensive_three_config_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_comprehensive_summary(all_config_data, mic_configs):
    """Generate comprehensive summary comparing all configurations"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE THREE-CONFIGURATION ANALYSIS")
    print("="*100)
    
    print("\n1. CONFIGURATION OVERVIEW:")
    print("-" * 60)
    print(f"{'Configuration':<25} {'Mic Count':<10} {'Geometry':<15} {'Max Distance':<12}")
    print("-" * 60)
    
    for config, data in all_config_data.items():
        mic_data = mic_configs[config]
        geom = calculate_geometric_properties(mic_data)
        config_label = data['metrics']['config_label'].iloc[0]
        
        geometry_type = "Linear" if config == 'soundbar3' else ("Compact" if config == 'tvplus' else "Distributed")
        
        print(f"{config_label:<25} {geom['num_mics']:<10} {geometry_type:<15} {geom['max_distance']:<12.3f}")
    
    print("\n2. PERFORMANCE BY SNR LEVEL:")
    print("-" * 80)
    print(f"{'Configuration':<25} {'SNR 0':<10} {'SNR 5':<10} {'SNR 10':<10} {'SNR 20':<10}")
    print("-" * 80)
    
    for config, data in all_config_data.items():
        metrics = data['metrics']
        config_label = metrics['config_label'].iloc[0]
        
        performance_row = [config_label]
        for snr in [0, 5, 10, 20]:
            snr_data = metrics[metrics['snr'] == snr]
            if len(snr_data) > 0:
                error = snr_data['mean_3d_error'].mean()
                performance_row.append(f"{error:.4f}")
            else:
                performance_row.append("N/A")
        
        print(f"{performance_row[0]:<25} {performance_row[1]:<10} {performance_row[2]:<10} {performance_row[3]:<10} {performance_row[4]:<10}")
    
    print("\n3. CONFIGURATION RANKINGS:")
    print("-" * 50)
    
    # Calculate average performance
    avg_performance = []
    for config, data in all_config_data.items():
        metrics = data['metrics']
        avg_error = metrics['mean_3d_error'].mean()
        config_label = metrics['config_label'].iloc[0]
        avg_performance.append((config_label, avg_error))
    
    avg_performance.sort(key=lambda x: x[1])
    
    print("Average Performance Ranking:")
    for i, (config_label, error) in enumerate(avg_performance, 1):
        print(f"{i}. {config_label}: {error:.4f} m")
    
    # Calculate best performance
    best_performance = []
    for config, data in all_config_data.items():
        metrics = data['metrics']
        best_data = metrics[metrics['snr'] == 20]
        if len(best_data) > 0:
            best_error = best_data['mean_3d_error'].mean()
            config_label = metrics['config_label'].iloc[0]
            best_performance.append((config_label, best_error))
    
    best_performance.sort(key=lambda x: x[1])
    
    print("\nBest Performance Ranking (SNR 20 dB):")
    for i, (config_label, error) in enumerate(best_performance, 1):
        print(f"{i}. {config_label}: {error:.4f} m")
    
    print("\n4. KEY INSIGHTS:")
    print("-" * 50)
    
    # 3-mic vs 4-mic comparison
    soundbar3_avg = all_config_data['soundbar3']['metrics']['mean_3d_error'].mean()
    tvplus_avg = all_config_data['tvplus']['metrics']['mean_3d_error'].mean()
    tvx_avg = all_config_data['tvx']['metrics']['mean_3d_error'].mean()
    
    print(f"• 3-mic vs 4-mic Performance:")
    print(f"  - Soundbar (3-mic): {soundbar3_avg:.4f} m")
    print(f"  - TV Plus (4-mic compact): {tvplus_avg:.4f} m")
    print(f"  - TV X (4-mic distributed): {tvx_avg:.4f} m")
    
    if soundbar3_avg < min(tvplus_avg, tvx_avg):
        print("  -> 3-microphone configuration outperforms 4-microphone setups")
    else:
        print("  -> 4-microphone configurations generally outperform 3-microphone setup")
    
    # SNR impact
    print(f"\n• SNR Impact Analysis:")
    for config, data in all_config_data.items():
        metrics = data['metrics']
        config_label = metrics['config_label'].iloc[0]
        
        snr_0_error = metrics[metrics['snr'] == 0]['mean_3d_error'].mean()
        snr_20_data = metrics[metrics['snr'] == 20]
        if len(snr_20_data) > 0:
            snr_20_error = snr_20_data['mean_3d_error'].mean()
            improvement = ((snr_0_error - snr_20_error) / snr_0_error) * 100
            print(f"  - {config_label}: {improvement:.1f}% improvement (SNR 0->20 dB)")
    
    print("="*100)
    
    # Save summary data
    summary_data = []
    for config, data in all_config_data.items():
        metrics = data['metrics']
        mic_data = mic_configs[config]
        geom = calculate_geometric_properties(mic_data)
        
        for snr in [0, 5, 10, 20]:
            snr_data = metrics[metrics['snr'] == snr]
            if len(snr_data) > 0:
                row = {
                    'Configuration': metrics['config_label'].iloc[0],
                    'Config_Code': config,
                    'Mic_Count': geom['num_mics'],
                    'SNR': snr,
                    'Mean_Error': snr_data['mean_3d_error'].mean(),
                    'Std_Error': snr_data['std_3d_error'].mean(),
                    'Max_Error': snr_data['max_3d_error'].mean(),
                    'P95_Error': snr_data['p95_3d_error'].mean(),
                    'Max_Distance': geom['max_distance'],
                    'Spread': geom['spread']
                }
                summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('report_figures/comprehensive_config_analysis_summary.csv', index=False)
    
    return summary_df

def main():
    """Main analysis function"""
    print("Loading all configuration data (TV Plus, TV X, Soundbar3)...")
    all_config_data, mic_configs = load_all_configurations()
    
    print(f"Loaded data for {len(all_config_data)} configurations:")
    for config, data in all_config_data.items():
        print(f"  - {config}: {len(data['metrics'])} simulation runs")
    
    print("Creating comprehensive analysis plots...")
    create_comprehensive_analysis_plots(all_config_data, mic_configs)
    
    print("Generating comprehensive summary...")
    summary_df = generate_comprehensive_summary(all_config_data, mic_configs)
    
    print("\nComprehensive analysis complete!")
    print("Generated files:")
    print("- report_figures/comprehensive_three_config_analysis.pdf")
    print("- report_figures/comprehensive_three_config_analysis.png")
    print("- report_figures/comprehensive_config_analysis_summary.csv")

if __name__ == "__main__":
    main()