import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def load_configuration_data():
    """Load simulation data separately for each configuration"""
    data_dir = Path("simulation_data")
    snr_levels = [0, 5, 10, 20]
    
    config_data = {}
    mic_configs = {}
    
    for config in ['tvplus', 'tvx']:
        config_metrics = []
        config_tracking = {}
        
        for snr in snr_levels:
            try:
                # Find files for this SNR and config
                metrics_pattern = f"{config}_SNR{snr}_*_metrics.csv"
                metrics_files = list(data_dir.glob(metrics_pattern))
                
                if not metrics_files:
                    continue
                
                metrics_file = metrics_files[0]
                filename_parts = metrics_file.stem.split('_')
                session_start = filename_parts[2] + '_' + filename_parts[3]
                
                true_pos_pattern = f"{config}_SNR{snr}_{session_start}_*_tracking_*_true_positions.csv"
                est_pos_pattern = f"{config}_SNR{snr}_{session_start}_*_tracking_*_estimated_positions.csv"
                filtered_pos_pattern = f"{config}_SNR{snr}_{session_start}_*_tracking_*_filtered_positions.csv"
                mic_pattern = f"{config}_SNR{snr}_{session_start}_*microphones.csv"
                
                true_pos_files = list(data_dir.glob(true_pos_pattern))
                est_pos_files = list(data_dir.glob(est_pos_pattern))
                filtered_pos_files = list(data_dir.glob(filtered_pos_pattern))
                mic_files = list(data_dir.glob(mic_pattern))
                
                if not all([true_pos_files, est_pos_files, filtered_pos_files, mic_files]):
                    continue
                
                # Load metrics
                metrics = pd.read_csv(metrics_file)
                metrics['snr'] = snr
                metrics['config'] = config
                config_metrics.append(metrics)
                
                # Load tracking data
                true_pos = pd.read_csv(true_pos_files[0])
                est_pos = pd.read_csv(est_pos_files[0])
                filtered_pos = pd.read_csv(filtered_pos_files[0])
                
                config_tracking[snr] = {
                    'true': true_pos,
                    'estimated': est_pos,
                    'filtered': filtered_pos
                }
                
                # Load microphone configuration (once per config)
                if config not in mic_configs:
                    mic_data = pd.read_csv(mic_files[0])
                    mic_configs[config] = mic_data
                    
            except Exception as e:
                print(f"Warning: Could not load data for {config} SNR{snr}: {e}")
                continue
        
        if config_metrics:
            config_data[config] = {
                'metrics': pd.concat(config_metrics, ignore_index=True),
                'tracking': config_tracking
            }
    
    return config_data, mic_configs

def analyze_microphone_geometry(mic_configs):
    """Analyze the geometric differences between configurations"""
    
    geometry_analysis = {}
    
    for config, mic_data in mic_configs.items():
        # Calculate geometric properties
        positions = mic_data[['x', 'y', 'z']].values
        
        # Calculate distances between all microphone pairs
        distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        # Calculate geometric spread
        centroid = np.mean(positions, axis=0)
        spread = np.mean([np.linalg.norm(pos - centroid) for pos in positions])
        
        # Calculate bounding box volume
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        volume = np.prod(max_coords - min_coords)
        
        geometry_analysis[config] = {
            'distances': distances,
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'max_distance': np.max(distances),
            'min_distance': np.min(distances),
            'spread': spread,
            'volume': volume,
            'centroid': centroid,
            'positions': positions
        }
    
    return geometry_analysis

def create_configuration_comparison_plots(config_data, mic_configs, geometry_analysis):
    """Create comprehensive comparison plots between configurations"""
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (18, 14),
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('TV X vs TV Plus Configuration Comparison Analysis', fontsize=16, fontweight='bold')
    
    snr_levels = [0, 5, 10, 20]
    colors = {'tvplus': 'blue', 'tvx': 'red'}
    markers = {'tvplus': 'o', 'tvx': 's'}
    
    # Plot 1: Direct Performance Comparison
    for config in ['tvplus', 'tvx']:
        if config in config_data:
            metrics = config_data[config]['metrics']
            snr_stats = metrics.groupby('snr')['mean_3d_error'].agg(['mean', 'std'])
            
            mean_errors = [snr_stats.loc[snr, 'mean'] if snr in snr_stats.index else np.nan for snr in snr_levels]
            error_stds = [snr_stats.loc[snr, 'std'] if snr in snr_stats.index else 0 for snr in snr_levels]
            
            valid_idx = ~np.isnan(mean_errors)
            valid_snr = np.array(snr_levels)[valid_idx]
            valid_means = np.array(mean_errors)[valid_idx]
            valid_stds = np.array(error_stds)[valid_idx]
            
            axes[0,0].errorbar(valid_snr, valid_means, yerr=valid_stds, 
                              marker=markers[config], linewidth=3, markersize=10, capsize=5,
                              label=f'{config.upper()}', color=colors[config])
    
    axes[0,0].set_xlabel('SNR (dB)')
    axes[0,0].set_ylabel('Mean 3D Error (m)')
    axes[0,0].set_title('Performance Comparison: Mean 3D Error vs SNR')
    axes[0,0].legend(fontsize=14)
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Error Variability Comparison
    for config in ['tvplus', 'tvx']:
        if config in config_data:
            metrics = config_data[config]['metrics']
            snr_stats = metrics.groupby('snr')['std_3d_error'].agg(['mean', 'std'])
            
            std_errors = [snr_stats.loc[snr, 'mean'] if snr in snr_stats.index else np.nan for snr in snr_levels]
            
            valid_idx = ~np.isnan(std_errors)
            valid_snr = np.array(snr_levels)[valid_idx]
            valid_stds = np.array(std_errors)[valid_idx]
            
            axes[0,1].plot(valid_snr, valid_stds, 
                          marker=markers[config], linewidth=3, markersize=10,
                          label=f'{config.upper()}', color=colors[config])
    
    axes[0,1].set_xlabel('SNR (dB)')
    axes[0,1].set_ylabel('Standard Deviation of Error (m)')
    axes[0,1].set_title('Error Variability Comparison')
    axes[0,1].legend(fontsize=14)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Maximum Error Comparison
    for config in ['tvplus', 'tvx']:
        if config in config_data:
            metrics = config_data[config]['metrics']
            snr_stats = metrics.groupby('snr')['max_3d_error'].agg(['mean'])
            
            max_errors = [snr_stats.loc[snr, 'mean'] if snr in snr_stats.index else np.nan for snr in snr_levels]
            
            valid_idx = ~np.isnan(max_errors)
            valid_snr = np.array(snr_levels)[valid_idx]
            valid_max = np.array(max_errors)[valid_idx]
            
            axes[0,2].plot(valid_snr, valid_max, 
                          marker=markers[config], linewidth=3, markersize=10,
                          label=f'{config.upper()}', color=colors[config])
    
    axes[0,2].set_xlabel('SNR (dB)')
    axes[0,2].set_ylabel('Maximum 3D Error (m)')
    axes[0,2].set_title('Maximum Error Comparison')
    axes[0,2].legend(fontsize=14)
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Microphone Configuration Geometry
    for i, (config, mic_data) in enumerate(mic_configs.items()):
        ax = axes[1, i]
        scatter = ax.scatter(mic_data['x'], mic_data['y'], 
                           c=mic_data['z'], s=300, cmap='viridis',
                           edgecolors='black', linewidth=3, marker=markers[config])
        
        # Add microphone labels
        for _, row in mic_data.iterrows():
            ax.annotate(row['mic_id'], (row['x'], row['y']), 
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=12, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # Add lines connecting microphones to show geometry
        positions = mic_data[['x', 'y']].values
        for j in range(len(positions)):
            for k in range(j+1, len(positions)):
                ax.plot([positions[j,0], positions[k,0]], 
                       [positions[j,1], positions[k,1]], 
                       'k--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{config.upper()} Configuration')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add colorbar for Z coordinates
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Z coordinate (m)')
    
    # Plot 5: Geometry Metrics Comparison
    config_names = list(geometry_analysis.keys())
    metrics = ['mean_distance', 'spread', 'volume']
    metric_labels = ['Mean Distance (m)', 'Spread (m)', 'Volume (m³)']
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    for i, config in enumerate(config_names):
        values = [geometry_analysis[config][metric] for metric in metrics]
        axes[1,2].bar(x_pos + i*width, values, width, 
                     label=config.upper(), color=colors[config], alpha=0.8)
    
    axes[1,2].set_xlabel('Geometric Metrics')
    axes[1,2].set_ylabel('Value')
    axes[1,2].set_title('Geometric Properties Comparison')
    axes[1,2].set_xticks(x_pos + width/2)
    axes[1,2].set_xticklabels(metric_labels, rotation=45, ha='right')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    # Plot 6: Performance Improvement Analysis
    improvement_data = []
    for config in ['tvplus', 'tvx']:
        if config in config_data:
            metrics = config_data[config]['metrics']
            snr_0_error = metrics[metrics['snr'] == 0]['mean_3d_error'].mean()
            
            for snr in [5, 10, 20]:
                snr_data = metrics[metrics['snr'] == snr]
                if len(snr_data) > 0:
                    snr_error = snr_data['mean_3d_error'].mean()
                    improvement = ((snr_0_error - snr_error) / snr_0_error) * 100
                    improvement_data.append([config, snr, improvement])
    
    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data, columns=['Config', 'SNR', 'Improvement'])
        
        for config in ['tvplus', 'tvx']:
            config_data_plot = improvement_df[improvement_df['Config'] == config]
            axes[2,0].plot(config_data_plot['SNR'], config_data_plot['Improvement'], 
                          marker=markers[config], linewidth=3, markersize=10,
                          label=f'{config.upper()}', color=colors[config])
    
    axes[2,0].set_xlabel('SNR (dB)')
    axes[2,0].set_ylabel('Improvement over SNR 0 dB (%)')
    axes[2,0].set_title('SNR Improvement Comparison')
    axes[2,0].legend(fontsize=14)
    axes[2,0].grid(True, alpha=0.3)
    
    # Plot 7: Error Distribution Box Plot
    all_errors = []
    for config in ['tvplus', 'tvx']:
        if config in config_data:
            metrics = config_data[config]['metrics']
            for _, row in metrics.iterrows():
                all_errors.append([config.upper(), row['snr'], row['mean_3d_error']])
    
    if all_errors:
        error_df = pd.DataFrame(all_errors, columns=['Config', 'SNR', 'Error'])
        
        # Create box plot using matplotlib
        config_colors = {'TVPLUS': 'blue', 'TVX': 'red'}
        positions = []
        box_data = []
        labels = []
        colors_list = []
        
        for i, snr in enumerate([0, 5, 10, 20]):
            for j, config in enumerate(['TVPLUS', 'TVX']):
                config_snr_data = error_df[(error_df['Config'] == config) & (error_df['SNR'] == snr)]
                if len(config_snr_data) > 0:
                    pos = i * 3 + j + 1
                    positions.append(pos)
                    box_data.append(config_snr_data['Error'].values)
                    labels.append(f'{config}\nSNR{snr}')
                    colors_list.append(config_colors[config])
        
        if box_data:
            bp = axes[2,1].boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
            
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[2,1].set_xticks(positions)
            axes[2,1].set_xticklabels(labels, rotation=45, ha='right')
            axes[2,1].set_ylabel('Mean 3D Error (m)')
            axes[2,1].set_title('Error Distribution by Configuration and SNR')
            axes[2,1].grid(True, alpha=0.3)
    
    # Plot 8: Relative Performance
    relative_perf = []
    for snr in snr_levels:
        tvplus_error = None
        tvx_error = None
        
        if 'tvplus' in config_data:
            tvplus_data = config_data['tvplus']['metrics']
            tvplus_snr = tvplus_data[tvplus_data['snr'] == snr]
            if len(tvplus_snr) > 0:
                tvplus_error = tvplus_snr['mean_3d_error'].mean()
        
        if 'tvx' in config_data:
            tvx_data = config_data['tvx']['metrics']
            tvx_snr = tvx_data[tvx_data['snr'] == snr]
            if len(tvx_snr) > 0:
                tvx_error = tvx_snr['mean_3d_error'].mean()
        
        if tvplus_error is not None and tvx_error is not None:
            relative = ((tvx_error - tvplus_error) / tvplus_error) * 100
            relative_perf.append([snr, relative])
    
    if relative_perf:
        rel_df = pd.DataFrame(relative_perf, columns=['SNR', 'Relative_Performance'])
        axes[2,2].bar(rel_df['SNR'], rel_df['Relative_Performance'], 
                     color=['red' if x > 0 else 'green' for x in rel_df['Relative_Performance']],
                     alpha=0.7, width=1.5)
        axes[2,2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[2,2].set_xlabel('SNR (dB)')
        axes[2,2].set_ylabel('TV X vs TV Plus Error Difference (%)')
        axes[2,2].set_title('Relative Performance\n(Positive = TV X worse, Negative = TV X better)')
        axes[2,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report_figures/configuration_comparison_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/configuration_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_comparison_report(config_data, geometry_analysis):
    """Generate detailed statistical comparison report"""
    
    print("\n" + "="*100)
    print("DETAILED CONFIGURATION COMPARISON ANALYSIS")
    print("="*100)
    
    # Overall performance comparison
    print("\n1. OVERALL PERFORMANCE SUMMARY:")
    print("-" * 50)
    
    for config in ['tvplus', 'tvx']:
        if config in config_data:
            metrics = config_data[config]['metrics']
            overall_mean = metrics['mean_3d_error'].mean()
            overall_std = metrics['mean_3d_error'].std()
            overall_min = metrics['mean_3d_error'].min()
            overall_max = metrics['mean_3d_error'].max()
            
            print(f"{config.upper()}:")
            print(f"  Average Error: {overall_mean:.4f} ± {overall_std:.4f} m")
            print(f"  Range: {overall_min:.4f} - {overall_max:.4f} m")
    
    # Performance by SNR
    print("\n2. PERFORMANCE BY SNR LEVEL:")
    print("-" * 50)
    print(f"{'SNR (dB)':<8} {'TV Plus':<12} {'TV X':<12} {'Difference':<12} {'Better':<10}")
    print("-" * 50)
    
    snr_comparison = []
    for snr in [0, 5, 10, 20]:
        tvplus_error = None
        tvx_error = None
        
        if 'tvplus' in config_data:
            tvplus_data = config_data['tvplus']['metrics']
            tvplus_snr = tvplus_data[tvplus_data['snr'] == snr]
            if len(tvplus_snr) > 0:
                tvplus_error = tvplus_snr['mean_3d_error'].mean()
        
        if 'tvx' in config_data:
            tvx_data = config_data['tvx']['metrics']
            tvx_snr = tvx_data[tvx_data['snr'] == snr]
            if len(tvx_snr) > 0:
                tvx_error = tvx_snr['mean_3d_error'].mean()
        
        if tvplus_error is not None and tvx_error is not None:
            diff = tvx_error - tvplus_error
            better = "TV Plus" if diff > 0 else "TV X"
            print(f"{snr:<8} {tvplus_error:<12.4f} {tvx_error:<12.4f} {diff:<+12.4f} {better:<10}")
            snr_comparison.append([snr, tvplus_error, tvx_error, diff])
    
    # Geometric analysis
    print("\n3. GEOMETRIC CONFIGURATION ANALYSIS:")
    print("-" * 50)
    print(f"{'Metric':<20} {'TV Plus':<12} {'TV X':<12} {'Difference':<12}")
    print("-" * 50)
    
    geometric_metrics = [
        ('Mean Distance', 'mean_distance'),
        ('Max Distance', 'max_distance'),
        ('Min Distance', 'min_distance'),
        ('Spread', 'spread'),
        ('Volume', 'volume')
    ]
    
    for metric_name, metric_key in geometric_metrics:
        if 'tvplus' in geometry_analysis and 'tvx' in geometry_analysis:
            tvplus_val = geometry_analysis['tvplus'][metric_key]
            tvx_val = geometry_analysis['tvx'][metric_key]
            diff = tvx_val - tvplus_val
            print(f"{metric_name:<20} {tvplus_val:<12.4f} {tvx_val:<12.4f} {diff:<+12.4f}")
    
    # SNR improvement analysis
    print("\n4. SNR IMPROVEMENT ANALYSIS:")
    print("-" * 50)
    print(f"{'Config':<10} {'SNR 5':<10} {'SNR 10':<10} {'SNR 20':<10} {'Best SNR':<10}")
    print("-" * 50)
    
    for config in ['tvplus', 'tvx']:
        if config in config_data:
            metrics = config_data[config]['metrics']
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
            
            best_snr = [5, 10, 20][np.argmax(improvements)]
            print(f"{config.upper():<10} {improvements[0]:<+10.1f}% {improvements[1]:<+10.1f}% {improvements[2]:<+10.1f}% {best_snr:<10}")
    
    # Statistical significance (if we had multiple runs)
    print("\n5. CONFIGURATION RECOMMENDATIONS:")
    print("-" * 50)
    
    if snr_comparison:
        comp_df = pd.DataFrame(snr_comparison, columns=['SNR', 'TVPlus', 'TVX', 'Diff'])
        avg_diff = comp_df['Diff'].mean()
        
        if abs(avg_diff) < 0.01:
            print("• Performance difference is minimal between configurations")
        elif avg_diff > 0:
            print(f"• TV Plus performs better on average by {abs(avg_diff):.4f} m")
            print("• TV Plus recommended for general use")
        else:
            print(f"• TV X performs better on average by {abs(avg_diff):.4f} m")
            print("• TV X recommended for general use")
        
        # SNR-specific recommendations
        best_snr_config = {}
        for _, row in comp_df.iterrows():
            snr = row['SNR']
            if row['Diff'] > 0:
                best_snr_config[snr] = 'TV Plus'
            else:
                best_snr_config[snr] = 'TV X'
        
        print("\n• SNR-specific recommendations:")
        for snr, config in best_snr_config.items():
            print(f"  SNR {snr} dB: {config}")
    
    # Geometric recommendations
    if 'tvplus' in geometry_analysis and 'tvx' in geometry_analysis:
        tvx_spread = geometry_analysis['tvx']['spread']
        tvplus_spread = geometry_analysis['tvplus']['spread']
        
        print(f"\n• Geometric considerations:")
        print(f"  TV Plus spread: {tvplus_spread:.3f} m (more compact)")
        print(f"  TV X spread: {tvx_spread:.3f} m (more distributed)")
        
        if tvx_spread > tvplus_spread:
            print("  TV X provides better spatial coverage")
            print("  TV Plus may be better for localized tracking")
    
    print("="*100)
    
    # Save detailed comparison
    if snr_comparison:
        comp_df = pd.DataFrame(snr_comparison, columns=['SNR', 'TVPlus_Error', 'TVX_Error', 'Difference'])
        comp_df.to_csv('report_figures/configuration_comparison_summary.csv', index=False)
    
    return snr_comparison

def main():
    """Main analysis function"""
    print("Loading configuration data...")
    config_data, mic_configs = load_configuration_data()
    
    print("Analyzing microphone geometry...")
    geometry_analysis = analyze_microphone_geometry(mic_configs)
    
    print("Creating configuration comparison plots...")
    create_configuration_comparison_plots(config_data, mic_configs, geometry_analysis)
    
    print("Generating detailed comparison report...")
    comparison_results = generate_detailed_comparison_report(config_data, geometry_analysis)
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("- report_figures/configuration_comparison_analysis.pdf")
    print("- report_figures/configuration_comparison_analysis.png")
    print("- report_figures/configuration_comparison_summary.csv")

if __name__ == "__main__":
    main()