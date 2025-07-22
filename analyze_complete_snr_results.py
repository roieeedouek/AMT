import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_complete_simulation_data():
    """Load simulation data for both microphone configurations and all SNR levels"""
    data_dir = Path("simulation_data")
    snr_levels = [0, 5, 10, 20]
    
    all_metrics = []
    tracking_data = {}
    mic_configs = {}
    
    for snr in snr_levels:
        for config in ['tvplus', 'tvx']:
            try:
                # Find files for this SNR and config using the correct naming pattern
                metrics_pattern = f"{config}_SNR{snr}_*_metrics.csv"
                metrics_files = list(data_dir.glob(metrics_pattern))
                
                if not metrics_files:
                    print(f"Warning: No metrics file found for {config} SNR{snr}")
                    continue
                
                metrics_file = metrics_files[0]
                
                # Extract session timestamp from filename
                filename_parts = metrics_file.stem.split('_')
                # Format: tvplus_SNR0_20250719_193950_202640_metrics
                session_start = filename_parts[2] + '_' + filename_parts[3]  # 20250719_193950
                
                true_pos_pattern = f"{config}_SNR{snr}_{session_start}_*_tracking_*_true_positions.csv"
                est_pos_pattern = f"{config}_SNR{snr}_{session_start}_*_tracking_*_estimated_positions.csv"
                filtered_pos_pattern = f"{config}_SNR{snr}_{session_start}_*_tracking_*_filtered_positions.csv"
                mic_pattern = f"{config}_SNR{snr}_{session_start}_*microphones.csv"
                
                true_pos_files = list(data_dir.glob(true_pos_pattern))
                est_pos_files = list(data_dir.glob(est_pos_pattern))
                filtered_pos_files = list(data_dir.glob(filtered_pos_pattern))
                mic_files = list(data_dir.glob(mic_pattern))
                
                if not all([true_pos_files, est_pos_files, filtered_pos_files, mic_files]):
                    print(f"Warning: Missing data files for {config} SNR{snr}")
                    continue
                
                true_pos_file = true_pos_files[0]
                est_pos_file = est_pos_files[0]
                filtered_pos_file = filtered_pos_files[0]
                mic_file = mic_files[0]
                
                # Load metrics
                metrics = pd.read_csv(metrics_file)
                metrics['snr'] = snr
                metrics['mic_config'] = config
                metrics['session_id'] = session_start
                all_metrics.append(metrics)
                
                # Load tracking data
                true_pos = pd.read_csv(true_pos_file)
                est_pos = pd.read_csv(est_pos_file)
                filtered_pos = pd.read_csv(filtered_pos_file)
                
                tracking_data[(config, snr)] = {
                    'true': true_pos,
                    'estimated': est_pos,
                    'filtered': filtered_pos
                }
                
                # Load microphone configuration
                if config not in mic_configs:
                    mic_data = pd.read_csv(mic_file)
                    mic_configs[config] = mic_data
                    
            except Exception as e:
                print(f"Warning: Could not load data for {config} SNR{snr}: {e}")
                continue
    
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    return combined_metrics, tracking_data, mic_configs

def create_comprehensive_snr_analysis(metrics_df, tracking_data, mic_configs):
    """Create comprehensive analysis comparing both microphone configurations"""
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'figure.figsize': (16, 12),
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # 1. Complete SNR Performance Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Complete SNR Performance Analysis: TV Plus vs TV X Configurations', fontsize=16, fontweight='bold')
    
    snr_levels = sorted(metrics_df['snr'].unique())
    configs = sorted(metrics_df['mic_config'].unique())
    colors = {'tvplus': 'blue', 'tvx': 'red'}
    markers = {'tvplus': 'o', 'tvx': 's'}
    
    # Plot 1: Mean 3D Error vs SNR for both configurations
    for config in configs:
        config_data = metrics_df[metrics_df['mic_config'] == config]
        snr_stats = config_data.groupby('snr')['mean_3d_error'].agg(['mean', 'std'])
        
        mean_errors = [snr_stats.loc[snr, 'mean'] if snr in snr_stats.index else np.nan for snr in snr_levels]
        error_stds = [snr_stats.loc[snr, 'std'] if snr in snr_stats.index else 0 for snr in snr_levels]
        
        valid_idx = ~np.isnan(mean_errors)
        valid_snr = np.array(snr_levels)[valid_idx]
        valid_means = np.array(mean_errors)[valid_idx]
        valid_stds = np.array(error_stds)[valid_idx]
        
        axes[0,0].errorbar(valid_snr, valid_means, yerr=valid_stds, 
                          marker=markers[config], linewidth=2, markersize=8, capsize=5,
                          label=f'{config.upper()}', color=colors[config])
    
    axes[0,0].set_xlabel('SNR (dB)')
    axes[0,0].set_ylabel('Mean 3D Error (m)')
    axes[0,0].set_title('Mean 3D Tracking Error vs SNR')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Error Distribution Comparison
    error_metrics = ['mean_3d_error', 'std_3d_error', 'p95_3d_error']
    x_pos = np.arange(len(snr_levels))
    width = 0.15
    
    for i, metric in enumerate(error_metrics):
        for j, config in enumerate(configs):
            config_data = metrics_df[metrics_df['mic_config'] == config]
            values = []
            for snr in snr_levels:
                snr_data = config_data[config_data['snr'] == snr]
                if len(snr_data) > 0:
                    values.append(snr_data[metric].mean())
                else:
                    values.append(0)
            
            offset = (i * len(configs) + j) * width - (len(error_metrics) * len(configs) - 1) * width / 2
            axes[0,1].bar(x_pos + offset, values, width, 
                         label=f'{config.upper()} {metric.replace("_", " ").title()}',
                         color=colors[config], alpha=0.7 + i*0.1)
    
    axes[0,1].set_xlabel('SNR (dB)')
    axes[0,1].set_ylabel('Error (m)')
    axes[0,1].set_title('Error Metrics Comparison')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(snr_levels)
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Configuration Performance Summary
    perf_summary = []
    for config in configs:
        config_data = metrics_df[metrics_df['mic_config'] == config]
        for snr in snr_levels:
            snr_data = config_data[config_data['snr'] == snr]
            if len(snr_data) > 0:
                avg_error = snr_data['mean_3d_error'].mean()
                max_error = snr_data['max_3d_error'].mean()
                perf_summary.append([config, snr, avg_error, max_error])
    
    perf_df = pd.DataFrame(perf_summary, columns=['Config', 'SNR', 'Avg_Error', 'Max_Error'])
    
    for config in configs:
        config_perf = perf_df[perf_df['Config'] == config]
        axes[0,2].plot(config_perf['SNR'], config_perf['Avg_Error'], 
                      marker=markers[config], linewidth=2, markersize=8,
                      label=f'{config.upper()} Average', color=colors[config])
        axes[0,2].plot(config_perf['SNR'], config_perf['Max_Error'], 
                      marker=markers[config], linewidth=2, markersize=8, linestyle='--',
                      label=f'{config.upper()} Maximum', color=colors[config], alpha=0.7)
    
    axes[0,2].set_xlabel('SNR (dB)')
    axes[0,2].set_ylabel('Error (m)')
    axes[0,2].set_title('Performance Summary by Configuration')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Microphone Configuration Visualization
    for i, (config, mic_data) in enumerate(mic_configs.items()):
        ax = axes[1, i]
        scatter = ax.scatter(mic_data['x'], mic_data['y'], 
                           c=mic_data['z'], s=200, cmap='viridis',
                           edgecolors='black', linewidth=2)
        
        # Add microphone labels
        for _, row in mic_data.iterrows():
            ax.annotate(row['mic_id'], (row['x'], row['y']), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{config.upper()} Configuration')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add colorbar for Z coordinates
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Z coordinate (m)')
    
    # Plot 5: Tracking Performance Over Time Comparison
    ax = axes[1, 2]
    for config in configs:
        for snr in [0, 20]:  # Show best and worst SNR for each config
            if (config, snr) in tracking_data:
                true_data = tracking_data[(config, snr)]['true']
                est_data = tracking_data[(config, snr)]['estimated']
                
                # Calculate 3D error over time for target 0
                target_0_true = true_data[true_data['target_id'] == 0]
                target_0_est = est_data[est_data['target_id'] == 0]
                
                if len(target_0_true) > 0 and len(target_0_est) > 0:
                    merged = pd.merge(target_0_true, target_0_est, on='step', suffixes=('_true', '_est'))
                    merged['3d_error'] = np.sqrt(
                        (merged['x_true'] - merged['x_est'])**2 + 
                        (merged['y_true'] - merged['y_est'])**2 + 
                        (merged['z_true'] - merged['z_est'])**2
                    )
                    
                    linestyle = '-' if snr == 20 else '--'
                    alpha = 0.8 if snr == 20 else 0.5
                    ax.plot(merged['step'], merged['3d_error'], 
                           label=f'{config.upper()} SNR {snr} dB', 
                           linewidth=2, linestyle=linestyle, alpha=alpha,
                           color=colors[config])
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('3D Tracking Error (m)')
    ax.set_title('Tracking Error Over Time (Target 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report_figures/complete_snr_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/complete_snr_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_trajectory_comparison_plots(tracking_data):
    """Create detailed trajectory comparison plots"""
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle('Trajectory Comparison: TV Plus vs TV X Configurations', fontsize=16, fontweight='bold')
    
    snr_levels = [0, 5, 10, 20]
    configs = ['tvplus', 'tvx']
    
    for i, snr in enumerate(snr_levels):
        for j, config in enumerate(configs):
            if (config, snr) in tracking_data:
                ax = axes[i, j*2]  # 3D plot
                ax2d = axes[i, j*2 + 1]  # 2D plot
                
                true_data = tracking_data[(config, snr)]['true']
                est_data = tracking_data[(config, snr)]['estimated']
                filtered_data = tracking_data[(config, snr)]['filtered']
                
                # Get target 0 data
                target_0_true = true_data[true_data['target_id'] == 0]
                target_0_est = est_data[est_data['target_id'] == 0]
                target_0_filt = filtered_data[filtered_data['target_id'] == 0]
                
                # 3D trajectory (projected to 2D for subplot)
                if len(target_0_true) > 0:
                    ax.plot(target_0_true['x'], target_0_true['y'], 'g-', linewidth=3, label='True', alpha=0.8)
                if len(target_0_est) > 0:
                    ax.plot(target_0_est['x'], target_0_est['y'], 'r--', linewidth=2, label='Estimated', alpha=0.7)
                if len(target_0_filt) > 0:
                    ax.plot(target_0_filt['x'], target_0_filt['y'], 'b:', linewidth=2, label='Filtered', alpha=0.7)
                
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title(f'{config.upper()} SNR {snr}dB - XY View')
                if i == 0 and j == 0:
                    ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
                # Z coordinate over time
                if len(target_0_true) > 0:
                    ax2d.plot(target_0_true['step'], target_0_true['z'], 'g-', linewidth=3, label='True', alpha=0.8)
                if len(target_0_est) > 0:
                    ax2d.plot(target_0_est['step'], target_0_est['z'], 'r--', linewidth=2, label='Estimated', alpha=0.7)
                if len(target_0_filt) > 0:
                    ax2d.plot(target_0_filt['step'], target_0_filt['z'], 'b:', linewidth=2, label='Filtered', alpha=0.7)
                
                ax2d.set_xlabel('Time Step')
                ax2d.set_ylabel('Z (m)')
                ax2d.set_title(f'{config.upper()} SNR {snr}dB - Z vs Time')
                if i == 0 and j == 0:
                    ax2d.legend()
                ax2d.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report_figures/complete_trajectory_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/complete_trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_comprehensive_summary(metrics_df):
    """Generate comprehensive summary statistics"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE SNR PERFORMANCE ANALYSIS")
    print("="*100)
    print(f"{'Config':<10} {'SNR (dB)':<10} {'Mean Error':<12} {'Std Error':<12} {'Max Error':<12} {'P95 Error':<12} {'Targets':<8}")
    print("-"*100)
    
    summary_data = []
    
    for config in sorted(metrics_df['mic_config'].unique()):
        config_data = metrics_df[metrics_df['mic_config'] == config]
        for snr in sorted(config_data['snr'].unique()):
            snr_data = config_data[config_data['snr'] == snr]
            
            mean_err = snr_data['mean_3d_error'].mean()
            std_err = snr_data['std_3d_error'].mean()
            max_err = snr_data['max_3d_error'].mean()
            p95_err = snr_data['p95_3d_error'].mean()
            n_targets = len(snr_data)
            
            print(f"{config.upper():<10} {snr:<10} {mean_err:<12.4f} {std_err:<12.4f} {max_err:<12.4f} {p95_err:<12.4f} {n_targets:<8}")
            
            summary_data.append({
                'config': config,
                'snr': snr,
                'mean_error': mean_err,
                'std_error': std_err,
                'max_error': max_err,
                'p95_error': p95_err,
                'n_targets': n_targets
            })
    
    print("="*100)
    
    # Configuration comparison
    print("\nCONFIGURATION PERFORMANCE COMPARISON:")
    print("-"*50)
    
    tvplus_data = metrics_df[metrics_df['mic_config'] == 'tvplus']
    tvx_data = metrics_df[metrics_df['mic_config'] == 'tvx']
    
    tvplus_avg = tvplus_data['mean_3d_error'].mean()
    tvx_avg = tvx_data['mean_3d_error'].mean()
    
    print(f"TV Plus average error: {tvplus_avg:.4f} m")
    print(f"TV X average error: {tvx_avg:.4f} m")
    
    if tvplus_avg < tvx_avg:
        improvement = ((tvx_avg - tvplus_avg) / tvx_avg) * 100
        print(f"TV Plus performs {improvement:.1f}% better than TV X")
    else:
        improvement = ((tvplus_avg - tvx_avg) / tvplus_avg) * 100
        print(f"TV X performs {improvement:.1f}% better than TV Plus")
    
    # SNR improvement analysis for each config
    print("\nSNR IMPROVEMENTS BY CONFIGURATION:")
    print("-"*50)
    
    for config in ['tvplus', 'tvx']:
        config_data = metrics_df[metrics_df['mic_config'] == config]
        snr_0_error = config_data[config_data['snr'] == 0]['mean_3d_error'].mean()
        
        print(f"\n{config.upper()} improvements relative to SNR 0 dB:")
        for snr in [5, 10, 20]:
            snr_data = config_data[config_data['snr'] == snr]
            if len(snr_data) > 0:
                snr_error = snr_data['mean_3d_error'].mean()
                improvement = ((snr_0_error - snr_error) / snr_0_error) * 100
                print(f"  SNR {snr} dB: {improvement:+.1f}% improvement")
    
    # Save comprehensive summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('report_figures/complete_snr_analysis_summary.csv', index=False)
    
    return summary_df

def main():
    """Main analysis function"""
    print("Loading complete simulation data...")
    metrics_df, tracking_data, mic_configs = load_complete_simulation_data()
    
    print(f"Loaded data for {len(metrics_df)} simulation runs")
    print(f"Configurations: {metrics_df['mic_config'].unique()}")
    print(f"SNR levels: {sorted(metrics_df['snr'].unique())}")
    
    print("Creating comprehensive SNR analysis plots...")
    create_comprehensive_snr_analysis(metrics_df, tracking_data, mic_configs)
    
    print("Creating trajectory comparison plots...")
    create_trajectory_comparison_plots(tracking_data)
    
    print("Generating comprehensive summary...")
    summary_df = generate_comprehensive_summary(metrics_df)
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("- report_figures/complete_snr_analysis.pdf")
    print("- report_figures/complete_snr_analysis.png")
    print("- report_figures/complete_trajectory_comparison.pdf")
    print("- report_figures/complete_trajectory_comparison.png")
    print("- report_figures/complete_snr_analysis_summary.csv")

if __name__ == "__main__":
    main()