import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_all_configuration_data():
    """Load all configuration data for analysis"""
    data_dir = Path("simulation_data")
    
    config_data = {}
    configs = ['tvplus', 'tvx', 'soundbar3']
    
    for config in configs:
        all_metrics = []
        for snr in [0, 5, 10, 20]:
            metrics_pattern = f"{config}_SNR{snr}_*_metrics.csv"
            metrics_files = list(data_dir.glob(metrics_pattern))
            if metrics_files:
                metrics = pd.read_csv(metrics_files[0])
                metrics['snr'] = snr
                metrics['config'] = config
                all_metrics.append(metrics)
        
        if all_metrics:
            config_data[config] = pd.concat(all_metrics, ignore_index=True)
    
    # Combine all configs for averaged results
    all_data = []
    for config_df in config_data.values():
        all_data.append(config_df)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data, config_data
    else:
        return pd.DataFrame(), config_data

def create_snr_vs_error_figure():
    """Create Figure 1: Mean and Maximum 3D Tracking Error vs SNR"""
    combined_data, _ = load_all_configuration_data()
    
    if combined_data.empty:
        print("No data found!")
        return
    
    # Group by SNR and calculate statistics for both mean and max errors
    snr_stats = combined_data.groupby('snr').agg({
        'mean_3d_error': ['mean', 'std'],
        'max_3d_error': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    snr_stats.columns = ['snr', 'mean_error_avg', 'mean_error_std', 'max_error_avg', 'max_error_std']
    
    plt.figure(figsize=(10, 6))
    
    # Plot mean error with error bars
    plt.errorbar(snr_stats['snr'], snr_stats['mean_error_avg'], yerr=snr_stats['mean_error_std'], 
                marker='o', linewidth=3, markersize=8, capsize=5, capthick=2,
                color='#2E86AB', markerfacecolor='#2E86AB', label='Mean 3D Error')
    
    # Plot maximum error with error bars
    plt.errorbar(snr_stats['snr'], snr_stats['max_error_avg'], yerr=snr_stats['max_error_std'], 
                marker='s', linewidth=3, markersize=8, capsize=5, capthick=2,
                color='#E74C3C', markerfacecolor='#E74C3C', label='Maximum 3D Error')
    
    plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    plt.ylabel('3D Tracking Error (m)', fontsize=12, fontweight='bold')
    plt.title('Mean and Maximum 3D Tracking Error vs SNR\n(Averaged Across All Configurations)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add improvement annotations for both metrics
    mean_baseline = snr_stats.iloc[0]['mean_error_avg']  # SNR 0 dB
    mean_best = snr_stats.iloc[-1]['mean_error_avg']     # SNR 20 dB
    mean_improvement = ((mean_baseline - mean_best) / mean_baseline) * 100
    
    max_baseline = snr_stats.iloc[0]['max_error_avg']    # SNR 0 dB
    max_best = snr_stats.iloc[-1]['max_error_avg']       # SNR 20 dB
    max_improvement = ((max_baseline - max_best) / max_baseline) * 100
    
    plt.text(0.02, 0.98, f'Mean Error: {mean_improvement:.1f}% improvement\nMax Error: {max_improvement:.1f}% improvement\n(SNR 0→20 dB)', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('report_figures/snr_vs_error_focused.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/snr_vs_error_focused.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_error_metrics_comparison():
    """Create Figure 2: Error Metrics Comparison by SNR"""
    combined_data, _ = load_all_configuration_data()
    
    if combined_data.empty:
        return
    
    # Calculate metrics by SNR
    metrics_by_snr = []
    for snr in [0, 5, 10, 20]:
        snr_data = combined_data[combined_data['snr'] == snr]
        metrics_by_snr.append({
            'snr': snr,
            'mean_error': snr_data['mean_3d_error'].mean(),
            'std_error': snr_data['std_3d_error'].mean(),
            'p95_error': snr_data['p95_3d_error'].mean()
        })
    
    metrics_df = pd.DataFrame(metrics_by_snr)
    
    plt.figure(figsize=(10, 6))
    
    x_pos = np.arange(len(metrics_df))
    width = 0.25
    
    # Create bars
    plt.bar(x_pos - width, metrics_df['mean_error'], width, 
           label='Mean 3D Error', color='#2E86AB', alpha=0.8)
    plt.bar(x_pos, metrics_df['std_error'], width, 
           label='Std 3D Error', color='#A23B72', alpha=0.8)
    plt.bar(x_pos + width, metrics_df['p95_error'], width, 
           label='P95 3D Error', color='#F18F01', alpha=0.8)
    
    plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    plt.ylabel('Error (m)', fontsize=12, fontweight='bold')
    plt.title('Error Metrics Comparison by SNR Level', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, metrics_df['snr'])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('report_figures/error_metrics_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/error_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_temporal_tracking_performance():
    """Create Figure 3: Real Temporal Tracking Error Over Time by SNR (Averaged Across All Configurations)"""
    data_dir = Path("simulation_data")
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    snr_levels = [0, 5, 10, 20]
    configs = ['tvplus', 'tvx', 'soundbar3']
    
    for i, snr in enumerate(snr_levels):
        all_config_errors = []  # Store errors from all configurations
        
        for config in configs:
            # Find trajectory files for this SNR level and configuration
            true_pattern = f"{config}_SNR{snr}_*_tracking_50steps_true_positions.csv"
            est_pattern = f"{config}_SNR{snr}_*_tracking_50steps_estimated_positions.csv"
            
            true_files = list(data_dir.glob(true_pattern))
            est_files = list(data_dir.glob(est_pattern))
            
            if true_files and est_files:
                # Load true and estimated positions
                true_pos = pd.read_csv(true_files[0])
                est_pos = pd.read_csv(est_files[0])
                
                # Calculate 3D tracking error over time for this configuration
                config_temporal_errors = []
                
                for step in range(50):  # 50 time steps
                    true_step = true_pos[true_pos['step'] == step]
                    est_step = est_pos[est_pos['step'] == step]
                    
                    if len(true_step) > 0 and len(est_step) > 0:
                        # Calculate error for all targets at this time step
                        step_errors = []
                        for target_id in true_step['target_id'].unique():
                            true_target = true_step[true_step['target_id'] == target_id]
                            est_target = est_step[est_step['target_id'] == target_id]
                            
                            if len(true_target) > 0 and len(est_target) > 0:
                                true_pos_3d = true_target[['x', 'y', 'z']].values[0]
                                est_pos_3d = est_target[['x', 'y', 'z']].values[0]
                                
                                error_3d = np.linalg.norm(true_pos_3d - est_pos_3d)
                                step_errors.append(error_3d)
                        
                        if step_errors:
                            config_temporal_errors.append(np.mean(step_errors))
                        else:
                            config_temporal_errors.append(np.nan)
                    else:
                        config_temporal_errors.append(np.nan)
                
                # Add this configuration's temporal errors to the collection
                all_config_errors.append(config_temporal_errors)
        
        # Average across all configurations for this SNR level
        if all_config_errors:
            # Convert to numpy array and calculate mean across configurations
            config_array = np.array(all_config_errors)
            # Handle NaN values by taking nanmean
            averaged_errors = np.nanmean(config_array, axis=0)
            
            # Remove NaN values and corresponding time steps
            valid_indices = ~np.isnan(averaged_errors)
            time_steps = np.arange(50)[valid_indices]
            averaged_errors = averaged_errors[valid_indices]
            
            if len(averaged_errors) > 0:
                # Smooth the signal slightly to reduce noise
                if len(averaged_errors) > 5:
                    smoothed_errors = np.convolve(averaged_errors, np.ones(3)/3, mode='same')
                else:
                    smoothed_errors = averaged_errors
                
                plt.plot(time_steps, smoothed_errors, 
                        label=f'SNR {snr} dB', color=colors[i], linewidth=2, 
                        marker='o' if snr == 20 else None, markersize=4 if snr == 20 else 0)
    
    plt.xlabel('Time Step', fontsize=12, fontweight='bold')
    plt.ylabel('3D Tracking Error (m)', fontsize=12, fontweight='bold')
    plt.title('Temporal Tracking Performance by SNR Level\n(Averaged Across All Three Configurations)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add performance insight text
    plt.text(0.02, 0.98, 'Higher SNR → More Stable Tracking\nAveraged across TV Plus, TV X, Soundbar3', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('report_figures/temporal_tracking_performance.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/temporal_tracking_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_summary_trends():
    """Create Figure 4: Performance Summary with Trends"""
    combined_data, _ = load_all_configuration_data()
    
    if combined_data.empty:
        return
    
    # Calculate summary statistics
    summary_stats = []
    for snr in [0, 5, 10, 20]:
        snr_data = combined_data[combined_data['snr'] == snr]
        summary_stats.append({
            'snr': snr,
            'avg_error': snr_data['mean_3d_error'].mean(),
            'max_error': snr_data['max_3d_error'].mean()
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    plt.figure(figsize=(10, 6))
    
    # Plot both average and maximum error trends
    plt.plot(summary_df['snr'], summary_df['avg_error'], 
            marker='o', linewidth=3, markersize=8, label='Average Error',
            color='#2E86AB', markerfacecolor='#2E86AB')
    plt.plot(summary_df['snr'], summary_df['max_error'], 
            marker='s', linewidth=3, markersize=8, label='Maximum Error',
            color='#E74C3C', markerfacecolor='#E74C3C')
    
    plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    plt.ylabel('Error (m)', fontsize=12, fontweight='bold')
    plt.title('Performance Summary: Error Trends vs SNR', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add improvement annotations
    avg_improvement = ((summary_df.iloc[0]['avg_error'] - summary_df.iloc[-1]['avg_error']) / 
                      summary_df.iloc[0]['avg_error']) * 100
    max_improvement = ((summary_df.iloc[0]['max_error'] - summary_df.iloc[-1]['max_error']) / 
                      summary_df.iloc[0]['max_error']) * 100
    
    plt.text(0.02, 0.98, f'Average Error Improvement: {avg_improvement:.1f}%\nMaximum Error Improvement: {max_improvement:.1f}%', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('report_figures/performance_summary_trends.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/performance_summary_trends.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create all individual figures"""
    print("Creating individual SNR analysis figures...")
    
    print("1. Creating SNR vs Error figure...")
    create_snr_vs_error_figure()
    
    print("2. Creating Error Metrics Comparison...")
    create_error_metrics_comparison()
    
    print("3. Creating Temporal Tracking Performance...")
    create_temporal_tracking_performance()
    
    print("4. Creating Performance Summary Trends...")
    create_performance_summary_trends()
    
    print("\nAll individual figures created!")
    print("Generated files:")
    print("- report_figures/snr_vs_error_focused.pdf/png")
    print("- report_figures/error_metrics_comparison.pdf/png")
    print("- report_figures/temporal_tracking_performance.pdf/png")
    print("- report_figures/performance_summary_trends.pdf/png")

if __name__ == "__main__":
    main()