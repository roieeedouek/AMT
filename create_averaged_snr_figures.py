import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_average_simulation_data():
    """Load simulation data for both configurations and average the results by SNR"""
    data_dir = Path("simulation_data")
    snr_levels = [0, 5, 10, 20]
    
    all_metrics = []
    tracking_data = {}
    
    for snr in snr_levels:
        snr_metrics = []
        snr_tracking = {'true': [], 'estimated': [], 'filtered': []}
        
        for config in ['tvplus', 'tvx']:
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
                
                true_pos_files = list(data_dir.glob(true_pos_pattern))
                est_pos_files = list(data_dir.glob(est_pos_pattern))
                filtered_pos_files = list(data_dir.glob(filtered_pos_pattern))
                
                if not all([true_pos_files, est_pos_files, filtered_pos_files]):
                    continue
                
                # Load metrics
                metrics = pd.read_csv(metrics_file)
                metrics['snr'] = snr
                metrics['config'] = config
                snr_metrics.append(metrics)
                
                # Load tracking data
                true_pos = pd.read_csv(true_pos_files[0])
                est_pos = pd.read_csv(est_pos_files[0])
                filtered_pos = pd.read_csv(filtered_pos_files[0])
                
                snr_tracking['true'].append(true_pos)
                snr_tracking['estimated'].append(est_pos)
                snr_tracking['filtered'].append(filtered_pos)
                
            except Exception as e:
                print(f"Warning: Could not load data for {config} SNR{snr}: {e}")
                continue
        
        if snr_metrics:
            # Combine metrics for this SNR level
            combined_snr_metrics = pd.concat(snr_metrics, ignore_index=True)
            all_metrics.append(combined_snr_metrics)
            
            # Combine tracking data for this SNR level
            if snr_tracking['true']:
                tracking_data[snr] = {
                    'true': pd.concat(snr_tracking['true'], ignore_index=True),
                    'estimated': pd.concat(snr_tracking['estimated'], ignore_index=True),
                    'filtered': pd.concat(snr_tracking['filtered'], ignore_index=True)
                }
    
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
    else:
        combined_metrics = pd.DataFrame()
    
    return combined_metrics, tracking_data

def create_averaged_snr_plots(metrics_df, tracking_data):
    """Create SNR comparison plots with averaged results from both configurations"""
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (15, 10),
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # Calculate averaged statistics by SNR
    snr_stats = metrics_df.groupby('snr').agg({
        'mean_3d_error': ['mean', 'std'],
        'std_3d_error': 'mean',
        'max_3d_error': 'mean',
        'p95_3d_error': 'mean'
    }).round(4)
    
    snr_levels = sorted(metrics_df['snr'].unique())
    
    # Create the main figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SNR Performance Analysis - Averaged Results from Both Configurations', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean 3D Error vs SNR (same as original)
    mean_errors = [snr_stats.loc[snr, ('mean_3d_error', 'mean')] for snr in snr_levels]
    error_stds = [snr_stats.loc[snr, ('mean_3d_error', 'std')] for snr in snr_levels]
    
    axes[0,0].errorbar(snr_levels, mean_errors, yerr=error_stds, 
                       marker='o', linewidth=2, markersize=8, capsize=5, color='blue')
    axes[0,0].set_xlabel('SNR (dB)')
    axes[0,0].set_ylabel('Mean 3D Error (m)')
    axes[0,0].set_title('Mean 3D Tracking Error vs SNR')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Error Distribution Comparison (same as original)
    error_metrics = ['mean_3d_error', 'std_3d_error', 'p95_3d_error']
    x_pos = np.arange(len(snr_levels))
    width = 0.25
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for i, metric in enumerate(error_metrics):
        if metric == 'std_3d_error':
            values = [snr_stats.loc[snr, (metric, 'mean')] for snr in snr_levels]
        else:
            values = [snr_stats.loc[snr, (metric, 'mean')] for snr in snr_levels]
        axes[0,1].bar(x_pos + i*width, values, width, 
                      label=metric.replace('_', ' ').title(), color=colors[i])
    
    axes[0,1].set_xlabel('SNR (dB)')
    axes[0,1].set_ylabel('Error (m)')
    axes[0,1].set_title('Error Metrics Comparison')
    axes[0,1].set_xticks(x_pos + width)
    axes[0,1].set_xticklabels(snr_levels)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Tracking Performance Over Time (averaged from both configs)
    for snr in snr_levels:
        if snr in tracking_data:
            true_data = tracking_data[snr]['true']
            est_data = tracking_data[snr]['estimated']
            
            # Calculate average 3D error over time for both targets and configs
            all_errors = []
            
            for target_id in true_data['target_id'].unique():
                target_true = true_data[true_data['target_id'] == target_id]
                target_est = est_data[est_data['target_id'] == target_id]
                
                if len(target_true) > 0 and len(target_est) > 0:
                    merged = pd.merge(target_true, target_est, on='step', suffixes=('_true', '_est'))
                    merged['3d_error'] = np.sqrt(
                        (merged['x_true'] - merged['x_est'])**2 + 
                        (merged['y_true'] - merged['y_est'])**2 + 
                        (merged['z_true'] - merged['z_est'])**2
                    )
                    all_errors.append(merged[['step', '3d_error']])
            
            if all_errors:
                # Average errors across all targets and configurations
                combined_errors = pd.concat(all_errors, ignore_index=True)
                avg_errors = combined_errors.groupby('step')['3d_error'].mean().reset_index()
                
                axes[1,0].plot(avg_errors['step'], avg_errors['3d_error'], 
                              label=f'SNR {snr} dB', linewidth=2, marker='o', markersize=4)
    
    axes[1,0].set_xlabel('Time Step')
    axes[1,0].set_ylabel('3D Tracking Error (m)')
    axes[1,0].set_title('Average Tracking Error Over Time')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Performance Summary (same as original)
    performance_summary = []
    for snr in snr_levels:
        snr_data = metrics_df[metrics_df['snr'] == snr]
        avg_error = snr_data['mean_3d_error'].mean()
        max_error = snr_data['max_3d_error'].mean()
        performance_summary.append([snr, avg_error, max_error])
    
    perf_df = pd.DataFrame(performance_summary, columns=['SNR', 'Avg_Error', 'Max_Error'])
    
    axes[1,1].plot(perf_df['SNR'], perf_df['Avg_Error'], 'o-', label='Average Error', 
                   linewidth=2, markersize=8, color='blue')
    axes[1,1].plot(perf_df['SNR'], perf_df['Max_Error'], 's-', label='Maximum Error', 
                   linewidth=2, markersize=8, color='red')
    axes[1,1].set_xlabel('SNR (dB)')
    axes[1,1].set_ylabel('Error (m)')
    axes[1,1].set_title('Performance Summary')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report_figures/averaged_snr_performance_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/averaged_snr_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return snr_stats

def create_simple_tracking_plots(tracking_data):
    """Create simple tracking trajectory plots averaged from both configurations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tracking Trajectories - Averaged Results from Both Configurations', fontsize=16, fontweight='bold')
    
    snr_levels = sorted(tracking_data.keys())
    
    for i, snr in enumerate(snr_levels):
        ax = axes[i//2, i%2]
        
        true_data = tracking_data[snr]['true']
        est_data = tracking_data[snr]['estimated']
        filtered_data = tracking_data[snr]['filtered']
        
        # Average trajectories across all targets and configurations
        for data_type, data, color, style, label in [
            ('true', true_data, 'green', '-', 'True'),
            ('estimated', est_data, 'red', '--', 'Estimated'),
            ('filtered', filtered_data, 'blue', ':', 'Filtered')
        ]:
            if len(data) > 0:
                # Group by step and average positions
                avg_positions = data.groupby('step')[['x', 'y', 'z']].mean().reset_index()
                
                ax.plot(avg_positions['x'], avg_positions['y'], 
                       color=color, linestyle=style, linewidth=3, 
                       label=label, alpha=0.8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'SNR {snr} dB - Average XY Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('report_figures/averaged_tracking_trajectories.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/averaged_tracking_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_averaged_summary_table(metrics_df):
    """Generate summary table with averaged results"""
    
    # Calculate averaged statistics by SNR
    summary_stats = metrics_df.groupby('snr').agg({
        'mean_3d_error': ['mean', 'std', 'min', 'max'],
        'std_3d_error': ['mean', 'std'],
        'max_3d_error': ['mean', 'std'],
        'p95_3d_error': ['mean', 'std'],
        'target_id': 'count'
    }).round(4)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    
    print("\n" + "="*80)
    print("AVERAGED SNR PERFORMANCE ANALYSIS SUMMARY")
    print("="*80)
    print(f"{'SNR (dB)':<10} {'Mean Error':<12} {'Std Error':<12} {'Max Error':<12} {'P95 Error':<12} {'Samples':<8}")
    print("-"*80)
    
    for snr in sorted(metrics_df['snr'].unique()):
        mean_err = summary_stats.loc[snr, 'mean_3d_error_mean']
        std_err = summary_stats.loc[snr, 'std_3d_error_mean'] 
        max_err = summary_stats.loc[snr, 'max_3d_error_mean']
        p95_err = summary_stats.loc[snr, 'p95_3d_error_mean']
        n_samples = int(summary_stats.loc[snr, 'target_id_count'])
        
        print(f"{snr:<10} {mean_err:<12.4f} {std_err:<12.4f} {max_err:<12.4f} {p95_err:<12.4f} {n_samples:<8}")
    
    print("="*80)
    
    # Performance improvement analysis
    snr_0_error = summary_stats.loc[0, 'mean_3d_error_mean']
    print("\nPERFORMANCE IMPROVEMENTS RELATIVE TO SNR 0 dB:")
    print("-"*50)
    for snr in [5, 10, 20]:
        if snr in summary_stats.index:
            snr_error = summary_stats.loc[snr, 'mean_3d_error_mean']
            improvement = ((snr_0_error - snr_error) / snr_0_error) * 100
            print(f"SNR {snr} dB: {improvement:+.1f}% improvement")
    
    # Save summary statistics
    summary_stats.to_csv('report_figures/averaged_snr_analysis_summary.csv')
    
    return summary_stats

def main():
    """Main analysis function"""
    print("Loading and averaging simulation data from both configurations...")
    metrics_df, tracking_data = load_and_average_simulation_data()
    
    if metrics_df.empty:
        print("Error: No data loaded!")
        return
    
    print(f"Loaded and averaged data from {len(metrics_df)} simulation runs")
    print(f"SNR levels: {sorted(metrics_df['snr'].unique())}")
    
    print("Creating averaged SNR comparison plots...")
    snr_stats = create_averaged_snr_plots(metrics_df, tracking_data)
    
    print("Creating averaged tracking trajectory plots...")
    create_simple_tracking_plots(tracking_data)
    
    print("Generating averaged summary...")
    summary_stats = generate_averaged_summary_table(metrics_df)
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("- report_figures/averaged_snr_performance_analysis.pdf")
    print("- report_figures/averaged_snr_performance_analysis.png")
    print("- report_figures/averaged_tracking_trajectories.pdf")
    print("- report_figures/averaged_tracking_trajectories.png")
    print("- report_figures/averaged_snr_analysis_summary.csv")

if __name__ == "__main__":
    main()