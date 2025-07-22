import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_simulation_data():
    """Load simulation data for all SNR levels"""
    data_dir = Path("simulation_data")
    snr_levels = [0, 5, 10, 20]
    
    all_metrics = []
    tracking_data = {}
    
    for snr in snr_levels:
        # Find files for this SNR
        metrics_file = list(data_dir.glob(f"SNR{snr}_*_metrics.csv"))[0]
        true_pos_file = list(data_dir.glob(f"SNR{snr}_*_tracking_*_true_positions.csv"))[0]
        est_pos_file = list(data_dir.glob(f"SNR{snr}_*_tracking_*_estimated_positions.csv"))[0]
        filtered_pos_file = list(data_dir.glob(f"SNR{snr}_*_tracking_*_filtered_positions.csv"))[0]
        
        # Load metrics
        metrics = pd.read_csv(metrics_file)
        metrics['snr'] = snr
        all_metrics.append(metrics)
        
        # Load tracking data
        true_pos = pd.read_csv(true_pos_file)
        est_pos = pd.read_csv(est_pos_file)
        filtered_pos = pd.read_csv(filtered_pos_file)
        
        tracking_data[snr] = {
            'true': true_pos,
            'estimated': est_pos,
            'filtered': filtered_pos
        }
    
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    return combined_metrics, tracking_data

def create_snr_comparison_plots(metrics_df, tracking_data):
    """Create comprehensive SNR comparison plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (15, 10),
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # 1. SNR vs Accuracy Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SNR Performance Analysis', fontsize=16, fontweight='bold')
    
    # Group by SNR and calculate statistics
    snr_stats = metrics_df.groupby('snr').agg({
        'mean_3d_error': ['mean', 'std'],
        'std_3d_error': 'mean',
        'max_3d_error': 'mean',
        'p95_3d_error': 'mean'
    }).round(4)
    
    snr_levels = sorted(metrics_df['snr'].unique())
    
    # Plot 1: Mean 3D Error vs SNR
    mean_errors = [snr_stats.loc[snr, ('mean_3d_error', 'mean')] for snr in snr_levels]
    error_stds = [snr_stats.loc[snr, ('mean_3d_error', 'std')] for snr in snr_levels]
    
    axes[0,0].errorbar(snr_levels, mean_errors, yerr=error_stds, 
                       marker='o', linewidth=2, markersize=8, capsize=5)
    axes[0,0].set_xlabel('SNR (dB)')
    axes[0,0].set_ylabel('Mean 3D Error (m)')
    axes[0,0].set_title('Mean 3D Tracking Error vs SNR')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Error Distribution Comparison
    error_metrics = ['mean_3d_error', 'std_3d_error', 'p95_3d_error']
    x_pos = np.arange(len(snr_levels))
    width = 0.25
    
    for i, metric in enumerate(error_metrics):
        if metric == 'std_3d_error':
            values = [snr_stats.loc[snr, (metric, 'mean')] for snr in snr_levels]
        else:
            values = [snr_stats.loc[snr, (metric, 'mean')] for snr in snr_levels]
        axes[0,1].bar(x_pos + i*width, values, width, 
                      label=metric.replace('_', ' ').title())
    
    axes[0,1].set_xlabel('SNR (dB)')
    axes[0,1].set_ylabel('Error (m)')
    axes[0,1].set_title('Error Metrics Comparison')
    axes[0,1].set_xticks(x_pos + width)
    axes[0,1].set_xticklabels(snr_levels)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Tracking Performance Over Time for Different SNRs
    for snr in snr_levels:
        true_data = tracking_data[snr]['true']
        est_data = tracking_data[snr]['estimated']
        
        # Calculate 3D error over time for target 0
        target_0_true = true_data[true_data['target_id'] == 0]
        target_0_est = est_data[est_data['target_id'] == 0]
        
        if len(target_0_true) > 0 and len(target_0_est) > 0:
            # Merge on step to calculate errors
            merged = pd.merge(target_0_true, target_0_est, on='step', suffixes=('_true', '_est'))
            merged['3d_error'] = np.sqrt(
                (merged['x_true'] - merged['x_est'])**2 + 
                (merged['y_true'] - merged['y_est'])**2 + 
                (merged['z_true'] - merged['z_est'])**2
            )
            
            axes[1,0].plot(merged['step'], merged['3d_error'], 
                          label=f'SNR {snr} dB', linewidth=2, marker='o', markersize=4)
    
    axes[1,0].set_xlabel('Time Step')
    axes[1,0].set_ylabel('3D Tracking Error (m)')
    axes[1,0].set_title('Tracking Error Over Time (Target 0)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: SNR vs Performance Summary
    performance_summary = []
    for snr in snr_levels:
        snr_data = metrics_df[metrics_df['snr'] == snr]
        avg_error = snr_data['mean_3d_error'].mean()
        max_error = snr_data['max_3d_error'].mean()
        performance_summary.append([snr, avg_error, max_error])
    
    perf_df = pd.DataFrame(performance_summary, columns=['SNR', 'Avg_Error', 'Max_Error'])
    
    axes[1,1].plot(perf_df['SNR'], perf_df['Avg_Error'], 'o-', label='Average Error', linewidth=2, markersize=8)
    axes[1,1].plot(perf_df['SNR'], perf_df['Max_Error'], 's-', label='Maximum Error', linewidth=2, markersize=8)
    axes[1,1].set_xlabel('SNR (dB)')
    axes[1,1].set_ylabel('Error (m)')
    axes[1,1].set_title('Performance Summary')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report_figures/snr_performance_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/snr_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return snr_stats

def create_tracking_trajectory_plots(tracking_data):
    """Create 3D trajectory plots for different SNR levels"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # Create subplots for each SNR
    snr_levels = sorted(tracking_data.keys())
    
    for i, snr in enumerate(snr_levels):
        # 3D trajectory plot
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        
        true_data = tracking_data[snr]['true']
        est_data = tracking_data[snr]['estimated']
        filtered_data = tracking_data[snr]['filtered']
        
        # Plot for target 0
        target_0_true = true_data[true_data['target_id'] == 0]
        target_0_est = est_data[est_data['target_id'] == 0]
        target_0_filt = filtered_data[filtered_data['target_id'] == 0]
        
        if len(target_0_true) > 0:
            ax.plot(target_0_true['x'], target_0_true['y'], target_0_true['z'], 
                   'g-', linewidth=3, label='True', alpha=0.8)
        if len(target_0_est) > 0:
            ax.plot(target_0_est['x'], target_0_est['y'], target_0_est['z'], 
                   'r--', linewidth=2, label='Estimated', alpha=0.7)
        if len(target_0_filt) > 0:
            ax.plot(target_0_filt['x'], target_0_filt['y'], target_0_filt['z'], 
                   'b:', linewidth=2, label='Filtered', alpha=0.7)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'SNR {snr} dB - Target 0 Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2D projection plots
        ax2d = fig.add_subplot(2, 4, i+5)
        
        if len(target_0_true) > 0:
            ax2d.plot(target_0_true['x'], target_0_true['y'], 'g-', linewidth=3, label='True', alpha=0.8)
        if len(target_0_est) > 0:
            ax2d.plot(target_0_est['x'], target_0_est['y'], 'r--', linewidth=2, label='Estimated', alpha=0.7)
        if len(target_0_filt) > 0:
            ax2d.plot(target_0_filt['x'], target_0_filt['y'], 'b:', linewidth=2, label='Filtered', alpha=0.7)
        
        ax2d.set_xlabel('X (m)')
        ax2d.set_ylabel('Y (m)')
        ax2d.set_title(f'SNR {snr} dB - XY Plane')
        ax2d.legend()
        ax2d.grid(True, alpha=0.3)
        ax2d.set_aspect('equal')
    
    plt.suptitle('Tracking Trajectories Across Different SNR Levels', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report_figures/tracking_trajectories_snr.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/tracking_trajectories_snr.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_results_summary_table(metrics_df):
    """Generate a summary table of results"""
    
    # Group by SNR and calculate comprehensive statistics
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
    print("SNR PERFORMANCE ANALYSIS SUMMARY")
    print("="*80)
    print(f"{'SNR (dB)':<10} {'Mean Error':<12} {'Std Error':<12} {'Max Error':<12} {'P95 Error':<12} {'Targets':<8}")
    print("-"*80)
    
    for snr in sorted(metrics_df['snr'].unique()):
        mean_err = summary_stats.loc[snr, 'mean_3d_error_mean']
        std_err = summary_stats.loc[snr, 'std_3d_error_mean'] 
        max_err = summary_stats.loc[snr, 'max_3d_error_mean']
        p95_err = summary_stats.loc[snr, 'p95_3d_error_mean']
        n_targets = int(summary_stats.loc[snr, 'target_id_count'])
        
        print(f"{snr:<10} {mean_err:<12.4f} {std_err:<12.4f} {max_err:<12.4f} {p95_err:<12.4f} {n_targets:<8}")
    
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
    
    return summary_stats

def main():
    """Main analysis function"""
    print("Loading simulation data...")
    metrics_df, tracking_data = load_simulation_data()
    
    print("Creating SNR comparison plots...")
    snr_stats = create_snr_comparison_plots(metrics_df, tracking_data)
    
    print("Creating tracking trajectory plots...")
    create_tracking_trajectory_plots(tracking_data)
    
    print("Generating results summary...")
    summary_stats = generate_results_summary_table(metrics_df)
    
    # Save summary statistics to CSV
    summary_stats.to_csv('report_figures/snr_analysis_summary.csv')
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("- report_figures/snr_performance_analysis.pdf")
    print("- report_figures/snr_performance_analysis.png") 
    print("- report_figures/tracking_trajectories_snr.pdf")
    print("- report_figures/tracking_trajectories_snr.png")
    print("- report_figures/snr_analysis_summary.csv")

if __name__ == "__main__":
    main()