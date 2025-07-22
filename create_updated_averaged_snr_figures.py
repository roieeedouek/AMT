import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_average_all_configurations():
    """Load simulation data for all configurations and average the results by SNR"""
    data_dir = Path("simulation_data")
    snr_levels = [0, 5, 10, 20]
    
    all_metrics = []
    tracking_data = {}
    
    # Define all configurations
    configs = ['tvplus', 'tvx', 'soundbar3']
    
    for snr in snr_levels:
        snr_metrics = []
        snr_tracking = {'true': [], 'estimated': [], 'filtered': []}
        
        for config in configs:
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

def create_updated_averaged_snr_plots(metrics_df, tracking_data):
    """Create SNR comparison plots with averaged results from all three configurations"""
    
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
    fig.suptitle('SNR Performance Analysis - Averaged Results from All Configurations\n(TV Plus + TV X + Soundbar3)', 
                fontsize=16, fontweight='bold')
    
    # Plot 1: Mean 3D Error vs SNR (same as original)
    mean_errors = [snr_stats.loc[snr, ('mean_3d_error', 'mean')] for snr in snr_levels]
    error_stds = [snr_stats.loc[snr, ('mean_3d_error', 'std')] for snr in snr_levels]
    
    axes[0,0].errorbar(snr_levels, mean_errors, yerr=error_stds, 
                       marker='o', linewidth=2, markersize=8, capsize=5, color='blue')
    axes[0,0].set_xlabel('SNR (dB)')
    axes[0,0].set_ylabel('Mean 3D Error (m)')
    axes[0,0].set_title('Mean 3D Tracking Error vs SNR')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add specific values as annotations
    for i, (snr, error) in enumerate(zip(snr_levels, mean_errors)):
        axes[0,0].annotate(f'{error:.3f}m', 
                          (snr, error), 
                          textcoords="offset points", 
                          xytext=(0,10), 
                          ha='center',
                          fontsize=10,
                          fontweight='bold')
    
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
    
    # Plot 3: Tracking Performance Over Time (averaged from all configs)
    for snr in snr_levels:
        if snr in tracking_data:
            true_data = tracking_data[snr]['true']
            est_data = tracking_data[snr]['estimated']
            
            # Calculate average 3D error over time for all targets and configs
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
    
    # Add performance improvement annotations
    baseline_error = perf_df.iloc[0]['Avg_Error']  # SNR 0 performance
    for i, row in perf_df.iterrows():
        if row['SNR'] > 0:
            improvement = ((baseline_error - row['Avg_Error']) / baseline_error) * 100
            axes[1,1].annotate(f'+{improvement:.1f}%', 
                              (row['SNR'], row['Avg_Error']), 
                              textcoords="offset points", 
                              xytext=(0,-15), 
                              ha='center',
                              fontsize=9,
                              color='green',
                              fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('report_figures/updated_averaged_snr_performance_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/updated_averaged_snr_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return snr_stats

def create_configuration_breakdown_plot(metrics_df):
    """Create a plot showing individual configuration contributions to averaged results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Configuration Breakdown: Individual Contributions to Averaged Results', fontsize=16, fontweight='bold')
    
    snr_levels = sorted(metrics_df['snr'].unique())
    configs = ['tvplus', 'tvx', 'soundbar3']
    config_labels = {
        'tvplus': 'TV Plus (4-mic Compact)',
        'tvx': 'TV X (4-mic Distributed)',
        'soundbar3': 'Soundbar (3-mic Linear)'
    }
    colors = {'tvplus': '#2E86AB', 'tvx': '#A23B72', 'soundbar3': '#F18F01'}
    markers = {'tvplus': 'o', 'tvx': 's', 'soundbar3': '^'}
    
    # Plot 1: Individual configuration performance
    for config in configs:
        config_data = metrics_df[metrics_df['config'] == config]
        if len(config_data) > 0:
            snr_stats = config_data.groupby('snr')['mean_3d_error'].agg(['mean', 'std'])
            
            mean_errors = [snr_stats.loc[snr, 'mean'] if snr in snr_stats.index else np.nan for snr in snr_levels]
            error_stds = [snr_stats.loc[snr, 'std'] if snr in snr_stats.index else 0 for snr in snr_levels]
            
            valid_idx = ~np.isnan(mean_errors)
            valid_snr = np.array(snr_levels)[valid_idx]
            valid_means = np.array(mean_errors)[valid_idx]
            valid_stds = np.array(error_stds)[valid_idx]
            
            axes[0,0].errorbar(valid_snr, valid_means, yerr=valid_stds, 
                              marker=markers[config], linewidth=2, markersize=8, capsize=5,
                              label=config_labels[config], color=colors[config])
    
    # Add averaged line
    snr_stats = metrics_df.groupby('snr')['mean_3d_error'].agg(['mean'])
    avg_errors = [snr_stats.loc[snr, 'mean'] for snr in snr_levels]
    axes[0,0].plot(snr_levels, avg_errors, 'k--', linewidth=3, label='Averaged Result', alpha=0.8)
    
    axes[0,0].set_xlabel('SNR (dB)')
    axes[0,0].set_ylabel('Mean 3D Error (m)')
    axes[0,0].set_title('Individual vs Averaged Performance')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Configuration count by SNR
    config_counts = []
    for snr in snr_levels:
        snr_data = metrics_df[metrics_df['snr'] == snr]
        count = len(snr_data['config'].unique())
        config_counts.append(count)
    
    bars = axes[0,1].bar(snr_levels, config_counts, color='lightblue', alpha=0.8, edgecolor='black')
    axes[0,1].set_xlabel('SNR (dB)')
    axes[0,1].set_ylabel('Number of Configurations')
    axes[0,1].set_title('Configurations Contributing to Average')
    axes[0,1].set_ylim(0, 4)
    axes[0,1].grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, config_counts):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                      str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Sample size information
    sample_info = []
    for snr in snr_levels:
        snr_data = metrics_df[metrics_df['snr'] == snr]
        total_samples = len(snr_data)
        sample_info.append(total_samples)
    
    bars = axes[1,0].bar(snr_levels, sample_info, color='lightgreen', alpha=0.8, edgecolor='black')
    axes[1,0].set_xlabel('SNR (dB)')
    axes[1,0].set_ylabel('Total Sample Size')
    axes[1,0].set_title('Sample Size per SNR Level')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add sample size labels
    for bar, samples in zip(bars, sample_info):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                      str(samples), ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Configuration summary table
    axes[1,1].axis('off')
    
    # Create summary table
    table_data = [['Configuration', 'Mic Count', 'Avg Error (m)', 'Contribution']]
    
    total_samples = len(metrics_df)
    for config in configs:
        config_data = metrics_df[metrics_df['config'] == config]
        if len(config_data) > 0:
            mic_count = "3" if config == 'soundbar3' else "4"
            avg_error = config_data['mean_3d_error'].mean()
            contribution = len(config_data) / total_samples * 100
            
            table_data.append([
                config_labels[config],
                mic_count,
                f'{avg_error:.4f}',
                f'{contribution:.1f}%'
            ])
    
    # Overall average
    overall_avg = metrics_df['mean_3d_error'].mean()
    table_data.append(['Overall Average', '3+4', f'{overall_avg:.4f}', '100.0%'])
    
    table = axes[1,1].table(cellText=table_data[1:],
                           colLabels=table_data[0],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style table
    for i in range(len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            elif i == len(table_data) - 1:  # Overall average row
                cell.set_facecolor('#FFE6CC')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F8F8F8')
    
    axes[1,1].set_title('Configuration Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('report_figures/configuration_breakdown_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/configuration_breakdown_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_updated_averaged_summary_table(metrics_df):
    """Generate updated summary table with all three configurations included"""
    
    print("\n" + "="*90)
    print("UPDATED AVERAGED SNR PERFORMANCE ANALYSIS")
    print("Including TV Plus + TV X + Soundbar3 Configurations")
    print("="*90)
    print(f"{'SNR (dB)':<10} {'Mean Error':<12} {'Std Error':<12} {'Max Error':<12} {'P95 Error':<12} {'Samples':<8}")
    print("-"*90)
    
    summary_stats = metrics_df.groupby('snr').agg({
        'mean_3d_error': ['mean', 'std', 'min', 'max'],
        'std_3d_error': ['mean', 'std'],
        'max_3d_error': ['mean', 'std'],
        'p95_3d_error': ['mean', 'std'],
        'target_id': 'count'
    }).round(4)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    
    for snr in sorted(metrics_df['snr'].unique()):
        mean_err = summary_stats.loc[snr, 'mean_3d_error_mean']
        std_err = summary_stats.loc[snr, 'std_3d_error_mean'] 
        max_err = summary_stats.loc[snr, 'max_3d_error_mean']
        p95_err = summary_stats.loc[snr, 'p95_3d_error_mean']
        n_samples = int(summary_stats.loc[snr, 'target_id_count'])
        
        print(f"{snr:<10} {mean_err:<12.4f} {std_err:<12.4f} {max_err:<12.4f} {p95_err:<12.4f} {n_samples:<8}")
    
    print("="*90)
    
    # Performance improvement analysis
    snr_0_error = summary_stats.loc[0, 'mean_3d_error_mean']
    print("\nPERFORMANCE IMPROVEMENTS RELATIVE TO SNR 0 dB:")
    print("-"*50)
    for snr in [5, 10, 20]:
        if snr in summary_stats.index:
            snr_error = summary_stats.loc[snr, 'mean_3d_error_mean']
            improvement = ((snr_0_error - snr_error) / snr_0_error) * 100
            print(f"SNR {snr} dB: {improvement:+.1f}% improvement")
    
    # Configuration breakdown
    print("\nCONFIGURATION BREAKDOWN:")
    print("-"*50)
    configs = ['tvplus', 'tvx', 'soundbar3']
    config_labels = {
        'tvplus': 'TV Plus (4-mic Compact)',
        'tvx': 'TV X (4-mic Distributed)',
        'soundbar3': 'Soundbar (3-mic Linear)'
    }
    
    for config in configs:
        config_data = metrics_df[metrics_df['config'] == config]
        if len(config_data) > 0:
            avg_error = config_data['mean_3d_error'].mean()
            n_runs = len(config_data)
            print(f"{config_labels[config]}: {avg_error:.4f} m ({n_runs} runs)")
    
    overall_avg = metrics_df['mean_3d_error'].mean()
    total_runs = len(metrics_df)
    print(f"Overall Average (All Configs): {overall_avg:.4f} m ({total_runs} total runs)")
    
    print("="*90)
    
    # Save updated summary statistics
    summary_stats.to_csv('report_figures/updated_averaged_snr_analysis_summary.csv')
    
    return summary_stats

def main():
    """Main analysis function"""
    print("Loading complete simulation data from all three configurations...")
    metrics_df, tracking_data = load_and_average_all_configurations()
    
    if metrics_df.empty:
        print("Error: No data loaded!")
        return
    
    print(f"Loaded and averaged data from {len(metrics_df)} simulation runs")
    print(f"Configurations included: {metrics_df['config'].unique()}")
    print(f"SNR levels: {sorted(metrics_df['snr'].unique())}")
    
    print("Creating updated averaged SNR comparison plots...")
    snr_stats = create_updated_averaged_snr_plots(metrics_df, tracking_data)
    
    print("Creating configuration breakdown analysis...")
    create_configuration_breakdown_plot(metrics_df)
    
    print("Generating updated averaged summary...")
    summary_stats = generate_updated_averaged_summary_table(metrics_df)
    
    print("\nUpdated analysis complete!")
    print("Generated files:")
    print("- report_figures/updated_averaged_snr_performance_analysis.pdf")
    print("- report_figures/updated_averaged_snr_performance_analysis.png")
    print("- report_figures/configuration_breakdown_analysis.pdf")
    print("- report_figures/configuration_breakdown_analysis.png")
    print("- report_figures/updated_averaged_snr_analysis_summary.csv")

if __name__ == "__main__":
    main()