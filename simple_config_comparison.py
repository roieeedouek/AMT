import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_simple_config_data():
    """Load and process configuration data for simple comparison"""
    data_dir = Path("simulation_data")
    snr_levels = [0, 5, 10, 20]
    
    config_results = {}
    mic_configs = {}
    
    for config in ['tvplus', 'tvx']:
        config_metrics = []
        
        for snr in snr_levels:
            try:
                # Find metrics file
                metrics_pattern = f"{config}_SNR{snr}_*_metrics.csv"
                metrics_files = list(data_dir.glob(metrics_pattern))
                
                if not metrics_files:
                    continue
                
                # Load metrics
                metrics = pd.read_csv(metrics_files[0])
                metrics['snr'] = snr
                metrics['config'] = config
                config_metrics.append(metrics)
                
                # Load microphone configuration (once per config)
                if config not in mic_configs:
                    filename_parts = metrics_files[0].stem.split('_')
                    session_start = filename_parts[2] + '_' + filename_parts[3]
                    mic_pattern = f"{config}_SNR{snr}_{session_start}_*microphones.csv"
                    mic_files = list(data_dir.glob(mic_pattern))
                    if mic_files:
                        mic_configs[config] = pd.read_csv(mic_files[0])
                        
            except Exception as e:
                continue
        
        if config_metrics:
            config_results[config] = pd.concat(config_metrics, ignore_index=True)
    
    return config_results, mic_configs

def calculate_comparison_metrics(config_results):
    """Calculate key comparison metrics"""
    
    comparison_data = {}
    
    for config, data in config_results.items():
        # Average across all SNR levels
        avg_metrics = {
            'mean_error_avg': data['mean_3d_error'].mean(),
            'mean_error_std': data['mean_3d_error'].std(),
            'std_error_avg': data['std_3d_error'].mean(),
            'max_error_avg': data['max_3d_error'].mean(),
            'p95_error_avg': data['p95_3d_error'].mean()
        }
        
        # Best SNR performance (SNR 20)
        best_snr_data = data[data['snr'] == 20]
        if len(best_snr_data) > 0:
            best_metrics = {
                'mean_error_best': best_snr_data['mean_3d_error'].mean(),
                'std_error_best': best_snr_data['std_3d_error'].mean(),
                'max_error_best': best_snr_data['max_3d_error'].mean(),
                'p95_error_best': best_snr_data['p95_3d_error'].mean()
            }
        else:
            best_metrics = {k: np.nan for k in ['mean_error_best', 'std_error_best', 'max_error_best', 'p95_error_best']}
        
        # Combine metrics
        comparison_data[config] = {**avg_metrics, **best_metrics}
    
    return comparison_data

def create_simple_comparison_plot(comparison_data, mic_configs):
    """Create simplified comparison visualization"""
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 14,
        'figure.figsize': (15, 10)
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Configuration Comparison: TV Plus vs TV X', fontsize=18, fontweight='bold')
    
    configs = ['tvplus', 'tvx']
    config_labels = ['TV Plus', 'TV X']
    colors = ['#2E86AB', '#A23B72']  # Blue and Red
    
    # Plot 1: Average Performance Comparison
    metrics = ['mean_error_avg', 'std_error_avg', 'max_error_avg']
    metric_labels = ['Mean Error', 'Std Error', 'Max Error']
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    for i, config in enumerate(configs):
        values = [comparison_data[config][metric] for metric in metrics]
        axes[0,0].bar(x_pos + i*width, values, width, 
                     label=config_labels[i], color=colors[i], alpha=0.8)
    
    axes[0,0].set_xlabel('Error Metrics')
    axes[0,0].set_ylabel('Error (m)')
    axes[0,0].set_title('Average Performance (All SNR Levels)')
    axes[0,0].set_xticks(x_pos + width/2)
    axes[0,0].set_xticklabels(metric_labels)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Best SNR Performance Comparison
    for i, config in enumerate(configs):
        values = [comparison_data[config][metric.replace('_avg', '_best')] for metric in metrics]
        axes[0,1].bar(x_pos + i*width, values, width, 
                     label=config_labels[i], color=colors[i], alpha=0.8)
    
    axes[0,1].set_xlabel('Error Metrics')
    axes[0,1].set_ylabel('Error (m)')
    axes[0,1].set_title('Best Performance (SNR 20 dB)')
    axes[0,1].set_xticks(x_pos + width/2)
    axes[0,1].set_xticklabels(metric_labels)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Microphone Configurations
    for i, (config, mic_data) in enumerate(mic_configs.items()):
        ax = axes[1, i]
        
        # Plot microphone positions
        scatter = ax.scatter(mic_data['x'], mic_data['y'], 
                           c=mic_data['z'], s=400, cmap='viridis',
                           edgecolors='black', linewidth=3, alpha=0.8)
        
        # Add microphone labels
        for _, row in mic_data.iterrows():
            ax.annotate(row['mic_id'], (row['x'], row['y']), 
                       xytext=(0, 0), textcoords='offset points',
                       fontsize=12, fontweight='bold', ha='center', va='center',
                       color='white')
        
        # Add connecting lines to show geometry
        positions = mic_data[['x', 'y']].values
        for j in range(len(positions)):
            for k in range(j+1, len(positions)):
                ax.plot([positions[j,0], positions[k,0]], 
                       [positions[j,1], positions[k,1]], 
                       'k--', alpha=0.4, linewidth=1)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{config_labels[i]} Microphone Layout')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Z coordinate (m)')
    
    plt.tight_layout()
    plt.savefig('report_figures/simple_config_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/simple_config_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_summary_plot(comparison_data):
    """Create a single summary performance comparison"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    configs = ['tvplus', 'tvx']
    config_labels = ['TV Plus', 'TV X']
    colors = ['#2E86AB', '#A23B72']
    
    # Compare mean errors (both average and best)
    scenarios = ['Average\n(All SNR)', 'Best\n(SNR 20 dB)']
    tvplus_values = [comparison_data['tvplus']['mean_error_avg'], 
                     comparison_data['tvplus']['mean_error_best']]
    tvx_values = [comparison_data['tvx']['mean_error_avg'], 
                  comparison_data['tvx']['mean_error_best']]
    
    x_pos = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, tvplus_values, width, 
                   label='TV Plus', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, tvx_values, width, 
                   label='TV X', color=colors[1], alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                   f'{height:.3f}m', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Performance Scenario')
    ax.set_ylabel('Mean 3D Tracking Error (m)')
    ax.set_title('Configuration Performance Summary', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add percentage difference annotations
    for i, scenario in enumerate(scenarios):
        tvplus_val = tvplus_values[i]
        tvx_val = tvx_values[i]
        diff_pct = ((tvx_val - tvplus_val) / tvplus_val) * 100
        
        y_pos = max(tvplus_val, tvx_val) + 0.015
        if diff_pct > 0:
            ax.text(x_pos[i], y_pos, f'TV Plus\n{diff_pct:.1f}% better', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        else:
            ax.text(x_pos[i], y_pos, f'TV X\n{abs(diff_pct):.1f}% better', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('report_figures/performance_summary_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/performance_summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_simple_summary_table(comparison_data):
    """Generate simplified summary table"""
    
    print("\n" + "="*80)
    print("SIMPLIFIED CONFIGURATION COMPARISON")
    print("="*80)
    
    print("\n1. AVERAGE PERFORMANCE (All SNR Levels):")
    print("-" * 50)
    print(f"{'Metric':<20} {'TV Plus':<12} {'TV X':<12} {'Winner':<10}")
    print("-" * 50)
    
    avg_metrics = [
        ('Mean Error (m)', 'mean_error_avg'),
        ('Std Error (m)', 'std_error_avg'),
        ('Max Error (m)', 'max_error_avg'),
        ('P95 Error (m)', 'p95_error_avg')
    ]
    
    tvplus_wins = 0
    tvx_wins = 0
    
    for metric_name, metric_key in avg_metrics:
        tvplus_val = comparison_data['tvplus'][metric_key]
        tvx_val = comparison_data['tvx'][metric_key]
        winner = "TV Plus" if tvplus_val < tvx_val else "TV X"
        
        if winner == "TV Plus":
            tvplus_wins += 1
        else:
            tvx_wins += 1
            
        print(f"{metric_name:<20} {tvplus_val:<12.4f} {tvx_val:<12.4f} {winner:<10}")
    
    print("\n2. BEST PERFORMANCE (SNR 20 dB):")
    print("-" * 50)
    print(f"{'Metric':<20} {'TV Plus':<12} {'TV X':<12} {'Winner':<10}")
    print("-" * 50)
    
    best_metrics = [
        ('Mean Error (m)', 'mean_error_best'),
        ('Std Error (m)', 'std_error_best'),
        ('Max Error (m)', 'max_error_best'),
        ('P95 Error (m)', 'p95_error_best')
    ]
    
    tvplus_wins_best = 0
    tvx_wins_best = 0
    
    for metric_name, metric_key in best_metrics:
        tvplus_val = comparison_data['tvplus'][metric_key]
        tvx_val = comparison_data['tvx'][metric_key]
        winner = "TV Plus" if tvplus_val < tvx_val else "TV X"
        
        if winner == "TV Plus":
            tvplus_wins_best += 1
        else:
            tvx_wins_best += 1
            
        print(f"{metric_name:<20} {tvplus_val:<12.4f} {tvx_val:<12.4f} {winner:<10}")
    
    print("\n3. OVERALL SUMMARY:")
    print("-" * 50)
    print(f"Average Performance Winner: {'TV Plus' if tvplus_wins > tvx_wins else 'TV X'} ({max(tvplus_wins, tvx_wins)}/{len(avg_metrics)} metrics)")
    print(f"Best Performance Winner: {'TV Plus' if tvplus_wins_best > tvx_wins_best else 'TV X'} ({max(tvplus_wins_best, tvx_wins_best)}/{len(best_metrics)} metrics)")
    
    # Performance differences
    avg_diff = comparison_data['tvx']['mean_error_avg'] - comparison_data['tvplus']['mean_error_avg']
    best_diff = comparison_data['tvx']['mean_error_best'] - comparison_data['tvplus']['mean_error_best']
    
    print(f"\nMean Error Differences:")
    print(f"Average Performance: {avg_diff:+.4f} m ({'TV Plus better' if avg_diff > 0 else 'TV X better'})")
    print(f"Best Performance: {best_diff:+.4f} m ({'TV Plus better' if best_diff > 0 else 'TV X better'})")
    
    print("\n4. RECOMMENDATION:")
    print("-" * 50)
    if abs(avg_diff) < 0.01 and abs(best_diff) < 0.02:
        print("• Performance difference is minimal between configurations")
        print("• Choose based on other factors (geometry, deployment constraints)")
    elif avg_diff > 0 and best_diff > 0:
        print("• TV Plus recommended for consistent performance across all conditions")
    elif avg_diff < 0 and best_diff < 0:
        print("• TV X recommended for superior performance in all scenarios")
    else:
        print("• TV Plus better for average conditions")
        print("• TV X better for optimal conditions (high SNR)")
    
    print("="*80)
    
    # Save summary data
    summary_data = []
    for config in ['tvplus', 'tvx']:
        for scenario in ['avg', 'best']:
            row = {
                'Configuration': config.upper(),
                'Scenario': 'Average' if scenario == 'avg' else 'Best SNR',
                'Mean_Error': comparison_data[config][f'mean_error_{scenario}'],
                'Std_Error': comparison_data[config][f'std_error_{scenario}'],
                'Max_Error': comparison_data[config][f'max_error_{scenario}']
            }
            if scenario == 'avg':
                row['P95_Error'] = comparison_data[config]['p95_error_avg']
            else:
                row['P95_Error'] = comparison_data[config]['p95_error_best']
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('report_figures/simple_config_comparison_summary.csv', index=False)
    
    return summary_df

def main():
    """Main analysis function"""
    print("Loading configuration data for simple comparison...")
    config_results, mic_configs = load_simple_config_data()
    
    print("Calculating comparison metrics...")
    comparison_data = calculate_comparison_metrics(config_results)
    
    print("Creating simple comparison plots...")
    create_simple_comparison_plot(comparison_data, mic_configs)
    
    print("Creating performance summary plot...")
    create_performance_summary_plot(comparison_data)
    
    print("Generating simple summary table...")
    summary_df = generate_simple_summary_table(comparison_data)
    
    print("\nSimple comparison complete!")
    print("Generated files:")
    print("- report_figures/simple_config_comparison.pdf")
    print("- report_figures/simple_config_comparison.png")
    print("- report_figures/performance_summary_comparison.pdf")
    print("- report_figures/performance_summary_comparison.png")
    print("- report_figures/simple_config_comparison_summary.csv")

if __name__ == "__main__":
    main()