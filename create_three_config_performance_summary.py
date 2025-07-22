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
        'tvplus': 'TV Plus\n(4-mic Compact)',
        'tvx': 'TV X\n(4-mic Distributed)', 
        'soundbar3': 'Soundbar3\n(3-mic Linear)'
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
    
    return config_data, mic_configs, config_labels

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
        'num_mics': len(positions),
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'spread': spread,
        'positions': positions
    }

def create_three_config_performance_summary():
    """Create performance summary comparison for all three configurations"""
    
    config_data, mic_configs, config_labels = load_all_configuration_data()
    
    # Calculate performance metrics
    performance_data = {}
    for config, data in config_data.items():
        mic_data = mic_configs[config]
        geom = calculate_geometric_properties(mic_data)
        
        performance_data[config] = {
            'config_label': config_labels[config],
            'mic_count': geom['num_mics'],
            'max_distance': geom['max_distance'],
            'avg_error': data['mean_3d_error'].mean(),
            'best_error': data[data['snr'] == 20]['mean_3d_error'].mean() if len(data[data['snr'] == 20]) > 0 else np.nan,
            'worst_error': data[data['snr'] == 0]['mean_3d_error'].mean() if len(data[data['snr'] == 0]) > 0 else np.nan,
        }
    
    # Create the figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    configs = ['tvplus', 'tvx', 'soundbar3']
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Red, Orange
    
    # Compare mean errors (both average and best)
    scenarios = ['Average\n(All SNR)', 'Best\n(SNR 20 dB)']
    config_names = [performance_data[config]['config_label'] for config in configs]
    
    avg_values = [performance_data[config]['avg_error'] for config in configs]
    best_values = [performance_data[config]['best_error'] for config in configs]
    
    x_pos = np.arange(len(configs))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x_pos - width/2, avg_values, width, 
                   label='Average (All SNR)', alpha=0.8, color=colors)
    bars2 = ax.bar(x_pos + width/2, best_values, width, 
                   label='Best (SNR 20 dB)', alpha=0.8, 
                   color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Average values
        height1 = bar1.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.003,
               f'{height1:.3f}m', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Best values
        height2 = bar2.get_height()
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.003,
               f'{height2:.3f}m', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Customize the plot
    ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean 3D Tracking Error (m)', fontsize=14, fontweight='bold')
    ax.set_title('Three-Configuration Performance Summary Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_names, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add configuration details as text annotations
    for i, config in enumerate(configs):
        mic_count = performance_data[config]['mic_count']
        max_dist = performance_data[config]['max_distance']
        
        # Add microphone count annotation
        ax.annotate(f'{mic_count} mics', 
                   xy=(i, 0.02), xytext=(0, 0),
                   textcoords='offset points', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.8))
        
        # Add max distance annotation
        ax.annotate(f'Max: {max_dist:.2f}m', 
                   xy=(i, 0.05), xytext=(0, 0),
                   textcoords='offset points', ha='center', va='bottom',
                   fontsize=9, style='italic', color='gray')
    
    # Add performance improvement annotations
    for i, config in enumerate(configs):
        avg_val = avg_values[i]
        best_val = best_values[i]
        improvement = ((avg_val - best_val) / avg_val) * 100
        
        # Add improvement percentage between bars
        mid_x = i
        mid_y = max(avg_val, best_val) + 0.02
        ax.annotate(f'{improvement:.1f}%\nimprovement', 
                   xy=(mid_x, mid_y), xytext=(0, 0),
                   textcoords='offset points', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='green',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    # Add ranking annotations
    avg_ranking = sorted(enumerate(avg_values), key=lambda x: x[1])
    best_ranking = sorted(enumerate(best_values), key=lambda x: x[1])
    
    # Add ranking text box
    ranking_text = "Performance Rankings:\n\nAverage Performance:\n"
    for rank, (idx, _) in enumerate(avg_ranking, 1):
        config_name = config_names[idx].replace('\n', ' ')
        ranking_text += f"{rank}. {config_name}\n"
    
    ranking_text += "\nBest Performance (SNR 20dB):\n"
    for rank, (idx, _) in enumerate(best_ranking, 1):
        config_name = config_names[idx].replace('\n', ' ')
        ranking_text += f"{rank}. {config_name}\n"
    
    # Add key insights
    insights_text = "\nKey Insights:\n"
    insights_text += f"• 4-mic avg: {(avg_values[0] + avg_values[1])/2:.3f}m\n"
    insights_text += f"• 3-mic: {avg_values[2]:.3f}m\n"
    mic_penalty = ((avg_values[2] - (avg_values[0] + avg_values[1])/2) / ((avg_values[0] + avg_values[1])/2)) * 100
    insights_text += f"• 3-mic penalty: {mic_penalty:.1f}%\n"
    insights_text += f"• Best overall: {min(best_values):.3f}m (TV X)"
    
    combined_text = ranking_text + insights_text
    
    ax.text(0.98, 0.98, combined_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
           fontsize=10, fontfamily='monospace')
    
    # Set y-axis limits with some padding
    max_val = max(max(avg_values), max(best_values))
    ax.set_ylim(0, max_val * 1.3)
    
    # Add horizontal lines for comparison
    avg_line = np.mean(avg_values)
    ax.axhline(y=avg_line, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(configs), avg_line, f'Avg: {avg_line:.3f}m', 
           verticalalignment='center', horizontalalignment='left',
           fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig('report_figures/three_config_performance_summary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/three_config_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return performance_data

def create_detailed_comparison_chart():
    """Create a more detailed comparison chart with additional metrics"""
    
    config_data, mic_configs, config_labels = load_all_configuration_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Three-Configuration Performance Analysis', fontsize=16, fontweight='bold')
    
    configs = ['tvplus', 'tvx', 'soundbar3']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Calculate all metrics
    metrics_data = {}
    for config in configs:
        data = config_data[config]
        mic_data = mic_configs[config]
        geom = calculate_geometric_properties(mic_data)
        
        metrics_data[config] = {
            'avg_error': data['mean_3d_error'].mean(),
            'best_error': data[data['snr'] == 20]['mean_3d_error'].mean() if len(data[data['snr'] == 20]) > 0 else np.nan,
            'worst_error': data[data['snr'] == 0]['mean_3d_error'].mean() if len(data[data['snr'] == 0]) > 0 else np.nan,
            'std_error': data['std_3d_error'].mean(),
            'max_distance': geom['max_distance'],
            'mic_count': geom['num_mics']
        }
    
    # Plot 1: Average vs Best Performance
    config_names_short = ['TV Plus', 'TV X', 'Soundbar3']
    avg_errors = [metrics_data[config]['avg_error'] for config in configs]
    best_errors = [metrics_data[config]['best_error'] for config in configs]
    
    x_pos = np.arange(len(configs))
    width = 0.35
    
    axes[0,0].bar(x_pos - width/2, avg_errors, width, label='Average', color=colors, alpha=0.8)
    axes[0,0].bar(x_pos + width/2, best_errors, width, label='Best (SNR 20dB)', 
                  color=colors, alpha=0.6, edgecolor='black')
    
    axes[0,0].set_xlabel('Configuration')
    axes[0,0].set_ylabel('Error (m)')
    axes[0,0].set_title('Average vs Best Performance')
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels(config_names_short)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Error Variability
    std_errors = [metrics_data[config]['std_error'] for config in configs]
    
    bars = axes[0,1].bar(config_names_short, std_errors, color=colors, alpha=0.8)
    axes[0,1].set_xlabel('Configuration')
    axes[0,1].set_ylabel('Standard Deviation (m)')
    axes[0,1].set_title('Error Variability')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, std_errors):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                      f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Microphone Count vs Performance
    mic_counts = [metrics_data[config]['mic_count'] for config in configs]
    
    scatter = axes[1,0].scatter(mic_counts, avg_errors, s=200, c=colors, alpha=0.8, edgecolors='black', linewidth=2)
    
    for i, config in enumerate(configs):
        axes[1,0].annotate(config_names_short[i], 
                          (mic_counts[i], avg_errors[i]),
                          xytext=(5, 5), textcoords='offset points',
                          fontweight='bold')
    
    axes[1,0].set_xlabel('Number of Microphones')
    axes[1,0].set_ylabel('Average Error (m)')
    axes[1,0].set_title('Microphone Count vs Performance')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xticks([3, 4])
    
    # Plot 4: Performance Improvement Potential
    improvements = []
    for config in configs:
        worst = metrics_data[config]['worst_error']
        best = metrics_data[config]['best_error']
        improvement = ((worst - best) / worst) * 100
        improvements.append(improvement)
    
    bars = axes[1,1].bar(config_names_short, improvements, color=colors, alpha=0.8)
    axes[1,1].set_xlabel('Configuration')
    axes[1,1].set_ylabel('Improvement (%)')
    axes[1,1].set_title('SNR Improvement Potential (0→20 dB)')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, improvements):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                      f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('report_figures/detailed_three_config_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/detailed_three_config_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    print("Creating three-configuration performance summary...")
    performance_data = create_three_config_performance_summary()
    
    print("Creating detailed comparison chart...")
    create_detailed_comparison_chart()
    
    print("\nPerformance Summary:")
    print("-" * 50)
    for config, data in performance_data.items():
        print(f"{data['config_label'].replace(chr(10), ' ')}: {data['avg_error']:.4f}m avg, {data['best_error']:.4f}m best")
    
    print("\nThree-configuration performance summary complete!")
    print("Generated files:")
    print("- report_figures/three_config_performance_summary.pdf")
    print("- report_figures/three_config_performance_summary.png")
    print("- report_figures/detailed_three_config_analysis.pdf")
    print("- report_figures/detailed_three_config_analysis.png")

if __name__ == "__main__":
    main()