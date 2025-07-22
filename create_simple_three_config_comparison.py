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
    
    return {
        'num_mics': len(positions),
        'max_distance': np.max(distances),
    }

def create_simple_three_config_comparison():
    """Create only the Average vs Best Performance comparison figure"""
    
    config_data, mic_configs = load_all_configuration_data()
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    configs = ['tvplus', 'tvx', 'soundbar3']
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Red, Orange
    config_names = ['TV Plus', 'TV X', 'Soundbar3']
    
    # Calculate metrics for each configuration
    avg_errors = []
    best_errors = []
    
    for config in configs:
        data = config_data[config]
        
        # Average error across all SNR levels
        avg_error = data['mean_3d_error'].mean()
        avg_errors.append(avg_error)
        
        # Best error (SNR 20 dB)
        best_data = data[data['snr'] == 20]
        if len(best_data) > 0:
            best_error = best_data['mean_3d_error'].mean()
            best_errors.append(best_error)
        else:
            best_errors.append(np.nan)
    
    # Create bar chart
    x_pos = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, avg_errors, width, 
                   label='Average (All SNR)', color=colors, alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, best_errors, width, 
                   label='Best (SNR 20 dB)', color=colors, alpha=0.6, 
                   edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Average values
        height1 = bar1.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.003,
               f'{height1:.4f}m', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Best values
        height2 = bar2.get_height()
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.003,
               f'{height2:.4f}m', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Customize the plot
    ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean 3D Tracking Error (m)', fontsize=14, fontweight='bold')
    ax.set_title('Average vs Best Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_names, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add microphone count annotations
    for i, config in enumerate(configs):
        mic_data = mic_configs[config]
        geom = calculate_geometric_properties(mic_data)
        mic_count = geom['num_mics']
        
        # Add microphone count annotation below x-axis
        ax.annotate(f'{mic_count} mics', 
                   xy=(i, -0.015), xytext=(0, 0),
                   textcoords='offset points', ha='center', va='top',
                   fontsize=10, fontweight='bold', color=colors[i])
    
    # Add improvement percentage annotations
    for i in range(len(configs)):
        avg_val = avg_errors[i]
        best_val = best_errors[i]
        if not np.isnan(best_val):
            improvement = ((avg_val - best_val) / avg_val) * 100
            
            # Add improvement annotation between bars
            mid_x = i
            mid_y = max(avg_val, best_val) + 0.015
            ax.annotate(f'{improvement:.1f}%', 
                       xy=(mid_x, mid_y), xytext=(0, 0),
                       textcoords='offset points', ha='center', va='bottom',
                       fontsize=10, fontweight='bold', color='green',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
    
    # Set y-axis limits with padding
    max_val = max(max(avg_errors), max([x for x in best_errors if not np.isnan(x)]))
    ax.set_ylim(0, max_val * 1.15)
    
    # Add summary statistics as text
    summary_text = "Performance Summary:\n"
    summary_text += f"4-mic average: {(avg_errors[0] + avg_errors[1])/2:.4f}m\n"
    summary_text += f"3-mic: {avg_errors[2]:.4f}m\n"
    penalty = ((avg_errors[2] - (avg_errors[0] + avg_errors[1])/2) / ((avg_errors[0] + avg_errors[1])/2)) * 100
    summary_text += f"3-mic penalty: {penalty:.1f}%"
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8),
           fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('report_figures/simple_three_config_performance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('report_figures/simple_three_config_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return avg_errors, best_errors

def main():
    """Main function"""
    print("Creating simple three-configuration performance comparison...")
    avg_errors, best_errors = create_simple_three_config_comparison()
    
    print("\nPerformance Results:")
    print("-" * 40)
    configs = ['TV Plus', 'TV X', 'Soundbar3']
    for i, config in enumerate(configs):
        print(f"{config}: {avg_errors[i]:.4f}m avg, {best_errors[i]:.4f}m best")
    
    print("\nSimple performance comparison complete!")
    print("Generated files:")
    print("- report_figures/simple_three_config_performance_comparison.pdf")
    print("- report_figures/simple_three_config_performance_comparison.png")

if __name__ == "__main__":
    main()