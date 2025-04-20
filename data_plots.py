import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
plt.rcParams['font.family'] = 'Times New Roman'

from pre_process import *

def extract_plotting_data(json_file: dict) -> Dict[str, List]:
    """
    Extracts plotting data from filtered proteins.
    """
    # Initialize data containers
    lengths = []
    disorder_percentages = []
    disorder_counts = []
    
    for sample in json_file['data']:
        # Apply the same filtering criteria
        disordered_regions = extract_disordered_regions(sample)
        if meets_filtering_criteria(sample, disordered_regions):
            lengths.append(sample['length'])
            disorder_percentages.append(sample['disorder_content'])
            disorder_counts.append(sample['regions_counter'])
    
    return {
        'lengths': lengths,
        'disorder_percentages': disorder_percentages,
        'disorder_counts': disorder_counts
    }

def create_analysis_plots(json_file: dict):
    data = extract_plotting_data(json_file)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.3)
    
    colors = ['#006D77', '#83C5BE', '#B8E1DD']
    
    # 1. Length Distribution
    sns.histplot(data=data['lengths'], bins=30, ax=ax1, color=colors[0])
    ax1.set_title('Protein Length Distribution', fontsize=12, pad=15)
    ax1.set_xlabel('Sequence Length', fontsize=10)
    ax1.set_ylabel('Count', fontsize=10)
    
    # Function to create consistent scatter plots
    def create_scatter_plot(ax, x, y, color, xlabel, ylabel, title):
        # Create scatter plot with consistent styling
        scatter = ax.scatter(x, y,
                           c=color,
                           edgecolor=color,
                           linewidth=1,
                           alpha=0.6,
                           s=50)
        
        # Set consistent limits based on data
        ax.set_xlim(min(data['lengths']) * 0.95, max(data['lengths']) * 1.05)
        
        # Set consistent styling
        ax.set_title(title, fontsize=12, pad=15)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        return scatter
    
    # 2. Length vs Disorder Percentage
    create_scatter_plot(ax2, data['lengths'], data['disorder_percentages'],
                       colors[1], 'Sequence Length', 'Disorder Content (%)',
                       'Protein Length vs Disorder Percentage')
    
    # 3. Length vs Number of Disordered Regions
    create_scatter_plot(ax3, data['lengths'], data['disorder_counts'],
                       colors[2], 'Sequence Length', 'Number of Disordered Regions',
                       'Protein Length vs Number of Disordered Regions')
    
    # Add statistics
    print (
        f"Total proteins: {len(data['lengths'])}\n"
        f"Mean length: {np.mean(data['lengths']):.1f}\n"
        f"Mean disorder %: {np.mean(data['disorder_percentages']):.1f}%\n"
        f"Mean regions: {np.mean(data['disorder_counts']):.1f}"
    )
    # plt.figtext(0.02, 0.02, stats_text, fontsize=10, ha='left', 
    #             font='Times New Roman')
    
    # Consistent styling for all plots
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(labelsize=9)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    
    plt.tight_layout()
    return fig

# Example usage:
json_path = "C:/Users/danas/OneDrive/Desktop/pro-disorder-predictor/DisProt_release_2024_12_Consensus_without_includes.json"
json_data = load_json_data(json_path)
fig = create_analysis_plots(json_data)
plt.show()