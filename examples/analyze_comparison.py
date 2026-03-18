#!/usr/bin/env python3
"""
Analyze comparison results between two planners.
Usage: python3 examples/analyze_comparison.py <experiment_folder> [output_file]
"""

import sys
import os
import numpy as np
from scipy import stats
from datetime import datetime


def load_costs(planner_folder: str) -> list:
    """Load final costs from costs.txt file."""
    costs_file = os.path.join(planner_folder, "costs.txt")
    if not os.path.exists(costs_file):
        raise FileNotFoundError(f"Costs file not found: {costs_file}")
    
    final_costs = []
    with open(costs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Each line has comma-separated costs, last one is final cost
                values = [float(x) for x in line.split(',') if x.strip()]
                if values:
                    final_costs.append(values[-1])
    return final_costs


def load_timestamps(planner_folder: str) -> tuple:
    """Load timestamps from timestamps.txt file.
    
    Returns:
        tuple: (first_solution_times, final_times, num_improvements)
            - first_solution_times: time to find first feasible solution per run
            - final_times: total planning time per run
            - num_improvements: number of solution improvements per run
    """
    timestamps_file = os.path.join(planner_folder, "timestamps.txt")
    if not os.path.exists(timestamps_file):
        raise FileNotFoundError(f"Timestamps file not found: {timestamps_file}")
    
    first_solution_times = []
    final_times = []
    num_improvements = []
    
    with open(timestamps_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                values = [float(x) for x in line.split(',') if x.strip()]
                if values:
                    first_solution_times.append(values[0])
                    final_times.append(values[-1])
                    num_improvements.append(len(values))
    
    return first_solution_times, final_times, num_improvements


def analyze(experiment_folder: str, output_file: str = None):
    """Analyze comparison between baseline and heuristic planners."""
    
    # Find planner folders
    baseline_folder = os.path.join(experiment_folder, "rrtstar_baseline")
    heuristic_folder = os.path.join(experiment_folder, "heuristic_rrtstar")
    
    if not os.path.exists(baseline_folder):
        # Try alternative naming
        for name in os.listdir(experiment_folder):
            full_path = os.path.join(experiment_folder, name)
            if not os.path.isdir(full_path):
                continue
            if "baseline" in name.lower() or name == "rrtstar":
                baseline_folder = full_path
            elif "heuristic" in name.lower():
                heuristic_folder = full_path
    
    print(f"Loading baseline from: {baseline_folder}")
    print(f"Loading heuristic from: {heuristic_folder}")
    
    # Load costs
    baseline_costs = load_costs(baseline_folder)
    heuristic_costs = load_costs(heuristic_folder)
    
    # Load timestamps
    baseline_first_times, baseline_final_times, baseline_num_impr = load_timestamps(baseline_folder)
    heuristic_first_times, heuristic_final_times, heuristic_num_impr = load_timestamps(heuristic_folder)
    
    n_runs = min(len(baseline_costs), len(heuristic_costs))
    baseline_costs = baseline_costs[:n_runs]
    heuristic_costs = heuristic_costs[:n_runs]
    baseline_first_times = baseline_first_times[:n_runs]
    heuristic_first_times = heuristic_first_times[:n_runs]
    baseline_num_impr = baseline_num_impr[:n_runs]
    heuristic_num_impr = heuristic_num_impr[:n_runs]
    
    print(f"Loaded {n_runs} runs from each planner")
    
    # Calculate metrics
    baseline_arr = np.array(baseline_costs)
    heuristic_arr = np.array(heuristic_costs)
    baseline_first_arr = np.array(baseline_first_times)
    heuristic_first_arr = np.array(heuristic_first_times)
    baseline_impr_arr = np.array(baseline_num_impr)
    heuristic_impr_arr = np.array(heuristic_num_impr)
    
    # Win/Loss/Tie analysis (cost)
    wins = np.sum(heuristic_arr < baseline_arr)
    losses = np.sum(heuristic_arr > baseline_arr)
    ties = np.sum(np.isclose(heuristic_arr, baseline_arr, rtol=1e-6))
    
    # Win/Loss/Tie analysis (time to first solution - lower is better)
    time_wins = np.sum(heuristic_first_arr < baseline_first_arr)
    time_losses = np.sum(heuristic_first_arr > baseline_first_arr)
    time_ties = np.sum(np.isclose(heuristic_first_arr, baseline_first_arr, rtol=1e-6))
    
    # Cost improvements
    improvements = baseline_arr - heuristic_arr
    pct_improvements = (improvements / baseline_arr) * 100
    
    # Statistical tests (cost)
    t_stat, t_pvalue = stats.ttest_rel(baseline_arr, heuristic_arr)
    try:
        wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(baseline_arr, heuristic_arr)
    except ValueError:
        # Wilcoxon test fails if all differences are zero
        wilcoxon_stat, wilcoxon_pvalue = float('nan'), float('nan')
    
    # Statistical tests (timing)
    t_stat_time, t_pvalue_time = stats.ttest_rel(baseline_first_arr, heuristic_first_arr)
    try:
        wilcoxon_stat_time, wilcoxon_pvalue_time = stats.wilcoxon(baseline_first_arr, heuristic_first_arr)
    except ValueError:
        wilcoxon_stat_time, wilcoxon_pvalue_time = float('nan'), float('nan')
    
    # Time improvements (positive = heuristic faster)
    time_improvements = baseline_first_arr - heuristic_first_arr
    pct_time_improvements = (time_improvements / baseline_first_arr) * 100
    
    # Generate report
    report = f"""================================================================================
HEURISTIC RRT* vs BASELINE RRT* COMPARISON REPORT
================================================================================
Date: {datetime.now().strftime("%d %B %Y %H:%M:%S")}
Experiment Folder: {experiment_folder}

================================================================================
SUMMARY
================================================================================

Number of Runs: {n_runs}

                        | Baseline RRT* | Heuristic RRT* |
  ----------------------|---------------|----------------|
  Mean Cost             | {np.mean(baseline_arr):13.4f} | {np.mean(heuristic_arr):14.4f} |
  Std Dev               | {np.std(baseline_arr):13.4f} | {np.std(heuristic_arr):14.4f} |
  Median Cost           | {np.median(baseline_arr):13.4f} | {np.median(heuristic_arr):14.4f} |
  Min Cost              | {np.min(baseline_arr):13.4f} | {np.min(heuristic_arr):14.4f} |
  Max Cost              | {np.max(baseline_arr):13.4f} | {np.max(heuristic_arr):14.4f} |

================================================================================
WIN/LOSS ANALYSIS
================================================================================

  Heuristic Wins (lower cost):  {wins:4d} / {n_runs} ({100*wins/n_runs:5.1f}%)
  Heuristic Losses:             {losses:4d} / {n_runs} ({100*losses/n_runs:5.1f}%)
  Ties (within 1e-6):           {ties:4d} / {n_runs} ({100*ties/n_runs:5.1f}%)

================================================================================
IMPROVEMENT ANALYSIS
================================================================================

  Mean Cost Improvement:        {np.mean(improvements):+.4f}
  Mean % Improvement:           {np.mean(pct_improvements):+.2f}%
  Median Cost Improvement:      {np.median(improvements):+.4f}
  Median % Improvement:         {np.median(pct_improvements):+.2f}%

  Best Improvement:             {np.max(improvements):+.4f} ({np.max(pct_improvements):+.2f}%)
  Worst Regression:             {np.min(improvements):+.4f} ({np.min(pct_improvements):+.2f}%)

================================================================================
STATISTICAL SIGNIFICANCE
================================================================================

  Paired t-test:
    t-statistic:                {t_stat:.4f}
    p-value:                    {t_pvalue:.6f}
    Significant (p<0.05):       {"YES" if t_pvalue < 0.05 else "NO"}

  Wilcoxon signed-rank test:
    W-statistic:                {wilcoxon_stat:.4f}
    p-value:                    {wilcoxon_pvalue:.6f}
    Significant (p<0.05):       {"YES" if wilcoxon_pvalue < 0.05 else "NO"}

================================================================================
TIMING COMPARISON (Time to First Solution)
================================================================================

                        | Baseline RRT* | Heuristic RRT* |
  ----------------------|---------------|----------------|
  Mean Time (s)         | {np.mean(baseline_first_arr):13.4f} | {np.mean(heuristic_first_arr):14.4f} |
  Std Dev               | {np.std(baseline_first_arr):13.4f} | {np.std(heuristic_first_arr):14.4f} |
  Median Time (s)       | {np.median(baseline_first_arr):13.4f} | {np.median(heuristic_first_arr):14.4f} |
  Min Time (s)          | {np.min(baseline_first_arr):13.4f} | {np.min(heuristic_first_arr):14.4f} |
  Max Time (s)          | {np.max(baseline_first_arr):13.4f} | {np.max(heuristic_first_arr):14.4f} |

  Mean # Improvements   | {np.mean(baseline_impr_arr):13.1f} | {np.mean(heuristic_impr_arr):14.1f} |

================================================================================
TIMING WIN/LOSS ANALYSIS (Faster First Solution)
================================================================================

  Heuristic Wins (faster):    {time_wins:4d} / {n_runs} ({100*time_wins/n_runs:5.1f}%)
  Heuristic Losses:           {time_losses:4d} / {n_runs} ({100*time_losses/n_runs:5.1f}%)
  Ties (within 1e-6):         {time_ties:4d} / {n_runs} ({100*time_ties/n_runs:5.1f}%)

  Mean Time Improvement:      {np.mean(time_improvements):+.4f}s
  Mean % Improvement:         {np.mean(pct_time_improvements):+.2f}%

================================================================================
TIMING STATISTICAL SIGNIFICANCE
================================================================================

  Paired t-test:
    t-statistic:                {t_stat_time:.4f}
    p-value:                    {t_pvalue_time:.6f}
    Significant (p<0.05):       {"YES" if t_pvalue_time < 0.05 else "NO"}

  Wilcoxon signed-rank test:
    W-statistic:                {wilcoxon_stat_time:.4f}
    p-value:                    {wilcoxon_pvalue_time:.6f}
    Significant (p<0.05):       {"YES" if wilcoxon_pvalue_time < 0.05 else "NO"}

================================================================================
CONCLUSION
================================================================================

"""
    # Add conclusion
    if wins > losses and t_pvalue < 0.05:
        conclusion = "Heuristic RRT* OUTPERFORMS baseline RRT* with statistical significance."
    elif losses > wins and t_pvalue < 0.05:
        conclusion = "Heuristic RRT* UNDERPERFORMS baseline RRT* with statistical significance."
    else:
        conclusion = "No statistically significant difference between the two planners."
    
    # Timing conclusion
    if time_wins > time_losses and t_pvalue_time < 0.05:
        time_conclusion = "Heuristic RRT* finds first solution FASTER with statistical significance."
    elif time_losses > time_wins and t_pvalue_time < 0.05:
        time_conclusion = "Heuristic RRT* finds first solution SLOWER with statistical significance."
    else:
        time_conclusion = "No statistically significant difference in time to first solution."

    report += f"""{conclusion}

Cost win rate: {100*wins/n_runs:.1f}%
Mean cost improvement: {np.mean(pct_improvements):+.2f}%

{time_conclusion}

Timing win rate: {100*time_wins/n_runs:.1f}%
Mean time improvement: {np.mean(pct_time_improvements):+.2f}%

================================================================================
RAW DATA
================================================================================

Run | Base Cost | Heur Cost | Cost Impr | Base Time | Heur Time | Time Impr | Cost Win
----|-----------|-----------|-----------|-----------|-----------|-----------|----------
"""
    
    for i in range(min(n_runs, 100)):  # Show first 100 runs
        winner = "Heuristic" if heuristic_arr[i] < baseline_arr[i] else ("Baseline" if heuristic_arr[i] > baseline_arr[i] else "Tie")
        report += f"{i+1:3d} | {baseline_arr[i]:9.4f} | {heuristic_arr[i]:9.4f} | {improvements[i]:+9.4f} | {baseline_first_arr[i]:9.4f} | {heuristic_first_arr[i]:9.4f} | {time_improvements[i]:+9.4f} | {winner}\n"
    
    report += "\n================================================================================"
    
    print(report)
    
    # Save to file
    if output_file is None:
        output_file = "results/heuristic_vs_baseline_comparison.txt"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_file}")
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 examples/analyze_comparison.py <experiment_folder> [output_file]")
        print("\nExample:")
        print("  python3 examples/analyze_comparison.py experiments/20260315_heuristic_comparison/")
        sys.exit(1)
    
    experiment_folder = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze(experiment_folder, output_file)
