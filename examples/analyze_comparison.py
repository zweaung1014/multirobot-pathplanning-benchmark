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
    
    n_runs = min(len(baseline_costs), len(heuristic_costs))
    baseline_costs = baseline_costs[:n_runs]
    heuristic_costs = heuristic_costs[:n_runs]
    
    print(f"Loaded {n_runs} runs from each planner")
    
    # Calculate metrics
    baseline_arr = np.array(baseline_costs)
    heuristic_arr = np.array(heuristic_costs)
    
    # Win/Loss/Tie analysis
    wins = np.sum(heuristic_arr < baseline_arr)
    losses = np.sum(heuristic_arr > baseline_arr)
    ties = np.sum(np.isclose(heuristic_arr, baseline_arr, rtol=1e-6))
    
    # Cost improvements
    improvements = baseline_arr - heuristic_arr
    pct_improvements = (improvements / baseline_arr) * 100
    
    # Statistical tests
    t_stat, t_pvalue = stats.ttest_rel(baseline_arr, heuristic_arr)
    try:
        wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(baseline_arr, heuristic_arr)
    except ValueError:
        # Wilcoxon test fails if all differences are zero
        wilcoxon_stat, wilcoxon_pvalue = float('nan'), float('nan')
    
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
    
    report += f"""{conclusion}

Win rate: {100*wins/n_runs:.1f}%
Mean improvement: {np.mean(pct_improvements):+.2f}%

================================================================================
RAW DATA
================================================================================

Run | Baseline Cost | Heuristic Cost | Improvement | Winner
----|---------------|----------------|-------------|-------
"""
    
    for i in range(min(n_runs, 100)):  # Show first 100 runs
        winner = "Heuristic" if heuristic_arr[i] < baseline_arr[i] else ("Baseline" if heuristic_arr[i] > baseline_arr[i] else "Tie")
        report += f"{i+1:3d} | {baseline_arr[i]:13.4f} | {heuristic_arr[i]:14.4f} | {improvements[i]:+11.4f} | {winner}\n"
    
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
