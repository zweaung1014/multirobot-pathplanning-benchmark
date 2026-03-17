# Changelog

All notable changes to the multirobot-pathplanning-benchmark project.

## [Unreleased] - 2026-03-15

### Added

#### New Planner: Heuristic-Guided RRT*
- **File**: `src/multi_robot_multi_goal_planning/planners/planner_heuristic_rrtstar.py`
- Implements adaptive three-way sampling strategy:
  - Uniform sampling
  - Goal-biased sampling  
  - Heuristic (line-to-goal band) sampling
- Probability blend shifts from exploration to refinement after finding first solution
- Configurable parameters via `HeuristicRRTConfig` dataclass

#### New Configuration Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `p_uniform_explore` | 0.7 | Uniform probability before solution |
| `p_goal_explore` | 0.2 | Goal-biased probability before solution |
| `p_heuristic_explore` | 0.1 | Heuristic probability before solution |
| `p_uniform_refine` | 0.3 | Uniform probability after solution |
| `p_goal_refine` | 0.1 | Goal-biased probability after solution |
| `p_heuristic_refine` | 0.6 | Heuristic probability after solution |
| `band_width` | 0.1 | Corridor width (fraction of config space) |
| `max_heuristic_attempts` | 100 | Max rejection sampling attempts |

#### Experiment Infrastructure
- **File**: `configs/experiments/heuristic_comparison.json` - 100-run comparison config
- **File**: `examples/analyze_comparison.py` - Statistical analysis script for comparing planners

#### Documentation
- **File**: `results/heuristic_rrtstar_implementation_report.txt` - Implementation report
- **File**: `results/CHANGELOG.md` - This changelog

### Modified

#### `src/multi_robot_multi_goal_planning/planners/__init__.py`
- Added exports: `HeuristicRRTstar`, `HeuristicRRTConfig`

#### `examples/run_planner.py`
- Added `heuristic_rrt_star` to planner choices
- Added `HeuristicRRTConfig` argument parsing with `--hrrt.` prefix
- Added planner instantiation block for heuristic RRT*

#### `examples/run_experiment.py`
- Added `HeuristicRRTstar`, `HeuristicRRTConfig` imports
- Added `heuristic_rrtstar` planner type in `setup_planner()` function

---

## Development Notes

### Branch
- `heuristic-guided`

### Testing
- Verified on `rai.simple` environment
- Single run produces comparable results to baseline RRT*
- Mode switching message confirmed: `[HeuristicRRT*] First solution found!`

### Usage Examples

```bash
# Single run
python3 examples/run_planner.py rai.simple --planner=heuristic_rrt_star --max_time=10

# 100-run comparison experiment
python3 examples/run_experiment.py configs/experiments/heuristic_comparison.json

# Analyze results (after experiment completes)
python3 examples/analyze_comparison.py experiments/<timestamp>_folder/ results/heuristic_vs_baseline_comparison.txt
```

---

## File Summary

### New Files Created
```
src/multi_robot_multi_goal_planning/planners/planner_heuristic_rrtstar.py
configs/experiments/heuristic_comparison.json
examples/analyze_comparison.py
results/heuristic_rrtstar_implementation_report.txt
results/CHANGELOG.md
```

### Modified Files
```
src/multi_robot_multi_goal_planning/planners/__init__.py
examples/run_planner.py
examples/run_experiment.py
```
