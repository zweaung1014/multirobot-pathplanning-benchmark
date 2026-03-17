import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode,
)
from multi_robot_multi_goal_planning.problems.configuration import Configuration

from .rrtstar_base import BaseRRTConfig
from .planner_rrtstar import RRTstar, Node, SingleTree
from .termination_conditions import PlannerTerminationCondition


@dataclass
class HeuristicRRTConfig(BaseRRTConfig):
    """Configuration for Heuristic-Guided RRT* planner.
    
    Extends BaseRRTConfig with parameters for adaptive sampling strategy
    that shifts from exploration to refinement after finding a solution.
    """
    # Exploration phase (before first solution) probabilities
    p_uniform_explore: float = 0.7
    p_goal_explore: float = 0.2
    p_heuristic_explore: float = 0.1
    
    # Refinement phase (after first solution) probabilities
    p_uniform_refine: float = 0.3
    p_goal_refine: float = 0.1
    p_heuristic_refine: float = 0.6
    
    # Band width as fraction of configuration space diagonal
    band_width: float = 0.1
    
    # Maximum rejection sampling attempts for heuristic sampling
    max_heuristic_attempts: int = 100


class HeuristicRRTstar(RRTstar):
    """Heuristic-Guided RRT* planner.
    
    Extends RRT* with a third sampling strategy: line-to-goal band sampling.
    The sampling probabilities adapt based on whether a feasible solution has been found:
    
    - Before solution (exploration): more uniform sampling (default 70%)
    - After solution (refinement): more heuristic sampling near start-goal line (default 60%)
    
    The heuristic samples configurations near the straight-line corridor from
    the start configuration to the first solution's goal configuration.
    """

    def __init__(self, env: BaseProblem, config: HeuristicRRTConfig):
        super().__init__(env=env, config=config)
        self.config: HeuristicRRTConfig = config
        
        # Reference goal configuration (set from first solution)
        self._reference_goal: Configuration | None = None
        
        # Compute configuration space extent for band width scaling
        self._config_space_extent: float = self._compute_config_space_extent()

    def _compute_config_space_extent(self) -> float:
        """Compute the diagonal extent of the configuration space.
        
        Returns:
            float: Euclidean length of the configuration space diagonal.
        """
        limits = self.env.limits  # Shape: (2, dim) where [0,:] = low, [1,:] = high
        extent = np.linalg.norm(limits[1, :] - limits[0, :])
        return extent

    def sample_configuration(self, mode: Mode) -> Configuration | None:
        """Sample a configuration using adaptive three-way probability blend.
        
        The sampling strategy adapts based on whether a solution has been found:
        - Before solution: exploration-focused (more uniform sampling)
        - After solution: refinement-focused (more heuristic sampling)
        
        Args:
            mode: Current operational mode.
            
        Returns:
            Configuration: Sampled collision-free configuration, or None if sampling fails.
        """
        # Select probabilities based on solution status
        if self.operation.init_sol:
            p_uniform = self.config.p_uniform_refine
            p_goal = self.config.p_goal_refine
            # p_heuristic = self.config.p_heuristic_refine (implied)
        else:
            p_uniform = self.config.p_uniform_explore
            p_goal = self.config.p_goal_explore
            # p_heuristic = self.config.p_heuristic_explore (implied)
        
        # Roll random value and select sampling strategy
        r = np.random.uniform(0, 1)
        
        if r < p_uniform:
            # Uniform sampling
            return self._sample_uniform(mode)
        elif r < p_uniform + p_goal:
            # Goal-biased sampling
            return self._sample_goal(mode, self.transition_node_ids, self.trees[mode].order)
        else:
            # Heuristic (line-to-goal band) sampling
            return self._sample_heuristic(mode)

    def _sample_heuristic(self, mode: Mode) -> Configuration | None:
        """Sample a configuration near the start-to-goal line corridor.
        
        Samples points in a band/corridor around the straight line from
        start to goal. Before a solution is found, falls back to uniform sampling.
        
        Args:
            mode: Current operational mode.
            
        Returns:
            Configuration: Collision-free configuration near the line, or None.
        """
        # Before first solution, no reference goal available - fallback to uniform
        if self._reference_goal is None:
            return self._sample_uniform(mode)
        
        start = self.env.start_pos.state()  # numpy array
        goal = self._reference_goal.state()  # numpy array
        
        # Compute line direction
        line_vec = goal - start
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 1e-10:
            # Start and goal are essentially the same - fallback to uniform
            return self._sample_uniform(mode)
        
        line_dir = line_vec / line_length
        
        # Band width in configuration space units
        band_std = self.config.band_width * self._config_space_extent
        
        limits_low = self.env.limits[0, :]
        limits_high = self.env.limits[1, :]
        
        for _ in range(self.config.max_heuristic_attempts):
            # Sample parameter t along the line [0, 1]
            t = np.random.uniform(0, 1)
            point_on_line = start + t * line_vec
            
            # Generate random noise
            noise = np.random.normal(0, band_std, size=start.shape)
            
            # Remove component along line direction (keep perpendicular part)
            noise_along_line = np.dot(noise, line_dir) * line_dir
            perpendicular_noise = noise - noise_along_line
            
            # Add perpendicular noise to point on line
            sampled_point = point_on_line + perpendicular_noise
            
            # Clip to configuration limits
            sampled_point = np.clip(sampled_point, limits_low, limits_high)
            
            # Create configuration from flat array
            q = self.env.start_pos.from_flat(sampled_point)
            
            # Check collision
            if self.env.is_collision_free(q, mode):
                return q
        
        # Exhausted attempts - fallback to uniform sampling
        return self._sample_uniform(mode)

    def manage_transition(self, mode: Mode, n_new: Node) -> None:
        """Handle mode transitions and capture reference goal from first solution.
        
        Extends parent method to capture the goal configuration from the first
        feasible solution, which is used for heuristic sampling.
        
        Args:
            mode: Current mode.
            n_new: Newly added node.
        """
        # Track if solution was found before this transition
        had_solution = self.operation.init_sol
        
        # Call parent implementation
        super().manage_transition(mode, n_new)
        
        # If solution was just found, capture reference goal
        if not had_solution and self.operation.init_sol:
            if self._reference_goal is None and self.operation.path is not None:
                self._reference_goal = self.operation.path[-1].q
                print(f"[HeuristicRRT*] First solution found! "
                      f"Reference goal captured. Switching to refinement mode.")
