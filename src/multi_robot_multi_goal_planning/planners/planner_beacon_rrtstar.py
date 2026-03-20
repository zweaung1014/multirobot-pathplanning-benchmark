"""Beacon-Guided RRT* Planner.

Combines RRT*-Smart's beacon-based sampling with corridor sampling for
improved convergence. Based on the RRT*-Smart algorithm by Nasir et al.
(2013): "Adaptive RRT*-Smart: Algorithm Characteristics and Behavior
Analysis in Complex Environments."

Key concepts:
- Path Optimization: After finding a solution, extract beacon waypoints
  by connecting directly visible nodes (greedy visibility optimization)
- Intelligent Sampling: Sample near beacon nodes at regular intervals
- Beacon Corridor: Sample along the piecewise-linear path connecting beacons
"""

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
class BeaconRRTConfig(BaseRRTConfig):
    """Configuration for Beacon-Guided RRT* planner.
    
    Extends BaseRRTConfig with parameters for beacon-based sampling strategy
    combining RRT*-Smart intelligent sampling with corridor sampling.
    """
    # Beacon sampling interval (sample at beacon every b iterations after first solution)
    beacon_interval: int = 5
    
    # Beacon radius as fraction of configuration space diagonal
    beacon_radius: float = 0.1
    
    # Corridor width as fraction of configuration space diagonal
    corridor_width: float = 0.1
    
    # Exploration phase (before first solution) probabilities
    p_uniform_explore: float = 0.7
    p_goal_explore: float = 0.3
    # No corridor sampling before first solution
    
    # Refinement phase (after first solution) probabilities
    # Note: Beacon point sampling is handled separately via beacon_interval
    p_uniform_refine: float = 0.3
    p_goal_refine: float = 0.1
    p_corridor_refine: float = 0.6  # Beacon corridor sampling
    
    # Maximum rejection sampling attempts
    max_beacon_attempts: int = 100
    max_corridor_attempts: int = 100


class BeaconRRTstar(RRTstar):
    """Beacon-Guided RRT* planner.
    
    Extends RRT* with RRT*-Smart's beacon concept and corridor sampling.
    
    Sampling strategy:
    - Before first solution (exploration):
        - Uniform sampling (default 70%)
        - Goal-biased sampling (default 30%)
    
    - After first solution (refinement):
        - Every b iterations: sample at a beacon point (deterministic)
        - Otherwise probabilistic:
            - Uniform sampling (default 30%)
            - Goal-biased sampling (default 10%)
            - Beacon corridor sampling (default 60%)
    
    The beacon corridor samples configurations along the piecewise-linear
    path connecting optimized beacon waypoints with Gaussian perpendicular noise.
    
    Beacons are re-extracted when a better path is found, via greedy
    visibility optimization (connecting directly visible nodes).
    """

    def __init__(self, env: BaseProblem, config: BeaconRRTConfig):
        super().__init__(env=env, config=config)
        self.config: BeaconRRTConfig = config
        
        # Beacon state
        self._beacons: List[Configuration] = []
        self._n_first_solution: int = 0  # Iteration when first solution found
        self._iteration_counter: int = 0  # Current iteration count
        self._last_beacon_cost: float = float('inf')  # Cost when beacons were last updated
        
        # Precomputed beacon corridor info (updated when beacons change)
        self._beacon_segments: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self._total_corridor_length: float = 0.0
        
        # Compute configuration space extent for radius/width scaling
        self._config_space_extent: float = self._compute_config_space_extent()

    def _compute_config_space_extent(self) -> float:
        """Compute the diagonal extent of the configuration space.
        
        Returns:
            float: Euclidean length of the configuration space diagonal.
        """
        limits = self.env.limits  # Shape: (2, dim) where [0,:] = low, [1,:] = high
        extent = np.linalg.norm(limits[1, :] - limits[0, :])
        return extent

    def _extract_beacons(self, path_nodes: List[Node]) -> List[Configuration]:
        """Extract beacon waypoints via greedy visibility optimization.
        
        Performs path optimization by connecting directly visible nodes,
        creating a simplified path with fewer waypoints (beacons).
        
        Args:
            path_nodes: List of nodes forming the current best path (in order).
            
        Returns:
            List[Configuration]: Beacon configurations (optimized waypoints).
        """
        if not path_nodes or len(path_nodes) < 2:
            return []
        
        beacons = []
        i = 0
        n = len(path_nodes)
        
        while i < n:
            # Add current node as beacon
            beacons.append(path_nodes[i].state.q)
            
            if i == n - 1:
                break
            
            # Find furthest directly visible node from current
            furthest_visible = i + 1
            current_mode = path_nodes[i].state.mode
            
            for j in range(n - 1, i + 1, -1):
                # Check if nodes are in same mode (or compatible modes)
                # and if there's a clear line of sight
                target_mode = path_nodes[j].state.mode
                
                # For cross-mode visibility, use the current node's mode
                if self.env.is_edge_collision_free(
                    path_nodes[i].state.q,
                    path_nodes[j].state.q,
                    current_mode
                ):
                    furthest_visible = j
                    break
            
            i = furthest_visible
        
        return beacons

    def _update_beacon_segments(self) -> None:
        """Precompute segment info for beacon corridor sampling.
        
        Caches segment endpoints, directions, and cumulative lengths
        for efficient corridor sampling.
        """
        self._beacon_segments = []
        self._total_corridor_length = 0.0
        
        if len(self._beacons) < 2:
            return
        
        for i in range(len(self._beacons) - 1):
            start = self._beacons[i].state()
            end = self._beacons[i + 1].state()
            segment_vec = end - start
            segment_length = np.linalg.norm(segment_vec)
            
            if segment_length > 1e-10:
                segment_dir = segment_vec / segment_length
            else:
                segment_dir = np.zeros_like(segment_vec)
            
            self._beacon_segments.append((start, segment_dir, segment_length))
            self._total_corridor_length += segment_length

    def _should_sample_beacon(self) -> bool:
        """Check if this iteration should sample at a beacon point.
        
        Returns:
            bool: True if beacon point sampling should occur.
        """
        if not self._beacons or self._n_first_solution == 0:
            return False
        
        # Sample at beacon every b iterations after first solution
        iterations_since_first = self._iteration_counter - self._n_first_solution
        if iterations_since_first <= 0:
            return False
        
        return (iterations_since_first % self.config.beacon_interval) == 0

    def _sample_at_beacon(self, mode: Mode) -> Configuration | None:
        """Sample a configuration near a randomly selected beacon point.
        
        Samples uniformly within a ball centered at a random beacon.
        
        Args:
            mode: Current operational mode.
            
        Returns:
            Configuration: Collision-free configuration near a beacon, or None.
        """
        if not self._beacons:
            return self._sample_uniform(mode)
        
        # Select random beacon
        beacon = self._beacons[np.random.randint(len(self._beacons))]
        beacon_state = beacon.state()
        
        # Radius in configuration space units
        radius = self.config.beacon_radius * self._config_space_extent
        
        limits_low = self.env.limits[0, :]
        limits_high = self.env.limits[1, :]
        
        for _ in range(self.config.max_beacon_attempts):
            # Sample uniformly in ball around beacon
            # Use Gaussian and normalize for uniform direction, then scale
            direction = np.random.normal(0, 1, size=beacon_state.shape)
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction = direction / norm
            
            # Uniform radius within ball (use cube root for uniform volume)
            r = radius * (np.random.uniform(0, 1) ** (1.0 / beacon_state.shape[0]))
            
            sampled_point = beacon_state + r * direction
            
            # Clip to configuration limits
            sampled_point = np.clip(sampled_point, limits_low, limits_high)
            
            # Create configuration from flat array
            q = self.env.start_pos.from_flat(sampled_point)
            
            # Check collision
            if self.env.is_collision_free(q, mode):
                return q
        
        # Exhausted attempts - fallback to uniform sampling
        return self._sample_uniform(mode)

    def _sample_beacon_corridor(self, mode: Mode) -> Configuration | None:
        """Sample a configuration along the beacon corridor.
        
        Samples uniformly along the piecewise-linear path connecting
        beacons, then adds Gaussian noise perpendicular to the local
        segment direction.
        
        Args:
            mode: Current operational mode.
            
        Returns:
            Configuration: Collision-free configuration near corridor, or None.
        """
        if not self._beacon_segments or self._total_corridor_length < 1e-10:
            return self._sample_uniform(mode)
        
        # Corridor width in configuration space units
        corridor_std = self.config.corridor_width * self._config_space_extent
        
        limits_low = self.env.limits[0, :]
        limits_high = self.env.limits[1, :]
        
        for _ in range(self.config.max_corridor_attempts):
            # Sample position along total corridor length
            t = np.random.uniform(0, self._total_corridor_length)
            
            # Find which segment this falls into
            cumulative = 0.0
            point_on_corridor = None
            segment_dir = None
            
            for start, seg_dir, seg_length in self._beacon_segments:
                if cumulative + seg_length >= t:
                    # This is the segment
                    local_t = t - cumulative
                    point_on_corridor = start + local_t * seg_dir
                    segment_dir = seg_dir
                    break
                cumulative += seg_length
            
            if point_on_corridor is None:
                # Fallback to last beacon
                point_on_corridor = self._beacons[-1].state()
                segment_dir = np.zeros_like(point_on_corridor)
            
            # Generate random noise
            noise = np.random.normal(0, corridor_std, size=point_on_corridor.shape)
            
            # Remove component along segment direction (keep perpendicular part)
            seg_norm = np.linalg.norm(segment_dir)
            if seg_norm > 1e-10:
                noise_along = np.dot(noise, segment_dir) * segment_dir
                perpendicular_noise = noise - noise_along
            else:
                perpendicular_noise = noise
            
            # Add perpendicular noise to point on corridor
            sampled_point = point_on_corridor + perpendicular_noise
            
            # Clip to configuration limits
            sampled_point = np.clip(sampled_point, limits_low, limits_high)
            
            # Create configuration from flat array
            q = self.env.start_pos.from_flat(sampled_point)
            
            # Check collision
            if self.env.is_collision_free(q, mode):
                return q
        
        # Exhausted attempts - fallback to uniform sampling
        return self._sample_uniform(mode)

    def sample_configuration(self, mode: Mode) -> Configuration | None:
        """Sample a configuration using hybrid beacon/corridor strategy.
        
        The sampling strategy uses deterministic beacon sampling at regular
        intervals, with probability-based selection for other iterations:
        
        - Before first solution: uniform + goal sampling
        - After first solution:
            - Every b iterations: sample at beacon point (deterministic)
            - Otherwise: uniform, goal, or corridor sampling (probabilistic)
        
        Args:
            mode: Current operational mode.
            
        Returns:
            Configuration: Sampled collision-free configuration, or None if sampling fails.
        """
        self._iteration_counter += 1
        
        # Check for deterministic beacon sampling
        if self._should_sample_beacon():
            return self._sample_at_beacon(mode)
        
        # Select probabilities based on solution status
        if self.operation.init_sol and self._beacons:
            # Refinement phase with beacons
            p_uniform = self.config.p_uniform_refine
            p_goal = self.config.p_goal_refine
            # p_corridor = self.config.p_corridor_refine (implied)
            
            r = np.random.uniform(0, 1)
            
            if r < p_uniform:
                return self._sample_uniform(mode)
            elif r < p_uniform + p_goal:
                return self._sample_goal(mode, self.transition_node_ids, self.trees[mode].order)
            else:
                return self._sample_beacon_corridor(mode)
        else:
            # Exploration phase (no solution yet or no beacons)
            p_uniform = self.config.p_uniform_explore
            p_goal = self.config.p_goal_explore
            
            r = np.random.uniform(0, 1)
            
            if r < p_uniform:
                return self._sample_uniform(mode)
            else:
                return self._sample_goal(mode, self.transition_node_ids, self.trees[mode].order)

    def manage_transition(self, mode: Mode, n_new: Node) -> None:
        """Handle mode transitions and update beacons on path improvement.
        
        Extends parent method to:
        1. Capture iteration number when first solution is found
        2. Extract beacons from first solution
        3. Re-extract beacons when a better path is found
        
        Args:
            mode: Current mode.
            n_new: Newly added node.
        """
        # Track if solution existed before this transition
        had_solution = self.operation.init_sol
        old_cost = self.operation.cost
        
        # Call parent implementation
        super().manage_transition(mode, n_new)
        
        # Check if first solution was just found
        if not had_solution and self.operation.init_sol:
            self._n_first_solution = self._iteration_counter
            self._extract_and_update_beacons()
            print(f"[BeaconRRT*] First solution found at iteration {self._iteration_counter}! "
                  f"Extracted {len(self._beacons)} beacons. Cost: {self.operation.cost:.4f}")
        
        # Check if path improved (new cost < last beacon cost)
        elif self.operation.init_sol and self.operation.cost < self._last_beacon_cost:
            self._extract_and_update_beacons()
            print(f"[BeaconRRT*] Path improved! New cost: {self.operation.cost:.4f}. "
                  f"Re-extracted {len(self._beacons)} beacons.")

    def _extract_and_update_beacons(self) -> None:
        """Extract beacons from current best path and update corridor info."""
        if self.operation.path_nodes is not None:
            self._beacons = self._extract_beacons(self.operation.path_nodes)
            self._update_beacon_segments()
            self._last_beacon_cost = self.operation.cost
