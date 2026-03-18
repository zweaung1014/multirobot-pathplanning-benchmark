"""
MAMP-RRT*: Mode-Adaptive Multi-Phase RRT*
========================================

Drop this file into:
    src/multi_robot_multi_goal_planning/planners/planner_mamp_rrtstar.py

Register in __init__.py of the planners folder:
    from .planner_mamp_rrtstar import MAMPRRTStar, MAMPRRTStarConfig

Run with:
    python3 examples/run_planner.py rai.2d_handover \
        --planner=mamp_rrtstar --max_time=60 --optimize \
        --distance_metric=euclidean \
        --per_agent_cost_function=euclidean \
        --cost_reduction=max

Design change
-------------
The original "Phase 1 = 100% uniform" idea is too weak for handover tasks.
This version changes Phase 1 into a transition-centric bootstrap phase:

    Phase 1  -> cached goal-seed bias + transition-node bias + sparse bootstrap + uniform
    Phase 2  -> PHS ellipsoid + local goal-seed bias + uniform
    Phase 3  -> Gaussian path tube + local goal-seed bias + uniform

Why this is needed
------------------
rai.2d_handover needs the planner to hit a very specific transition manifold.
Pure uniform exploration wastes too many samples before the first transition.

Important implementation notes
------------------------------
1. p_goal is forced to 0.0 so we do not inherit the base 0.4 goal bias.
2. We still keep a sparse bootstrap call to the repo's _sample_goal() before
   the first solution so the planner can discover the first transition region.
3. Every successful bootstrap target is cached as a cheap local "goal seed".
   Then many later samples are drawn near those seeds without re-running the NLP.
4. Once real transition nodes are discovered, the planner aggressively samples
   near them because those are the best cheap hints of where mode changes happen.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from multi_robot_multi_goal_planning.problems.configuration import Configuration
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    Mode,
    State,
)

from .planner_rrtstar import RRTstar
from .rrtstar_base import BaseRRTConfig, Node
from .sampling_informed import compute_PHS_matrices, sample_phs_with_given_matrices
from .termination_conditions import PlannerTerminationCondition


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class MAMPRRTStarConfig(BaseRRTConfig):
    """
    Extends BaseRRTConfig with MAMP-RRT* parameters.

    Important defaults for this planner:
    - p_goal = 0.0 so we do not inherit the base planner's 0.4 goal bias
    - informed_sampling = False because MAMP uses its own phase samplers
    - with_mode_validation = False because the repo's mode validation can
      prune successor modes after real transition hits, and some abstract
      environments do not implement is_collision_free_for_robot()
    """

    informed_sampling: bool = False
    p_goal: float = 0.0
    with_mode_validation: bool = False

    # Sparse expensive bootstrap before first solution.
    bootstrap_goal_frac: float = 0.06

    # Cheap local reuse of successful bootstrap targets.
    goal_seed_bias: float = 0.15
    goal_seed_sigma: float = 0.20
    max_goal_seeds_per_mode: int = 64

    # Cheap local reuse of discovered transition nodes.
    transition_node_bias: float = 0.50
    transition_node_sigma: float = 0.03

    # Additional Phase 1 local focus once seeds exist.
    explore_local_frac: float = 0.55

    # Phase 2 -> 3 switch rule.
    near_optimal_ratio: float = 0.20

    # Path tube parameters.
    tube_sigma_init: float = 0.15
    tube_decay: float = 0.995

    # Mixture fractions.
    p2_ellipsoid_frac: float = 0.85
    p3_tube_frac: float = 0.95

    # Stagnation fallback.
    stagnation_window: int = 200
    stagnation_delta: float = 1e-4

# -----------------------------------------------------------------------------
# Planner
# -----------------------------------------------------------------------------

class MAMPRRTStar(RRTstar):
    """
    Mode-Adaptive Multi-Phase RRT*.

    Main differences from the previous version:
    - no inherited p_goal=0.4 surprise
    - Phase 1 is transition-centric, not 100% uniform
    - successful expensive bootstrap targets are cached and reused cheaply
    - discovered transition nodes are exploited aggressively
    - per-mode phase logic still stays local to the best contiguous segment
    """

    _EXPLORE = 1
    _INFORMED = 2
    _EXPLOIT = 3

    def __init__(self, env: BaseProblem, config: MAMPRRTStarConfig):
        super().__init__(env=env, config=config)

        self._phase: Dict[Mode, int] = {}
        self._tube_r: Dict[Mode, float] = {}
        self._mode_c_best: Dict[Mode, float] = {}
        self._mode_c_lo: Dict[Mode, float] = {}
        self._stag_buf: Dict[Mode, List[float]] = {}

        # Cached best contiguous segment per mode from current best path.
        self._mode_segment_states: Dict[Mode, List[State]] = {}
        self._mode_segment_cost: Dict[Mode, float] = {}

        # Cached successful bootstrap targets per mode.
        self._goal_seed_cache: Dict[Mode, List[np.ndarray]] = {}

        cfg = config
        self._near_opt = cfg.near_optimal_ratio
        self._tube_init = cfg.tube_sigma_init
        self._tube_dec = cfg.tube_decay
        self._p2_ell = cfg.p2_ellipsoid_frac
        self._p3_tube = cfg.p3_tube_frac
        self._stag_win = cfg.stagnation_window
        self._stag_dlt = cfg.stagnation_delta

        self._bootstrap_goal_frac = cfg.bootstrap_goal_frac
        self._goal_seed_bias = cfg.goal_seed_bias
        self._goal_seed_sigma = cfg.goal_seed_sigma
        self._max_goal_seeds = cfg.max_goal_seeds_per_mode

        self._transition_node_bias = cfg.transition_node_bias
        self._transition_sigma = cfg.transition_node_sigma
        self._explore_local_frac = cfg.explore_local_frac

        self._stats = {
            "uniform": 0,
            "bootstrap_goal": 0,
            "goal_seed": 0,
            "transition_bias": 0,
            "ellipsoid": 0,
            "path_tube": 0,
            "transition_hits": 0,
            "new_modes": 0,
        }

    # -------------------------------------------------------------------------
    # Phase helpers
    # -------------------------------------------------------------------------

    def _init_mode_phase(self, mode: Mode) -> None:
        self._phase[mode] = self._EXPLORE
        self._tube_r[mode] = self._tube_init
        self._mode_c_best[mode] = math.inf
        self._mode_c_lo[mode] = math.inf
        self._stag_buf[mode] = []
        self._goal_seed_cache.setdefault(mode, [])

    def _advance_phase(self, mode: Mode, mode_segment_cost: float) -> None:
        if mode_segment_cost >= self._mode_c_best.get(mode, math.inf):
            return

        self._mode_c_best[mode] = mode_segment_cost
        self._tube_r[mode] = max(
            self._tube_r.get(mode, self._tube_init) * self._tube_dec,
            1e-4,
        )

        cur = self._phase.get(mode, self._EXPLORE)

        if cur == self._EXPLORE:
            self._phase[mode] = self._INFORMED
            return

        if cur == self._INFORMED:
            c_lo = self._mode_c_lo.get(mode, math.inf)
            if c_lo > 1e-9:
                ratio = (mode_segment_cost - c_lo) / (c_lo + 1e-9)
                if ratio < self._near_opt:
                    self._phase[mode] = self._EXPLOIT

    def _phase_name(self, phase: int) -> str:
        if phase == self._EXPLORE:
            return "EXPLORE"
        if phase == self._INFORMED:
            return "INFORMED"
        if phase == self._EXPLOIT:
            return "EXPLOIT"
        return "UNKNOWN"

    # -------------------------------------------------------------------------
    # Path bookkeeping
    # -------------------------------------------------------------------------

    def _edge_cost(self, q_from: Configuration, q_to: Configuration) -> float:
        batch = np.asarray([q_to.state()], dtype=np.float64)
        val = self.env.batch_config_cost(q_from, batch)
        return float(np.asarray(val).reshape(-1)[0])

    def _segment_cost(self, states: List[State]) -> float:
        if len(states) < 2:
            return 0.0

        total = 0.0
        for i in range(len(states) - 1):
            total += self._edge_cost(states[i].q, states[i + 1].q)
        return total

    def _refresh_mode_path_cache(self) -> None:
        if not self.operation.path:
            return

        current_segments: Dict[Mode, Dict[str, Any]] = {}
        path = self.operation.path
        n = len(path)
        start = 0

        while start < n:
            mode = path[start].mode
            end = start
            while end + 1 < n and path[end + 1].mode == mode:
                end += 1

            seg_states = path[start : end + 1]
            seg_cost = self._segment_cost(seg_states)

            q_start = seg_states[0].q.state()
            q_end = seg_states[-1].q.state()
            c_lo = float(np.linalg.norm(q_end - q_start))

            existing = current_segments.get(mode)
            if existing is None or seg_cost < existing["cost"]:
                current_segments[mode] = {
                    "states": seg_states,
                    "cost": seg_cost,
                    "c_lo": c_lo,
                }

            start = end + 1

        for mode, data in current_segments.items():
            if mode not in self._phase:
                self._init_mode_phase(mode)
            self._mode_segment_states[mode] = data["states"]
            self._mode_segment_cost[mode] = data["cost"]
            self._mode_c_lo[mode] = data["c_lo"]

    # -------------------------------------------------------------------------
    # Goal-seed helpers
    # -------------------------------------------------------------------------

    def _record_goal_seed(self, mode: Mode, q: Configuration) -> None:
        seed = np.asarray(q.state(), dtype=np.float64).copy()
        cache = self._goal_seed_cache.setdefault(mode, [])

        # Keep simple deduplication so the cache does not fill with near-identical points.
        for old in cache:
            if np.linalg.norm(old - seed) < 1e-3:
                return

        cache.append(seed)
        if len(cache) > self._max_goal_seeds:
            cache.pop(0)

    def _has_goal_seeds(self, mode: Mode) -> bool:
        return len(self._goal_seed_cache.get(mode, [])) > 0

    # -------------------------------------------------------------------------
    # Sampling helpers
    # -------------------------------------------------------------------------

    def _sample_near_vector(
        self,
        center: np.ndarray,
        mode: Mode,
        sigma: float,
        max_tries: int = 25,
    ) -> Optional[Configuration]:
        try:
            lims = self.env.limits
            template = self.env.get_start_pos()
        except Exception:
            return None

        n_dim = len(center)
        if lims.shape[1] != n_dim:
            return None

        for _ in range(max_tries):
            q_flat = np.clip(
                center + np.random.normal(0.0, sigma, size=n_dim),
                lims[0],
                lims[1],
            )
            try:
                q = template.from_flat(q_flat)
            except Exception:
                continue
            if self.env.is_collision_free(q, mode):
                return q

        return None

    def _sample_goal_seed_bias(self, mode: Mode) -> Optional[Configuration]:
        seeds = self._goal_seed_cache.get(mode, [])
        if not seeds:
            return None

        center = random.choice(seeds)
        return self._sample_near_vector(
            center=center,
            mode=mode,
            sigma=self._goal_seed_sigma,
            max_tries=25,
        )

    def _sample_transition_node_bias(self, mode: Mode) -> Optional[Configuration]:
        ids = self.transition_node_ids.get(mode, [])
        if not ids:
            return None

        node_id = random.choice(ids)
        node = self.trees[mode].subtree.get(node_id)
        if node is None:
            return None

        return self._sample_near_vector(
            center=node.state.q.state(),
            mode=mode,
            sigma=self._transition_sigma,
            max_tries=25,
        )

    def _current_bootstrap_frac(self, mode: Mode) -> float:
        """
        Use more bootstrap only when we still do not have cheap local guidance.
        """
        if self.operation.init_sol:
            return 0.0

        has_transition_nodes = len(self.transition_node_ids.get(mode, [])) > 0
        has_goal_seeds = self._has_goal_seeds(mode)

        if has_transition_nodes:
            return 0.0
        if has_goal_seeds:
            return self._bootstrap_goal_frac * 0.25
        return self._bootstrap_goal_frac

    def _sample_bootstrap_goal(self, mode: Mode) -> Optional[Configuration]:
        if self.operation.init_sol:
            return None
        if mode not in self.trees:
            return None

        try:
            q = self._sample_goal(
                mode,
                self.transition_node_ids,
                self.trees[mode].order,
            )
        except Exception:
            return None

        if q is not None:
            self._record_goal_seed(mode, q)

        return q

    # -------------------------------------------------------------------------
    # Override: initialize_planner
    # -------------------------------------------------------------------------

    def initialize_planner(self) -> None:
        super().initialize_planner()
        if self.modes:
            self._init_mode_phase(self.modes[0])

    # -------------------------------------------------------------------------
    # Override: sample_configuration
    # -------------------------------------------------------------------------
def sample_configuration(self, mode: Mode) -> Optional[Configuration]:
    if mode not in self.modes:
        return None

    phase = self._phase.get(mode, self._EXPLORE)

    # ----------------------------
    # BEFORE FIRST SOLUTION
    # ----------------------------
    if not self.operation.init_sol:
        if self.transition_node_ids.get(mode) and np.random.random() < self._transition_node_bias:
            q = self._sample_transition_node_bias(mode)
            if q is not None:
                self._stats["transition_bias"] += 1
                return q

        if self._has_goal_seeds(mode) and np.random.random() < self._goal_seed_bias:
            q = self._sample_goal_seed_bias(mode)
            if q is not None:
                self._stats["goal_seed"] += 1
                return q

        bootstrap_frac = self._current_bootstrap_frac(mode)
        if bootstrap_frac > 0.0 and np.random.random() < bootstrap_frac:
            q = self._sample_bootstrap_goal(mode)
            if q is not None:
                self._stats["bootstrap_goal"] += 1
                return q

        self._stats["uniform"] += 1
        return self._sample_uniform(mode)

    # ----------------------------
    # AFTER FIRST SOLUTION
    # Use actual MAMP phase sampler first
    # ----------------------------
    if phase == self._EXPLORE:
        if self._has_goal_seeds(mode) and np.random.random() < 0.20:
            q = self._sample_goal_seed_bias(mode)
            if q is not None:
                self._stats["goal_seed"] += 1
                return q

        if self.transition_node_ids.get(mode) and np.random.random() < 0.20:
            q = self._sample_transition_node_bias(mode)
            if q is not None:
                self._stats["transition_bias"] += 1
                return q

        self._stats["uniform"] += 1
        return self._sample_uniform(mode)

    if phase == self._INFORMED:
        if np.random.random() < self._p2_ell:
            q = self._sample_ellipsoid(mode)
            if q is not None:
                self._stats["ellipsoid"] += 1
                return q

        if self._has_goal_seeds(mode) and np.random.random() < 0.15:
            q = self._sample_goal_seed_bias(mode)
            if q is not None:
                self._stats["goal_seed"] += 1
                return q

        if self.transition_node_ids.get(mode) and np.random.random() < 0.15:
            q = self._sample_transition_node_bias(mode)
            if q is not None:
                self._stats["transition_bias"] += 1
                return q

        self._stats["uniform"] += 1
        return self._sample_uniform(mode)

    if phase == self._EXPLOIT:
        if np.random.random() < self._p3_tube:
            q = self._sample_path_tube(mode)
            if q is not None:
                self._stats["path_tube"] += 1
                return q

        if self._has_goal_seeds(mode) and np.random.random() < 0.10:
            q = self._sample_goal_seed_bias(mode)
            if q is not None:
                self._stats["goal_seed"] += 1
                return q

        if self.transition_node_ids.get(mode) and np.random.random() < 0.10:
            q = self._sample_transition_node_bias(mode)
            if q is not None:
                self._stats["transition_bias"] += 1
                return q

        self._stats["uniform"] += 1
        return self._sample_uniform(mode)

    self._stats["uniform"] += 1
    return self._sample_uniform(mode)

    # -------------------------------------------------------------------------
    # Override: manage_transition
    # -------------------------------------------------------------------------

    def manage_transition(self, mode: Mode, n_new: Node) -> None:
        modes_before = set(self.modes)
        was_transition = n_new.transition

        super().manage_transition(mode, n_new)

        if (not was_transition) and n_new.transition:
            self._stats["transition_hits"] += 1

        new_modes = set(self.modes) - modes_before
        self._stats["new_modes"] += len(new_modes)

        for new_mode in new_modes:
            self._init_mode_phase(new_mode)

        if self.operation.init_sol and self.operation.path:
            self._refresh_mode_path_cache()
    
    # -------------------------------------------------------------------------
    # Override: plan
    # -------------------------------------------------------------------------

    def plan(
        self,
        ptc: PlannerTerminationCondition,
        optimize: bool = True,
    ) -> Tuple[Optional[List[State]], Dict[str, Any]]:
        i = 0
        prev_cost = math.inf
        self.initialize_planner()

        while True:
            i += 1

            if not self.modes:
                print("[MAMP-RRT*] No active modes remain.")
                break

            active_mode = self.random_mode()

            q_rand = self.sample_configuration(active_mode)
            if q_rand is None:
                continue

            n_nearest, dist, set_dists, n_nearest_idx = self.nearest(active_mode, q_rand)
            state_new = self.steer(active_mode, n_nearest, q_rand, dist)
            if not state_new:
                continue

            if self.env.is_collision_free(
                state_new.q, active_mode
            ) and self.env.is_edge_collision_free(
                n_nearest.state.q, state_new.q, active_mode
            ):
                n_new = Node(state_new, self.operation)

                if np.equal(n_new.state.q.state(), q_rand.state()).all():
                    N_near_batch, n_near_costs, node_indices = self.near(
                        active_mode, n_new, n_nearest_idx, set_dists
                    )
                else:
                    N_near_batch, n_near_costs, node_indices = self.near(
                        active_mode, n_new, n_nearest_idx
                    )

                batch_cost = self.env.batch_config_cost(n_new.state.q, N_near_batch)

                self.find_parent(
                    active_mode,
                    node_indices,
                    n_new,
                    n_nearest,
                    batch_cost,
                    n_near_costs,
                )

                if self.rewire(
                    active_mode,
                    node_indices,
                    n_new,
                    batch_cost,
                    n_near_costs,
                ):
                    self.update_cost(active_mode, n_new)

                self.manage_transition(active_mode, n_new)

            if self.operation.init_sol and self.operation.cost < prev_cost:
                prev_cost = self.operation.cost
                self._refresh_mode_path_cache()

                for mode, seg_cost in self._mode_segment_cost.items():
                    self._advance_phase(mode, seg_cost)

            if self.operation.init_sol:
                seg_cost = self._mode_segment_cost.get(active_mode)
                if seg_cost is not None:
                    buf = self._stag_buf.setdefault(active_mode, [])
                    buf.append(seg_cost)
                    if len(buf) > self._stag_win:
                        buf.pop(0)

                    if (
                        self._phase.get(active_mode) == self._EXPLOIT
                        and len(buf) >= self._stag_win
                        and (max(buf) - min(buf)) < self._stag_dlt
                    ):
                        self._phase[active_mode] = self._INFORMED
                        self._stag_buf[active_mode] = []

            if not optimize and self.operation.init_sol:
                self.save_tree_data()
                break

            if ptc.should_terminate(i, time.time() - self.start_time):
                phase_report = {
                    str(m.task_ids): self._phase_name(self._phase.get(m, self._EXPLORE))
                    for m in self.modes
                }
                print(f"[MAMP-RRT*] iterations: {i}")
                print(f"[MAMP-RRT*] phases: {phase_report}")
                print(f"[MAMP-RRT*] sampler stats: {self._stats}")
                if not self.operation.init_sol:
                    print("[MAMP-RRT*] No solution found within time limit.")
                break

        self.update_results_tracking(self.operation.cost, self.operation.path)
        info = {
            "costs": self.costs,
            "times": self.times,
            "paths": self.all_paths,
            "sampler_stats": dict(self._stats),
        }

        # Same post-processing as repo RRTstar.
        path_w_doubled_modes: List[State] = []
        for idx in range(len(self.operation.path)):
            path_w_doubled_modes.append(self.operation.path[idx])
            if (
                idx + 1 < len(self.operation.path)
                and self.operation.path[idx].mode != self.operation.path[idx + 1].mode
            ):
                path_w_doubled_modes.append(
                    State(
                        self.operation.path[idx].q,
                        self.operation.path[idx + 1].mode,
                    )
                )
        self.operation.path = path_w_doubled_modes

        return self.operation.path, info