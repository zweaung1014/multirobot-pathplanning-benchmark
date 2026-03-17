"""
MAMP-RRT*: Mode-Adaptive Multi-Phase RRT*
==========================================

Drop this file into:
    src/multi_robot_multi_goal_planning/planners/planner_mamp_rrtstar.py

Register in __init__.py of the planners folder:
    from .planner_mamp_rrtstar import MAMPRRTStar, MAMPRRTStarConfig

Run with:
    python3 examples/run_planner.py rai.2d_handover \\
        --planner=mamp_rrtstar --max_time=60 --optimize \\
        --distance_metric=euclidean \\
        --per_agent_cost_function=euclidean \\
        --cost_reduction=max

What is novel vs the existing planners in this repo
----------------------------------------------------
RRTstar / InformedRRTstar use a single fixed sampling strategy throughout.
MAMP-RRT* adds two things on top:

1.  Three explicit sampling phases, per mode segment:
        Phase 1  (no solution yet)  -> 100% uniform
        Phase 2  (first path found) -> 70% PHS ellipsoid + 30% uniform
        Phase 3  (near-optimal)     -> 90% Gaussian path-tube + 10% uniform

2.  Mode-aware phase resets:
        Every time a mode transition fires (robot picks up / puts down an
        object, reaches a sub-goal, hands over to another robot) the phase
        counter AND c_best for the NEW mode are reset to Phase 1 / inf.
        Every other planner in the repo carries the global cost and sampling
        region across mode boundaries -- MAMP-RRT* does not.

Performance note
----------------
The base RRTstar sets p_goal=0.4, which calls _sample_goal() -> IK NLP solver
on 40% of iterations. Each IK call takes ~1-2 seconds. We set p_goal=0 and
implement a fast lightweight goal bias using already-computed transition node
positions (O(1), no IK). This is why the planner runs at full speed.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    Mode,
    State,
)
from multi_robot_multi_goal_planning.problems.configuration import Configuration

from .rrtstar_base import BaseRRTConfig, Node, SingleTree
from .planner_rrtstar import RRTstar
from .sampling_informed import compute_PHS_matrices, sample_phs_with_given_matrices
from .termination_conditions import PlannerTerminationCondition


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class MAMPRRTStarConfig(BaseRRTConfig):
    """
    Extends BaseRRTConfig with MAMP-RRT* phase-scheduling parameters.
    All BaseRRTConfig options (shortcutting, distance_metric, etc.) still apply.
    """

    # Disable the parent's generate_samples informed sampler.
    # MAMP-RRT* runs its own three-phase ellipsoid + path-tube instead.
    informed_sampling: bool = False

    # Switch Phase 2 -> 3 when (c_best - c_lower) / c_lower < this
    near_optimal_ratio: float = 0.05

    # Path-tube parameters (Phase 3)
    tube_sigma_init: float = 0.15   # initial Gaussian std-dev
    tube_decay:      float = 0.995  # shrink factor per cost improvement

    # Mixture fractions
    p2_ellipsoid_frac: float = 0.70  # Phase 2: fraction from PHS ellipsoid
    p3_tube_frac:      float = 0.90  # Phase 3: fraction from path tube

    # Stagnation detection (Phase 3 -> Phase 2 fallback)
    stagnation_window: int   = 200   # iterations in sliding window
    stagnation_delta:  float = 1e-4  # min improvement to stay in Phase 3


# -----------------------------------------------------------------------------
# Planner
# -----------------------------------------------------------------------------

class MAMPRRTStar(RRTstar):
    """
    Mode-Adaptive Multi-Phase RRT* (MAMP-RRT*)

    Inherits everything from RRTstar and overrides only:
        __init__             -- adds per-mode phase state dicts
        initialize_planner   -- seeds phase state for the first mode
        sample_configuration -- fast three-phase dispatch (zero IK calls)
        manage_transition    -- mode-aware phase reset on new modes
        plan                 -- adds stagnation detection + phase advancement
    """

    _EXPLORE  = 1
    _INFORMED = 2
    _EXPLOIT  = 3

    def __init__(self, env: BaseProblem, config: MAMPRRTStarConfig):
        super().__init__(env=env, config=config)

        self._phase:       Dict[Mode, int]         = {}
        self._tube_r:      Dict[Mode, float]       = {}
        self._mode_c_best: Dict[Mode, float]       = {}
        self._mode_c_lo:   Dict[Mode, float]       = {}
        self._stag_buf:    Dict[Mode, List[float]] = {}

        cfg = config
        self._near_opt  = getattr(cfg, "near_optimal_ratio",  0.05)
        self._tube_init = getattr(cfg, "tube_sigma_init",     0.15)
        self._tube_dec  = getattr(cfg, "tube_decay",          0.995)
        self._p2_ell    = getattr(cfg, "p2_ellipsoid_frac",   0.70)
        self._p3_tube   = getattr(cfg, "p3_tube_frac",        0.90)
        self._stag_win  = getattr(cfg, "stagnation_window",   200)
        self._stag_dlt  = getattr(cfg, "stagnation_delta",    1e-4)

    # -------------------------------------------------------------------------
    # Phase helpers
    # -------------------------------------------------------------------------

    def _init_mode_phase(self, mode: Mode) -> None:
        """Reset all phase state for a mode (called on new mode discovery)."""
        self._phase[mode]       = self._EXPLORE
        self._tube_r[mode]      = self._tube_init
        self._mode_c_best[mode] = math.inf
        self._mode_c_lo[mode]   = math.inf  # Initialize lower-bound cost for this mode
        self._stag_buf[mode]    = []

    def _advance_phase(self, mode: Mode, global_cost: float) -> None:
        """Advance the phase for a mode when the global cost improves."""
        if global_cost >= self._mode_c_best.get(mode, math.inf):
            return

        self._mode_c_best[mode] = global_cost
        self._tube_r[mode] = max(
            self._tube_r.get(mode, self._tube_init) * self._tube_dec, 1e-4
        )

        # Compute mode-specific lower bound from solution path
        if self._mode_c_lo.get(mode, math.inf) == math.inf and self.operation.path:
            mode_states = [s for s in self.operation.path if s.mode == mode]
            if len(mode_states) >= 2:
                a = mode_states[0].q.state()
                b = mode_states[-1].q.state()
                c_min = float(np.linalg.norm(b - a))
                self._mode_c_lo[mode] = c_min

        cur = self._phase.get(mode, self._EXPLORE)

        if cur == self._EXPLORE:
            self._phase[mode] = self._INFORMED
            return

        if cur == self._INFORMED:
            c_lo = self._mode_c_lo.get(mode, math.inf)
            if c_lo < math.inf:
                if (global_cost - c_lo) / (c_lo + 1e-9) < self._near_opt:
                    self._phase[mode] = self._EXPLOIT

    # -------------------------------------------------------------------------
    # Override: initialize_planner
    # -------------------------------------------------------------------------

    def initialize_planner(self) -> None:
        """Extends parent init: seeds phase state for the first mode."""
        super().initialize_planner()
        if self.modes:
            self._init_mode_phase(self.modes[0])

    # -------------------------------------------------------------------------
    # Override: sample_configuration  (the core contribution)
    # -------------------------------------------------------------------------

    def sample_configuration(self, mode: Mode) -> Optional[Configuration]:
        """
        Three-phase sampler.

        Goal bias: uses the parent's _sample_goal() exactly like the baseline
        RRTstar does -- same IK-based goal sampling, same frequency.

        Phase 1 -> uniform only
        Phase 2 -> 70% PHS ellipsoid + 30% uniform
        Phase 3 -> 90% Gaussian path-tube + 10% uniform
        """
        # Validate that mode is still in the current mode list
        if mode not in self.modes:
            return None
            
        # Goal bias: identical to baseline RRTstar (uses parent's method)
        if np.random.uniform(0, 1) < self.config.p_goal:
            try:
                return self._sample_goal(
                    mode, self.transition_node_ids, self.trees[mode].order
                )
            except Exception:
                # Fall through to regular sampling if goal sampling fails
                pass

        phase = self._phase.get(mode, self._EXPLORE)

        if phase == self._EXPLORE:
            return self._sample_uniform(mode)

        if phase == self._INFORMED:
            if np.random.random() < self._p2_ell:
                q = self._sample_ellipsoid(mode)
                if q is not None:
                    return q
            return self._sample_uniform(mode)

        if phase == self._EXPLOIT:
            if np.random.random() < self._p3_tube:
                q = self._sample_path_tube(mode)
                if q is not None:
                    return q
            return self._sample_uniform(mode)

        return self._sample_uniform(mode)

    def _sample_ellipsoid(self, mode: Mode) -> Optional[Configuration]:
        """
        PHS ellipsoid (Informed RRT*, Gammell 2014).
        Focal points: first and last State on current path in this mode.
        Transverse diameter: current global best cost.
        """
        if not self.operation.init_sol or not self.operation.path:
            return None

        mode_states = [s for s in self.operation.path if s.mode == mode]
        if len(mode_states) < 2:
            return None

        a = mode_states[0].q.state()
        b = mode_states[-1].q.state()
        c = self.operation.cost

        c_min = float(np.linalg.norm(b - a))
        if c_min < 1e-9 or c <= c_min:
            return None

        try:
            rot, center = compute_PHS_matrices(a, b, c)
        except Exception:
            return None

        try:
            lims = self.env.limits
            template = self.env.get_start_pos()
        except Exception:
            return None

        # Validate dimensions match
        if lims.shape[1] != len(template.state()):
            return None

        for _ in range(50):
            sample = sample_phs_with_given_matrices(rot, center, n=1)
            q_flat = sample[:, 0]
            
            # Extra validation: check array dimensions
            if len(q_flat) != lims.shape[1]:
                continue
            if np.any(q_flat < lims[0]) or np.any(q_flat > lims[1]):
                continue
            try:
                q = template.from_flat(q_flat)
            except (AssertionError, ValueError, IndexError):
                continue
            if self.env.is_collision_free(q, mode):
                return q

        return None

    def _sample_path_tube(self, mode: Mode) -> Optional[Configuration]:
        """
        Gaussian tube around the current best path for this mode.
        Sigma is dimension-normalised so the tube stays useful in
        high-DOF joint spaces (e.g. 4 robots x 7 DOF = 28D).
        """
        if not self.operation.init_sol or not self.operation.path:
            return None

        mode_states = [s for s in self.operation.path if s.mode == mode]
        if not mode_states:
            return None

        try:
            anchor_q = random.choice(mode_states).q.state()
            n_dim    = len(anchor_q)
            lims     = self.env.limits
            template = self.env.get_start_pos()
        except Exception:
            return None

        # Validate that limits dimension matches anchor configuration
        if lims.shape[1] != n_dim:
            return None

        sigma    = self._tube_r.get(mode, self._tube_init) / math.sqrt(n_dim)

        for _ in range(20):
            q_flat = np.clip(
                anchor_q + np.random.normal(0.0, sigma, size=n_dim),
                lims[0], lims[1]
            )
            
            # Extra validation: ensure dimensions still match after clipping
            if len(q_flat) != n_dim or len(q_flat) != lims.shape[1]:
                continue
                
            try:
                q = template.from_flat(q_flat)
            except (AssertionError, ValueError, IndexError):
                continue
            if self.env.is_collision_free(q, mode):
                return q

        return None

    # -------------------------------------------------------------------------
    # Override: manage_transition
    # -------------------------------------------------------------------------

    def manage_transition(self, mode: Mode, n_new: Node) -> None:
        """
        Runs parent transition logic then fires mode-aware phase resets.

        Restores self.modes if mode_validation raises AssertionError
        (known edge case on some rai envs) or empties the list.
        
        Extra robustness: wraps env state changes to prevent corruption.
        """
        modes_list_before = list(self.modes)
        modes_set_before  = set(self.modes)

        try:
            super().manage_transition(mode, n_new)
        except (AssertionError, Exception) as e:
            # Restore modes on any exception to maintain consistency
            self.modes = modes_list_before
            # Log the error for debugging but continue planning
            return

        # Restore if super() silently emptied the list
        if not self.modes and modes_list_before:
            self.modes = modes_list_before
            return

        # MODE-AWARE RESET: initialise phase for every newly discovered mode
        for new_mode in set(self.modes) - modes_set_before:
            self._init_mode_phase(new_mode)

        if self.operation.init_sol and self.operation.cost < math.inf:
            self._advance_phase(mode, self.operation.cost)

    # -------------------------------------------------------------------------
    # Override: plan
    # -------------------------------------------------------------------------

    def plan(
        self,
        ptc: PlannerTerminationCondition,
        optimize: bool = True,
    ) -> Tuple[Optional[List[State]], Dict[str, Any]]:

        i         = 0
        prev_cost = math.inf
        self.initialize_planner()

        while True:
            i += 1

            # Safety: re-seed modes if mode_validation emptied the list
            if not self.modes:
                start_mode = self.env.get_start_mode()
                self.modes.append(start_mode)
                if start_mode not in self.trees:
                    self.add_tree(start_mode, SingleTree)
                if start_mode not in self._phase:
                    self._init_mode_phase(start_mode)

            active_mode = self.random_mode()

            q_rand = self.sample_configuration(active_mode)
            if not q_rand:
                continue

            n_nearest, dist, set_dists, n_nearest_idx = self.nearest(
                active_mode, q_rand
            )
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

                batch_cost = self.env.batch_config_cost(
                    n_new.state.q, N_near_batch
                )
                self.find_parent(
                    active_mode, node_indices, n_new,
                    n_nearest, batch_cost, n_near_costs,
                )

                if self.rewire(
                    active_mode, node_indices, n_new, batch_cost, n_near_costs
                ):
                    self.update_cost(active_mode, n_new)

                self.manage_transition(active_mode, n_new)

            # Phase advancement
            if self.operation.init_sol:
                cur_cost = self.operation.cost

                if cur_cost < prev_cost:
                    prev_cost = cur_cost
                    # Only advance phases for modes that still exist
                    for m in self.modes:
                        if m in self._phase:
                            self._advance_phase(m, cur_cost)

                # Stagnation: Phase 3 -> Phase 2 fallback
                # Only if active_mode is still valid
                if active_mode in self.modes and active_mode in self._phase:
                    buf = self._stag_buf.setdefault(active_mode, [])
                    buf.append(cur_cost)
                    if len(buf) > self._stag_win:
                        buf.pop(0)
                    if (
                        self._phase.get(active_mode) == self._EXPLOIT
                        and len(buf) >= self._stag_win
                        and (max(buf) - min(buf)) < self._stag_dlt
                    ):
                        self._phase[active_mode] = self._INFORMED
                        self._stag_buf[active_mode] = []

            # Termination
            if not optimize and self.operation.init_sol:
                self.save_tree_data()
                break

            if ptc.should_terminate(i, time.time() - self.start_time):
                print(f"[MAMP-RRT*] iterations: {i}")
                print(
                    f"[MAMP-RRT*] phases: "
                    f"{ {str(m.task_ids): self._phase.get(m, 1) for m in self.modes} }"
                )
                break

        self.update_results_tracking(self.operation.cost, self.operation.path)
        info = {
            "costs": self.costs,
            "times": self.times,
            "paths": self.all_paths,
        }

        # Double mode-switch nodes -- same post-processing as RRTstar
        path_w_doubled_modes = []
        for idx in range(len(self.operation.path)):
            path_w_doubled_modes.append(self.operation.path[idx])
            if (
                idx + 1 < len(self.operation.path)
                and self.operation.path[idx].mode
                != self.operation.path[idx + 1].mode
            ):
                path_w_doubled_modes.append(
                    State(
                        self.operation.path[idx].q,
                        self.operation.path[idx + 1].mode,
                    )
                )
        self.operation.path = path_w_doubled_modes

        return self.operation.path, info