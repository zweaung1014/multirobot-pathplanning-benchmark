import robotic as ry
import numpy as np
import random

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.dependency_graph import DependencyGraph

import multi_robot_multi_goal_planning.problems.rai.rai_config as rai_config
from ..configuration import config_dist

# from multi_robot_multi_goal_planning.problems.rai_config import *
from ..planning_env import (
    BaseModeLogic,
    SequenceMixin,
    DependencyGraphMixin,
    State,
    Task,
    ProblemSpec,
    AgentType,
    GoalType,
    ConstraintType,
    DynamicsType,
    ManipulationType,
    DependencyType,
    SafePoseType,
)

from ..skills import (
    EEPoseGoalReaching,
    Screw
)

from ..goals import (
    SingleGoal,
    GoalSet,
    GoalRegion,
    ConditionalGoal,
)
from ..rai_base_env import rai_env

from ..registry import register

############
# Debugging/testing envs: single agent
############

@register("rai.single_agent_screw")
class rai_single_agent_screw(SequenceMixin, rai_env):
    def __init__(self):
        self.C, self.robots = rai_config.make_ur10_screwing_env()
        # self.C.view(True)

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        self.tasks = [
            Task(
                "pick",
                ["a1"],
                SingleGoal(np.array([-0.5, -0.5, 0])),
                frames=["a1_ur_ee_marker", "obj1"]
            ),
            Task(
                "pre_screw",
                ["a1"],
                SingleGoal(np.array([0.5, 0.5, 0])),
            ),
            Task(
                "screw",
                ["a1"],
                SingleGoal(np.array([0.5, 0.5, 0])),
                frames=["table", "obj1"],
                skill = Screw()
            ),
            Task(
                "terminal",
                ["a1"],
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["pick", "pre_screw", "screw", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


@register("rai.single_agent_drawing")
class rai_single_agent_drawing(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_single_agent_drawing()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        path = LineSegment()

        self.tasks = [
            Task(
                "pre_draw",
                ["a1"],
                SingleGoal(np.array([0.5, 0.5, 0])),
            ),
            Task(
                "draw",
                ["a1"],
                SingleGoal(np.array([0.5, 0.5, 0])),
                skill = EndEffectorPositionFollowing(path)
            ),
            Task(
                "terminal",
                ["a1"],
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["pre_draw", "draw", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


@register("rai.single_agent_lego")
class rai_single_agent_lego(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_single_agent_lego()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        lego_placement_path = CubicSpline()

        self.tasks = [
            Task(
                "pick",
                ["a1"],
                SingleGoal(np.array([0.5, 0.5, 0])),
                frames=["a1_ur_ee_marker", "obj1"]
            ),
            Task(
                "pre_place",
                ["a1"],
                SingleGoal(home_pose),
            ),
            Task(
                "place",
                ["a1"],
                SingleGoal(np.array([0.5, 0.5, 0])),
                skill = EndEffectorPositionFollowing(lego_placement_path),
                frames=["table", "obj1"]
            ),
            Task(
                "terminal",
                ["a1"],
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["pick", "pre_place", "place", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


@register("rai.single_agent_pick_and_place")
class rai_single_agent_pick_and_place(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_single_agent_pick_and_place()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        placement_pose = None

        self.tasks = [
            Task(
                "pre_pick",
                ["a1"],
                SingleGoal(home_pose),
            ),
            Task(
                "pick",
                ["a1"],
                SingleGoal(np.array([0.5, 0.5, 0])),
                frames=["a1_ur_ee_marker", "obj1"]
            ),
            Task(
                "pre_place",
                ["a1"],
                SingleGoal(home_pose),
            ),
            Task(
                "place",
                ["a1"],
                SingleGoal(np.array([0.5, 0.5, 0])),
                skill = EEPoseGoalReaching(placement_pose),
                frames=["table", "obj1"]
            ),
            Task(
                "terminal",
                ["a1"],
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["pre_pick", "pick", "pre_place", "place",
            "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

@register("rai.single_agent_scripted_insert")
class rai_single_agent_scripted_insert(SequenceMixin, rai_env):
  pass

@register("rai.single_agent_learned_insert")
class rai_single_agent_learned_insert(SequenceMixin, rai_env):
  pass

@register("rai.multi_agent_pick_and_place")
class rai_multi_agent_pick_and_place(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_multi_agent_pick_and_place()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        self.tasks = [
        ]

        self.sequence = self._make_sequence_from_names(
          []
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

# draw the crl logo with 3 robots
@register("rai.multi_agent_drawing")
class rai_multi_agent_insert(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_multi_agent_drawing()
        # self.C.view(True)

        self.robots = ["a1", "a2", "a3"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        self.tasks = [
            None
        ]

        self.sequence = self._make_sequence_from_names(
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


# four robot, same welding env as before
# welding lines here
@register("rai.multi_agent_line_weld")
class rai_multi_agent_weld(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_simple_skill_welding_env()
        # self.C.view(True)

        self.robots = ["a1", "a2", "a3"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        self.tasks = [
            None
        ]

        self.sequence = self._make_sequence_from_names(
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

# kids game:
# vision based insertion?
# vision based grasping?
@register("rai.multi_agent_insert")
class rai_multi_agent_insert(SequenceMixin, rai_env):
  pass

# skills: 
# - multiple robots -> fast pcb assembly?
# - bimanual skill with reorientation of obj? holding?
@register("rai.bimanual_assembly")
class rai_bimanual_assembly(SequenceMixin, rai_env):
  # one holding, the other adding something
  # skills might be 
  # - dual insertion where both do something
  # - single robot pick up
  # - idally both at some pt.
  pass

# skills: 
# - screwing
# - placing
# - scaffolding stuff
@register("rai.husky_assembly")
class rai_husky_assembly(SequenceMixin, rai_env):
  pass

# skills: 
# - grasping -> deterministic
# - insertion -> stochastic
# - do with four arms -> assemble fast
@register("rai.yijiang_corl")
class rai_yijiang_corl(SequenceMixin, rai_env):
  pass

# skills: 
# - tying wire knots
# - inserting rods?
@register("rai.mesh")
class rai_mesh(SequenceMixin, rai_env):
  pass
