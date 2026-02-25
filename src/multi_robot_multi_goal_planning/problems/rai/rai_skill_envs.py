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
    JogJoint,
    EndEffectorPositionFollowing,
    StochasticBinPick,
    DualRobotGrasping
)

from ..constraints import (
    relative_pose
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
        self.C, self.robots, [pick_pose, pre_screw_pose] = rai_config.make_ur10_screwing_env()
        # self.C.view(True)

        print(pick_pose, pre_screw_pose)

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        post_screw_pose = pre_screw_pose * 1.
        post_screw_pose[-1] += 2. * np.pi/2.

        self.tasks = [
            Task(
                "pick",
                [self.robots[0]],
                SingleGoal(pick_pose),
                type="pick",
                frames=["a1_ur_gripper_center", "obj1"]
            ),
            Task(
                "pre_screw",
                [self.robots[0]],
                SingleGoal(pre_screw_pose),
            ),
            Task(
                "screw",
                [self.robots[0]],
                SingleGoal(post_screw_pose),
                frames=["table", "obj1"],
                type="place",
                skill = JogJoint(speed=np.pi/2., idx=5, duration=2.) # just moving the final joint for a fixed time
            ),
            Task(
                "terminal",
                [self.robots[0]],
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
        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])

# Debugging for single agent timed skill
@register("rai.single_agent_drawing")
class rai_single_agent_drawing(SequenceMixin, rai_env):
    def __init__(self):
        self.C, poses = rai_config.make_single_agent_drawing()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        #table_height = 0.1 
        table_height = 0.24 # Table top at z = 0.23
        pts = [
            np.array([-0.5, 0, table_height]), 
            np.array([0.5, 0, table_height])
        ]
        # path = LineSegment(pts)

        self.tasks = [
            Task(
                "pre_draw",
                ["a1"],
                SingleGoal(poses[0]),
            ),
            Task(
                "draw",
                ["a1"],
                SingleGoal(poses[0]), # TODO: figure out how to do skill goal checking
                skill = EndEffectorPositionFollowing(*pts, "a1_stick_ee")
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
        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])

# TODO unfinished
@register("rai.single_agent_lego")
class rai_single_agent_lego(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_single_agent_lego()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        lego_placement_path = CubicSpline()
        
        pick_pose = None
        pre_place_pose = None

        self.tasks = [
            Task(
                "pick",
                ["a1"],
                SingleGoal(pick_pose),
                type="pick",
                frames=["a1_ur_ee_marker", "obj1"]
            ),
            Task(
                "pre_place",
                ["a1"],
                SingleGoal(pre_place_pose),
            ),
            Task(
                "place",
                ["a1"],
                SingleGoal(np.array([0.5, 0.5, 0])),
                skill = EndEffectorPositionFollowing(lego_placement_path),
                type="place",
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


# TODO: enable mode to only plan for a subset of dofs
@register("rai.single_agent_pick_and_place")
class rai_single_agent_pick_and_place(SequenceMixin, rai_env):
    def __init__(self):
        self.C, [pre_pick, pre_place] = rai_config.make_single_agent_pick_and_place()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        pick_position = self.C.getFrame("obj1").getPose()
        place_position = self.C.getFrame("goal1").getPose()

        self.tasks = [
            Task(
                "pre_pick",
                ["a1"],
                SingleGoal(pre_pick),
            ),
            Task(
                "pick",
                ["a1"],
                SingleGoal(home_pose),
                frames=["a1_ur_ee_marker", "obj1"],
                type="pick",
                skill = EEPoseGoalReaching(pick_position, "a1_ur_ee_marker")
            ),
            Task(
                "pre_place",
                ["a1"],
                SingleGoal(pre_place),
            ),
            Task(
                "place",
                ["a1"],
                SingleGoal(home_pose),
                skill = EEPoseGoalReaching(place_position, "obj1"),
                type="place",
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

# TODO unfinished
@register("rai.single_agent_scripted_insert")
class rai_single_agent_scripted_insert(SequenceMixin, rai_env):
  pass

# TODO unfinished
@register("rai.single_agent_learned_insert")
class rai_single_agent_learned_insert(SequenceMixin, rai_env):
  pass

# TODO unfinished
# multi agent rearrangement with skills
@register("rai.multi_agent_rearrangement")
class rai_multi_agent_pick_and_place(SequenceMixin, rai_env):
    def __init__(self, num_agents=4, num_objects=4):
        self.C, poses = rai_config.make_multi_agent_pick_and_place()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        self.tasks = []
        named_sequence = []

        for robot, task in poses:
            pass

        self.sequence = self._make_sequence_from_names(
          named_sequence
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


# TODO unfinished
# multi agent rearrangement with skills
@register("rai.multi_agent_stacking")
class rai_multi_agent_stacking(SequenceMixin, rai_env):
    def __init__(self, num_agents=4, num_objects=4):
        self.C, poses = rai_config.make_multi_agent_stacking()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        self.tasks = []
        named_sequence = []

        for robot, task in poses:
            pass

        self.sequence = self._make_sequence_from_names(
          named_sequence
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


# TODO unfinished
# multi agent rearrangement with skills
@register("rai.multi_agent_dexterous_stacking")
class rai_multi_agent_stacking(SequenceMixin, rai_env):
    def __init__(self, num_agents=4, num_objects=4):
        self.C, poses = rai_config.make_multi_agent_stacking()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        self.tasks = []
        named_sequence = []

        for robot, task in poses:
            pass

        self.sequence = self._make_sequence_from_names(
          named_sequence
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

# TODO unfinished
# draw the crl logo with 3 robots:
# TODO this should be an unordered problem
@register("rai.multi_agent_drawing")
class rai_multi_agent_drawing(SequenceMixin, rai_env):
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


# TODO unfinished
# four robot, same welding env as before
# welding lines here
@register("rai.multi_agent_line_weld")
class rai_multi_agent_weld(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_multi_agent_skill_welding_env(num_robots=4, num_lines=4)
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

# TODO unfinished
@register("rai.multi_agent_pcb")
class rai_multi_agent_pcb(SequenceMixin, rai_env):
  pass

# TODO unfinished
# kids game: https://www.youtube.com/watch?v=Ddertj2CG3I
# vision based insertion?
# vision based grasping?
@register("rai.multi_agent_insert")
class rai_multi_agent_insert(SequenceMixin, rai_env):
  pass

@register("rai.dual_arm_transport")
class rai_dual_arm_transport(SequenceMixin, rai_env):
    def __init__(self):
        self.C, _, [self.pick_pose, _] = rai_config.make_bimanual_grasping_env(obstacle=False, rotate=False)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        obj_start_pose = self.C.getFrame("obj1").getPose()
        obj_goal_pose = self.C.getFrame("goal1").getPose()

        ee_names = ["a1_ur_ee_marker", "a2_ur_ee_marker"]

        self.C.setJointState(self.pick_pose)
        
        a1_pose = self.C.getFrame("a1_ur_ee_marker").getPose()
        a2_pose = self.C.getFrame("a2_ur_ee_marker").getPose()
        obj_pose = self.C.getFrame("obj1").getPose()

        a1_transformation = relative_pose(a1_pose, obj_pose)
        a2_transformation = relative_pose(a2_pose, obj_pose)

        self.C.setJointState(home_pose)

        self.tasks = [
            Task(
                "pick_1",
                ["a1", "a2"],
                SingleGoal(self.pick_pose),
                frames=["a1_ur_ee_marker", "obj1"],
                type="pick",
            ),
            Task(
                "move",
                ["a1", "a2"],
                SingleGoal(self.pick_pose),
                skill = DualRobotGrasping(ee_names, [a1_transformation, a2_transformation], obj_start_pose, obj_goal_pose),
                type="place",
                frames=["table", "obj1"]
            
            ),
            Task(
                "terminal",
                ["a1", "a2"],
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["pick_1", "move", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


# TODO unfinished
# pick 'any' item from a bin
# Stochastic skill
@register("rai.single_agent_bin_picking")
class rai_single_agent_bin_picking(SequenceMixin, rai_env):
    def __init__(self):
        self.C, [pre_pick_o1, pre_place_o1, pre_pick_o2, pre_place_o2] = rai_config.make_single_agent_bin_picking_env()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        pick_position = self.C.getFrame("obj1").getPose()
        place_position = self.C.getFrame("goal1").getPose()

        # assuming here that we have a place to set down an object, and we only need to go to a 'generic' position
        # above the bin for picking

        # the planning itself is not that interesting here, since we only do a single robot, but this is a stochastic
        # skill, therefore needing to deal with this.
        
        self.tasks = [
            Task(
                "pre_pick_1",
                ["a1"],
                SingleGoal(pre_pick_o1),
            ),
            Task(
                "pick_1",
                ["a1"],
                SingleGoal(pre_pick_o1),
                frames=["a1_ur_gripper_center", "obj1"],
                type="pick",
                skill = StochasticBinPick()
            ),
            Task(
                "pre_place_1",
                ["a1"],
                SingleGoal(pre_place_o1),
            ),
            Task(
                "place_1",
                ["a1"],
                SingleGoal(pre_place_o1),
                skill = StochasticBinPick(),
                type="place",
                frames=["table", "obj1"]
            ),
            Task(
                "pre_pick_2",
                ["a1"],
                SingleGoal(pre_pick_o2),
            ),
            Task(
                "pick_2",
                ["a1"],
                SingleGoal(pre_pick_o2),
                frames=["a1_ur_gripper_center", "obj2"],
                type="pick",
                skill = StochasticBinPick()
            ),
            Task(
                "pre_place_2",
                ["a1"],
                SingleGoal(pre_place_o2),
            ),
            Task(
                "place_2",
                ["a1"],
                SingleGoal(pre_place_o2),
                skill = StochasticBinPick(),
                type="place",
                frames=["table", "obj2"]
            ),
            Task(
                "terminal",
                ["a1"],
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["pre_pick_1", "pick_1", "pre_place_1", "place_1", "pre_pick_2", "pick_2", "pre_place_2", "place_2", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

# TODO unfinished
# pick 'any' item from a bin
# Stochastic skill
# TODO: can a skill determine which frame will be linked??
# possible solution: just add the object where the robot ends up
@register("rai.multi_agent_bin_picking")
class rai_multi_agent_bin_picking(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_multi_agent_bin_picking_env()
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

# TODO unfinished
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

# TODO unfinished
# inspiration: https://arxiv.org/pdf/2511.04758
@register("rai.bimanual_sorting")
class rai_bimanual_sorting(SequenceMixin, rai_env):
    def __init__(self):
        self.C, a1_keyframes, a2_keyframes = rai_config.make_bimanual_sorting()
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)
        self.manipulating_env = True

        home_pose = self.C.getJointState()

        self.tasks = []

        for robot_name, [pre_pick, pre_place], obj_name, goal_name in zip(self.robots, [a1_keyframes, a2_keyframes], ["obj1", "obj2"], ["goal1", "goal2"]):
            pick_position = self.C.getFrame(obj_name).getPose()
            place_position = self.C.getFrame(goal_name).getPose()
            
            self.tasks.extend([
                Task(
                    robot_name + "_pre_pick",
                    [robot_name],
                    SingleGoal(pre_pick),
                ),
                Task(
                    robot_name + "_pick",
                    [robot_name],
                    SingleGoal(pre_pick),
                    frames=[robot_name + "_ur_gripper_center", obj_name],
                    type="pick",
                    skill = EEPoseGoalReaching(pick_position, robot_name + "ur_gripper_center")
                ),
                Task(
                    robot_name + "_pre_place",
                    [robot_name],
                    SingleGoal(pre_place),
                ),
                Task(
                    robot_name + "_place",
                    [robot_name],
                    SingleGoal(pre_place),
                    skill = EEPoseGoalReaching(place_position, obj_name),
                    type="place",
                    frames=["table", obj_name]
                )
            ]
            )

        self.tasks.append(
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
            ),
        )

        self.sequence = self._make_sequence_from_names(
            ["a1_pre_pick", "a1_pick", "a2_pre_pick", "a2_pick", "a1_pre_place", "a1_place", "a2_pre_place", "a2_place", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

# TODO unfinished
# inspiration: https://arxiv.org/pdf/2511.04758
@register("rai.so_100_sorting")
class rai_so_100_sorting(SequenceMixin, rai_env):
  pass

# TODO unfinished
# skills: 
# - screwing
# - placing
# - scaffolding stuff
@register("rai.husky_assembly")
class rai_husky_assembly(SequenceMixin, rai_env):
  pass

# TODO unfinished
# skills: 
# - grasping -> deterministic
# - insertion -> stochastic
# - do with four arms -> assemble fast
@register("rai.yijiang_corl")
class rai_yijiang_corl(SequenceMixin, rai_env):
  pass

# TODO unfinished
# skills: 
# - tying wire knots
# - inserting rods?
@register("rai.mesh")
class rai_mesh(SequenceMixin, rai_env):
  pass

# TODO unfinished
# skill to follow curved surface
@register("rai.polishing")
class rai_polising(SequenceMixin, rai_env):
  pass