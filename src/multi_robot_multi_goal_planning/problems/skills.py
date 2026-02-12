import numpy as np

from abc import ABC, abstractmethod
import robotic

# abstract class for skills.
class DeterministicBaseSkill(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def step(self, q, env):
    pass


# abstract class for stochastic skills.
class StochasticBaseSkill(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def step(self, q, env):
    pass

# abstract class for deterministic timed skills.
class BaseDeterministicTimedSkill(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def step(self, q, t, env):
    raise NotImplementedError

# abstract class for stochastic timed skills.
class BaseStochasticTimedSkill(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def step(self, q, t, env):
    raise NotImplementedError

class EEPositionGoalReaching(DeterministicBaseSkill):
  def __init__(self, goal, ee_name):
    self.goal_position = goal
    self.ee_name = ee_name

    self.kp = 1
    self.kd = 0

  def step(self, q, env, dt=0.1):
    # get jacobian
    env.C.setJointState(q)
    [err, jac] = env.C.eval(robotic.FS.position, [self.ee_name], 1, self.goal_position)
    
    # compute pid law
    q_dot = np.linalg.pinv(jac) @ err

    # integrate to get next pos
    q_new = q - dt * q_dot
    return q_new

# simple pid controller
class EEPoseGoalReaching(DeterministicBaseSkill):
  def __init__(self, goal, ee_name):
    self.goal_pose = goal
    self.ee_name = ee_name

  def step(self, q, env, dt=0.1):
    # get jacobian
    env.C.setJointState(q)
    [err, jac] = env.C.eval(robotic.FS.pose, [self.ee_name], 1, self.goal_pose)
    
    # compute pid law
    q_dot = np.linalg.pinv(jac) @ err

    # integrate to get next pos
    q_new = q - dt * q_dot
    return q_new

# question: can the mode be changed in a skill?
# or does it need to be two skills?
class VacuumGrasping(BaseDeterministicTimedSkill):
  def __init__(self, box_pos):
    pass

  def step(self, t, q, env):
    raise NotImplementedError

class EndEffectorPoseFollowing(BaseDeterministicTimedSkill):
  def __init__(self, line_start_pos, line_goal_pos, ee_name):
    self.line_start_pos = line_start_pos
    self.line_goal_pos = line_goal_pos

    self.ee_name = ee_name

  def step(self, t, q, env):
    # look up where we are on the trajctory
    desired_next_pos = 0
    current_ee_pos = 0
    jac = 0

    # return pt
    pos_error = 0
    rot_error = log_map_rot_error()
    err = [pos_error, rot_error]
    q_dot = gain * jac @ err

    q_new = q + q_dot * dt

    raise NotImplementedError


class EndEffectorPositionFollowing(BaseDeterministicTimedSkill):
  def __init__(self, line_start_pos, line_goal_pos, ee_name):
    self.line_start_pos = line_start_pos
    self.line_goal_pos = line_goal_pos

    self.ee_name = ee_name

  def step(self, t, q, env):
    # look up where we are on the trajctory and get next position
    # compute control input -> pose is free/might be constrained
    # integrate
    # return pt
    raise NotImplementedError

# cool because it includes multiple robots.
class DualRobotGrasping(BaseDeterministicTimedSkill):
  def __init__(self):
    self.obj_path = 0
    self.ee_names = []

    self.obj_name = 0

  def step(self, t, q, env):
    # get desired position of obj at time
    # get ee-pos
    # get jacobians
    # do ik to compute the positions of the end effectors
    raise NotImplementedError


# Note: might be a cooler demo if we also have skills that are 'env aware'
# might also be more interesting planning wise.
