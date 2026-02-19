import numpy as np

from abc import ABC, abstractmethod
import robotic

##########
# Note: might be a cooler demo if we also have skills that are 'env aware'
# might also be more interesting planning wise.
##########

# abstract class for skills.
class DeterministicBaseSkill(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def step(self, q, env):
    pass

  # TODO: move to step itself? two return values?
  @abstractmethod
  def done(self, q, env):
    pass

# abstract class for stochastic skills.
class StochasticBaseSkill(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def step(self, q, env):
    pass

  @abstractmethod
  def done(self, q, env):
    pass

# abstract class for deterministic timed skills.
class BaseDeterministicTimedSkill(ABC):
  def __init__(self):
    pass

  # TODO: should likely simply merge q and t to 'state'
  @abstractmethod
  def step(self, q, t, env):
    raise NotImplementedError

  @abstractmethod
  def done(self, q, t, env):
    pass

# abstract class for stochastic timed skills.
class BaseStochasticTimedSkill(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def step(self, q, t, env):
    raise NotImplementedError

  @abstractmethod
  def done(self, q, t, env):
    pass

class EEPositionGoalReaching(DeterministicBaseSkill):
  def __init__(self, goal, ee_name):
    self.goal_position = goal
    self.ee_name = ee_name

    self.qdot_clip = 0.2

  def step(self, q, env, dt=0.1):
    # get jacobian
    env.C.setJointState(q)
    [err, jac] = env.C.eval(robotic.FS.position, [self.ee_name], 1, self.goal_position)
    
    # compute pid law
    q_dot = np.linalg.pinv(jac) @ err
    q_dot = np.clip(q_dot, a_min=-self.qdot_clip*np.ones(q_dot.shape), a_max=self.qdot_clip*np.ones(q_dot.shape))

    # integrate to get next pos
    q_new = q - dt * q_dot
    return q_new

  def done(self, q, env):
    env.C.setJointState(q)
    [err, jac] = env.C.eval(robotic.FS.position, [self.ee_name], 1, self.goal_position)
    
    if np.linalg.norm(err) < 1e-3:
      return True

    return False

# simple pid controller
class EEPoseGoalReaching(DeterministicBaseSkill):
  def __init__(self, goal, ee_name):
    self.goal_pose = goal
    self.ee_name = ee_name

    self.qdot_clip = 0.2

  def step(self, q, env, dt=0.1):
    # get jacobian
    env.C.setJointState(q)
    [err, jac] = env.C.eval(robotic.FS.pose, [self.ee_name], 1, self.goal_pose)
    
    # compute pid law
    q_dot = np.linalg.pinv(jac) @ err
    q_dot = np.clip(q_dot, a_min=-self.qdot_clip*np.ones(q_dot.shape), a_max=self.qdot_clip*np.ones(q_dot.shape))

    # integrate to get next pos
    q_new = q - dt * q_dot
    return q_new

  def done(self, q, env):
    # get jacobian
    env.C.setJointState(q)
    [err, jac] = env.C.eval(robotic.FS.pose, [self.ee_name], 1, self.goal_pose)

    if np.linalg.norm(err) < 1e-3:
      return True

    return False

# question: can the mode be changed in a skill?
# or does it need to be two skills?
class VacuumGrasping(BaseDeterministicTimedSkill):
  def __init__(self, box_pos):
    pass

  def step(self, t, q, env):
    raise NotImplementedError

  def done(self, t, q, env):
    raise NotImplementedError

class EndEffectorPoseFollowing(BaseDeterministicTimedSkill):
  def __init__(self, line_start_pos, line_goal_pos, ee_name):
    self.line_start_pos = line_start_pos
    self.line_goal_pos = line_goal_pos

    self.duration = 1

    self.ee_name = ee_name

  def _get_desired_pose_at_time(self, t):
    return self.line_start_pos + t * (self.line_goal_pos - self.line_start_pos)

  def step(self, t, q, env):
    # look up where we are on the trajctory
    desired_next_pos = self._get_desired_pose_at_time(t)

    env.C.setJointState(q)
    [err, jac] = env.C.eval(robotic.FS.pose, [self.ee_name], 1, desired_next_pos)
    
    # compute pid law
    q_dot = np.linalg.pinv(jac) @ err

    # integrate to get next pos
    q_new = q - dt * q_dot
    return q_new

  def done(self, t, q, env):
    desired_next_pos = self._get_desired_pose_at_time(self.duration)

    env.C.setJointState(q)
    [err, jac] = env.C.eval(robotic.FS.pose, [self.ee_name], 1, desired_next_pos)
    
    if np.linalg.norm(err) < 1e-3:
      return True

    return False

class EndEffectorPositionFollowing(BaseDeterministicTimedSkill):
  def __init__(self, line_start_pos, line_goal_pos, ee_name):
    self.line_start_pos = line_start_pos
    self.line_goal_pos = line_goal_pos

    self.duration = 1

    self.ee_name = ee_name

  def _get_desired_position_at_time(self, t):
    return self.line_start_pos + t * (self.line_goal_pos - self.line_start_pos)

  def step(self, t, q, env, dt=0.1):
    # look up where we are on the trajctory and get next position
    env.C.setJointState(q)

    desired_position = self._get_desired_position_at_time(t)
    [err, jac] = env.C.eval(robotic.FS.position, [self.ee_name], 1, desired_position)
    
    # compute pid law
    q_dot = np.linalg.pinv(jac) @ err

    # integrate to get next pos
    q_new = q - dt * q_dot
    return q_new

  def done(self, t, q, env):
    desired_next_pos = self._get_desired_pose_at_time(self.duration)

    env.C.setJointState(q)
    [err, jac] = env.C.eval(robotic.FS.position, [self.ee_name], 1, desired_position)

    if np.linalg.norm(err) < 1e-3:
      return True

    return False

# cool because it includes multiple robots.
class DualRobotGrasping(BaseDeterministicTimedSkill):
  def __init__(self, ee_names, obj_name, obj_start_pos, obj_end_pos):
    self.obj_start_pos = obj_start_pos
    self.obj_end_pos = obj_end_pos
    
    self.ee_names = []
    self.obj_name = obj_name

  def _get_desired_obj_pose_at_time(self, t):
    return self.obj_start_pos + t * (self.obj_end_pos - self.obj_start_pos)

  def step(self, t, q, env, dt=0.1):
    # get desired position of obj at time
    desired_pos = self._get_desired_obj_pose_at_time(t)
    
    # get ee-pos
    # get jacobians
    # do ik to compute the positions of the end effectors
    raise NotImplementedError

  def done(self, t, q, env):
    raise NotImplementedError

class Insertion(StochasticBaseSkill):
  def __init__():
    pass

  def step(self, q, env, dt=0.1):
    # query the policy
    # onnx?
    # decide noise level ourselves?
    pass

  def done(self, q, env):
    raise NotImplementedError

class DexterousGrasping(StochasticBaseSkill):
  def __init__():
    pass

  def step(self, q, env, dt=0.1):
    pass

  def done(self, q, env):
    raise NotImplementedError

class Handover(DeterministicBaseSkill):
  def __init__():
    pass

  def step(self, q, env, dt=0.1):
    pass

  def done(self, q, env):
    raise NotImplementedError

class JogJoint(BaseDeterministicTimedSkill):
  def __init__(self, speed, idx, duration):
    self.speed = speed
    self.idx = idx
    self.duration = duration

  def step(self, t, q, env, dt=0.1):
    qn = q
    qn[idx] += speed
    return qn

  def done(self, t, q, env, dt=0.1):
    if t > self.duration:
      return True

    return False

class Screw(DeterministicBaseSkill):
  def __init__(self,speed, ee_name):
    self.speed = speed
    self.ee_name = ee_name

  def step(self, q, env, dt=0.1):
    pass

  def done(self, q, env):
    raise NotImplementedError

class PrecomputedSkillDistribution(StochasticBaseSkill):
  def __init__():
    pass

  def step(self, q, env, dt=0.1):
    pass

  def done(self, q, env):
    raise NotImplementedError

# this can model bin picking form a bin where we do not care which item we take
# could e.g. be a bin of all the same objects, and we do not care
class StochasticBinPick(StochasticBaseSkill):
  def __init__(self):
    pass

  def step(self, q, env, dt=0.1):
    pass

  def done(self, q, env):
    raise NotImplementedError
