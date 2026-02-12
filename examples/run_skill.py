from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.problems.skills import EEPositionGoalReaching, EEPoseGoalReaching


import numpy as np
import random

def main():
    np.random.seed(0)
    random.seed(0)
    
    env = get_env_by_name("rai.single_stick_upright")

    # ee_reaching_skill = EEPositionGoalReaching(np.array([0, 0, 0]), "a1_stick_ee")
    ee_reaching_skill = EEPoseGoalReaching(np.array([0, 0, 0, 0, 1, 0, 0]), "a1_stick_ee")
    
    for _ in range(100):
      qnew = ee_reaching_skill.step(env.C.getJointState(), env)

      env.C.setJointState(qnew)
      env.C.view(True)

if __name__ == "__main__":
    main()