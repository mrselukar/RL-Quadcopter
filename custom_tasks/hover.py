import numpy as np
from physics_sim import PhysicsSim

class hover():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=np.array([0., 0., 10.]), init_velocities=None,
        init_angle_velocities=None, runtime=10., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles pitch yaw roll
            runtime: time limit for each episode (This makes it an episodic task)
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0 ## lowest roter speed
        self.action_high = 900 ## highest roter speed

        self.action_size = 4 ## 4 roter speeds

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        reward = -0.001
        # Reward for being close to the z target
        reward -= 1.5*(abs(self.sim.pose[ 2] - self.target_pos[2])) 
        
        # reward for being close to the x,y target 
        # lower sensitivity as grvity is not in x,y plane 
        reward -= 1.2*(abs(self.sim.pose[:2] - self.target_pos[:2]).sum())
        
        
        dist_reward = reward

        scale_fac = max(np.exp(reward/25),0.0)*0.5
        
        # angular velocity Very High Penalty
        reward -= (abs(self.sim.angular_v[:3]).sum())*0.6
           
        #rewards for normal velocity
        reward -= scale_fac*(abs(self.sim.v[0:2]).sum())
        
        

        # rewards for euler angles 
        reward -=  scale_fac*(abs(self.sim.pose[3:-1]).sum())*2
        
        return reward
    
    
    
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        
        if abs(self.sim.pose[:3]-self.target_pos).sum() > 50:
            reward -= 50
            done = True
        
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
