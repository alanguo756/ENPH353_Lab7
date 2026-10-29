import cv2
import rclpy
import time
import gym_gazebo
import gymnasium as gym
import numpy as np
from cv_bridge import CvBridge
from gym_gazebo.core.gazebo_env import GazeboEnv
from rclpy.node import Node as RosNode
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist as RosTwist

NUM_OBS_STATES = 3
NUM_ACTIONS = 3

class LineFollowerEnv(GazeboEnv):
    def __init__(self):
        # Intializing ROS2
        rclpy.init(args=None)
        
        super(LineFollowerEnv, self).__init__(launch_pkg='linefollow_ros',
                                    launch_file='my_agent.launch.xml',
                                    world_name='monza_gym')

        # Setting up ROS2 node
        self.ros_node = RosNode('line_follower_RL')

        # Initializing variables
        self.is_resetting = False
        self.latest_obs = None
        self.new_obs_event = False

        # Setting the observation and action spaces
        self.observation_space = gym.spaces.Discrete(NUM_OBS_STATES)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # ROS2 topics
        observation_topic = "/camera/image_raw"
        velocity_topic = "/cmd_vel"

        # Publishing and subscribing to ROS2 topics
        self.pub_cmd_vel_msg = self.ros_node.create_publisher(
            RosTwist, velocity_topic, 10)
            
        self.sub_obs = self.ros_node.create_subscription(
            RosImage, observation_topic, self.callback_obs_feed_, 10)


    def callback_obs_feed_(self, msg: RosImage):
        '''
        @brief observation callback resets the new_obs_event flag
        @param msg: Image from Gazebo sim
        '''
        # Takes in observation, returns a flag an observation is recieved
        if not self.is_resetting:
            self.latest_obs = msg
            self.new_obs_event = True


    ## MAIN FUNCTIONS:
    def reset(self, seed=None, options=None):
        '''
        @brief resets the simulation
        '''
        self.is_resetting = True
        print("Resetting")

        # Zero out velocities
        vel_cmd = RosTwist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Reset position and joints
        self._reset_agents()

        # Unpause the sim
        self._pause_sim(False)

        # Allow some time to settle
        time.sleep(0.1) # Wait 100 ms
        
        self.is_resetting = False

        # Wait for next observation to come in before kickstarting the data loop
        self.new_obs_event = False
        while not self.new_obs_event:
            # Telling executor (node) to spin
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # # Pause the sim after the first observation comes in
        self._pause_sim(True)
        
        state, _ = self.process_obs(self.latest_obs)

        print("Done resetting")
        return state, {}
    
    
    def step(self, action: int):
        '''
        @brief takes a step in the simulation
        @param action
        @retval state
        @retval reward
        @retval terminated
        @retval truncated
        '''
        # Process the action
        vel_cmd = self.process_action(action)

        # Unpause the sim
        self._pause_sim(False)

        # Moving the agent
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Leave running until next observation comes in
        self.new_obs_event = False
        while not self.new_obs_event:
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # Pause the simulation after the next observation comes in
        self._pause_sim(True)

        state, terminated = self.process_obs(self.latest_obs)
        
        reward = self.process_reward(action, state, terminated)

        truncated = False # Used if we ran maximum number of steps.
        return state, reward, terminated, truncated, {}

    
    ## HELPER FUNCTIONS
    def process_action(self, action: int):
        vel_cmd = RosTwist()
        if action == 0:      # left
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = 0.5
        elif action == 1:    # forward
            vel_cmd.linear.x = 0.6
            vel_cmd.angular.z = 0.0
        elif action == 2:    # right
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = -0.5
        else:
            raise ValueError(f"Invalid action: {action}")
        return vel_cmd


    def process_reward(self, action, state, terminated):
        '''
        @brief depending on the action, state and terminated values assign
            a reward
        @retval reward: the reward for the current step
        '''
        # Lost the line: the failure we most want to avoid
        if terminated or state == -1:
            return -100.0

        reward = 0.0

        if state == 1:
            # Line centered: the goal
            reward += 2.0
            if action == 1:
                reward += 1.0     # holding a straight line while centered
            else:
                reward -= 0.5     # turning while centered = wobble
        else:
            # Line at an edge: drifting, about to lose it
            reward -= 1.0

            toward = 0 if state == 0 else 2   # line left -> turn left, line right -> turn right
            away = 2 if state == 0 else 0

            if action == toward:
                reward += 1.5     # correcting: best option at an edge
            elif action == away:
                reward -= 1.0     # making it worse

        return reward


    def process_obs(self, obs: RosImage):
        # ROS Image -> numpy array
        image = np.frombuffer(obs.data, dtype=np.uint8).reshape(
            obs.height, obs.width, -1)
        gray = image.mean(axis=2) if image.shape[2] > 1 else image[:, :, 0]

        # Binary mask: dark line on light floor (flip to > 100 if the line is lighter)
        mask = gray < 100

        # Region of interest: bottom part of the image, closest to the robot
        roi_start = int(obs.height * 0.80)
        roi = mask[roi_start:, :]

        # No line (or only noise) -> terminate
        if roi.sum() < 20:
            return -1, True

        # Centroid: mean x position of all line pixels in the ROI
        _, xs = np.nonzero(roi)
        centroid_x = xs.mean()

        # Which third of the image is the centroid in?
        third = obs.width / 3
        if centroid_x < third:
            state = 0          # line left
        elif centroid_x < 2 * third:
            state = 1          # line center
        else:
            state = 2          # line right

        return state, False


    # Shutting down ROS2
    def user_close(self):
        self.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()