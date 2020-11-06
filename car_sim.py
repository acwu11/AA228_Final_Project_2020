import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
from time import process_time
import numpy as np
import cv2
import statistics
import math

from model_free import *

IM_WIDTH = 640
IM_HEIGHT = 480
SHOW_PREVIEW = False

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    #STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    control_diff = 0.4

    #- ENV SETUP  --------------------------------------------------------------------------
    # intialize world and car
    def __init__(self):

        # world and spectator
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()

        # car
        self.blueprint_library = self.world.get_blueprint_library()
        self.bp = self.blueprint_library.filter('model3')[0]
        print(self.bp)

        # specify available actions [throttle, steer, brake]
        self.ActionDict = {0 : [1.0, -0.1, 0], 
                           1 : [1.0, 0, 0],
                           2 : [1.0, 0.1, 0],
                           3 : [0.5, 0.1, 0],
                           4 : [0.5, -0.1, 0],
                           5 : [0, 0.1, 0],
                           6 : [0, -0.1, 0],
                           7 : [0, 0.1, 0.5],
                           8 : [0, -0.1, 0.5],
                           9 : [0, 0.1, 1],
                           10: [0, -0.1, 1]}
        self.nAction = len(self.ActionDict)

    # reset environment
    def reset(self):
        self.actor_list = []
        self.collision_hist = []
        self.steering_amt = 0
        self.throttle_amt = 0
        self.depth = 50
        self.azimuth = 0
        
        # spawn vehicle at chosen pawn point and append to actor list
        self.spawn_point = (self.world.get_map().get_spawn_points())[5]
        self.vehicle = self.world.spawn_actor(self.bp, self.spawn_point)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
        time.sleep(1)
        self.actor_list.append(self.vehicle)
        
        # get sensor blueprints
        self.blueprint = self.blueprint_library.find('sensor.other.radar')
        self.blueprintOD = self.blueprint_library.find('sensor.other.obstacle')
        self.blueprintCD = self.blueprint_library.find('sensor.other.collision')
        self.blueprintRGB = self.blueprint_library.find('sensor.camera.rgb')
        
        # # change the dimensions of the image
        self.blueprintRGB.set_attribute('image_size_x', f'{self.im_width}')
        self.blueprintRGB.set_attribute('image_size_y', f'{self.im_height}')
        self.blueprintRGB.set_attribute('fov', '110')

        # Adjust sensor relative to vehicle
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        spawn_pointcam = carla.Transform(carla.Location(x=0, z=2))

        # spawn the sensor and attach to vehicle.
        self.sensor = self.world.spawn_actor(self.blueprint, spawn_point, attach_to=self.vehicle)
        self.sensorOD = self.world.spawn_actor(self.blueprintOD, spawn_point, attach_to=self.vehicle)
        self.sensorCD = self.world.spawn_actor(self.blueprintCD, spawn_point, attach_to=self.vehicle)
        self.sensorRGB = self.world.spawn_actor(self.blueprintRGB, spawn_pointcam, attach_to=self.vehicle)

        # add sensor to list of actors
        self.actor_list.append(self.sensor)
        self.actor_list.append(self.sensorOD)
        self.actor_list.append(self.sensorCD)
        self.actor_list.append(self.sensorRGB)
            
        # do something with this sensor

        self.sensor.listen(lambda data: self.process_data(data))
        self.sensorOD.listen(lambda dataOD: self.register_obstacle(dataOD))
        self.sensorCD.listen(lambda dataCD: self.register_collision(dataCD))
        self.sensorRGB.listen(lambda dataRGB: self.process_img(dataRGB))

    #- DATA PROCESSING --------------------------------------------------------------------------
    # radar
    def process_data(self, data):
        # img = data.raw_data
        self.points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        self.points = np.reshape(self.points, (len(data), 4))
        self.depth = statistics.mean([item[3] for item in self.points])
        self.azimuth = statistics.mean([item[2] for item in self.points])*(180/math.pi)
        self.detect_count = data.get_detection_count()

        return self.depth

    # obstacle detection
    def register_obstacle(self, data):
        self.obstacle = data
    
    # collision detection
    def register_collision(self, data):
        collided_with = data.other_actor
        print(collided_with)
        self.collision_hist.append(data)
    
    # returns vehicle speed, 
    def get_obs(self):
        self.v = self.vehicle.get_velocity()
        self.kmh = int(3.6*math.sqrt(self.v.x**2 + self.v.y**2 + self.v.z**2))
        return [self.kmh, self.depth, 0,  len(self.collision_hist), self.azimuth] 

    # applies action to vehicle, computes reward received
    def step(self, action):
        # given action apply throttle and steer
        t = self.ActionDict[action][0]
        s = self.ActionDict[action][1]
        b = self.ActionDict[action][2]
        self.vehicle.apply_control(carla.VehicleControl(throttle=t, steer=s, brake=b))
        
        # car velocity
        self.v = self.vehicle.get_velocity()
        self.kmh = int(3.6*math.sqrt(self.v.x**2 + self.v.y**2 + self.v.z**2))
        
        # reward
        if len(self.collision_hist) != 0:
            done = True
            reward = 0
            #reward = -200
            
        elif (self.kmh < 10) and self.depth > 1:
            done = False
            reward = -10
        elif self.kmh > 60:
            done = False
            reward = -1
        elif 1 < self.depth < 5:
            done = False
            reward = -50
        elif self.depth < 1:
            done = False
            reward = -1
        else:
            done = False
            reward = 10

        obs =  [self.kmh, self.depth, 0,  len(self.collision_hist), self.azimuth]  
        return obs, reward, done, action

    #- VISUALIZATION  --------------------------------------------------------------------------
    # show image from camera and update world spectator position
    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3
        self.spectator.set_transform(self.sensorRGB.get_transform())


if __name__ == "__main__":
    # simulation parameters
    EPSILON = 0.3           # exploration 
    DISCOUNT = 0.9          # discount (gamma)
    LEARNING_RATE = 0.9     # learning rate (alpha)
    NUM_EPISODES = 100      # number of training episodes
    FPS = 5                 # frames per second
    DR_ALPHA = 0.999        # discount rate on epsilon (alpha)
        
    # method selection
    qlearning = 0
    sarsa = 0
    sarsa_lambda = 1

    # set up environment and agent
    env = CarEnv()
    learn_time = np.zeros(NUM_EPISODES)
    if qlearning == 1:
        agent = QAgent(80, 0, 50, 0, env.nAction)
    elif sarsa == 1:
        agent = SARSA_agent(80, 0, 50, 0, env.nAction)
    elif sarsa_lambda == 1:
        agent = SLambda_agent(80, 0, 50, 0, env.nAction, 0.5)

    # run sim
    for i in range(NUM_EPISODES):
        env.reset()
        t_start = process_time()
        print('epsilon: ' + str(EPSILON))

        while True:
            current_state = env.get_obs()
            
            # epsilon greedy action
            if qlearning == 1: 
                if np.random.uniform() > EPSILON:
                    action = np.argmax(agent.get_Q(current_state))
                    time.sleep(1/FPS)
                    print(f"Action {action} was taken.")
                else:
                    action = np.random.randint(0, env.nAction)
                    print(f"Random action {action} was taken.")
                    time.sleep(1/FPS)
            elif sarsa == 1 or sarsa_lambda == 1:
                if np.random.uniform() > EPSILON:
                    action = agent.lastexp[1]
                    time.sleep(1/FPS)
                    print(f"Action {action} was taken.")
                else:
                    action = np.random.randint(0, env.nAction)
                    time.sleep(1/FPS)
                    print(f"Random action {action} was taken.")
            
            # advance controls with that action
            new_state, reward, done, _ = env.step(action)

            #Update Q_table
            if qlearning == 1:
                agent.update_Q(current_state, reward, action, new_state)
            elif sarsa == 1:
                agent.sarsa_update(reward, action, new_state)
            elif sarsa_lambda == 1:
                agent.sarsa_lambda_update(reward, action, new_state)

            if done:
                break

        # record episode runtime
        t_intermediate = process_time()
        learn_time[i] = t_intermediate - t_start

        # reduce epsilon 
        EPSILON = DR_ALPHA * EPSILON

        # save Q-table and time data
        np.save('qlearn.npy', agent.Q_table)
        np.save('qlearn_runtime.npy', learn_time)

        # clean up actors
        for actor in env.actor_list:
            print('destroying actor')
            actor.destroy()
        print(f"Episode {i}")
