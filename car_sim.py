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
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    #- ENV SETUP  --------------------------------------------------------------------------
    # intialize world and car
    def __init__(self):

        # world and spectator
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

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

        self.depth = 50
        
        self.Obdist = 20
        self.ObdistL = 15
        self.ObdistR = 15
        
        self.azimuth = 0
        self.zrate = 0
        
        self.spawn_point = (self.world.get_map().get_spawn_points())[5]

        self.vehicle = self.world.spawn_actor(self.bp, self.spawn_point)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
        time.sleep(1)
        #vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

        self.actor_list.append(self.vehicle)
        

        # https://carla.readthedocs.io/en/latest/cameras_and_sensors
        # get the blueprint for this sensor
        self.blueprint = self.blueprint_library.find('sensor.other.radar')
        
        self.blueprintOD = self.blueprint_library.find('sensor.other.obstacle')
        self.blueprintODside = self.blueprint_library.find('sensor.other.obstacle')
        
        self.blueprintCD = self.blueprint_library.find('sensor.other.collision')
        self.blueprintRGB = self.blueprint_library.find('sensor.camera.rgb')
        self.blueprintIMU = self.blueprint_library.find('sensor.other.imu')
        
        # change the dimensions of the image
        self.blueprintRGB.set_attribute('image_size_x', f'{self.im_width}')
        self.blueprintRGB.set_attribute('image_size_y', f'{self.im_height}')
        self.blueprintRGB.set_attribute('fov', '110')
        
        self.blueprint.set_attribute('horizontal_fov', '15')
        self.blueprint.set_attribute('vertical_fov', '5')
        self.blueprint.set_attribute('range', '15')
        
        self.blueprintOD.set_attribute('distance', '40')
        self.blueprintOD.set_attribute('debug_linetrace', 'False')
        self.blueprintOD.set_attribute('hit_radius', '2.5')
        
        self.blueprintODside.set_attribute('distance', '25')
        self.blueprintODside.set_attribute('debug_linetrace', 'False')
        self.blueprintODside.set_attribute('hit_radius', '2')

        # Adjust sensor relative to vehicle
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        spawn_pointD = carla.Transform(carla.Location(x=2.5, z=4))
        spawn_pointL = carla.Transform(carla.Location(x=2.5, z=3), carla.Rotation(0, -50, 0))
        spawn_pointR = carla.Transform(carla.Location(x=2.5, z=3), carla.Rotation(0, 50, 0))
        
        spawn_pointcam = carla.Transform(carla.Location(x=0, z=2))

        # spawn the sensor and attach to vehicle.
        #self.sensor = self.world.spawn_actor(self.blueprint, spawn_point, attach_to=self.vehicle)
        
        self.sensorOD = self.world.spawn_actor(self.blueprintOD, spawn_pointD, attach_to=self.vehicle)
        self.sensorODR = self.world.spawn_actor(self.blueprintODside, spawn_pointR, attach_to=self.vehicle)
        self.sensorODL = self.world.spawn_actor(self.blueprintODside, spawn_pointL, attach_to=self.vehicle)
        
        self.sensorCD = self.world.spawn_actor(self.blueprintCD, spawn_point, attach_to=self.vehicle)
        self.sensorRGB = self.world.spawn_actor(self.blueprintRGB, spawn_pointcam, attach_to=self.vehicle)
        self.sensorIMU = self.world.spawn_actor(self.blueprintIMU, spawn_point, attach_to=self.vehicle)

        # add sensor to list of actors
        #self.actor_list.append(self.sensor)
        self.actor_list.append(self.sensorOD)
        self.actor_list.append(self.sensorODL)
        self.actor_list.append(self.sensorODR)
        self.actor_list.append(self.sensorCD)
        self.actor_list.append(self.sensorRGB)
        self.actor_list.append(self.sensorIMU)
          
        # do something with this sensor

        #self.sensor.listen(lambda data: self.process_data(data))
        
        self.sensorOD.listen(lambda dataOD: self.register_obstacle(dataOD))   
        self.sensorODL.listen(lambda dataODL: self.register_obstacleL(dataODL))
        self.sensorODR.listen(lambda dataODR: self.register_obstacleR(dataODR))
        
        self.sensorCD.listen(lambda dataCD: self.register_collision(dataCD))
        
        #self.sensorRGB.listen(lambda dataRGB: self.process_img(dataRGB))
        
        self.sensorIMU.listen(lambda dataIMU: self.process_IMU(dataIMU))

    #- DATA PROCESSING --------------------------------------------------------------------------
    # radar
    def process_data(self, data):
        img = data.raw_data
        self.points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        self.points = np.reshape(self.points, (len(data), 4))
        if len(self.points) > 0:
            #self.depth = statistics.mean([item[3] for item in self.points])
            #self.depth = min([item[3] for item in self.points])
            self.azimuth = statistics.mean([item[2] for item in self.points])*(180/math.pi)
            self.detect_count = data.get_detection_count()
            vel = [item[0] for item in self.points]
            maxvel = min(vel)
            maxveldata = self.points[[i for i,x in enumerate(vel) if x==maxvel]]
            #print(maxveldata)
            azi_of_maxvel = [item[2] for item in maxveldata][0]*180/math.pi
            #print(maxvel)
            #print(azi_of_maxvel)
            
        #print(f"Azimuth is {self.azimuth}")
        #print(f"Depth is {self.depth}")
        #print(self.detect_count)
        return

    # obstacle detection
    def register_obstacle(self, data):
        self.Obdist = data.distance
        #return data
        #self.obstacle = data
        
    def register_obstacleL(self, data):
        self.ObdistL = data.distance
        #print(f"Left side collision: {self.ObdistL}")
        #return data
        #self.obstacle = data
    
    def register_obstacleR(self, data):
        self.ObdistR = data.distance
        #print(f"Right side collision: {self.ObdistR}")
        #return data
        #self.obstacle = data
    
    # collision detection
    def register_collision(self, data):
        collided_with = data.other_actor
        print(collided_with)
        self.collision_hist.append(data)
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0, reverse=True))
        #return collided_with
        
    # IMU
    def process_IMU(self, data):
        gyro = data.gyroscope
        self.zrate = gyro.z*180/math.pi
        #print(gyro)
        #print(self.zrate)
        self.spectator.set_transform(self.sensorRGB.get_transform())
        
    # returns vehicle speed, 
    def get_obs(self):
        self.v = self.vehicle.get_velocity()
        self.kmh = int(3.6*math.sqrt(self.v.x**2 + self.v.y**2 + self.v.z**2))
        return [self.kmh, self.Obdist, self.ObdistR, self.ObdistL, len(self.collision_hist)]

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
        if self.Obdist < 25 and action == 1:
            done = False
            reward = -30
        elif self.Obdist < 15 and action == 1:
            done = False
            reward = -200
        elif self.ObdistL < 20 and (action == 0 or action == 4 or action == 6):
            reward = -50
            done = False
        elif self.ObdistL < 10 and (action == 0 or action == 4 or action == 6):
            reward = -500
            done = False
        elif self.ObdistR < 20 and (action == 2 or action == 3 or action == 5):
            reward = -50
            done = False
        elif self.ObdistR < 10 and (action == 2 or action == 3 or action == 5):
            reward = -500
            done = False
        else:
            done = False
            reward = 0
            
        if len(self.collision_hist) != 0:
            done = True
            reward = reward - 500
            
        obs =  [self.kmh, self.Obdist, self.ObdistR, self.ObdistL, len(self.collision_hist)]  
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
        # max_speed, min_speed, Sbins, max_dist, min_dist, Dbins, max_distside, min_distside, DSbins, nAction
        agent = QAgent(80, 0, 10, 30, 0, 10, 20, 0, 10, 5)
    elif sarsa == 1:
        # max_speed, min_speed, Sbins, max_dist, min_dist, Dbins, max_distside, min_distside, DSbins, nAction
        agent = SARSA_agent(80, 0, 10, 30, 0, 10, 20, 0, 10, 5)
    elif sarsa_lambda == 1:
        # max_speed, min_speed, Sbins, max_dist, min_dist, Dbins, max_distside, min_distside, DSbins, nAction, lambda
        agent = SLambda_agent(80, 0, 10, 30, 0, 10, 20, 0, 10, 5, 0.5)

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
                    #time.sleep(1/FPS)
                    print(f"Action {action} was taken.")
                else:
                    action = np.random.randint(0, 5)
                    print(f"Random action {action} was taken.")
                    #time.sleep(1/FPS)
            elif sarsa == 1 or sarsa_lambda == 1:
                if np.random.uniform() > EPSILON:
                    action = agent.lastexp[1]
                    #time.sleep(1/FPS)
                    print(f"Action {action} was taken.")
                else:
                    action = np.random.randint(0, 5)
                    #time.sleep(1/FPS)
                    print(f"Random action {action} was taken.")
            
            # advance controls with that action
            new_state, reward, done, _ = env.step(action)

            #Update Q_table
            if done:
                break
            else:
                if qlearning == 1:
                    agent.update_Q(current_state, reward, action, new_state)
                elif sarsa == 1:
                    agent.sarsa_update(reward, action, new_state)
                elif sarsa_lambda == 1:
                    agent.sarsa_lambda_update(reward, action, new_state)

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
