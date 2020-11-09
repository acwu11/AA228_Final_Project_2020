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

from representation import *

IM_WIDTH = 640
IM_HEIGHT = 480
SHOW_PREVIEW = False

class Test:
    SHOW_CAM = SHOW_PREVIEW
    #STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    control_diff = 0.4

    # intialize world, ego vehicle, other vehicles and testing parameters
    def __init__(self, method, action_file=None, others=0):
        # store test being run
        self.method = method
        self.others = others
        print("Running Test on " + method + " policy")

        # initialize world and spectator
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()
        self.actor_list = []

        # initialize ego car info
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

        # create state space representation and load Q table
        self.saRep = Descritize(80, 0, 10, 30, 0, 10, 20, 0, 10, 5)
        if self.method != 'auto':
            self.Q_table = np.load(action_file)    # load extracted policy
        
        # test metrics for single episode
        self.epReward = 0
        self.epCollidedWith = ''
    
    # reset environment
    def reset(self, test_ind):
        self.actor_list = []
        self.collision_hist = []
        self.epReward = 0

        self.depth = 50
        
        self.Obdist = 20
        self.ObdistL = 15
        self.ObdistR = 15
        
        self.azimuth = 0
        self.zrate = 0
        
        # spawn at spawn point corresponding to test index
        self.spawn_point = (self.world.get_map().get_spawn_points())[test_ind]

        # spawn vehicle at chosen spawn point and append to actor list
        self.ego_vehicle = self.world.spawn_actor(self.bp, self.spawn_point)
        self.actor_list.append(self.ego_vehicle)
        if self.method == 'auto':
            self.ego_vehicle.set_autopilot(True)
        else:
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
        time.sleep(1)
        print('Added ego vehicle %s' %self.ego_vehicle.type_id)

        # initialize other random cars in environment
        if self.others != 0: 
            for _ in range(0, 10):
                transform = random.choice(self.world.get_map().get_spawn_points())
                bp = random.choice(self.blueprint_library.filter('vehicle'))
                npc = self.world.try_spawn_actor(bp, transform)
                if npc is not None:
                    self.actor_list.append(npc)
                    npc.set_autopilot(True)
                    print('created %s' % npc.type_id)

        # get sensors to attach to ego vehicle
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
        self.sensorOD = self.world.spawn_actor(self.blueprintOD, spawn_pointD, attach_to=self.ego_vehicle)
        self.sensorODR = self.world.spawn_actor(self.blueprintODside, spawn_pointR, attach_to=self.ego_vehicle)
        self.sensorODL = self.world.spawn_actor(self.blueprintODside, spawn_pointL, attach_to=self.ego_vehicle)
        
        self.sensorCD = self.world.spawn_actor(self.blueprintCD, spawn_point, attach_to=self.ego_vehicle)
        self.sensorRGB = self.world.spawn_actor(self.blueprintRGB, spawn_pointcam, attach_to=self.ego_vehicle)
        self.sensorIMU = self.world.spawn_actor(self.blueprintIMU, spawn_point, attach_to=self.ego_vehicle)

        # add sensor to list of actors
        self.actor_list.append(self.sensorOD)
        self.actor_list.append(self.sensorODL)
        self.actor_list.append(self.sensorODR)
        self.actor_list.append(self.sensorCD)
        self.actor_list.append(self.sensorRGB)
        self.actor_list.append(self.sensorIMU)

        # do something with this sensor
        self.sensorOD.listen(lambda dataOD: self.register_obstacle(dataOD))   
        self.sensorODL.listen(lambda dataODL: self.register_obstacleL(dataODL))
        self.sensorODR.listen(lambda dataODR: self.register_obstacleR(dataODR))
        self.sensorCD.listen(lambda dataCD: self.register_collision(dataCD))
        self.sensorIMU.listen(lambda dataIMU: self.process_IMU(dataIMU))
        
    #- DATA PROCESSING --------------------------------------------------------------------------
    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3
        
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
        
    def register_collision(self, data):
        collided_with = data.other_actor
        print(collided_with)
        self.collision_hist.append(data)

        self.epCollidedWith = collided_with
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0, reverse=True))
        #return collided_with
        
    def process_IMU(self, data):
        gyro = data.gyroscope
        self.zrate = gyro.z*180/math.pi
        self.spectator.set_transform(self.sensorRGB.get_transform())
                
    def get_obs(self):
        self.v = self.ego_vehicle.get_velocity()
        self.kmh = int(3.6*math.sqrt(self.v.x**2 + self.v.y**2 + self.v.z**2))
        return [self.kmh, self.Obdist, self.ObdistR, self.ObdistL, len(self.collision_hist)]

    def step(self, action):
        # given action apply throttle and steer
        t = self.ActionDict[action][0]
        s = self.ActionDict[action][1]
        b = self.ActionDict[action][2]
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=t, steer=s, brake=b))
        
        # car velocity
        self.v = self.ego_vehicle.get_velocity()
        self.kmh = int(3.6*math.sqrt(self.v.x**2 + self.v.y**2 + self.v.z**2))

        # reward
        if 15 < self.Obdist < 25 and action == 1:
            done = False
            reward = -30
        elif self.Obdist < 15 and action == 1:
            done = False
            reward = -200
        elif 10 < self.ObdistL < 20 and  (action == 0 or action == 4 or action == 6):
            reward = -50
            done = False
        elif self.ObdistL < 10 and (action == 0 or action == 4 or action == 6):
            reward = -500
            done = False
        elif 10 < self.ObdistR < 20 and (action == 2 or action == 3 or action == 5):
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
            
        return reward, done
    
    # pseudo reward for the auto pilot
    def step_auto(self):
        self.v = self.ego_vehicle.get_velocity()
        self.kmh = int(3.6*math.sqrt(self.v.x**2 + self.v.y**2 + self.v.z**2))

        # reward
        if 15 < self.Obdist < 25:
            done = False
            reward = -30
        elif self.Obdist < 15:
            done = False
            reward = -200
        elif 10 < self.ObdistL < 20:
            reward = -50
            done = False
        elif self.ObdistL < 10:
            reward = -500
            done = False
        elif 10 < self.ObdistR < 20:
            reward = -50
            done = False
        elif self.ObdistR < 10:
            reward = -500
            done = False
        else:
            done = False
            reward = 0
            
        if len(self.collision_hist) != 0:
            done = True
            reward = reward - 500

        return reward, done
    
     #- Qtable --------------------------------------------------------------------------
    def get_Q(self, obs):
        state = self.saRep.get_state_ind(obs[0], obs[1], obs[2], obs[3])
        return self.Q_table[state, :]

if __name__ == "__main__":
    # set method: test output files will be saved starting with method string
    #   'auto' for autopilot
    #   'Qlearning for Q Learning'
    #   'sarsa' for sarsa
    #   'sarsaL' for sarsa lambda
    method = 'sarsa'
    if method == 'auto':
        test = Test(method, others=1)

    # TODO: fill in correct action file (q-table) file for corresponding method
    else:
        test = Test(method, action_file='sarsa.npy',others=1)

    # test metrics and storage
    numTest = 150
    ep_reward = np.zeros(numTest)
    ep_col_times = np.zeros(numTest)
    ep_times =  np.zeros(numTest)
    ep_col_items = np.zeros(numTest)

    # for k in range(numTest):
    print(f"Running {numTest} Tests")
    for k in range(numTest):
        print(f"Starting Test {k}")
        stop = 0
        t_epStart = process_time()
        test.reset(k)
        while not stop:
            current_state = test.get_obs()

            # if vehicle is near traffic light, set traffic light to green
            try:
                if test.ego_vehicle.is_at_traffic_light():
                    traffic_light = test.ego_vehicle.get_traffic_light()
                    if traffic_light.get_state() == carla.TrafficLightState.Red:
                        traffic_light.set_state(carla.TrafficLightState.Green)
            except AttributeError:
                pass

            # execute optimal action
            if method == 'auto':
                reward, done = test.step_auto()
            else:
                curr_qs = test.get_Q(current_state)
                action = np.argmax(curr_qs)
                reward, done = test.step(action)

            test.epReward += reward
            t_epEnd = process_time()
            if done:
                print("Collision, starting next test")
                ep_col_times[k] = t_epEnd - t_epStart
                break
            elif t_epEnd - t_epStart >= 65:
                print("Times up, starting next test")
                stop = 1
        
        ep_times[k] = t_epEnd - t_epStart
        ep_reward[k] = test.epReward
        ep_col_items[k] = test.epCollidedWith

        # save metrics
        print('Saving Metrics')
        np.save(f'{method}_reward.npy', ep_reward)
        np.save(f'{method}_epTimes.npy', ep_times)
        np.save(f'{method}_collision.npy', ep_col_times)
        np.save(f'{method}_collItems.npy', ep_col_items)

        # clean up actors
        for actor in test.actor_list:
            try:
                actor.set_autopilot(False)
            except AttributeError:
                pass

            print('destroying actor')
            actor.destroy()


    

    
    




