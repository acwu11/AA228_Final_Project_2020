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
import numpy as np
import cv2
import statistics
import math

from model_free import *

IM_WIDTH = 640
IM_HEIGHT = 480
LEARNING_RATE = 0.9
DISCOUNT = 0.9
EPSILON = 0.3
SHOW_PREVIEW = True
NUM_EPISODES = 200

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    #STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    control_diff = 0.4
    
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)

        self.world = self.client.get_world()

        self.blueprint_library = self.world.get_blueprint_library()

        self.bp = self.blueprint_library.filter('model3')[0]
        print(self.bp)
    
    def reset(self):
        self.actor_list = []
        self.collision_hist = []
        self.steering_amt = 0
        self.throttle_amt = 0
        self.depth = 50
        
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
        self.blueprintCD = self.blueprint_library.find('sensor.other.collision')
        self.blueprintRGB = self.blueprint_library.find('sensor.camera.rgb')
        
        # change the dimensions of the image
        self.blueprintRGB.set_attribute('image_size_x', f'{self.im_width}')
        self.blueprintRGB.set_attribute('image_size_y', f'{self.im_height}')
        self.blueprintRGB.set_attribute('fov', '110')

        # Adjust sensor relative to vehicle
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

        # spawn the sensor and attach to vehicle.
        self.sensor = self.world.spawn_actor(self.blueprint, spawn_point, attach_to=self.vehicle)
        self.sensorOD = self.world.spawn_actor(self.blueprintOD, spawn_point, attach_to=self.vehicle)
        self.sensorCD = self.world.spawn_actor(self.blueprintCD, spawn_point, attach_to=self.vehicle)
        self.sensorRGB = self.world.spawn_actor(self.blueprintRGB, spawn_point, attach_to=self.vehicle)

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
        
        
    #May have to add a reset function too if we do episodic training
    
    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3


    def process_data(self, data):
        img = data.raw_data
        self.points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        self.points = np.reshape(self.points, (len(data), 4))
        self.depth = statistics.mean([item[3] for item in self.points])
        self.detect_count = data.get_detection_count()
            
        #print(self.depth)
        #print(self.detect_count)
        return self.depth
        
    def register_obstacle(self, data):
        #print(data)
        #return data
        self.obstacle = data
        
    def register_collision(self, data):
        collided_with = data.other_actor
        print(collided_with)
        self.collision_hist.append(data)
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0, reverse=True))
        #return collided_with
        
    def get_obs(self):
        self.v = self.vehicle.get_velocity()
        self.kmh = int(3.6*math.sqrt(self.v.x**2 + self.v.y**2 + self.v.z**2))
        return [self.kmh, self.depth, 0,  len(self.collision_hist)]  

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-.1))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=.1))
                
        # if action == 1:
            # self.throttle_amt = self.control_diff
            # self.steering_amt = self.control_diff
                
        # # Subtract throttle
        # elif action == 2:
            # self.throttle_amt = -self.control_diff
            # self.steering_amt = self.control_diff
        # # Add to steering.
        # elif action == 3:
            # self.throttle_amt = self.control_diff
            # self.steering_amt = -self.control_diff
                

        # # Subtract steering
        # elif action == 4:
            # self.throttle_amt = -self.control_diff
            # self.steering_amt = -self.control_diff

            
        # # Add throttle and steering.
        # elif action == 5:
            # self.throttle_amt = self.control_diff
            # self.steering_amt = 0
                

        # # Add throttle substract steering
        # elif action == 6:
            # self.throttle_amt = -self.control_diff
            # self.steering_amt = 0

        # # Subtract throttle add steering
        # elif action == 7:
            # self.throttle_amt = 0
            # self.steering_amt = self.control_diff

        # # Subtract throttle and steering
        # elif action == 8:
            # self.throttle_amt = 0
            # self.steering_amt = -self.control_diff
            
        # else:
            # self.throttle_amt = 0
            # self.steering_amt = 0
        
        # if self.throttle_amt > 0:
            # self.vehicle.apply_control(carla.VehicleControl(throttle=self.throttle_amt, steer=self.steering_amt, brake=0.0))
        # else:
            # self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=self.steering_amt, brake=abs(self.throttle_amt)))
        
        self.v = self.vehicle.get_velocity()
        self.kmh = int(3.6*math.sqrt(self.v.x**2 + self.v.y**2 + self.v.z**2))
        
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
            
        elif self.kmh < 1:
            done = False
            reward = -10
            
        elif self.kmh > 70:
            done = False
            reward = -1
            
        else:
            done = False
            reward = 10
            
        reward = reward - 100*(1 - self.depth/60)
        # if self.episode_start + SECONDS_PER_EPISODE < time.time():
            # done = True
        obs =  [self.kmh, self.depth, 0,  len(self.collision_hist)]  
        #print(self.depth)
        return obs, reward, done, action

if __name__ == "__main__":
    
    # which method
    qlearning = 0
    sarsa = 0
    sarsa_lambda = 1

    FPS = 5

    if qlearning == 1:
        agent = QAgent(80, 0, 50, 0)
    elif sarsa == 1:
        agent = SARSA_agent(80, 0, 50, 0)
    elif sarsa_lambda == 1:
        agent = SLambda_agent(80, 0, 50, 0, 0.5)

    env = CarEnv()

    #agent.get_qs(np.ones((env.IM_HEIGHT, env.IM_WIDTH, 3)))
    for i in range(NUM_EPISODES):
        env.reset()
        while True:
            current_state = env.get_obs()
            
            # Optimize for the best action
            if qlearning == 1: 
                if np.random.random() > EPSILON:
                    action = np.argmax(agent.get_Q(current_state))
                    time.sleep(1/FPS)
                    print(f"Action {action} was taken.")
                else:
                    action = np.random.randint(0, 2)
                    #print("Random action was taken.")
                    time.sleep(1/FPS)
            elif sarsa == 1 or sarsa_lambda == 1:
                if np.random.random() > EPSILON:
                    action = agent.lastexp[1]
                    time.sleep(1/FPS)
                else:
                    action = np.random.randint(0, 2)
                    time.sleep(1/FPS)
            
            #Advance controls with that action
            new_state, reward, done, _ = env.step(action)
            #print(new_state)
            
            #Update Q_table
            if qlearning == 1:
                agent.update_Q(current_state, reward, action, new_state)
            #print(f"Reward is {reward}")
            elif sarsa == 1:
                agent.sarsa_update(reward, new_state)
            elif sarsa_lambda == 1:
                agent.sarsa_lambda_update(reward, new_state)
            
            if done:
                break
            
        for actor in env.actor_list:
            actor.destroy()
        print(f"Episode {i}")