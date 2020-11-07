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

IM_WIDTH = 640
IM_HEIGHT = 480
LEARNING_RATE = 0.6
DISCOUNT = 0.95
EPSILON = 0.3
SHOW_PREVIEW = False
NUM_EPISODES = 1000
crashed = 1

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()

        self.blueprint_library = self.world.get_blueprint_library()

        self.bp = self.blueprint_library.filter('model3')[0]
        print(self.bp)
    
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
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0, reverse=True))
        #return collided_with
        
    def process_IMU(self, data):
        gyro = data.gyroscope
        self.zrate = gyro.z*180/math.pi
        #print(gyro)
        #print(self.zrate)
        self.spectator.set_transform(self.sensorRGB.get_transform())
                
    def get_obs(self):
        self.v = self.vehicle.get_velocity()
        self.kmh = int(3.6*math.sqrt(self.v.x**2 + self.v.y**2 + self.v.z**2))
        return [self.kmh, self.Obdist, self.ObdistR, self.ObdistL, len(self.collision_hist)]

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=.8, steer=-.1))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=.8, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=.8, steer=.1))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=.1))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-.1))
        # elif action == 5:
            # self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=.2))
        # elif action == 6:
            # self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=-.2))
        # elif action == 7:
            # self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=.2, brake=0.5))
        # elif action == 8:
            # self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=-.2, brake=0.5))
        # elif action == 9:
            # self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=.2, brake=1))
        # elif action == 10:
            # self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
        else:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=-.1, brake=1))
        
        self.v = self.vehicle.get_velocity()
        self.kmh = int(3.6*math.sqrt(self.v.x**2 + self.v.y**2 + self.v.z**2))

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

class QAgent:
    def __init__(self):
        self.Q_table = np.ones((9*4*3*3, 5))
        for a in range(9):
            self.Q_table[:, 1] = np.ones(9*4*3*3) * 5
        
    #Outputs vector Q(s,a) for every possible a
    def update_Q(self, obs, reward, action, next_state):
    
        #Current states
        self.speed = math.floor(obs[0]/10)
        if self.speed > 8:
            self.speed = 8
            
        self.dist = math.floor(obs[1]/10)
        if self.dist > 3:
            self.dist = 3
            
        self.distR = math.floor(obs[2]/10)
        if self.distR >= 2:
            self.distR = 2
        
        self.distL = math.floor(obs[3]/10)
        if self.distL >= 2:
            self.distL = 2
            
        # self.rev = obs[2]
        self.col = obs[4]
        if self.col > 1:
            self.col = 1
        
        # self.zr = math.floor(obs[4]/5)
        # if self.zr < -2:
            # self.zr = -2
        # elif self.zr >= 2:
            # self.zr = 1
            
        # self.zr = self.zr + 2
        
        
        # Separating next states
        self.speednex = math.floor(next_state[0]/10)
        if self.speednex > 8:
            self.speednex = 8
            
        self.distnex = math.floor(next_state[1]/10)
        if self.distnex > 3:
            self.distnex = 3
            
        self.distRnex = math.floor(next_state[2]/10)
        if self.distRnex >= 2:
            self.distRnex = 2
        
        self.distLnex = math.floor(next_state[3]/10)
        if self.distLnex >= 2:
            self.distLnex = 2
            
        # self.revnex = next_state[2]
        self.colnex = next_state[4]
        if self.colnex > 1:
            self.colnex = 1
            
        # self.zrnex = math.floor(next_state[4]/5)
        # #print(self.zrnex)
        # if self.zrnex < -2:
            # self.zrnex = -2
        # elif self.zrnex >= 2:
            # self.zrnex = 1
            
        # self.zrnex = self.zrnex + 2
        
        #Convert to linear index
        self.state = np.ravel_multi_index((self.speed, self.dist, self.distR, self.distL), (9,4,3,3))
        self.nex_state = np.ravel_multi_index((self.speednex, self.distnex, self.distRnex, self.distLnex), (9,4,3,3))
        
        # Calculate Q(s, a) <- Q(s, a) + LEARNING_RATE*(r + DISCOUNT*max(Q(s', a') - Q(s, a)))
        self.Q_table[self.state, action] = self.Q_table[self.state, action] + LEARNING_RATE*(reward + DISCOUNT*max(self.Q_table[self.nex_state, :]) - self.Q_table[self.state, action])
        return self.Q_table[self.state, :]
        
    def get_Q(self, obs):
        self.speed = math.floor(obs[0]/10)
        if self.speed > 8:
            self.speed = 8
            
        self.dist = math.floor(obs[1]/10)
        if self.dist > 3:
            self.dist = 3
        
        self.distR = math.floor(obs[2]/10)
        if self.distR >= 2:
            self.distR = 2
        
        self.distL = math.floor(obs[3]/10)
        if self.distL >= 2:
            self.distL = 2
            
        #self.rev = obs[2]
        self.col = obs[4]
        if self.col > 1:
            self.col = 1
            
        #self.zr = math.floor(obs[4]/5)
        #if self.zr < -2:
        #    self.zr = -2
        #elif self.zr >= 2:
        #    self.zr = 1
            
        #self.zr = self.zr + 2
        
        self.state = np.ravel_multi_index((self.speed, self.dist, self.distR, self.distL), (9,4,3,3))
        
        return self.Q_table[self.state, :]

if __name__ == "__main__":
       
    FPS = 1
    agent = QAgent()
    env = CarEnv()
    
    if crashed == 1:
        agent.Q_table = np.load('Q_table.npy')

    for i in range(NUM_EPISODES):
        env.reset()
        if i > 50:
            EPSILON = 0.1
        elif i > 100:
            EPSILON = 0.05
        else:
            EPSILON = 0.01
        while True:
            current_state = env.get_obs()
            #print([current_state[i] for i in [0, 1, 2, 3]])
            #Optimize for the best action
            if np.random.random() > EPSILON:
                curr_qs = agent.get_Q(current_state)
                m = max(curr_qs)
                #print(curr_qs)                
                if len([i for i,x in enumerate(curr_qs) if x==m]) > 1:
                    action = np.random.choice([i for i,x in enumerate(curr_qs) if x==m])
                else:
                    action = np.argmax(curr_qs)
                #print(f"Action {action} was taken.")
            else:
                action = np.random.randint(0, 5)
                #print(f"Random action {action} was taken.")
            
            #Advance controls with that action
            new_state, reward, done, _ = env.step(action)
          
            #time.sleep(1/FPS)
            #Update Q_table
            
            if done:
                break
            else:
                agent.update_Q(current_state, reward, action, new_state)
                
        for actor in env.actor_list:
            actor.destroy()
        print(f"Episode {i}")
        
        if (i % 25 == 0):
            np.save('Q_table', agent.Q_table)