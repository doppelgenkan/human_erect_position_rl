"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random

class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        #'video.frames_per_second' : 50
        'video.frames_per_second' : 200 #100
    }

    def __init__(self):
        self.gravity = 9.8

        # self.masscart = 1.0

        ##### input below data  #####
        self.total_mass = 69.6
        self.length_GP = 1.734 * 0.56 #0.5 # actually half the pole's length
        self.length_AP = 0.06                           # Ankle hight
        self.length = self.length_GP - self.length_AP   # length to COG from Ankle
        self.masscart = self.total_mass * 0.014 * 2
        self.masspole = self.total_mass - self.masscart #0.1
        self.polemass_length = (self.masspole * self.length)
        self.i_moment = self.masspole * self.length**2 * (1.0/3.0 + (3.0/25.0)**2)

        # Ajust belowe parameters
        self.tau = 0.001  # 0.02 # seconds between state updates -> 0.005 -> 0.001
        self.series = 25  # self.series * self.tau is the keeping time(sec) of the active-torque. #50

        self.odr = abs(int(np.log10(self.tau)))
        self.kappa = 0.2 * 0.7 #0.2 * 0.5  # weight of torque added to the default active torque
        ##### up to here #####

        ##### added by me #####
        #----- Case 3. Pssive Torque Parameters by Misgeld et al. 2017
        self.k1 = 5.63682
        self.k2 = 5.09939
        self.T_passive0_mag = 5.6744 * 2 * (self.length / self.length_AP)  # ~171.9
        #--------------------------------------------------------------

        # Angle at which to fail the episode
        #self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.theta_threshold_radians_front = math.radians(4) #4 * math.pi / 180 #5*
        self.theta_threshold_radians_back = math.radians(3) #2 * math.pi / 180 #3*
        self.theta_threshol_region = math.radians(1)
        self.theta_threshold_radians = self.theta_threshold_radians_front
        self.x_threshold_front = 0.01 #0.021
        self.x_threshold_back = 0.01 #0.014
        self.x_threshold = self.x_threshold_front #0.03 #0.015 #2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        """
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])
        """
        high_back = np.array([
            np.finfo(np.float32).max,
            self.theta_threshold_radians_back,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])
        high_front = np.array([
            np.finfo(np.float32).max,
            self.theta_threshold_radians_front,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])
        self.action_space = spaces.Discrete(7)   #Discrete(2) takes a value of 0 or 1.
        #self.observation_space = spaces.Box(-high, high)
        self.observation_space = spaces.Box(-high_back, high_front)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        #x0, x_dot0, theta0, theta_dot0, thetaacc0, torque0 = state
        #t0, theta0, theta_dot0, thetaacc0, cop0, torque0 = state
        series, theta0, theta_dot0, thetaacc0, T_acctive, torque0 = state

        costheta = math.cos(theta0)
        sintheta = math.sin(theta0)

        ##### added by me #####
        #self.torque_mag = abs(self.total_mass * np.random.normal(0, 1.17)) / 10
        #self.add_torque_mag = self.kappa * self.T_passive0_mag * (0.5 + np.random.rand()) # * np.random.rand()    #additional active torque
        #self.add_torque_mag = self.masspole * np.random.normal(0, 1.18) * self.length # *self.kappa
        #self.add_torque_mag = self.masspole * np.random.normal(0, 1.18/2) * self.length # *self.kappa
        #self.add_torque_mag = self.kappa * self.masspole * (np.random.random()-0.5) * self.length
        #self.add_torque_mag = self.kappa * self.masspole * (np.random.random()) * self.length
        #self.add_torque_mag = self.kappa * self.masspole * self.length #np.random.uniform(low=0.3, high=0.7)
        add_torque_mag = self.T_passive0_mag * 0.1

        ##### up to here #####

        #if int( ( t/self.tau ) % self.series ) == 0:
            #T_acctive = self.T_passive0_mag + self.add_torque_mag if action==1 else self.T_passive0_mag - self.add_torque_mag
        base_T_passive_mag = self.T_passive0_mag * (1 + np.random.uniform(low=-0.01, high=0.01))
        if series == 0:
            series = 25
            if action == 0:
                T_acctive = base_T_passive_mag - add_torque_mag * 1.5
            elif action == 1:
                T_acctive = base_T_passive_mag - add_torque_mag * 1.25
            elif action == 2:
                T_acctive = base_T_passive_mag - add_torque_mag
            elif action == 3:
                T_acctive = base_T_passive_mag
            elif action == 4:
                T_acctive = base_T_passive_mag + add_torque_mag
            elif action == 5:
                T_acctive = base_T_passive_mag + add_torque_mag * 1.25
            else:
                T_acctive = base_T_passive_mag + add_torque_mag * 1.5

        #T_acctive = self.T_passive0_mag + self.add_torque_mag if action==1 else self.T_passive0_mag - self.add_torque_mag

        #temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass

        ##### added by me #####        #T_passive = 2 * (self.length / self.length_AP) * (2 * self.beta_2 * math.sinh(self.k_2 * theta))   # first coefficient is length_COG/(length_foot/2)
        T_passive = 2 * (self.length / self.length_AP) * (-math.exp(self.k1 * (theta0 + math.pi / 10)) + math.exp(-self.k2 * (theta0 + math.pi / 10)))
        torque = T_acctive + T_passive
        thetaacc = (self.gravity * self.polemass_length * sintheta + torque) / (self.masspole * self.length**2 + self.i_moment)
        ##### up to here #####

        ##### added by me <- The integral accomplished by making use of the method of trapezoid sum #####
        theta_dot = theta_dot0 + self.tau * (thetaacc0 + thetaacc) / 2  ### added
        theta = theta0 + self.tau * (theta_dot0 + theta_dot) / 2

        #cop = self.length * sintheta
        #x_dot = x_dot0
        #x = self.length * sintheta
        #x_dot = self.length * theta_dot * costheta
        ##### up to here #####

        #self.state = (t, theta, theta_dot, thetaacc, cop, torque)
        series -= 1
        self.state = (series, theta, theta_dot, thetaacc, T_acctive, torque)
        '''
        done =  x < -self.x_threshold_back \
                or x > self.x_threshold_front \
                or theta < -self.theta_threshold_radians_back \
                or theta > self.theta_threshold_radians_front
        '''
        done =  theta < -self.theta_threshold_radians_back \
                or theta > self.theta_threshold_radians_front
        done = bool(done)

        #if not done:
            #reward = 1.0
        ##### added by me #####
        #if (-self.theta_threshold_radians_back/2 <= theta <= self.theta_threshold_radians_front/2) \
            #and (-self.x_threshold_back/2 <= x <= self.x_threshold_front/2):
        #if (-self.theta_threshold_radians_back/2.5 <= theta <= self.theta_threshold_radians_front/2.5): # add
            #reward = 3.0 #2.0                                                                           # add
        #if (-self.theta_threshold_radians_back/2 <= theta <= self.theta_threshold_radians_front/2):
            #reward = 2.0
        if (-self.theta_threshold_radians_back/3 <= theta <= self.theta_threshold_radians_front/3):
            reward = 3.0
        elif (-self.theta_threshold_radians_back/2 <= theta <= self.theta_threshold_radians_front/2):
            reward = 2.0
        elif (-self.theta_threshold_radians_back/1.5 <= theta <= self.theta_threshold_radians_front/1.5):
            reward = 1.0
        elif not done:
            reward = 0.0 #-1.0
        elif done:
            reward = -100.0 #-100.0
        ##### up to here #####
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0 #1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}


    def reset(self):
        ##### added by me #####
        theta_init = self.np_random.uniform(low=-math.radians(1), high=math.radians(1))
        #theta_init = 0
        #cop_init = self.length * math.sin(theta_init)
        self.state = \
        [0,
         theta_init,
         self.np_random.uniform(low=-math.radians(1), high=math.radians(1)),
         0,
         self.T_passive0_mag,
         self.T_passive0_mag + 2 * (self.length / self.length_AP) * (-math.exp(self.k1 * (theta_init + math.pi/10)) + math.exp(-self.k2 * (theta_init + math.pi/10)))]
        ##### up tp here #####

        self.steps_beyond_done = None
        return np.array(self.state)


    def render(self, mode='human'):
        screen_width = 200 #600
        screen_height = 800 #400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0 #10.0
        polelen = scale * 0.05 #scale * 1.0 -> scale * 0.1
        cartwidth = 50.0 # 100.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = screen_width/2.0 # x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer: self.viewer.close()
