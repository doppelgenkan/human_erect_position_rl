"""
H. Tachibana
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
        'video.frames_per_second' : 200
    }

    def __init__(self):
        self.gravity = 9.8

        ##### input below parameters #####
        self.total_mass = 65.7
        self.length_GP = 1.723 * 0.56  # actually 56% of the pole's length
        self.length_AP = 0.06                           # Ankle hight
        self.length = self.length_GP - self.length_AP   # length to COG from Ankle
        self.masscart = self.total_mass * 0.014 * 2
        self.masspole = self.total_mass - self.masscart #0.1
        self.polemass_length = (self.masspole * self.length)
        delta = 2.8
        self.i_moment = self.masspole * self.length**2 * (1.0/3.0 + delta)

        # Ajust belowe parameters
        self.tau = 0.001
        self.epsilon = 0.01

        ##### added by me #####
        #----- Pssive Torque Parameters by Misgeld et al. 2017 --------
        self.k1 = 5.63682
        self.k2 = 5.09939
        self.beta = math.pi / 10
        self.T_passive0_mag = 5.6744 * 2 * (self.length / self.length_AP)
        self.T_passive0_mag = abs( 2 * (self.length / self.length_AP) * (-math.exp(self.k1 * self.beta) + math.exp(-self.k2 * self.beta)))  # ~172.32
        #--------------------------------------------------------------

        # Angle at which to fail the episode
        self.theta_threshold_radians_front = math.radians(5.1)
        self.theta_threshold_radians_back = math.radians(3.2)
        self.theta_threshol_region = math.radians(1)
        self.theta_threshold_radians = self.theta_threshold_radians_front
        self.x_threshold_front = 0.01
        self.x_threshold_back = 0.01
        self.x_threshold = self.x_threshold_front

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
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
        self.action_space = spaces.Discrete(6)   #Discrete(6) takes a value of 0, 1, 2, 3, 4 or 5.
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
        ttl, theta0, theta_dot0, thetaacc0, T_active, torque0 = state

        costheta = math.cos(theta0)
        sintheta = math.sin(theta0)

        T_base = self.T_passive0_mag
        rho = np.random.uniform(low=-0.03, high=0.03)

        if ttl == 0:
            ttl = np.random.randint(20, 41)
            if action == 0:
                alpha = -15
            elif action == 1:
                alpha = -9
            elif action == 2:
                alpha = -3
            elif action == 3:
                alpha = 3
            elif action == 4:
                alpha = 9
            else:
                alpha = 15
            T_active = T_base * (1 + self.epsilon * alpha + rho)
        ttl -= 1

        T_passive = 2 * (self.length / self.length_AP) * (-math.exp(self.k1 * (theta0 + self.beta)) + math.exp(-self.k2 * (theta0 + self.beta)))
        torque = T_active + T_passive
        thetaacc = (self.gravity * self.polemass_length * sintheta + torque) / (self.masspole * self.length**2 + self.i_moment)

        theta_dot = theta_dot0 + self.tau * (thetaacc0 + thetaacc) / 2  ### added
        theta = theta0 + self.tau * (theta_dot0 + theta_dot) / 2
        self.state = (ttl, theta, theta_dot, thetaacc, T_active, torque)

        done =  theta < -math.radians(3.2) or theta > math.radians(5.1)
        done = bool(done)

        if math.radians(-0.9) <= theta <= math.radians(1.3):
            reward = 3.0
        elif math.radians(-1.3) <= theta <= math.radians(1.9):
            reward = 2.0
        elif math.radians(-1.9) <= theta <= math.radians(2.5):
            reward = 1.0
        elif not done:
            reward = -1.0
        elif done:
            reward = -100.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}


    def reset(self):
        self.state = [0,
                      0,
                      self.np_random.uniform(low=-math.radians(2), high=math.radians(2)),
                      0,
                      self.T_passive0_mag,
                      0]

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
