'''
Created on Apr 4, 2012

@author: tambetm
'''

import msgParser
import carState
import carControl
import numpy as np
import random
import csv

from replay_memory2 import ReplayMemory
from deepqnetwork_duel import DeepQNetwork

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, args):
        '''Constructor'''
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        self.speeds = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        # self.num_inputs = 19 + 4
        self.num_inputs = 19 + 2
        self.num_speeds = len(self.speeds)
          
        self.net = DeepQNetwork(self.num_inputs, (self.num_speeds,), args)
        self.num_actions = 1
        
        self.mem = ReplayMemory(args.replay_size, self.num_inputs, self.num_actions)
        self.minibatch_size = args.batch_size

        self.target_speed = 70  # The speed we want to maintain without crashing.

        if args.load_replay:
            self.mem.load(args.load_replay)
        if args.load_weights:
            self.net.load_weights(args.load_weights)
        self.save_weights_prefix = args.save_weights_prefix
        self.save_interval = args.save_interval
        self.save_replay = args.save_replay

        self.enable_training = args.enable_training
        self.enable_exploration = args.enable_exploration

        self.total_train_steps = 0
        self.exploration_decay_steps = args.exploration_decay_steps
        self.exploration_rate_start = args.exploration_rate_start
        self.exploration_rate_end = args.exploration_rate_end
        self.skip = args.skip
        self.repeat_train = args.repeat_train

        self.show_sensors = args.show_sensors
        self.show_qvalues = args.show_qvalues

        self.steer_lock = 0.785398

        self.loss_sum = self.loss_steps = 0
        self.maxQ_sum = self.maxQ_steps = 0

        self.episode = 0
        self.distances = []
        self.onRestart()
        
        if self.show_sensors:
            from sensorstats import Stats
            self.stats = Stats(inevery=8)
        
        if self.show_qvalues:
            from plotq import PlotQ
            self.plotq = PlotQ(self.num_steers, self.num_speeds, args.update_qvalues_interval)

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})

    def getState(self):
        #state = np.array(self.state.getTrack() + [self.state.getSpeedX()] +
        #                 [self.state.getAngle()] + [self.state.getTrackPos()] +
        #                 [self.state.getSpeedX() - self.target_speed])
        state = np.array(self.state.getTrack() + [self.state.getSpeedX()] +
                         [self.state.getSpeedX() - self.target_speed])
        assert state.shape == (self.num_inputs,)
        return state

    def getReward(self, terminal):
        if terminal:
            reward = -500
        else:
            # First, get the distance away from the target speed.
            speed_diff = abs(self.target_speed - self.state.getSpeedX())

            # Try to get within 20 mph.
            # r_speed_diff = 20 - speed_diff
            if speed_diff < 1:
                reward = 500
            elif speed_diff < 5:
                reward = 100
            elif speed_diff < 20:
                reward = 0
            else:
                reward = -100

            # Substract sideways driving.
            # r_angle = 50 * abs(self.state.getAngle())

            # Subtract being off center.
            # r_track_pos = 20 * abs(self.state.getTrackPos())

            # Compile it all.
            # reward = r_speed_diff - r_angle - r_track_pos

        return reward

    def getTerminal(self):
        return np.all(np.array(self.state.getTrack()) == -1)

    def getEpsilon(self):
        # calculate decaying exploration rate
        if self.total_train_steps < self.exploration_decay_steps:
            return self.exploration_rate_start - self.total_train_steps * (self.exploration_rate_start - self.exploration_rate_end) / self.exploration_decay_steps
        else:
            return self.exploration_rate_end
 
    def drive(self, msg):
        # parse incoming message
        self.state.setFromMsg(msg)
        
        # show sensors
        if self.show_sensors:
            self.stats.update(self.state)

        # training
        if self.enable_training and self.mem.count > 0:
          for i in range(self.repeat_train):
            minibatch = self.mem.getMinibatch(self.minibatch_size)
            self.loss_sum += self.net.train(minibatch)
            self.loss_steps += 1

        # skip frame and use the same action as previously
        if self.skip > 0:
            self.frame = (self.frame + 1) % self.skip
            if self.frame != 0:
                return self.control.toMsg()

        # fetch state, calculate reward and terminal indicator
        state = self.getState()
        terminal = self.getTerminal()
        reward = self.getReward(terminal)

        # store new experience in replay memory
        if self.enable_training and self.prev_state is not None and self.prev_action is not None:
            self.mem.add(self.prev_state, self.prev_action, reward, state, terminal)

        # if terminal state (out of track), then restart game
        if terminal:
            print("Terminal, restarting.")
            self.control.setMeta(1)
            return self.control.toMsg()
        elif self.enable_training and self.state.getDistRaced() > 3608:
            print("Going too long, ending episode.")
            self.control.setMeta(1)
            return self.control.toMsg()
        else:
            self.control.setMeta(0)

        # use broadcasting to efficiently produce minibatch of desired size
        minibatch = state + np.zeros((self.minibatch_size, 1))
        Q = self.net.predict(minibatch)
        if self.show_qvalues:
            self.plotq.update(Q[0])
        self.maxQ_sum += np.max(Q[0])
        self.maxQ_steps += 1

        # choose actions for wheel and speed
        epsilon = self.getEpsilon()
        self.steer()

        if self.enable_exploration and random.random() < epsilon:
            speed = random.randrange(self.num_speeds)
        else:
            speed = np.argmax(Q[0, -self.num_speeds:])
        self.setSpeedAction(speed)
        
        # gears are always automatic
        gear = self.gear()
        self.setGearAction(gear)

        # remember state and actions 
        self.prev_state = state
        self.prev_action = np.array([speed])

        self.total_train_steps += 1

        return self.control.toMsg()

    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
    
    def gear(self):
        speed = self.state.getSpeedX()
        gear = self.state.getGear()

        if speed < 25:
            gear = 1
        elif 30 < speed < 55:
            gear = 2
        elif 60 < speed < 85:
            gear = 3
        elif 90 < speed < 115:
            gear = 4
        elif 120 < speed < 145:
            gear = 5
        elif speed > 150:
            gear = 6

        return gear

    def setSteerAction(self, steer):
        assert 0 <= steer <= self.num_steers
        self.control.setSteer(self.steers[steer])

    def setGearAction(self, gear):
        assert -1 <= gear <= 6
        self.control.setGear(gear)

    def setSpeedAction(self, speed):
        assert 0 <= speed <= self.num_speeds
        accel = self.speeds[speed]
        if accel >= 0:
            #print "accel", accel
            self.control.setAccel(accel)
            self.control.setBrake(0)
        else:
            #print "brake", -accel
            self.control.setAccel(0)
            self.control.setBrake(-accel)
    
    def onShutDown(self):
        if self.save_replay:
            self.mem.save(self.save_replay)

    def onRestart(self):
    
        self.prev_rpm = None
        self.prev_dist = None
        self.prev_state = None
        self.prev_action = None
        self.frame = -1

        if self.episode > 0:
            dist = self.state.getDistRaced()
            self.distances.append(dist)
            epsilon = self.getEpsilon()
            avgloss = self.loss_sum / max(self.loss_steps, 1)
            self.loss_sum = self.loss_steps = 0
            avgmaxQ = self.maxQ_sum / max(self.maxQ_steps, 1)
            self.maxQ_sum = self.maxQ_steps = 0
            print("Ep:", self.episode, "\tDist:", dist, "\tMax:", max(self.distances), "\tMedian10:", np.median(self.distances[-10:]), \
                "\tEps:", epsilon, "\tAvg loss:", avgloss, "\tAvg maxQ", avgmaxQ) 

            if self.save_weights_prefix and self.save_interval > 0 and self.episode % self.save_interval == 0:
                self.net.save_weights(self.save_weights_prefix + "_" + str(self.episode) + ".pkl")

        self.episode += 1
