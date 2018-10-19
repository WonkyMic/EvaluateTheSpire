import numpy as np
import tensorflow as tf
from keras import backend as K
import json
import random
import time
import pickle
import gc

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork


class Brain:
    def __init__(self, state_processor):
        self.BUFFER_SIZE = 1000
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.TAU = 0.001     # Target Network HyperParameters
        self.LRA = 0.0001    # Learning rate for Actor
        self.LRC = 0.001     # Learning rate for Critic

        self.action_dim = 21  # Target/Action
        self.state_dim = 131055  # columns in input state

        np.random.seed(1337)
        self.total_reward = 0.
        self.loss = 0

        self.EXPLORE = 100000.
        self.reward = 0
        self.done = False
        self.epsilon = 1
        self.indicator = 0

        self.train_indicator = 1  # 1 means Train, 0 means simply Run

        # Tensorflow GPU optimization
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        K.set_session(self.sess)

        self.actor = ActorNetwork(self.sess, self.state_dim, self.action_dim, self.BATCH_SIZE, self.TAU, self.LRA)
        self.critic = CriticNetwork(self.sess, self.state_dim, self.action_dim, self.BATCH_SIZE, self.TAU, self.LRC)
        self.buff = ReplayBuffer(self.BUFFER_SIZE)    # Create replay buffer
        self.combat_buff = ReplayBuffer(self.BUFFER_SIZE)    # Create combat replay buffer
        self.turn_buff = ReplayBuffer(self.BUFFER_SIZE)    # Create turn replay buffer
        global graph
        graph = tf.get_default_graph()

        self.state_processor = state_processor

    def move_turn_buff_to_combat_buff(self):
        starttime = time.time()
        batch = self.turn_buff.getBatch(self.turn_buff.count())
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])

        if rewards.size > 0:
            total_reward = np.sum(rewards)

            for x in range(rewards.size):
                rewards[x] = total_reward
                print("REWARD FOR THE TURN: " + str(total_reward.item()))
                self.combat_buff.add(states[x], actions[x], rewards[x], new_states[x], dones[x])

        self.turn_buff.erase()
        endtime = time.time()
        print("move_turn_buff_to_combat_buff took {} seconds.".format(endtime-starttime))

    def move_combat_buff_to_buff(self):
        starttime = time.time()
        batch = self.combat_buff.getBatch(self.combat_buff.count())
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])

        # rewards[rewards.size-1] = np.sum(rewards)

        # print("current reward: ", int(rewards[rewards.size-1]))

        for x in range(rewards.size):
            # rewards[x] = rewards[rewards.size-1]
            self.buff.add(states[x], actions[x], rewards[x], new_states[x], dones[x])

        self.combat_buff.erase()
        endtime = time.time()
        print("move_combat_buff_to_buff took {} seconds.".format(endtime-starttime))

    def load_models(self):
        # Now load the weight
        print("Now we load the weights")
        try:
            self.actor.model.load_weights("actormodel.h5")
            self.critic.model.load_weights("criticmodel.h5")
            self.actor.target_model.load_weights("actormodel.h5")
            self.critic.target_model.load_weights("criticmodel.h5")
            print("Models loaded successfully")
        except:
            print("Cannot find the models")

    def save_models(self):
        starttime = time.time()
        print("Now we save model")
        self.actor.model.save_weights("actormodel.h5", overwrite=True)
        with open("actormodel.json", "w") as outfile:
            json.dump(self.actor.model.to_json(), outfile)

        self.critic.model.save_weights("criticmodel.h5", overwrite=True)
        with open("criticmodel.json", "w") as outfile:
            json.dump(self.critic.model.to_json(), outfile)
        print("Finished saving model")
        endtime = time.time()
        print("save_models took {} seconds.".format(endtime-starttime))

    def get_action(self):
        starttime = time.time()
        s_t1 = self.state_processor.previous_state_array
        a_t1 = self.state_processor.previous_action
        r_t1 = self.state_processor.get_reward()
        s_t = self.state_processor.current_state_array  # set s_t based on data input from Java
        if s_t1.any():
            self.turn_buff.add(s_t1, a_t1, r_t1, s_t, False)      # Add replay buffer

        self.epsilon -= 1.0 / self.EXPLORE

        if random.random() < self.epsilon:
            print("getting random action")
            a_t = np.append(self.state_processor.get_rand_valid_monster(), self.state_processor.get_rand_valid_action())
        else:
            print("predicting action")
            global graph
            with graph.as_default():
                a_t = self.actor.model.predict(np.expand_dims(s_t, axis=0))
                a_t = np.append(self.state_processor.get_valid_monster(a_t), self.state_processor.get_valid_action(a_t))

        self.state_processor.previous_action = self.state_processor.current_action
        self.state_processor.current_action = a_t

        endtime = time.time()
        print("get_action took {} seconds.".format(endtime-starttime))
        return np.nonzero(a_t[:5]), np.nonzero(a_t[5:])

    def train_model(self):
        # Do the batch update
        batch = self.buff.getBatch(self.BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        global graph
        with graph.as_default():
            starttime = time.time()
            target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])
            endtime = time.time()
            print("critic target predict took {} seconds.".format(endtime-starttime))

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + self.GAMMA*target_q_values[k][0]

            if self.train_indicator:
                starttime = time.time()
                self.loss += self.critic.model.train_on_batch([states, actions], y_t)
                endtime = time.time()
                print("critic train_on_batch took {} seconds.".format(endtime-starttime))
                starttime = time.time()
                a_for_grad = self.actor.model.predict(states)
                endtime = time.time()
                print("actor predict for gradient took {} seconds.".format(endtime-starttime))
                starttime = time.time()
                grads = self.critic.gradients(states, a_for_grad)
                endtime = time.time()
                print("critic gradients took {} seconds.".format(endtime-starttime))
                starttime = time.time()
                self.actor.train(states, grads)
                endtime = time.time()
                print("actor train took {} seconds.".format(endtime-starttime))
                starttime = time.time()
                self.actor.target_train()
                endtime = time.time()
                print("actor target train took {} seconds.".format(endtime-starttime))
                starttime = time.time()
                self.critic.target_train()
                endtime = time.time()
                print("critic target train took {} seconds.".format(endtime-starttime))

        self.total_reward += self.state_processor.get_reward()

    def reset(self):
        starttime=time.time()
        self.state_processor.previousStateDataDict = {}
        self.state_processor.previous_state_array = self.state_processor.current_state_array
        self.state_processor.current_state_array = np.zeros((131055,), dtype=int)
        s_t1 = self.state_processor.previous_state_array
        a_t1 = self.state_processor.previous_action
        r_t1 = self.state_processor.get_reward() + 10
        self.total_reward = 0
        s_t = self.state_processor.current_state_array
        if s_t1.any():
            self.turn_buff.add(s_t1, a_t1, r_t1, s_t, True)      # Add replay buffer
            self.move_turn_buff_to_combat_buff()
            self.move_combat_buff_to_buff()

        self.epsilon -= 1.0 / self.EXPLORE

        self.state_processor.previous_action = np.zeros(21, dtype=int)
        self.state_processor.current_action = np.zeros(21, dtype=int)
        self.state_processor.previous_state_array = np.zeros((131055,), dtype=int)

        if self.buff.count() >= self.BATCH_SIZE:
            self.train_model()

        self.save_models()
        endtime = time.time()
        print("reset took {} seconds.".format(endtime-starttime))

    def reset_after_death(self):
        starttime=time.time()
        self.state_processor.previousStateDataDict = {}
        self.state_processor.previous_state_array = self.state_processor.current_state_array
        self.state_processor.current_state_array = np.zeros((131055,), dtype=int)
        s_t1 = self.state_processor.previous_state_array
        a_t1 = self.state_processor.previous_action
        r_t1 = self.state_processor.get_reward() - 100
        self.total_reward = 0
        s_t = self.state_processor.current_state_array
        if s_t1.any():
            self.turn_buff.add(s_t1, a_t1, r_t1, s_t, True)      # Add replay buffer
            self.move_turn_buff_to_combat_buff()
            self.move_combat_buff_to_buff()

        self.epsilon -= 1.0 / self.EXPLORE

        self.state_processor.previous_action = np.zeros(21, dtype=int)
        self.state_processor.current_action = np.zeros(21, dtype=int)
        self.state_processor.previous_state_array = np.zeros((131055,), dtype=int)

        if self.buff.count() >= self.BATCH_SIZE:
            self.train_model()

        self.save_models()
        endtime = time.time()
        print("reset_after_death took {} seconds.".format(endtime-starttime))

    def serialize(self):
        K.clear_session()
        gc.collect()
        # filehandler = open("replaybuff.deq", 'w')
        # pickle.dump(self.buff, filehandler)


    def deserialize(self):
        try:
            filehandler = open("replaybuff.deq", 'r')
            self.buff = pickle.load(filehandler)
        except:
            print("Cannot find the deque")

