import numpy as np
import tensorflow as tf
import json
import random

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
from dqn import DQNAgent

OU = OU()       # Ornstein-Uhlenbeck Process
#state_processor = DQNAgent()


class Brain:
    def __init__(self):
        self.BUFFER_SIZE = 100000
        self.BATCH_SIZE = 2  # 32
        self.GAMMA = 0.99
        self.TAU = 0.001     # Target Network HyperParameters
        self.LRA = 0.0001    # Learning rate for Actor
        self.LRC = 0.001     # Learning rate for Critic

        self.action_dim = 21  # Target/Action
        self.state_dim = 5951  # columns in input state

        np.random.seed(1337)
        self.total_reward = 0.
        self.loss = 0

        self.vision = False

        self.EXPLORE = 100000.
        self.episode_count = 2000
        self.max_steps = 100000
        self.reward = 0
        self.done = False
        self.step = 0
        self.epsilon = 1
        self.indicator = 0

        self.train_indicator = 1  # 1 means Train, 0 means simply Run

        # Tensorflow GPU optimization
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        from keras import backend as K
        K.set_session(self.sess)

        self.actor = ActorNetwork(self.sess, self.state_dim, self.action_dim, self.BATCH_SIZE, self.TAU, self.LRA)
        self.critic = CriticNetwork(self.sess, self.state_dim, self.action_dim, self.BATCH_SIZE, self.TAU, self.LRC)
        self.buff = ReplayBuffer(self.BUFFER_SIZE)    # Create replay buffer
        global graph
        graph = tf.get_default_graph()

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
        print("Now we save model")
        self.actor.model.save_weights("actormodel.h5", overwrite=True)
        with open("actormodel.json", "w") as outfile:
            json.dump(self.actor.model.to_json(), outfile)

        self.critic.model.save_weights("criticmodel.h5", overwrite=True)
        with open("criticmodel.json", "w") as outfile:
            json.dump(self.critic.model.to_json(), outfile)
        print("Finished saving model")

    def get_action(self, state_processor):
        s_t1 = state_processor.previous_state_array
        a_t1 = state_processor.previous_action
        r_t1 = state_processor.get_reward()
        s_t = state_processor.current_state_array  # set s_t based on data input from Java
        if not s_t1.size:
            s_t1 = [np.zeros(5), np.zeros(16)]
        #else:
            #s_t1 = np.transpose(s_t1)
        s_t = np.transpose(s_t)
        self.buff.add(s_t1, a_t1, r_t1, s_t, False)      # Add replay buffer

        self.epsilon -= 1.0 / self.EXPLORE
        # a_t = np.zeros([1, action_dim])

        if random.random() < self.epsilon:
            print("getting random action")
            a_t = [state_processor.get_rand_valid_monster(), state_processor.get_rand_valid_action()]
        else:
            print("predicting action")
            a_t = self.actor.model.predict(s_t)
            a_t = [state_processor.get_valid_monster(a_t), state_processor.get_valid_action(a_t)]

        state_processor.previous_action = state_processor.current_action
        state_processor.current_action = np.append(a_t[0], a_t[1])
        # ob, r_t, done, info = env.step(a_t[0]) # send action to Java, wait for next state
        return np.nonzero(a_t[0]), np.nonzero(a_t[1])

    def train_model(self, state_processor):
        # Do the batch update
        batch = self.buff.getBatch(self.BATCH_SIZE)
        states = np.squeeze(np.asarray([e[0] for e in batch]), axis=1)
        actions = np.asarray([e[1] for e in batch])
        #actions2 = np.asarray([e[1][1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.squeeze(np.asarray([e[3] for e in batch]), axis=2)
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        global graph
        with graph.as_default():
            target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + self.GAMMA*target_q_values[0][k]

            if self.train_indicator:
                print(actions)
                self.loss += self.critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = self.actor.model.predict(states)
                grads = self.critic.gradients(states, a_for_grad)
                self.actor.train(states, grads)
                self.actor.target_train()
                self.critic.target_train()

        self.total_reward += state_processor.get_reward()
