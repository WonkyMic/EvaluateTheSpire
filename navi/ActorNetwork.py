from keras.models import Model
from keras.layers import Dense, Input, concatenate
import tensorflow as tf
import keras.backend as K
from keras.optimizers import RMSprop

HIDDEN1_UNITS = 500
HIDDEN2_UNITS = 1000

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network(state_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    # Note: pass in_keras=False to use this function with raw numbers of numpy arrays for testing
    @staticmethod
    def huber_loss(a, b, in_keras=True):
        error = a - b
        quadratic_term = error*error / 2
        linear_term = abs(error) - 1/2
        use_linear_term = (abs(error) > 1.0)
        if in_keras:
            # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
            use_linear_term = K.cast(use_linear_term, 'float32')
        return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

    @staticmethod
    def create_actor_network(state_size):
        print("Now we build the model")
        S = Input(shape=(state_size,))
        h0 = Dense(1000, activation="relu")(S)
        h1 = Dense(500, activation="relu")(h0)
        Target = Dense(5, activation="softmax")(h1)
        Action = Dense(16, activation="softmax")(h1)
        V = concatenate([Target, Action])
        model = Model(inputs=S, outputs=V)
        model.summary()
        # model.compile(self.optimizer, loss=self.huber_loss)
        global graph
        graph = tf.get_default_graph()
        return model, model.trainable_weights, S
