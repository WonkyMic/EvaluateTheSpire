import json
import psycopg2
import pandas as pds
import keras.backend as K
import numpy as np
import time
import configparser

graph = []
GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP  = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient


class DQNAgent:
    def __init__(self):
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.stateData = pds.DataFrame()
        self.combatDataDict = {}
        self.stateDataDict = {}
        self.current_state_array = np.zeros((130873,), dtype=int)
        self.previous_state_array = np.zeros((130873,), dtype=int)
        self.current_action = np.zeros(21, dtype=int)
        self.previous_action = np.zeros(21, dtype=int)
        self.action_list = []

    def loadStateDataToDatabase(self):
        data_file = "C:\\Users\\Hafez\\IdeaProjects\\NavigateTheSpire\\json\\StateDataDumpjsonDump.json"
        with open(data_file, "r") as f:
            data = json.load(f)
        self.loadDataToDatabase(data, "stateData")

    def loadCombatDataToDatabase(self):
        data_file = "C:\\Users\\Hafez\\IdeaProjects\\NavigateTheSpire\\json\\CombatDataDumpjsonDump.json"
        with open(data_file, "r") as f:
            data = json.load(f)
        i=0
        for enemy in data["jsonEnemyArrayList"]:
            corrected_dict = {(k + str(i)): v for k, v in enemy.items()}
            data.update(corrected_dict)
            i=i+1
        del data["jsonEnemyArrayList"]

        j=0
        for card in data["jsonCardArrayListHand"]:
            corrected_dict = {(k + str(j)): v for k, v in card.items()}
            data.update(corrected_dict)
        j=j+1
        del data["jsonCardArrayListHand"]

        l=0
        for card in data["jsonCardArrayListExhaustPile"]:
            corrected_dict = {(k + str(l)): v for k, v in card.items()}
            data.update(corrected_dict)
        l=l+1
        del data["jsonCardArrayListExhaustPile"]

        m=0
        for card in data["jsonCardArrayListDiscardPile"]:
            corrected_dict = {(k + str(m)): v for k, v in card.items()}
            data.update(corrected_dict)
        m=m+1
        del data["jsonCardArrayListDiscardPile"]

        n=0
        for card in data["jsonCardArrayListDrawPile"]:
            corrected_dict = {(k + str(n)): v for k, v in card.items()}
            data.update(corrected_dict)
        n=n+1
        del data["jsonCardArrayListDrawPile"]

        self.loadDataToDatabase(data, "combatStateData")

        config = configparser.ConfigParser()
        config.read('config.ini')
        #conn = psycopg2.connect(host=config['postgresql']['host'], database=config['postgresql']['database'],
        #                        user=config['postgresql']['user'], password=config['postgresql']['password'], port=config['postgresql']['port'])
        #print("Database Connected")

        #with conn.cursor() as cursor:
        #    keys = data.keys()
        #    columns = ','.join(keys)
        #    values = ','.join(["%("+k+")s" for k in data])
        #    insert = 'INSERT into {0} ({1}) VALUES ({2})'.format("combatStateData", columns, values)
        #    print(insert, data)
        #    cursor.execute(insert, data)
        #conn.commit()


    def loadDataToDatabase(self, data_dict, table_name):
        conn = psycopg2.connect(host="localhost", database="stsData",
                            user="postgres", password="postgres", port='5433')
        print("Database Connected")

        with conn.cursor() as cursor:
            keys = data_dict.keys()
            columns = ','.join(keys)
            values = ','.join(["%("+k+")s" for k in data_dict])
            insert = 'INSERT into {0} ({1}) VALUES ({2})'.format(table_name, columns, values)
            cursor.execute(insert, data_dict)
        conn.commit()

    def loadAllDataToDatabase(self):
        self.loadStateDataToDatabase()
        self.loadCombatDataToDatabase()


    def createCombatDict(self, combatData):
        starttime = time.time()
        #data_file = "C:\\Users\\Hafez\\IdeaProjects\\NavigateTheSpire\\json\\CombatDataDumpjsonDump.json"
        #with open(data_file, "r") as f:
        #    data = json.load(f)

        data = combatData

        i=0
        for enemy in data["jsonEnemyArrayList"]:
            data.update({(k + str(i)): v for k, v in enemy.items()})
            i=i+1
        del data["jsonEnemyArrayList"]

        j=0
        for card in data["jsonCardArrayListHand"]:
            data.update({("hand"+k + str(j)): v for k, v in card.items()})
            j=j+1
        del data["jsonCardArrayListHand"]

        l=0
        for card in data["jsonCardArrayListExhaustPile"]:
            data.update({("exhaust"+k + str(l)): v for k, v in card.items()})
            l=l+1
        del data["jsonCardArrayListExhaustPile"]

        m=0
        for card in data["jsonCardArrayListDiscardPile"]:
            data.update({("discard"+k + str(m)): v for k, v in card.items()})
            m=m+1
        del data["jsonCardArrayListDiscardPile"]

        n=0
        for card in data["jsonCardArrayListDrawPile"]:
            data.update({("draw"+k + str(n)): v for k, v in card.items()})
            n=n+1
        del data["jsonCardArrayListDrawPile"]

        # delete currently unnecessary data
        del data["gameID"]
        del data["combatStateID"]
        del data["currentStateID"]

        self.combatDataDict = data

        endtime = time.time()
        print("createCombatDict took {} seconds.".format(endtime-starttime))
        return data


    def combat_data_to_array(self, combatData):
        starttime = time.time()
        data = self.createCombatDict(combatData)
        arr = np.fromiter(data.values(), dtype=int)

        endtime = time.time()
        print("combat_data_to_array took {} seconds.".format(endtime-starttime))
        return arr

    def createStateDict(self, stateData):
        #data_file = "C:\\Users\\Hafez\\IdeaProjects\\NavigateTheSpire\\json\\StateDataDumpjsonDump.json"
        #with open(data_file, "r") as f:
        #    data = json.load(f)

        data = stateData

        # delete currently unnecessary fields
        del data["gameID"]
        del data["currentStateID"]

        self.stateDataDict = data
        return data

    def state_data_to_array(self, stateData):
        starttime = time.time()
        data = self.createStateDict(stateData)
        arr = np.fromiter(data.values(), dtype=int)

        endtime = time.time()
        print("state_data_to_array took {} seconds.".format(endtime-starttime))
        return arr

    def get_reward(self):
        if True:  # .empty
            reward = 0
        else:
            health_reward = self.current_state['currentHealth'] - self.previous_state['currentHealth']
            enemy_health_reward = self.previous_state['currentHealth0'] - self.current_state['currentHealth0'] + self.previous_state['currentHealth1'] - self.current_state['currentHealth1'] + self.previous_state['currentHealth2'] - self.current_state['currentHealth2'] + self.previous_state['currentHealth3'] - self.current_state['currentHealth3'] + self.previous_state['currentHealth4'] - self.current_state['currentHealth4']
            # currentGoldReward = self.current_state['currentGold'] / 10
            # floorNumReward = self.current_state['floorNum']
            # actNumReward = state['actNum'] * 5 #maybe this should multiply all other rewards instead of being its own constant reward?
            # relic_id_cols = [col for col in state.columns if 'relic' in col]
            # relicReward = state.groupby(relic_id_cols).ngroups
            reward = health_reward + enemy_health_reward # + currentGoldReward + floorNumReward + actNumReward + relicReward
        print("current reward: ", int(reward))
        return reward

    def discount_rewards(self, r, gamma=0.99):
        """Takes 1d float array of rewards and computes discounted reward
        e.g. f([1, 1, 1], 0.99) -> [2.9701, 1.99, 1]
        """
        prior = 0
        out = []
        for val in r:
            new_val = val + prior * gamma
            out.append(new_val)
            prior = new_val
        return np.array(out[::-1])

    def custom_loss(self, y_true, y_pred):
        log_lik = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
        return K.mean(log_lik, keepdims=True)

    # Note: pass in_keras=False to use this function with raw numbers of numpy arrays for testing
    def huber_loss(self, a, b, in_keras=True):
        error = a - b
        quadratic_term = error*error / 2
        linear_term = abs(error) - 1/2
        use_linear_term = (abs(error) > 1.0)
        if in_keras:
            # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
            use_linear_term = K.cast(use_linear_term, 'float32')
        return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

    def copy_model(self, model):
        """Returns a copy of a keras model."""
        import keras.models
        model.save('tmp_model')
        return keras.models.load_model('tmp_model', custom_objects={'huber_loss': self.huber_loss})

    def create_combined_dataframe(self, stateData, combatData):
        starttime = time.time()
        state_array = self.state_data_to_array(stateData)
        combat_array = self.combat_data_to_array(combatData)
        result = np.append(state_array, combat_array)
        self.previous_state_array = self.current_state_array
        self.current_state_array = result
        endtime = time.time()
        print("create_combined_dataframe took {} seconds.".format(endtime-starttime))

    def get_valid_action(self, predicted_action_values):
        cards = {k: v for k, v in self.combatDataDict.items() if 'handisPlayable' in k}
        playable_cards = np.fromiter(cards.values(), dtype=int)

        potions = {k: 1 if v != "" and v != "Potion Slot" else 0 for k, v in self.stateDataDict.items() if 'potions' in k}
        playable_potions = np.fromiter(potions.values(), dtype=int)

        playable_cards_and_actions = np.append(playable_cards, playable_potions)
        playable_cards_and_actions = np.append(playable_cards_and_actions, np.array([1]))

        final_action_values = np.multiply(playable_cards_and_actions, predicted_action_values[1])

        return np.argmax(final_action_values[0])

    def get_valid_monster(self, predicted_action_values):
        monsters = {k: 1 if v != 0 else 0 for k, v in self.combatDataDict.items() if 'currentHealth' in k}
        targetable_monsters = np.fromiter(monsters.values(), dtype=int)

        final_action_values = np.multiply(targetable_monsters, predicted_action_values[0])

        return np.argmax(final_action_values[0])

    def get_rand_valid_action(self):
        starttime = time.time()
        cards = {k: v for k, v in self.combatDataDict.items() if 'handisPlayable' in k}
        playable_cards = np.fromiter(cards.values(), dtype=int)

        potions = {k: 0 if v == 1 else 1 for k, v in self.stateDataDict.items() if 'IsPotionSlot' in k}
        playable_potions = np.fromiter(potions.values(), dtype=int)

        playable_cards_and_actions = np.append(playable_cards, playable_potions)
        playable_cards_and_actions = np.append(playable_cards_and_actions, np.array([1]))

        final_action_value_index = np.random.choice(np.nonzero(playable_cards_and_actions == 1)[0], replace=False)
        final_action_values = np.zeros(playable_cards_and_actions.shape)
        final_action_values[final_action_value_index] = 1
        endtime = time.time()
        print("get_rand_valid_action took {} seconds.".format(endtime-starttime))

        return final_action_values

    def get_rand_valid_monster(self):
        starttime = time.time()
        monsters = {k: 1 if v != 0 else 0 for k, v in self.combatDataDict.items() if 'currentHealth' in k}
        targetable_monsters = np.fromiter(monsters.values(), dtype=int)

        if np.nonzero(targetable_monsters == 1)[0].size == 0:
            final_action_values = np.zeros(targetable_monsters.shape)
            final_action_values[0] = 1
        else:
            final_action_value_index = np.random.choice(np.nonzero(targetable_monsters == 1)[0], replace=False)
            final_action_values = np.zeros(targetable_monsters.shape)
            final_action_values[final_action_value_index] = 1

        endtime = time.time()
        print("get_rand_valid_monster took {} seconds.".format(endtime-starttime))

        return final_action_values
