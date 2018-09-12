from flask import Flask
from flask import jsonify
from flask import request

from dqn import DQNAgent
from brain import Brain

import numpy as np

app = Flask(__name__)
state_processor = DQNAgent()
brain = Brain()

@app.route('/navi/StateDataDumptest-endpoint', methods=['GET', 'POST'])
def state():
    content = request.get_json()
    state_processor.stateData = content
    return jsonify(content)

@app.route('/navi/CombatDataDumptest-endpoint', methods=['GET', 'POST'])
def combat():
    content = request.get_json()
    state_processor.create_combined_dataframe(state_processor.stateData, content)

    monster_to_target, card_or_potion_to_use = brain.get_action(state_processor)
    mydict = {
        'monster_to_target': monster_to_target[0][0].item(),
        'card_or_potion_to_use': card_or_potion_to_use[0][0].item()
    }

    if brain.buff.count() >= brain.BATCH_SIZE:
        brain.train_model(state_processor)

    return jsonify(mydict)

@app.route('/navi/EventDataDumptest-endpoint', methods=['GET', 'POST'])
def event():
    content = request.get_json()
    #return jsonify(content)

if __name__ == '__main__':
    app.run()