from flask import Flask
from flask import jsonify
from flask import request

from StateHolder import StateHolder
from brain import Brain

import signal
import sys

app = Flask(__name__)
state_processor = StateHolder()
brain = Brain(state_processor)


def signal_handler(sig, frame):
    print('You stopped the process!')
    # brain.serialize()
    sys.exit(0)


@app.route('/navi/StateDataDumptest-endpoint', methods=['GET', 'POST'])
def state():
    content = request.get_json()
    state_processor.current_state_data = content
    return jsonify(content)


@app.route('/navi/CombatDataDumptest-endpoint', methods=['GET', 'POST'])
def combat():
    content = request.get_json()
    state_processor.create_combined_array(state_processor.current_state_data, content)

    monster_to_target, card_or_potion_to_use = brain.get_action()
    mydict = {
        'monster_to_target': monster_to_target[0].item(),
        'card_or_potion_to_use': card_or_potion_to_use[0].item()
    }

    state_processor.generation += 1
    return jsonify(mydict)


@app.route('/navi/EventDataDumptest-endpoint', methods=['GET', 'POST'])
def event():
    content = request.get_json()
    return jsonify({})


@app.route('/navi/reset-endpoint', methods=['GET', 'POST'])
def reset():
    brain.reset()
    if state_processor.generation == 0:
        state_processor.generation = 0
        return jsonify({'restart': 'true'})
    else:
        return jsonify({'restart': 'false'})


@app.route('/navi/death-endpoint', methods=['GET', 'POST'])
def death():
    brain.reset_after_death()
    if state_processor.generation == 0:
        state_processor.generation = 0
        return jsonify({'restart': 'true'})
    else:
        return jsonify({'restart': 'false'})


@app.route('/navi/turn-start-endpoint', methods=['GET', 'POST'])
def turn_start():
    brain.move_turn_buff_to_combat_buff()
    return jsonify({})


if __name__ == '__main__':
    brain.load_models()
    signal.signal(signal.SIGINT, signal_handler)
    brain.deserialize()
    app.run()
