from flask import Flask
from flask import jsonify
from flask import request

from dqn import DQNAgent

app = Flask(__name__)
agent = DQNAgent()

@app.route('/navi/StateDataDumptest-endpoint', methods=['GET', 'POST'])
def state():
    content = request.get_json()
    agent.stateData = content
    return True
    #return jsonify(content)

@app.route('/navi/CombatDataDumptest-endpoint', methods=['GET', 'POST'])
def combat():
    content = request.get_json()
    current_state, combat_state = agent.create_combined_dataframe(agent.stateData, content)

    data = current_state.values
    combat_action = agent.predict_combat_action(data)
    mydict = {
        'x1': [x.tolist() for x in combat_action]
    }

    return jsonify(mydict)

@app.route('/navi/EventDataDumptest-endpoint', methods=['GET', 'POST'])
def event():
    content = request.get_json()
    #return jsonify(content)

if __name__ == '__main__':
    app.run()