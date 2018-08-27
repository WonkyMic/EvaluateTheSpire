from flask import Flask 
from flask import jsonify
from flask import request

app = Flask(__name__)

@app.route('/navi/test-endpoint', methods=['GET', 'POST'])
def navi():
	content = request.get_json()
	return jsonify(content)

if __name__ == '__main__':
	app.run()