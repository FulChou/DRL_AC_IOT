'''
Author: Ful Chou
Date: 2021-03-22 12:04:41
LastEditors: Ful Chou
LastEditTime: 2021-03-31 10:37:22
FilePath: /RL-Adventure-2/http_server.py
Description: What this document does
'''


from flask import Flask, app
from flask.globals import request
from a2c_train import get_status_return_model_parameters
from a2c_train import a2c_env_param
from test import test
import json
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'hellollll'


@app.route('/get_a2c_env_param')
def get_a2c_env_param():
    return json.dumps(a2c_env_param)


@app.route('/sendStatus', methods=['POST'])
def sendStatus():
    # print(request.data)
    data = request.data
    with open('status.txt', 'w+') as f:
        f.write(str(data))
    data = json.loads(data)
    parameters = get_status_return_model_parameters(status=data)
    return json.dumps(parameters)


if __name__ == '__main__':
    # macos local ip: 192.168.199.128
    app.run(host='0.0.0.0', port=5000, debug=True)
