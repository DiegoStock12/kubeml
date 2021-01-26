import logging
from flask import Flask, request, jsonify
from kubeml.kubeml.exceptions import KubeMLException

# set some basic logging params
FORMAT = '[%(asctime)s] %(levelname)-8s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

app = Flask(__name__)

import module_usage


@app.errorhandler(KubeMLException)
def handle_exception(error: KubeMLException):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/')
def do():
    logging.info(f'Called api with args, {request.args}')
    raise KubeMLException("Test exception", 500)
    return module_usage.main()


if __name__ == '__main__':
    app.run(debug=True)
