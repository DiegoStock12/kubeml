import logging

from flask import Flask, request
import module_usage

# set some basic logging params
FORMAT = '[%(asctime)s] %(levelname)-8s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

app = Flask(__name__)


@app.route('/')
def do():
    logging.info(f'Called api with args, {request.args}')
    return module_usage.main()


if __name__ == '__main__':
    app.run(debug=True)
