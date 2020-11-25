from flask import current_app, jsonify
import torch
import redisai

def other_func():
	current_app.logger.info(f'Talking from another func')
	a = torch.Tensor([1,2,3,4])
	return a

def main():
	a = other_func()
	# current_app.logger.info(torch.cuda.current_device())
	return jsonify(tensor=str(a))
	

