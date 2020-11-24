from flask import current_app
import torch

def main():
	a = torch.Tensor([1,2,3,4])
	current_app.logger.info(f'This is a log, {str(a)}')
	current_app.logger.info(torch.cuda.is_available())
	# current_app.logger.info(torch.cuda.current_device())
	return f'Hello I am a function and my result is {str(a)}'
	

