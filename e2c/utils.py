import torch
import os
import json
from .e2c import compute_loss
import types
def load_model(m,dir,filename):
	dicr_dir = os.path.join(dir,'model',filename)
	m.load_state_dict(torch.load(dicr_dir))

class DateEnc(json.JSONEncoder):
	def default(self, o):
		if isinstance(o,types.FunctionType):
			return o.__name__


