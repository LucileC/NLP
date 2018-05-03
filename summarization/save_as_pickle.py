import pickle

def save_obj(obj, name ):
	print('Saving %s...'%name)
	with open('obj/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
	print('%s saved'%name)

def load_obj(name ):
	print('Loading %s...'%name)
	with open('obj/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)
	print('%s loaded'%name)