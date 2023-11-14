import base64
import pickle


def serialize(obj, binary=True):
    pickled_string = pickle.dumps(obj)
    if binary:
        pickled_string = base64.b64encode(pickled_string)
    return pickled_string


def deserialize(pickled_string, binary=True):
    if binary:
        pickled_string = base64.b64decode(pickled_string)
    obj = pickle.loads(pickled_string)
    return obj
