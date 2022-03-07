import os
import pickle
from functools import wraps


def cache_file(file_name):
    def cache_file_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.isfile(file_name):
                with open(file_name, 'rb') as f:
                    return pickle.load(f)
            else:
                out = func(*args, **kwargs)
                with open(file_name, 'wb') as f:
                    pickle.dump(out, f)
                return out
        return wrapper
    return cache_file_decorator
