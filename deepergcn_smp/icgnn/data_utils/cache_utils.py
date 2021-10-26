import gzip
import pickle


def read_cache(filename, compression=False):
    print(f"Cache file exists: {filename}")
    if compression:
        with gzip.open(filename, "rb") as f:
            obj = pickle.load(f)
    else:
        with open(filename, "rb") as f:
            obj = pickle.load(f)

    return obj


def write_cache(filename, obj, compression=False):
    print(f"Creating cache file: {filename}")
    if compression:
        with gzip.open(filename, "wb") as f:
            pickle.dump(obj, f)
    else:
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
