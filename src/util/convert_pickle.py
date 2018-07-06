"""
Convert the pickle2 file into a pickle3 file that can be used in a python3
environment. Script must be ran in a python 2 environment
"""
import os
import chumpy
import numpy as np
import cPickle as pickle


# path variables
py_dir = os.path.dirname(os.path.abspath(__file__))

# model variables
pkl_path = os.path.join(py_dir, "..", "..", "models", "neutral_smpl_with_cocoplus_reg.pkl")
np_path = os.path.join(py_dir, "..", "..", "models", "neutral_smpl_with_cocoplus_reg.npy")


# util function
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


# do conversion
def convertPickleToNumpy():
    # load pickle
    with open(pkl_path, 'rb') as f:
        dd = pickle.load(f)

    # convert shapedirs from chumpy to numpy
    # chumpy doesn't quite work in python3
    dd['shapedirs'] = undo_chumpy(dd['shapedirs'])

    # save numpy file
    np.save(np_path, dd)

    print("Succesfully converted to numpy file")


if __name__ == "__main__":
    convertPickleToNumpy()
