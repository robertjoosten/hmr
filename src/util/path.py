"""
Path util
"""
import os

joint_names = [
    'r_ankle', 'r_knee', 'r_hip',
    'l_hip', 'l_knee', 'l_ankle',
    'r_wrist', 'r_elbow', 'r_shoulder',
    'l_shoulder', 'l_elbow', 'l_wrist',
    'neck', 'head', 'nose',
    'l_eye', 'r_eye', 'l_ear',
    'r_ear'
]



def listdir(path, filetype):
    """
    Extend list dir function with a filetype option. Ideal to be used to get
    all files in a folder from a particular filetype.

    :param str path:
    :param str filetype:
    :return: Files in the path matching the filetype
    :rtype: list
    """
    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(filetype.lower())
    ]


def list_to_joint_name_dict(data):
    """
    Create a dictionary with the joint names and the data
    """
    return {
        joint: d
        for joint, d in zip(joint_names, data)
    }