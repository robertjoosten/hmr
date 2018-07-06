"""
Export of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

This module will allow for the exporting of the skeleton animated over time ( multiple images ). 3D matrices and 3D camera points are exported. These can be
used to rebuild the skeleton in a seperate application. If a json_dir is provided keypoints will be extracted and matched depending on distance from each other
and a distance threshold. This way the number of people and people over different frames are nicely matched. There is also a possibility to exclude people if
they are only present in the animation under the presence_threshold. These values can be adjusted in the openpose util.

Sample usage:

# on multiple images and open pose output json ( json_dir is optional )
python3 -m export --img_dir data/images --json_dir data/json --output_path output

# on a single image and single open pose output json ( json_path is optional )
python3 -m export --img_path data/images/image0001.jpg --json_dir data/json/json0001.json --output_path output
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import image as img_util
from src.util import openpose as op_util
from src.util import path as path_util
from src.util import logging as logging_util
import src.config
from src.RunModel import RunModel


# add flag for image and json dir
flags.DEFINE_string('img_dir', None, 'Folder containing images to process')
flags.DEFINE_string('img_path', None, 'Image to process')
flags.DEFINE_string('json_dir', None, 'Folder containing json to process.')
flags.DEFINE_string('json_path', None, 'Json to process.')
flags.DEFINE_string('output_path', None, 'Path to save data too.')


def preprocess_image(img_path, kp=None):
    img = io.imread(img_path)

    if img.shape[2] == 4:
        img = img[:, :, :3]

    if kp is None:
        if np.max(img.shape[:2]) != config.img_size:
            logger.debug(
                "Resizing Image:        Max size ({})".format(config.img_size)
            )
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.

        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(kp)
        if not scale and not center:
            logger.warning(
                "Image Process Error:   {}".format(img_path)
            )
            return None, None, None

    crop, proc_param = img_util.scale_and_crop(
        img,
        scale,
        center,
        config.img_size
    )

    # normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def shift_matrices(proc_param, cam, matrices):
    img_size = proc_param['img_size']
    undo_scale = 1. / np.array(proc_param['scale'])

    flength = 50.

    start_pt = proc_param['start_pt'] - 0.5 * img_size
    principal_pt = np.array([img_size, img_size]) / 2.
    final_principal_pt = (principal_pt + start_pt) * undo_scale

    tz = flength / (0.5 * img_size * cam[0][0])
    tx = cam[0][1] + (final_principal_pt[0] * 0.01)
    ty = cam[0][2] + (final_principal_pt[1] * 0.01)

    trans = [tx, ty, tz]

    for i, matrix in enumerate(matrices):
        matrices[i][12] = matrices[i][12] + trans[0]
        matrices[i][13] = matrices[i][13] + trans[1]
        matrices[i][14] = matrices[i][14] + trans[2]

    return matrices

# ----------------------------------------------------------------------------


def single_image(img_path, json_path=None, ordered_keypoints=None):
    # variable
    data = []

    # get keypoints to loop
    json_keypoints = op_util.get_keypoints(json_path) if json_path else [None]
    kps = ordered_keypoints if ordered_keypoints else json_keypoints

    # loop keypoints
    for kp in kps:
        # append none of no data is found
        if kp is False:
            data.append(None)
            continue

        # process images
        input_img, proc_param, img = preprocess_image(img_path, kp)

        # add non if image cannot be process
        if input_img is None:
            data.append(None)
            continue

        input_img = np.expand_dims(input_img, 0)

        # predict model
        _, _, cams3d, _, _, matrices3d = model.predict(input_img)

        # add matrices
        matrices3d = shift_matrices(proc_param, cams3d, matrices3d)
        matrices_dict = path_util.list_to_joint_name_dict(matrices3d)

        # add cams
        data.append(
            {
                "cams_3d": cams3d.tolist(),
                "matrices_3d": matrices_dict
            }
        )

    return {"people": data}


def multiple_images(img_dir, json_dir=None):
    # get files
    img_paths = path_util.listdir(img_dir, "png")
    json_paths = path_util.listdir(json_dir, "json")

    # order keypoints to find best matching person
    ordered_keypoints = op_util.sort_keypoints(json_paths)

    return {
        "frames": [
            single_image(img_path, ordered_keypoints=keypoints)
            for img_path, keypoints in zip(img_paths, ordered_keypoints)
        ]
    }


# ----------------------------------------------------------------------------


if __name__ == '__main__':
    # set config
    config = flags.FLAGS
    config(sys.argv)
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1

    # run model
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    
    # get logger
    logger = logging_util.get_logger(config.log_dir)

    # get data
    data = {}

    # process data
    if config.img_dir:
        logger.info("Process Mode:          Directory")
        data = multiple_images(config.img_dir, config.json_dir)
    elif config.img_path:
        logger.info("Process Mode:          File")
        frame = single_image(config.img_path, config.json_path)
        data["frames"] = [frame]

    # write to file
    with open(config.output_path, "w") as f:
        json.dump(data, f)
