"""
Script to convert openpose output into bbox
"""
import json
import numpy as np
import src.util.logging as logging_util


def match_keypoints(kp1, kp2):
    """
    Get difference between two sets of keypoints.

    :param numpy.array kp1:
    :param numpy.array kp2:
    :return: Distantial difference
    :rtype: float
    """
    return sum(
        np.linalg.norm(k1 - k2)
        for k1, k2 in zip(kp1, kp2)
    )


def get_last_valid_keypoints(keypoints):
    """
    Get the last valid set of keypoints from a keypoints list.
    These lists can contain None values.

    :param list keypoints:
    :return: Valid keypoints
    :rtype: numpy.array
    """
    for i in range(len(keypoints)-1, -1, -1):
        if keypoints[i] is not None:
            return keypoints[i]


def sort_keypoints(
        json_paths, distance_threshold=5000, presence_threshold=0.1
):
    """
    OpenPose doesn't match up the people indices between different json files.
    As OpenPose only looks at an image at a time. This function will process
    all of the json files and will match the keypoints with either and
    generate an accurate sequence of people and keypoints.

    :param list json_paths:
    :param float distance_threshold:
    :param float presence_threshold:
    :return: Sorted keypoints
    :rtype: list
    """
    # log title
    logging_util.logger.info("---- Match Keypoints Over Multiple Frames ----")

    for i, json_path in enumerate(json_paths):
        kps = get_keypoints(json_path)

        # initialize keypoint data
        if i == 0:
            person_data = [[kp] for kp in kps]
            continue

        person_matches = []
        for j, person in enumerate(person_data):
            # get last available keypoints
            last_kp = get_last_valid_keypoints(person)

            # match with keypoints of this frame
            match_scores = [
                match_keypoints(kp, last_kp)
                for kp in kps
            ]

            # get best score and index of that score
            match_score = min(match_scores)
            match_index = match_scores.index(match_score)
            person_matches.append([match_score, match_index, j])

        person_indices = list(range(len(person_data)))
        kps_indices = list(range(len(kps)))
        for distance, match_index, person_index in sorted(person_matches):
            # see if distance fits threshold
            if distance > distance_threshold:
                continue

            # see if person hasn't been matched already
            if match_index not in kps_indices:
                continue

            # attach keypoints to the right person index
            kp = kps[match_index]
            kps_indices.remove(match_index)

            person_data[person_index].append(kp)
            person_indices.remove(person_index)

        # create new persons
        for index in kps_indices:
            logging_util.logger.debug(
                "New Person:            Frame {}".format(i + 1)
            )
            person_data.append([None]*(i-1) + [kps[index]])

        # add to non resolved persons
        for index in person_indices:
            person_data[index].append(None)

    # omit person with less than presence threshold
    for i in range(len(person_data)-1, -1, -1):
        person = person_data[i]
        presence = [None for d in person if d is not None]
        presence_percentage = len(presence)/float(len(person))
        if presence_percentage < presence_threshold:
            logging_util.logger.debug(
                "Omit Person:           Presence Percentage {}".format(
                    round(presence_percentage, 2)
                )
            )
            person_data.pop(i)

    logging_util.logger.info("People:                {}".format(len(person_data)))

    # reformat person data
    person_data_reformat = []
    for i in range(len(person_data[0])):
        person_data_reformat.append(
            [
                person_data[j][i]
                for j in range(len(person_data))
            ]
        )

    return person_data_reformat


def get_keypoints(json_path):
    with open(json_path) as f:
        data = json.load(f)

    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        kps.append(kp)

    return kps


def get_num_people(json_path):
    return len(get_keypoints(json_path))


def get_bbox(kp, vis_thr=0.2):
    vis = kp[:, 2] > vis_thr
    vis_kp = kp[vis, :2]
    min_pt = np.min(vis_kp, axis=0)
    max_pt = np.max(vis_kp, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        return None, None
    center = (min_pt + max_pt) / 2.
    scale = 150. / person_height

    return scale, center
