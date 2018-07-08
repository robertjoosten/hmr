# End-to-end Recovery of Human Shape and Pose

This project is forked from https://github.com/akanazawa/hmr

Angjoo Kanazawa, Michael J. Black, David W. Jacobs, Jitendra Malik
CVPR 2018

[Project Page](https://akanazawa.github.io/hmr/)

The export module will allow for the exporting of the skeleton animated over time ( multiple images ). 3D matrices and 3D camera points are exported. These can be used to rebuild the skeleton in a seperate application. If a json_dir is provided keypoints will be extracted and matched depending on distance from each other and a distance threshold. This way the number of people and people over different frames are nicely matched. There is also a possibility to exclude people if they are only present in the animation under the presence_threshold. These values can be adjusted in the openpose util.

Sample usage:

On multiple images and open pose output json ( json_dir is optional ):

    python3 -m export --img_dir data/images --json_dir data/json --output_path output

On a single image and single open pose output json ( json_path is optional ):

    python3 -m export --img_path data/images/image0001.jpg --json_dir data/json/json0001.json --output_path output
