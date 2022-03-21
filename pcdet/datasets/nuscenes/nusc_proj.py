from pathlib import Path
import numpy as np
import cv2

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

def project_lidar_to_img(nusc, single_frame = True):
    def _rotate(pc, rot):
        pc[:3, :] = np.dot(rot, pc[:3, :])
        return pc

    def _translate(pc, trans):
        for i in range(3):
            pc[i, :] = pc[i, :] + trans[i]
        return pc

    for index, sample in enumerate(nusc.sample):
        pointsensor_token = sample['data']['LIDAR_TOP']
        camera_token = sample['data']['CAM_FRONT']
        cam = nusc.get('sample_data', camera_token)
        pointsensor = nusc.get('sample_data', pointsensor_token)
        pcl_path = Path(nusc.dataroot) / Path(pointsensor['filename'])
        points = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)
        points = points.T

        cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        points = _rotate(points, Quaternion(cs_record['rotation']).rotation_matrix)
        points = _translate(points, np.array(cs_record['translation']))

        poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        points = _rotate(points, Quaternion(poserecord['rotation']).rotation_matrix)
        points = _translate(points, np.array(poserecord['translation']))

        poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
        points = _translate(points, -np.array(poserecord['translation']))
        points = _rotate(points, Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        points = _translate(points, -np.array(cs_record['translation']))
        points = _rotate(points, Quaternion(cs_record['rotation']).rotation_matrix.T)

        points[0, :] = points[0, :] / points[2, :]
        points[1, :] = points[1, :] / points[2, :]
        points = points.T

        cam_front_path = Path(nusc.dataroot) / Path(cam['filename'])
        img = cv2.imread(cam_front_path, cv2.IMREAD_COLOR)

        for i in range(int(points.shape[0] / 2)):
            y = points[i][0]
            x = points[i][1]
            img = cv2.circle(img, (int(y), int(x)), 2, (0, 255, 255), -1)
        out_path = Path('./output_img') / 'point.jpg'
        cv2.imwrite(out_path, img)

        if single_frame:
            break

    return

if __name__ == "__main__":
    data_path = Path('/home/xuhang/Data/nuscenes-pcdet/')
    version = 'v1.0-trainval'
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    project_lidar_to_img(nusc)