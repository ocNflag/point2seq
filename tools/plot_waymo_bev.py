import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm
import numpy as np
import pickle
import os


def rotate_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def plot_waymo_gt_on_bev():
    gt = pickle.load(open('/home/xuhang/xueyj/datasets/Waymo/waymo_infos_val.pkl', 'rb'))
    for gt_one in tqdm(gt):
        fig, ax = plt.subplots(figsize=(6, 6))
        frame_id = gt_one['frame_id']
        frame_name, frame_seq = frame_id.rsplit('_', 1)
        box_lidar = gt_one['annos']['gt_boxes_lidar']
        vehicle_id = gt_one['annos']['name'] == 'Vehicle'
        box_lidar = box_lidar[vehicle_id, :]
        if len(box_lidar) != 0:
            corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]])
            corners = corners * box_lidar[:, [3, 4]].reshape(-1, 1, 2)
            rot_matrix = np.stack(list([rotate_z(box[-1]).T for box in box_lidar]), axis=0)
            corners = np.einsum('nij,njk->nik', corners, rot_matrix) + box_lidar[:, :2].reshape((-1, 1, 2))
            for corner in corners:
                ax.add_patch(Polygon(corner, closed=True, facecolor='none', edgecolor='blue'))

        np_filepath = os.path.join(
            '/home/xuhang/xueyj/datasets/Waymo/waymo_processed_data/', frame_name, '0' + str(frame_seq) + '.npy'
        )

        data = np.load(np_filepath)
        ax.scatter(data[:, 0], data[:, 1], marker="x", s=0.01, c='g')

        ax.set_xlim(-80, 80)
        ax.set_ylim(-80, 80)
        ax.invert_yaxis()
        plt.tight_layout()
        save_path = '/home/xuhang/xueyj/vis/waymo_gt_bev/{}_bev_gt.png'.format(frame_id)
        plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    plot_waymo_gt_on_bev()
