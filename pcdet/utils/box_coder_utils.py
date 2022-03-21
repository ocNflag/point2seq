import numpy as np
import torch

from pcdet.utils import common_utils
from pcdet.utils.common_utils import limit_period, check_numpy_to_torch


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


class ResidualCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if self.encode_angle_by_sincos:
            self.code_size += 1

    def encode_torch(self, boxes, anchors):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualRoIDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = ra - rt

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PointResidualCoder(object):
    def __init__(self, code_size=8, use_mean_size=True, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.use_mean_size = use_mean_size
        if self.use_mean_size:
            self.mean_size = torch.from_numpy(np.array(kwargs['mean_size'])).cuda().float()
            assert self.mean_size.min() > 0

    def encode_torch(self, gt_boxes, points, gt_classes=None):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            points: (N, 3) [x, y, z]
            gt_classes: (N) [1, num_classes]
        Returns:
            box_coding: (N, 8 + C)
        """
        gt_boxes[:, 3:6] = torch.clamp_min(gt_boxes[:, 3:6], min=1e-5)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert gt_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[gt_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = torch.log(dxg / dxa)
            dyt = torch.log(dyg / dya)
            dzt = torch.log(dzg / dza)
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
            dxt = torch.log(dxg)
            dyt = torch.log(dyg)
            dzt = torch.log(dzg)

        cts = [g for g in cgs]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, torch.cos(rg), torch.sin(rg), *cts], dim=-1)

    def decode_torch(self, box_encodings, points, pred_classes=None):
        """
        Args:
            box_encodings: (N, 8 + C) [x, y, z, dx, dy, dz, cos, sin, ...]
            points: [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:

        """
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert pred_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[pred_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za

            dxg = torch.exp(dxt) * dxa
            dyg = torch.exp(dyt) * dya
            dzg = torch.exp(dzt) * dza
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za
            dxg, dyg, dzg = torch.split(torch.exp(box_encodings[..., 3:6]), 1, dim=-1)

        rg = torch.atan2(sint, cost)

        cgs = [t for t in cts]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


# this coder is only for e2e cases which has quite different logic
class CenterCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, period=2 * np.pi, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        self.period = period
        if self.encode_angle_by_sincos:
            self.code_size += 1

    @staticmethod
    def _rotate_points_along_z(points, angle):
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot

    @staticmethod
    def boxes_to_corners_3d(boxes3d):
        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        corners3d = CenterCoder._rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
        corners3d += boxes3d[:, None, 0:3]
        return corners3d

    @staticmethod
    def _prep(input):
        slices = [input[..., :3], input[..., 3:6], input[..., 6:]]
        slices[1] = torch.clamp_min(slices[1], min=1e-5)
        return torch.cat(slices, dim=-1)

    def encode(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            tmp = CenterCoder._prep(gt_boxes[i])

            xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(tmp, 1, dim=-1)

            dxn = torch.log(dxg)
            dyn = torch.log(dyg)
            dzn = torch.log(dzg)

            if self.encode_angle_by_sincos:
                cosg = torch.cos(rg)
                sing = torch.sin(rg)
                rgs = [cosg, sing]
            else:
                rgs = [rg, ]

            rlt.append(torch.cat([xg, yg, zg, dxn, dyn, dzn, *rgs, *cgs], dim=-1))

        return rlt

    def gt_to_corner(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            boxes3d = CenterCoder._prep(gt_boxes[i])
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
        return rlt

    def pred_to_corner(self, preds):

        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)
            rt = torch.atan2(sint, cost)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        all_boxes = torch.cat([xt, yt, zt, dxg, dyg, dzg], dim=-1)


        rlt = []
        for i in range(len(all_boxes)):
            boxes3d = all_boxes[i]
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
            rlt.append(corners3d)

        return rlt

    def get_delta(self, gt_boxes, preds):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            gts: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        gt_boxes = CenterCoder._prep(gt_boxes)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        if not self.encode_angle_by_sincos:
            xp, yp, zp, dxp, dyp, dzp, rp, *cps = torch.split(preds, 1, dim=-1)
        else:
            xp, yp, zp, dxp, dyp, dzp, cosp, sinp, *cps = torch.split(preds, 1, dim=-1)

        xt = xg - xp
        yt = yg - yp
        zt = zg - zp

        dxt = torch.log(dxg) - dxp
        dyt = torch.log(dyg) - dyp
        dzt = torch.log(dzg) - dzp

        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - cosp
            rt_sin = torch.sin(rg) - sinp
            rts = [rt_cos, rt_sin]
        else:
            rts = [(rg / self.period) - rp]

        cts = [g - a for g, a in zip(cgs, cps)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, preds):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
        Returns:

        """
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        if self.encode_angle_by_sincos:
            rg = torch.atan2(sint, cost)
        else:
            rg = rt * self.period

        cgs = [t for t in cts]
        return torch.cat([xt, yt, zt, dxg, dyg, dzg, rg, *cgs], dim=-1)


    def encode_with_rois(self, boxes, rois):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        rois[:, 3:6] = torch.clamp_min(rois[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(rois, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)



class TokenCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, period=2 * np.pi, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        self.period = period
        assert self.encode_angle_by_sincos

    @staticmethod
    def _prep(input):
        slices = [input[..., :3], input[..., 3:6], input[..., 6:]]
        slices[1] = torch.clamp_min(slices[1], min=1e-5)
        return torch.cat(slices, dim=-1)

    def encode(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            tmp = CenterCoder._prep(gt_boxes[i])

            xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(tmp, 1, dim=-1)

            cosg = torch.cos(rg)
            sing = torch.sin(rg)
            rgs = [cosg, sing]

            rlt.append(torch.cat([xg, yg, zg, dxg, dyg, dzg, *rgs, *cgs], dim=-1))

        return rlt


    def get_tokenized_labels(self, gt_boxes, anchor_xy, anchor_zwlh, bin_size, bin_num):
        def _get_tokenized_labels(gt, anc, attr_key):
            rlt = torch.clamp(
                ((gt - anc) / bin_size[attr_key] + int(bin_num[attr_key] / 2)).int(),
                min=0,
                max=bin_num[attr_key] - 1
            )
            return rlt

        gt_boxes = self._prep(gt_boxes)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)

        xa, ya = torch.split(anchor_xy, 1, dim=-1)
        za, wa, la, ha = torch.split(anchor_zwlh, 1, dim=-1)

        xt = _get_tokenized_labels(xg, xa, 'xy')
        yt = _get_tokenized_labels(yg, ya, 'xy')
        zt = _get_tokenized_labels(zg, za, 'z')

        wt = _get_tokenized_labels(dxg, wa, 'wlh')
        lt = _get_tokenized_labels(dyg, la, 'wlh')
        ht = _get_tokenized_labels(dzg, ha, 'wlh')

        cost = _get_tokenized_labels(torch.cos(rg), 0, 'theta')
        sint = _get_tokenized_labels(torch.sin(rg), 0, 'theta')

        return [xt, yt, zt, wt, lt, ht, cost, sint]

    def decode_matcher(self, preds, anchor_xy, anchor_zwlh, bin_size, bin_num):
        def _get_box_params(pred, anc, attr_key):
            rlt = (pred.argmax(dim=1) - int(bin_num[attr_key] / 2) + 0.5) * bin_size[attr_key] + anc
            return rlt
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = preds

        xa, ya = torch.split(anchor_xy, 1, dim=1)
        za, wa, la, ha = torch.split(anchor_zwlh, 1, dim=-1)

        xp = _get_box_params(xt, xa.squeeze(dim=1), 'xy')
        yp = _get_box_params(yt, ya.squeeze(dim=1), 'xy')
        zp = _get_box_params(zt, za, 'z')

        wp = _get_box_params(dxt, wa, 'wlh')
        lp = _get_box_params(dyt, la, 'wlh')
        hp = _get_box_params(dzt, ha, 'wlh')
        cosp = _get_box_params(cost, 0, 'theta')
        sinp = _get_box_params(sint, 0, 'theta')

        cgs = [t for t in cts]
        return torch.stack([xp, yp, zp, wp, lp, hp, cosp, sinp, *cgs], dim=-1)

    def decode_torch(self, preds, anchor_xy, anchor_zwlh, bin_size, bin_num):
        def _get_box_params(pred, anc, attr_key):
            rlt = (pred.argmax(dim=1) - int(bin_num[attr_key] / 2) + 0.5) * bin_size[attr_key] + anc
            return rlt
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = preds

        xa, ya = torch.split(anchor_xy, 1, dim=1)
        za, wa, la, ha = torch.split(anchor_zwlh, 1, dim=-1)

        xp = _get_box_params(xt, xa.squeeze(dim=1), 'xy')
        yp = _get_box_params(yt, ya.squeeze(dim=1), 'xy')
        zp = _get_box_params(zt, za, 'z')

        wp = _get_box_params(dxt, wa, 'wlh')
        lp = _get_box_params(dyt, la, 'wlh')
        hp = _get_box_params(dzt, ha, 'wlh')
        cosp = _get_box_params(cost, 0, 'theta')
        sinp = _get_box_params(sint, 0, 'theta')


        rp = torch.atan2(sinp, cosp)

        cgs = [t for t in cts]
        return torch.stack([xp, yp, zp, wp, lp, hp, rp, *cgs], dim=-1)