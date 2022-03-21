import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from ...utils import box_utils
from . import roipoint_pool3d_stack_cuda


class RoIPointPool3dStack(nn.Module):
    def __init__(self, num_sampled_points=512, pool_extra_width=1.0):
        super().__init__()
        self.num_sampled_points = num_sampled_points
        self.pool_extra_width = pool_extra_width

    def forward(self, points, point_features, points_batch_id, point_batch_cnt, boxes3d):
        """
        Args:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, M, 7), [x, y, z, dx, dy, dz, heading]

        Returns:
            pooled_features: (B, M, 512, 3 + C)
            pooled_empty_flag: (B, M)
        """
        return RoIPointPool3dStackFunction.apply(
            points, point_features, points_batch_id, point_batch_cnt,
            boxes3d, self.pool_extra_width, self.num_sampled_points
        )


class RoIPointPool3dStackFunction(Function):
    @staticmethod
    def forward(
            ctx, points, point_features, points_batch_id,
            point_batch_cnt, boxes3d, pool_extra_width, num_sampled_points=512
    ):
        """
        Args:
            ctx:
            points: (P, 3)
            point_features: (P, C)
            boxes3d: (B, num_boxes, 7), [x, y, z, dx, dy, dz, heading]
            pool_extra_width:
            num_sampled_points:

        Returns:
            pooled_features: (B, num_boxes, 512, 3 + C)
            pooled_empty_flag: (B, num_boxes)
        """
        assert points.shape.__len__() == 2 and points.shape[1] == 3
        pool_extra_width = (pool_extra_width, pool_extra_width, pool_extra_width)
        batch_size, boxes_num, feature_len = boxes3d.shape[0], boxes3d.shape[1], point_features.shape[1]
        pooled_boxes3d = box_utils.enlarge_box3d(boxes3d.view(-1, 7), pool_extra_width).view(batch_size, -1, 7)

        pooled_features = point_features.new_zeros((batch_size, boxes_num, num_sampled_points, 3 + feature_len))
        pooled_empty_flag = point_features.new_zeros((batch_size, boxes_num)).int()
        point_empty_flag = point_features.new_ones((batch_size, boxes_num, num_sampled_points)).int() * -1

        roipoint_pool3d_stack_cuda.forward(
            pooled_boxes3d.contiguous(), points_batch_id.contiguous(), point_batch_cnt.contiguous(),
            points.contiguous(), point_features.contiguous(),
            pooled_features, pooled_empty_flag, point_empty_flag
        )

        return pooled_features, pooled_empty_flag, point_empty_flag

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class RoIPointQueryStack(nn.Module):
    def __init__(self, num_sampled_points=512, pool_extra_width=1.0):
        super().__init__()
        self.num_sampled_points = num_sampled_points
        self.pool_extra_width = pool_extra_width

    def forward(self, points, points_batch_id, point_batch_cnt, boxes3d):
        """
        Args:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, M, 7), [x, y, z, dx, dy, dz, heading]

        Returns:
            point_ids: (B, M, 512, 3 + C)
        """
        return RoIPointQueryStackFunction.apply(
            points, points_batch_id, point_batch_cnt,
            boxes3d, self.pool_extra_width, self.num_sampled_points
        )


class RoIPointQueryStackFunction(Function):
    @staticmethod
    def forward(
            ctx, points, points_batch_id,
            point_batch_cnt, boxes3d, pool_extra_width, num_sampled_points=512
    ):
        """
        Args:
            ctx:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, num_boxes, 7), [x, y, z, dx, dy, dz, heading]
            pool_extra_width:
            num_sampled_points:

        Returns:
            point_idx: (B, num_boxes, 512, 3 + C)
        """
        assert points.shape.__len__() == 2 and points.shape[1] == 3
        pool_extra_width = (pool_extra_width, pool_extra_width, pool_extra_width)
        batch_size, boxes_num = boxes3d.shape[0], boxes3d.shape[1]
        pooled_boxes3d = box_utils.enlarge_box3d(boxes3d.view(-1, 7), pool_extra_width).view(batch_size, boxes_num, 7)

        point_idx = points.new_ones((batch_size, boxes_num, num_sampled_points)).int() * -1

        roipoint_pool3d_stack_cuda.point_query(
            pooled_boxes3d.contiguous(), points_batch_id.contiguous(),
            point_batch_cnt.contiguous(), points.contiguous(),
            point_idx
        )

        return point_idx

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, features_batch_cnt: torch.Tensor,
                idx: torch.Tensor, idx_batch_cnt: torch.Tensor):
        """
        Args:
            ctx:
            features: (N1 + N2 ..., C) tensor of features to group
            features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
            idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
            idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with

        Returns:
            output: (M1 + M2, C, nsample) tensor
        """
        assert features.is_contiguous()
        assert features_batch_cnt.is_contiguous()
        assert idx.is_contiguous()
        assert idx_batch_cnt.is_contiguous()

        assert features.shape[0] == features_batch_cnt.sum(), \
            'features: %s, features_batch_cnt: %s' % (str(features.shape), str(features_batch_cnt))
        assert idx.shape[0] == idx_batch_cnt.sum(), \
            'idx: %s, idx_batch_cnt: %s' % (str(idx.shape), str(idx_batch_cnt))

        M, nsample = idx.size()
        N, C = features.size()
        B = idx_batch_cnt.shape[0]
        output = torch.cuda.FloatTensor(M, C, nsample).zero_()

        roipoint_pool3d_stack_cuda.group_features_wrapper(B, M, C, nsample, features, features_batch_cnt, idx,
                                                          idx_batch_cnt, output)

        ctx.for_backwards = (B, N, idx, features_batch_cnt, idx_batch_cnt)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward

        Returns:
            grad_features: (N1 + N2 ..., C) gradient of the features
        """
        B, N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards

        M, C, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(N, C).zero_())

        grad_out_data = grad_out.data.contiguous()
        roipoint_pool3d_stack_cuda.group_features_grad_wrapper(B, M, C, N, nsample, grad_out_data, idx,
                                                               idx_batch_cnt, features_batch_cnt, grad_features.data)
        return grad_features, None, None, None


grouping_operation = GroupingOperation.apply

if __name__ == '__main__':
    pass
