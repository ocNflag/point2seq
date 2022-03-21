import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn

from . import point_voxel_ops_cuda as point_voxel_ops

class FillDense(Function):

    @staticmethod
    def forward(ctx, v_indices, xyz_to_vidx, v_indices_range):
        """
        Args:
            v_indices: [num_voxels, 3] zyx voxel indices for one sample
            xyz_to_vidx: [x_max, y_max, z_max] dense 3d
            v_indices_range: z_max, y_max, x_max -> voxel range
        """
        z_max, y_max, x_max = v_indices_range
        num_voxels = v_indices.shape[0]
        point_voxel_ops.fill_dense_wrapper(x_max, y_max, z_max, num_voxels, v_indices, xyz_to_vidx)
        return xyz_to_vidx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

fill_dense = FillDense.apply

class FillDenseStack(Function):

    @staticmethod
    def forward(ctx, v_indices, xyz_to_vidx, v_bs_cnt, v_indices_range):
        """
        Args:
            v_indices: [num_voxels, 4] bs zyx voxel indices for stacked batch
            v_bs_cnt: [B] num_voxels for each sample in batch
            xyz_to_vidx: [B, max_hash_size, 2] xyz to v_idx hash table
            v_indices_range: z_max, y_max, x_max -> voxel range
        """
        z_max, y_max, x_max = v_indices_range
        num_total_voxels = v_indices.shape[0]
        max_hash_size = xyz_to_vidx.shape[1]
        point_voxel_ops.fill_dense_hash_wrapper(x_max, y_max, z_max, num_total_voxels, max_hash_size, v_indices, v_bs_cnt, xyz_to_vidx)
        return xyz_to_vidx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

fill_dense_stack = FillDenseStack.apply

class FillDenseStackSimple(Function):

    @staticmethod
    def forward(ctx, v_indices, xyz_to_vidx, v_bs_cnt, v_indices_range):
        """
        Args:
            v_indices: [num_voxels, 4] bs zyx voxel indices for stacked batch
            v_bs_cnt: [B] num_voxels for each sample in batch
            xyz_to_vidx: [B, max_hash_size, 2] xyz to v_idx hash table
            v_indices_range: z_max, y_max, x_max -> voxel range
        """
        z_max, y_max, x_max = v_indices_range
        num_total_voxels = v_indices.shape[0]
        max_hash_size = xyz_to_vidx.shape[1]
        point_voxel_ops.fill_dense_simple_wrapper(x_max, y_max, z_max, num_total_voxels, max_hash_size, v_indices, v_bs_cnt, xyz_to_vidx)
        return xyz_to_vidx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

fill_dense_stack_simple = FillDenseStackSimple.apply

class PointVoxelQuery(Function):

    @staticmethod
    def forward(ctx, p_coords, v_indices, xyz_to_vidx, v_size, v_indices_range, num_samples):
        """
        Args:
            p_coords: [num_points, 3] point cloud xyz coordinates for one sample
            v_indices: [num_voxels, 3] zyx voxel indices for one sample
            xyz_to_vidx: [x_max, y_max, z_max] dense 3d
            v_size: x_size, y_size, z_size -> voxel actual size
            v_indices_range: z_max, y_max, x_max -> voxel range
            num_samples: max points in a voxel
        Returns:
            v_map: [num_voxels, num_samples] points indices for each voxel
            v_mask: [num_voxels, 1] num_points in each voxel
        """
        x_size, y_size, z_size = v_size
        z_max, y_max, x_max = v_indices_range
        num_points = p_coords.shape[0]
        num_voxels = v_indices.shape[0]
        v_map = torch.cuda.IntTensor(num_voxels, num_samples).fill_(-1) # filled with -1
        v_mask = torch.cuda.IntTensor(num_voxels, 1).zero_()
        point_voxel_ops.point_to_voxel_query_wrapper(x_size, y_size, z_size, x_max, y_max, z_max,
                                                    num_points, num_voxels, num_samples,
                                                    p_coords, xyz_to_vidx,
                                                    v_map, v_mask)
        return v_map, v_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None

point_voxel_query = PointVoxelQuery.apply

class PointVoxelQueryStack(Function):

    @staticmethod
    def forward(ctx, p_coords, v_indices, p_bs_cnt, v_bs_cnt, xyz_to_vidx, v_size, v_indices_range, num_samples):
        """
        Args:
            p_coords: [num_points, 4] point cloud bs xyz coordinates for stacked batch
            v_indices: [num_voxels, 4] bs zyx voxel indices for stacked batch
            p_bs_cnt: [B] num_points for each sample in batch
            v_bs_cnt: [B] num_voxels for each sample in batch
            xyz_to_vidx: [B, max_hash_size, 2] xyz to v_idx hash table
            v_size: x_size, y_size, z_size -> voxel actual size
            v_indices_range: z_max, y_max, x_max -> voxel range
            num_samples: max points in a voxel
        Returns:
            v_map: [num_voxels, num_samples] points indices for each voxel
            v_mask: [num_voxels, 1] num_points in each voxel
        """
        x_size, y_size, z_size = v_size
        z_max, y_max, x_max = v_indices_range
        num_total_points = p_coords.shape[0]
        num_total_voxels = v_indices.shape[0]
        max_hash_size = xyz_to_vidx.shape[1]
        v_map = torch.cuda.IntTensor(num_total_voxels, num_samples).fill_(-1) # filled with -1
        v_mask = torch.cuda.IntTensor(num_total_voxels, 1).zero_()
        point_voxel_ops.point_to_voxel_query_hash_wrapper(x_size, y_size, z_size, x_max, y_max, z_max,
                                                            num_total_points, num_total_voxels, num_samples, max_hash_size,
                                                            p_bs_cnt, v_bs_cnt, p_coords, xyz_to_vidx,
                                                            v_map, v_mask)
        return v_map, v_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None, None, None, None

point_voxel_query_stack = PointVoxelQueryStack.apply

class VoxelPointQuery(Function):

    @staticmethod
    def forward(ctx, p_coords, v_indices, xyz_to_vidx, v_size, v_indices_range, num_samples, step_range = 1):
        """
        Args:
            p_coords: [num_points, 3] point cloud xyz coordinates for one sample
            v_indices: [num_voxels, 3] zyx voxel indices for one sample
            xyz_to_vidx: [x_max, y_max, z_max] dense 3d
            v_size: x_size, y_size, z_size -> voxel actual size
            v_indices_range: z_max, y_max, x_max -> voxel range
            num_samples: max voxel centers around each point
        Returns:
            p_map: [num_points, num_samples] voxel indices around each point
            p_mask: [num_points, 1] num_voxel_centers around each point
        """
        x_size, y_size, z_size = v_size
        z_max, y_max, x_max = v_indices_range
        num_points = p_coords.shape[0]
        num_voxels = v_indices.shape[0]
        p_map = torch.cuda.IntTensor(num_points, num_samples).fill_(-1) # filled with -1
        p_mask = torch.cuda.IntTensor(num_points, 1).zero_()
        point_voxel_ops.voxel_to_point_query_wrapper(x_size, y_size, z_size, x_max, y_max, z_max,
                                                    num_points, num_voxels, num_samples, step_range,
                                                    p_coords, xyz_to_vidx,
                                                    p_map, p_mask)
        return p_map, p_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None

voxel_point_query = VoxelPointQuery.apply

class VoxelPointQueryBin(Function):

    @staticmethod
    def forward(ctx, p_coords, v_indices, p_bs_cnt, v_bs_cnt, xyz_to_vidx, v_size, v_indices_range, num_samples, v_step, v_dilate, v_bin):
        """
        Args:
            p_coords: [num_points, 4] point cloud bs xyz coordinates for stacked batch
            v_indices: [num_voxels, 4] bs zyx voxel indices for stacked batch
            p_bs_cnt: [B] num_points for each sample in batch
            v_bs_cnt: [B] num_voxels for each sample in batch
            xyz_to_vidx: [B, max_hash_size, 2] xyz to v_idx hash table
            v_size: x_size, y_size, z_size -> voxel actual size
            v_indices_range: z_max, y_max, x_max -> voxel range
            num_samples: max points in a voxel
        Returns:
            p_map: [num_points, num_samples] voxel indices around each point
            p_mask: [num_points, 1] num_voxel_centers around each point
        """
        x_size, y_size, z_size = v_size
        z_max, y_max, x_max = v_indices_range
        x_step, y_step, z_step = v_step
        x_dilate, y_dilate, z_dilate = v_dilate
        x_divide, y_divide, z_divide = v_bin

        num_bins = x_divide * y_divide * z_divide
        num_total_points = p_coords.shape[0]
        num_total_voxels = v_indices.shape[0]
        max_hash_size = xyz_to_vidx.shape[1]

        # p_map = torch.cuda.IntTensor(num_total_points, num_samples).fill_(-1) # filled with -1
        p_map = torch.cuda.IntTensor(num_total_points, num_samples).zero_() # filled with 0
        p_mask = torch.cuda.IntTensor(num_total_points, 1).zero_()
        p_bin = torch.cuda.IntTensor(num_total_points, num_bins, num_samples).fill_(-1)

        point_voxel_ops.voxel_to_point_query_bin_wrapper(x_size, y_size, z_size, x_max, y_max, z_max,
                                                            x_step, y_step, z_step, x_dilate, y_dilate, z_dilate,
                                                            x_divide, y_divide, z_divide,
                                                            num_total_points, num_total_voxels, num_samples, max_hash_size,
                                                            p_bs_cnt, p_coords, xyz_to_vidx, p_map, p_mask, p_bin)
        return p_map, p_mask, p_bin

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None, None, None

voxel_point_query_bin = VoxelPointQueryBin.apply

class VoxelPointQueryBinSimple(Function):

    @staticmethod
    def forward(ctx, p_coords, v_indices, p_bs_cnt, v_bs_cnt, xyz_to_vidx, v_size, v_indices_range, num_samples, v_step, v_dilate, v_bin):
        """
        Args:
            p_coords: [num_points, 4] point cloud bs xyz coordinates for stacked batch
            v_indices: [num_voxels, 4] bs zyx voxel indices for stacked batch
            p_bs_cnt: [B] num_points for each sample in batch
            v_bs_cnt: [B] num_voxels for each sample in batch
            xyz_to_vidx: [B, max_hash_size, 2] xyz to v_idx hash table
            v_size: x_size, y_size, z_size -> voxel actual size
            v_indices_range: z_max, y_max, x_max -> voxel range
            num_samples: max points in a voxel
        Returns:
            p_map: [num_points, num_samples] voxel indices around each point
            p_mask: [num_points, 1] num_voxel_centers around each point
        """
        x_size, y_size, z_size = v_size
        z_max, y_max, x_max = v_indices_range
        x_step, y_step, z_step = v_step
        x_dilate, y_dilate, z_dilate = v_dilate
        x_divide, y_divide, z_divide = v_bin

        num_bins = x_divide * y_divide * z_divide
        num_total_points = p_coords.shape[0]
        num_total_voxels = v_indices.shape[0]
        max_hash_size = xyz_to_vidx.shape[1]

        # p_map = torch.cuda.IntTensor(num_total_points, num_samples).fill_(-1) # filled with -1
        p_map = torch.cuda.IntTensor(num_total_points, num_samples).zero_() # filled with 0
        p_mask = torch.cuda.IntTensor(num_total_points, 1).zero_()
        p_bin = torch.cuda.IntTensor(num_total_points, num_bins, num_samples).fill_(-1)

        point_voxel_ops.voxel_to_point_query_simple_wrapper(x_size, y_size, z_size, x_max, y_max, z_max,
                                                            x_step, y_step, z_step, x_dilate, y_dilate, z_dilate,
                                                            x_divide, y_divide, z_divide,
                                                            num_total_points, num_total_voxels, num_samples, max_hash_size,
                                                            p_bs_cnt, p_coords, xyz_to_vidx, p_map, p_mask, p_bin)
        return p_map, p_mask, p_bin

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None, None, None

voxel_point_query_bin_simple = VoxelPointQueryBinSimple.apply

class VoxelPointQueryStack(Function):

    @staticmethod
    def forward(ctx, p_coords, v_indices, p_bs_cnt, v_bs_cnt, xyz_to_vidx, v_size, v_indices_range, num_samples, v_step, v_dilate):
        """
        Args:
            p_coords: [num_points, 4] point cloud bs xyz coordinates for stacked batch
            v_indices: [num_voxels, 4] bs zyx voxel indices for stacked batch
            p_bs_cnt: [B] num_points for each sample in batch
            v_bs_cnt: [B] num_voxels for each sample in batch
            xyz_to_vidx: [B, max_hash_size, 2] xyz to v_idx hash table
            v_size: x_size, y_size, z_size -> voxel actual size
            v_indices_range: z_max, y_max, x_max -> voxel range
            num_samples: max points in a voxel
        Returns:
            p_map: [num_points, num_samples] voxel indices around each point
            p_mask: [num_points, 1] num_voxel_centers around each point
        """
        x_size, y_size, z_size = v_size
        z_max, y_max, x_max = v_indices_range
        x_step, y_step, z_step = v_step
        x_dilate, y_dilate, z_dilate = v_dilate

        num_total_points = p_coords.shape[0]
        num_total_voxels = v_indices.shape[0]
        max_hash_size = xyz_to_vidx.shape[1]

        # p_map = torch.cuda.IntTensor(num_total_points, num_samples).fill_(-1) # filled with -1
        p_map = torch.cuda.IntTensor(num_total_points, num_samples).zero_() # filled with 0
        p_mask = torch.cuda.IntTensor(num_total_points, 1).zero_()

        point_voxel_ops.voxel_to_point_query_hash_wrapper(x_size, y_size, z_size, x_max, y_max, z_max,
                                                            x_step, y_step, z_step, x_dilate, y_dilate, z_dilate,
                                                            num_total_points, num_total_voxels, num_samples, max_hash_size,
                                                            p_bs_cnt, p_coords, xyz_to_vidx, p_map, p_mask)
        return p_map, p_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None, None, None

voxel_point_query_stack = VoxelPointQueryStack.apply