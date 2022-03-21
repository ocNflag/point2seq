/*
Query voxel centers for each point
Written by Jiageng Mao
*/


#ifndef VOXEL_TO_POINT_GPU_H
#define VOXEL_TO_POINT_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int voxel_to_point_query_wrapper(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                 int num_points, int num_voxels, int num_samples, int step_range,
                                 at::Tensor xyz_tensor, at::Tensor xyz_to_vidx_tensor,
                                 at::Tensor p_map_tensor, at::Tensor p_mask_tensor);

void voxel_to_point_query_kernel_launcher(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                          int num_points, int num_voxels, int num_samples, int step_range,
                                          const float *xyz, const int *xyz_to_vidx, int *p_map, int *p_mask);


#endif
