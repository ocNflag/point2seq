/*
Query points for each voxel
Written by Jiageng Mao
*/


#ifndef POINT_TO_VOXEL_GPU_H
#define POINT_TO_VOXEL_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int point_to_voxel_query_wrapper(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                 int num_points, int num_voxels, int num_samples,
                                 at::Tensor xyz_tensor, at::Tensor xyz_to_vidx_tensor,
                                 at::Tensor v_map_tensor, at::Tensor v_mask_tensor);

void point_to_voxel_query_kernel_launcher(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                          int num_points, int num_voxels, int num_samples,
                                          const float *xyz, const int *xyz_to_vidx, int *v_map, int *v_mask);


#endif
