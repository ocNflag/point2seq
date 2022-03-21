/*
Fill dense 3d voxels with indices
Written by Jiageng Mao
*/


#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "fill_dense_simple_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int fill_dense_simple_wrapper(int x_max, int y_max, int z_max, int num_total_voxels, int max_hash_size,
                            at::Tensor v_indices_tensor, at::Tensor v_bs_cnt_tensor, at::Tensor xyz_to_vidx_tensor) {

    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(v_bs_cnt_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);

    const int *v_indices = v_indices_tensor.data<int>();
    const int *v_bs_cnt = v_bs_cnt_tensor.data<int>();
    int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();

    fill_dense_simple_kernel_launcher(x_max, y_max, z_max, num_total_voxels,
                                    max_hash_size, v_indices, v_bs_cnt, xyz_to_vidx);
    return 1;
}
