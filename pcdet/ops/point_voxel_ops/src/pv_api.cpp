#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "fill_dense_gpu.h"
#include "point_to_voxel_gpu.h"
#include "voxel_to_point_gpu.h"
#include "fill_dense_hash_gpu.h"
#include "point_to_voxel_hash_gpu.h"
#include "voxel_to_point_hash_gpu.h"
#include "voxel_to_point_bin_gpu.h"
#include "voxel_to_point_simple_gpu.h"
#include "fill_dense_simple_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fill_dense_wrapper", &fill_dense_wrapper, "fill_dense_wrapper");
    m.def("point_to_voxel_query_wrapper", &point_to_voxel_query_wrapper, "point_to_voxel_query_wrapper");
    m.def("voxel_to_point_query_wrapper", &voxel_to_point_query_wrapper, "voxel_to_point_query_wrapper");
    m.def("fill_dense_hash_wrapper", &fill_dense_hash_wrapper, "fill_dense_hash_wrapper");
    m.def("point_to_voxel_query_hash_wrapper", &point_to_voxel_query_hash_wrapper, "point_to_voxel_query_hash_wrapper");
    m.def("voxel_to_point_query_hash_wrapper", &voxel_to_point_query_hash_wrapper, "voxel_to_point_query_hash_wrapper");
    m.def("voxel_to_point_query_bin_wrapper", &voxel_to_point_query_bin_wrapper, "voxel_to_point_query_bin_wrapper");
    m.def("voxel_to_point_query_simple_wrapper", &voxel_to_point_query_simple_wrapper, "voxel_to_point_query_simple_wrapper");
    m.def("fill_dense_simple_wrapper", &fill_dense_simple_wrapper, "fill_dense_simple_wrapper");
}
