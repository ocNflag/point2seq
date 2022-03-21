/*
Point cloud feature pooling
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <math.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
// #define DEBUG


__device__ inline void lidar_to_local_coords(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y){
    float cosa = cos(-rot_angle), sina = sin(-rot_angle);
    local_x = shift_x * cosa + shift_y * (-sina);
    local_y = shift_x * sina + shift_y * cosa;
}


__device__ inline int check_pt_in_box3d(const float *pt, const float *box3d, float &local_x, float &local_y){
    // param pt: (x, y, z)
    // param box3d: [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center

    const float MARGIN = 1e-5;
    float x = pt[0], y = pt[1], z = pt[2];
    float cx = box3d[0], cy = box3d[1], cz = box3d[2];
    float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];

    if (fabsf(z - cz) > dz / 2.0) return 0;
    lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
    float in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    return in_flag;
}


__global__ void assign_pts_to_box3d_stack(
    int batch_size, int pts_num, int boxes_num,
    const int *pts_batch_id, const float *xyz, const float *boxes3d, int *pts_assign
){
    // params xyz: (P, 3)
    // params boxes3d: (B, M, 7)
    // params pts_assign: (B, N, M): idx of the corresponding box3d, -1 means background points

    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_idx = blockIdx.y;
    int bs_idx = pts_batch_id[pt_idx];

    if (pt_idx >= pts_num || box_idx >= boxes_num){
        return;
    }
    int assign_idx = pt_idx * boxes_num + box_idx;
    pts_assign[assign_idx] = 0;

    int box_offset = bs_idx * boxes_num * 7 + box_idx * 7;
    int pt_offset = pt_idx * 3;


    float local_x = 0, local_y = 0;
    int cur_in_flag = check_pt_in_box3d(xyz + pt_offset, boxes3d + box_offset, local_x, local_y);
    pts_assign[assign_idx] = cur_in_flag;
}


__global__ void get_pooled_idx_stack(int batch_size, int pts_num, int boxes_num, int sampled_pts_num, const int *pts_batch_cnt,
                               const int *pts_assign, int *pts_idx, int *pooled_empty_flag, int *point_empty_flag){
    // params xyz: (B, N, 3)
    // params pts_feature: (B, N, C)
    // params pts_assign: (B, N)
    // params pts_idx: (B, M, 512)
    // params pooled_empty_flag: (B, M)

    int boxes_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (boxes_idx >= boxes_num){
        return;
    }

    int bs_idx = blockIdx.y;
    int pts_start_idx = 0;

    for (int k = 0; k < bs_idx; k++) {
        pts_start_idx += pts_batch_cnt[k];
    }

    pts_assign += pts_start_idx * boxes_num;

    int cnt = 0;
    int n = pts_batch_cnt[bs_idx];
    for (int k = 0; k < n; k++){
        if (pts_assign[k * boxes_num + boxes_idx]){
            if (cnt < sampled_pts_num){
                pts_idx[bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num + cnt] = k + pts_start_idx;
                point_empty_flag[bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num + cnt] = 1;
                cnt++;
            }
            else break;
        }
    }

    if (cnt == 0){
        pooled_empty_flag[bs_idx * boxes_num + boxes_idx] = 1;
    }

    else if (cnt < sampled_pts_num){
        // duplicate same points for sampling
        for (int k = cnt; k < sampled_pts_num; k++){
            int base_offset = bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num;
            pts_idx[base_offset + k] = -1;
        }
    }
}


__global__ void roipool3d_forward_stack(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num,
                                   const float *xyz, const int *pts_idx, const float *pts_feature,
                                   float *pooled_features, int *pooled_empty_flag){
    // params xyz: (P, 3)
    // params pts_idx: (B, M, 512)
    // params pts_feature: (P, C)
    // params pooled_features: (P, 512, 3+C)
    // params pooled_empty_flag: (B, M)

    int sample_pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_idx = blockIdx.y;
    int bs_idx = blockIdx.z;

    if (sample_pt_idx >= sampled_pts_num || box_idx >= boxes_num || bs_idx >= batch_size){
        return;
    }

    if (pooled_empty_flag[bs_idx * boxes_num + box_idx]){
        return;
    }

    int temp_idx = bs_idx * boxes_num * sampled_pts_num + box_idx * sampled_pts_num + sample_pt_idx;
    int src_pt_idx = pts_idx[temp_idx];
    if (src_pt_idx == -1){
        return;
    }

    int dst_feature_offset = temp_idx * (3 + feature_in_len);

    for (int j = 0; j < 3; j++)
        pooled_features[dst_feature_offset + j] = xyz[src_pt_idx * 3 + j];

    int src_feature_offset = src_pt_idx * feature_in_len;
    for (int j = 0; j < feature_in_len; j++)
        pooled_features[dst_feature_offset + 3 + j] = pts_feature[src_feature_offset + j];
}


__global__ void get_query_idx_stack(int batch_size, int pts_num, int boxes_num, int sampled_pts_num, const int *pts_batch_cnt,
                               const int *pts_assign, int *pts_idx){
    // params xyz: (B, N, 3)
    // params pts_feature: (B, N, C)
    // params pts_assign: (B, N)
    // params pts_idx: (B, M, 512)
    // params pooled_empty_flag: (B, M)

    int boxes_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (boxes_idx >= boxes_num){
        return;
    }

    int bs_idx = blockIdx.y;
    int pts_start_idx = 0;

    for (int k = 0; k < bs_idx; k++) {
        pts_start_idx += pts_batch_cnt[k];
    }

    pts_assign += pts_start_idx * boxes_num;

    int cnt = 0;
    int n = pts_batch_cnt[bs_idx];
    for (int k = 0; k < n; k++){
        if (pts_assign[k * boxes_num + boxes_idx]){
            if (cnt < sampled_pts_num){
                pts_idx[bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num + cnt] = k + pts_start_idx;
                cnt++;
            }
            else break;
        }
    }
}


void ROIPoolStackLauncher(
    int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num,
    const int *pts_batch_id, const int *pts_batch_cnt, const float *xyz,
    const float *boxes3d, const float *pts_feature,
    float *pooled_features, int *pooled_empty_flag, int *point_empty_flag
){

    int *pts_assign = NULL;
    cudaMalloc(&pts_assign, pts_num * boxes_num * sizeof(int));  // (batch_size, N, M)
    // cudaMemset(&pts_assign, -1, batch_size * pts_num * boxes_num * sizeof(int));

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    assign_pts_to_box3d_stack<<<blocks, threads>>>(
        batch_size, pts_num, boxes_num,
        pts_batch_id, xyz, boxes3d, pts_assign
    );

    int *pts_idx = NULL;
    cudaMalloc(&pts_idx, batch_size * boxes_num * sampled_pts_num * sizeof(int));  // (batch_size, M, sampled_pts_num)

    dim3 blocks2(DIVUP(boxes_num, THREADS_PER_BLOCK), batch_size);  // blockIdx.x(col), blockIdx.y(row)
    get_pooled_idx_stack<<<blocks2, threads>>>(
        batch_size, pts_num, boxes_num, sampled_pts_num,
        pts_batch_cnt, pts_assign, pts_idx, pooled_empty_flag, point_empty_flag
    );

    dim3 blocks_pool(DIVUP(sampled_pts_num, THREADS_PER_BLOCK), boxes_num, batch_size);
    roipool3d_forward_stack<<<blocks_pool, threads>>>(
        batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
        xyz, pts_idx, pts_feature, pooled_features, pooled_empty_flag
    );

    cudaFree(pts_assign);
    cudaFree(pts_idx);

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}


void ROIPointQueryStackLauncher(
    int batch_size, int pts_num, int boxes_num, int sampled_pts_num,
    const int *pts_batch_id, const int *pts_batch_cnt, const float *xyz,
    const float *boxes3d, int *point_idx
){

    int *pts_assign = NULL;
    cudaMalloc(&pts_assign, pts_num * boxes_num * sizeof(int));  // (batch_size, N, M)
    // cudaMemset(&pts_assign, -1, batch_size * pts_num * boxes_num * sizeof(int));

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    assign_pts_to_box3d_stack<<<blocks, threads>>>(
        batch_size, pts_num, boxes_num,
        pts_batch_id, xyz, boxes3d, pts_assign
    );

    dim3 blocks2(DIVUP(boxes_num, THREADS_PER_BLOCK), batch_size);  // blockIdx.x(col), blockIdx.y(row)
    get_query_idx_stack<<<blocks2, threads>>>(
        batch_size, pts_num, boxes_num, sampled_pts_num,
        pts_batch_cnt, pts_assign, point_idx
    );

    cudaFree(pts_assign);

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}


__global__ void group_features_grad_kernel(int B, int M, int C, int N, int nsample,
    const float *grad_out, const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt, float *grad_features) {
    // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :return:
    //     grad_features: (N1 + N2 ..., C) gradient of the features
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = index % nsample;
    int C_idx = (index / nsample) % C;
    int pt_idx = (index / nsample / C);

    if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;

    idx += pt_idx * nsample + sample_idx;
    if (idx[0] < 0) return; // don't care neg indices

    int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += idx_batch_cnt[k];
        bs_idx = k;
    }

    int features_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) features_batch_start_idx += features_batch_cnt[k];

    grad_out += pt_idx * C * nsample + C_idx * nsample + sample_idx;

    grad_features += (idx[0] * C) + C_idx;
    atomicAdd(grad_features, grad_out[0]);
}

void group_features_grad_kernel_launcher(int B, int M, int C, int N, int nsample,
    const float *grad_out, const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt, float *grad_features) {
    // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :return:
    //     grad_features: (N1 + N2 ..., C) gradient of the features

    cudaError_t err;
    // dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 blocks(DIVUP(M * C * nsample, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_features_grad_kernel<<<blocks, threads>>>(B, M, C, N, nsample, grad_out, idx, idx_batch_cnt, features_batch_cnt, grad_features);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void group_features_kernel(int B, int M, int C, int nsample,
    const float *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, float *out) {
    // :param features: (N1 + N2 ..., C) tensor of features to group
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :return:
    //     output: (M1 + M2, C, nsample) tensor
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = index % nsample;
    int C_idx = (index / nsample) % C;
    int pt_idx = (index / nsample / C);

    if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;

    idx += pt_idx * nsample + sample_idx;
    if (idx[0] < 0) return; // don't care neg indices

    int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += idx_batch_cnt[k];
        bs_idx = k;
    }

    int features_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) features_batch_start_idx += features_batch_cnt[k];
    // features += features_batch_start_idx * C;

    int in_idx = idx[0] * C + C_idx;
    int out_idx = pt_idx * C * nsample + C_idx * nsample + sample_idx;

    out[out_idx] = features[in_idx];
}


void group_features_kernel_launcher(int B, int M, int C, int nsample,
    const float *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, float *out) {
    // :param features: (N1 + N2 ..., C) tensor of features to group
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :return:
    //     output: (M1 + M2, C, nsample) tensor

    cudaError_t err;
    dim3 blocks(DIVUP(M * C * nsample, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_features_kernel<<<blocks, threads>>>(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
