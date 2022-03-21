#include <math.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

__device__ float limit_period_all(float val, float offset, float period){
    float rval = val - floor(val / period + offset) * period;
    return rval;
}

__device__ float gaussian_radius_all(float height, float width, float min_overlap){
    float a1 = 1;
    float b1 = (height + width);
    float c1 = width * height * (1 - min_overlap) / (1 + min_overlap);
    float sq1 = sqrt(b1 * b1 - 4 * a1 * c1);
    float r1 = (b1 + sq1) / 2;

    float a2 = 4;
    float b2 = 2 * (height + width);
    float c2 = (1 - min_overlap) * width * height;
    float sq2 = sqrt(b2 * b2 - 4 * a2 * c2);
    float r2 = (b2 + sq2) / 2;

    float a3 = 4 * min_overlap;
    float b3 = -2 * min_overlap * (height + width);
    float c3 = (min_overlap - 1) * width * height;
    float sq3 = sqrt(b3 * b3 - 4 * a3 * c3);
    float r3 = (b3 + sq3) / 2;
    return min(min(r1, r2), r3);
}

__global__ void draw_all_kernel(int batch_size, int max_boxes, int max_objs, int num_cls, int H, int W, int code_size,
                                    int min_radius, float voxel_x, float voxel_y, float range_x, float range_y,
                                    float out_factor, float gaussian_overlap, float fg_enlarge_ratio,
                                    const float *gt_boxes, const float *gt_corners, float *center_map,
                                    float *corner_map, float *foreground_map, int *gt_ind, int *gt_mask, int *gt_cat,
                                    float *gt_box_encoding, int *gt_cnt){

    /*
        Args:
            gt_boxes: (B, max_boxes, 8 or 10) with class labels
            gt_corners: (B, max_boxes, 4, 2) 4 corner coords
            center_map: (B, num_cls, H, W)
            corner_map: (B, num_cls, 4, H, W) each corner has a seperate map
            foreground_map: (B, num_cls, H, W)
            gt_ind: (B, num_cls, max_objs)
            gt_mask: (B, num_cls, max_objs)
            gt_cat: (B, num_cls, max_objs)
            gt_box_encoding: (B, num_cls, max_objs, code_size) sin/cos
            gt_cnt: (B, num_cls)
    */

    int bs_idx = blockIdx.x;
    int box_idx = threadIdx.x;
    if (bs_idx >= batch_size || box_idx >= max_boxes) return;

    // move pointer
    gt_boxes += bs_idx * max_boxes * code_size;
    gt_corners += bs_idx * max_boxes * 4 * 2;
    center_map += bs_idx * num_cls * H * W;
    corner_map += bs_idx * num_cls * 4 * H * W;
    foreground_map += bs_idx * num_cls * H * W;
    gt_ind += bs_idx * num_cls * max_objs;
    gt_mask += bs_idx * num_cls * max_objs;
    gt_cat += bs_idx * num_cls * max_objs;
    gt_box_encoding += bs_idx * num_cls * max_objs * code_size;
    gt_cnt += bs_idx * num_cls;

    // gt box parameters
    float x = gt_boxes[box_idx * code_size + 0];
    float y = gt_boxes[box_idx * code_size + 1];
    float z = gt_boxes[box_idx * code_size + 2];
    float dx = gt_boxes[box_idx * code_size + 3];
    float dy = gt_boxes[box_idx * code_size + 4];
    float dz = gt_boxes[box_idx * code_size + 5];
    float rot = gt_boxes[box_idx * code_size + 6];
    float vel_x = 0;
    float vel_y = 0;
    float cls = 0;
    if (code_size == 10) {
        vel_x = gt_boxes[box_idx * code_size + 7];
        vel_y = gt_boxes[box_idx * code_size + 8];
        cls = gt_boxes[box_idx * code_size + 9];
    } else if (code_size == 8) {
        cls = gt_boxes[box_idx * code_size + 7];
    } else {
        return;
    }

    // box not defined
    if (dx == 0 || dy == 0 || dz == 0) return;

    // gt corner coordinates
    float corner_x0 = gt_corners[box_idx * 4 * 2 + 0 * 2 + 0];
    float corner_y0 = gt_corners[box_idx * 4 * 2 + 0 * 2 + 1];
    float corner_x1 = gt_corners[box_idx * 4 * 2 + 1 * 2 + 0];
    float corner_y1 = gt_corners[box_idx * 4 * 2 + 1 * 2 + 1];
    float corner_x2 = gt_corners[box_idx * 4 * 2 + 2 * 2 + 0];
    float corner_y2 = gt_corners[box_idx * 4 * 2 + 2 * 2 + 1];
    float corner_x3 = gt_corners[box_idx * 4 * 2 + 3 * 2 + 0];
    float corner_y3 = gt_corners[box_idx * 4 * 2 + 3 * 2 + 1];


    // cls begin from 1
    int cls_idx = (int) cls - 1;
    center_map += cls_idx * H * W;
    corner_map += cls_idx * 4 * H * W;
    foreground_map += cls_idx * H * W;
    gt_ind += cls_idx * max_objs;
    gt_mask += cls_idx * max_objs;
    gt_cat += cls_idx * max_objs;
    gt_box_encoding += cls_idx * max_objs * code_size;
    gt_cnt += cls_idx;

    // transform the center to bev map coords
    float offset = 0.5;
    float period = 6.283185307179586;
    rot = limit_period_all(rot, offset, period);
    float coor_dx = dx / voxel_x / out_factor;
    float coor_dy = dy / voxel_y / out_factor;
    float radius = gaussian_radius_all(coor_dy, coor_dx, gaussian_overlap);
    // note that gaussian radius is shared both center_map and corner_map
    // this can be modified according to needs
    int radius_int = max(min_radius, (int) radius);
    float coor_x = (x - range_x) / voxel_x / out_factor;
    float coor_y = (y - range_y) / voxel_y / out_factor;
    int coor_x_int = (int) coor_x;
    int coor_y_int = (int) coor_y;
    // if center is outside, directly return
    if (coor_x_int >= W || coor_x_int < 0 || coor_y_int >= H || coor_y_int < 0) return;

    // transform corners to bev map coords
    float coor_corner_x0 = (corner_x0 - range_x) / voxel_x / out_factor;
    float coor_corner_y0 = (corner_y0 - range_y) / voxel_y / out_factor;
    int coor_corner_x0_int = (int) coor_corner_x0;
    int coor_corner_y0_int = (int) coor_corner_y0;
    float coor_corner_x1 = (corner_x1 - range_x) / voxel_x / out_factor;
    float coor_corner_y1 = (corner_y1 - range_y) / voxel_y / out_factor;
    int coor_corner_x1_int = (int) coor_corner_x1;
    int coor_corner_y1_int = (int) coor_corner_y1;
    float coor_corner_x2 = (corner_x2 - range_x) / voxel_x / out_factor;
    float coor_corner_y2 = (corner_y2 - range_y) / voxel_y / out_factor;
    int coor_corner_x2_int = (int) coor_corner_x2;
    int coor_corner_y2_int = (int) coor_corner_y2;
    float coor_corner_x3 = (corner_x3 - range_x) / voxel_x / out_factor;
    float coor_corner_y3 = (corner_y3 - range_y) / voxel_y / out_factor;
    int coor_corner_x3_int = (int) coor_corner_x3;
    int coor_corner_y3_int = (int) coor_corner_y3;


    // draw gaussian center_map and corner_map
    float div_factor = 6.0;
    float sigma = (2 * radius_int + 1) / div_factor;
    for (int scan_y = -radius_int; scan_y < radius_int + 1; scan_y++){
        for (int scan_x = -radius_int; scan_x < radius_int + 1; scan_x++){
            float weight = exp(-(scan_x * scan_x + scan_y * scan_y) / (2 * sigma * sigma)); // force convert float sigma
            float eps = 0.0000001;
            if (weight < eps) weight = 0;

            // draw center
            if (coor_x_int + scan_x < 0 || coor_x_int + scan_x >= W || coor_y_int + scan_y < 0 || coor_y_int + scan_y >= H){
                ;
            } else{
                float *center_addr = center_map + (coor_y_int + scan_y) * W + (coor_x_int + scan_x);
                float center_exch_weight = atomicExch(center_addr, weight);
                if (center_exch_weight > weight) center_exch_weight = atomicExch(center_addr, center_exch_weight);
            }

            // draw 4 corners
            if (coor_corner_x0_int + scan_x < 0 || coor_corner_x0_int + scan_x >= W || coor_corner_y0_int + scan_y < 0 || coor_corner_y0_int + scan_y >= H){
                ;
            } else{
                float *corner_addr0 = corner_map + 0 * W * H + (coor_corner_y0_int + scan_y) * W + (coor_corner_x0_int + scan_x);
                float corner_exch_weight0 = atomicExch(corner_addr0, weight);
                if (corner_exch_weight0 > weight) corner_exch_weight0 = atomicExch(corner_addr0, corner_exch_weight0);
            }
            if (coor_corner_x1_int + scan_x < 0 || coor_corner_x1_int + scan_x >= W || coor_corner_y1_int + scan_y < 0 || coor_corner_y1_int + scan_y >= H){
                ;
            } else{
                float *corner_addr1 = corner_map + 1 * W * H + (coor_corner_y1_int + scan_y) * W + (coor_corner_x1_int + scan_x);
                float corner_exch_weight1 = atomicExch(corner_addr1, weight);
                if (corner_exch_weight1 > weight) corner_exch_weight1 = atomicExch(corner_addr1, corner_exch_weight1);
            }
            if (coor_corner_x2_int + scan_x < 0 || coor_corner_x2_int + scan_x >= W || coor_corner_y2_int + scan_y < 0 || coor_corner_y2_int + scan_y >= H){
                ;
            } else{
                float *corner_addr2 = corner_map + 2 * W * H + (coor_corner_y2_int + scan_y) * W + (coor_corner_x2_int + scan_x);
                float corner_exch_weight2 = atomicExch(corner_addr2, weight);
                if (corner_exch_weight2 > weight) corner_exch_weight2 = atomicExch(corner_addr2, corner_exch_weight2);
            }
            if (coor_corner_x3_int + scan_x < 0 || coor_corner_x3_int + scan_x >= W || coor_corner_y3_int + scan_y < 0 || coor_corner_y3_int + scan_y >= H){
                ;
            } else{
                float *corner_addr3 = corner_map + 3 * W * H + (coor_corner_y3_int + scan_y) * W + (coor_corner_x3_int + scan_x);
                float corner_exch_weight3 = atomicExch(corner_addr3, weight);
                if (corner_exch_weight3 > weight) corner_exch_weight3 = atomicExch(corner_addr3, corner_exch_weight3);
            }
        }
    }

    // draw foreground map
    int box_range = (0.5 * coor_dx + 0.5 * coor_dy) * fg_enlarge_ratio;
    float cosa = cos(-rot), sina = sin(-rot);
    float fg_dx = dx * fg_enlarge_ratio, fg_dy = dy * fg_enlarge_ratio;
    for (int scan_y = -box_range; scan_y < box_range + 1; scan_y++){
        for (int scan_x = -box_range; scan_x < box_range + 1; scan_x++){
            if (coor_x_int + scan_x < 0 || coor_x_int + scan_x >= W || coor_y_int + scan_y < 0 || coor_y_int + scan_y >= H) continue;
            float box_x = (coor_x_int + scan_x + 0.5) * voxel_x * out_factor + range_x;
            float box_y = (coor_y_int + scan_y + 0.5) * voxel_y * out_factor + range_y;
            float local_x = (box_x - x) * cosa + (box_y - y) * (-sina);
            float local_y = (box_x - x) * sina + (box_y - y) * cosa;
            if ((fabs(local_x) <= fg_dx / 2.0) && (fabs(local_y) <= fg_dy / 2.0)){
                float *fg_addr = foreground_map + (coor_y_int + scan_y) * W + (coor_x_int + scan_x);
                // assign all fg pixels to 1
                // this assignment can be modified according to needs (e.g. decay)
                float fg_exch_weight = atomicExch(fg_addr, 1.0);
            }
        }
    }

    int obj_idx = atomicAdd(gt_cnt, 1);
    if (obj_idx >= max_objs) return;
    gt_ind[obj_idx] = coor_y_int * W + coor_x_int;
    gt_mask[obj_idx] = 1;
    gt_cat[obj_idx] = cls_idx + 1; // begin from 1
    gt_box_encoding[obj_idx * code_size + 0] = coor_x - coor_x_int;
    gt_box_encoding[obj_idx * code_size + 1] = coor_y - coor_y_int;
    gt_box_encoding[obj_idx * code_size + 2] = z;
    gt_box_encoding[obj_idx * code_size + 3] = dx;
    gt_box_encoding[obj_idx * code_size + 4] = dy;
    gt_box_encoding[obj_idx * code_size + 5] = dz;
    gt_box_encoding[obj_idx * code_size + 6] = sin(rot);
    gt_box_encoding[obj_idx * code_size + 7] = cos(rot);
    if (code_size == 10) {
        gt_box_encoding[obj_idx * code_size + 8] = vel_x;
        gt_box_encoding[obj_idx * code_size + 9] = vel_y;
    }
    return;
}

void draw_all_kernel_launcher(int batch_size, int max_boxes, int max_objs, int num_cls, int H, int W, int code_size, int min_radius,
                              float voxel_x, float voxel_y, float range_x, float range_y, float out_factor,
                              float gaussian_overlap, float fg_enlarge_ratio,
                              const float *gt_boxes, const float *gt_corners,
                              float *center_map, float *corner_map, float *foreground_map,
                              int *gt_ind, int *gt_mask, int *gt_cat, float *gt_box_encoding, int *gt_cnt){
    cudaError_t err;

    dim3 blocks(batch_size);
    dim3 threads(THREADS_PER_BLOCK);
    draw_all_kernel<<<blocks, threads>>>(batch_size, max_boxes, max_objs, num_cls, H, W, code_size, min_radius,
                                         voxel_x, voxel_y, range_x, range_y, out_factor,
                                         gaussian_overlap, fg_enlarge_ratio,
                                         gt_boxes, gt_corners,
                                         center_map, corner_map, foreground_map,
                                         gt_ind, gt_mask, gt_cat, gt_box_encoding, gt_cnt);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}