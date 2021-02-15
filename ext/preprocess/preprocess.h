#pragma once

#include "common.h"

py::array_t<int> count_nbs(array_f32_d tgt_dm_np, array_f32_d tgt_K_np,
                           array_f32_d tgt_R_np, array_f32_d tgt_t_np,
                           array_f32_d src_dms_np, array_f32_d src_Ks_np,
                           array_f32_d src_Rs_np, array_f32_d src_ts_np,
                           float bwd_depth_thresh);

std::tuple<py::array_t<float>, py::array_t<int>, py::array_t<float>,
           py::array_t<float>, py::array_t<int>, py::array_t<int>>
map_source_points(array_f32_d tgt_dm_np, array_f32_d tgt_K_np,
                  array_f32_d tgt_R_np, array_f32_d tgt_t_np,
                  array_i32_d tgt_count, array_f32_d src_dms_np,
                  array_f32_d src_Ks_np, array_f32_d src_Rs_np,
                  array_f32_d src_ts_np, float bwd_depth_thresh,
                  int n_max_sources = -1, std::string rank_mode = "pointdir");

std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<float>>
inverse_map_to_list(array_i32_d src_idx_np, array_f32_d src_pos_np, int n_maps);

py::array_t<int> keys_to_fc_edges(array_i32_d keys_np);

py::array_t<int> point_edges(array_i32_d point_keys_np, array_i32_d tgt_idx_np,
                             std::string mode);

py::array_t<long> prefix_to_indicator(array_i64_d prefix_np);
