#include "preprocess.h"

#include <chrono>
#include <iostream>
#include <unordered_set>

bool is_valid_projection(int h, int w, float proj_d, const depthmap_t& dm,
                         int height, int width, float bwd_depth_thresh) {
    bool in_domain = proj_d > 0 && w >= 0 && h >= 0 && w < width && h < height;
    if (!in_domain) {
        return false;
    }

    float ds = dm(h, w);
    if (ds <= 0) {
        return false;
    }

    bool valid_depth_diff =
        (bwd_depth_thresh <= 0) || (ds > 1e6) ||
        std::abs(ds - proj_d) < bwd_depth_thresh * std::min(ds, proj_d);
    return valid_depth_diff;
}

py::array_t<int> count_nbs(array_f32_d tgt_dm_np, array_f32_d tgt_K_np,
                           array_f32_d tgt_R_np, array_f32_d tgt_t_np,
                           array_f32_d src_dms_np, array_f32_d src_Ks_np,
                           array_f32_d src_Rs_np, array_f32_d src_ts_np,
                           float bwd_depth_thresh) {
    if (tgt_dm_np.ndim() != 2) {
        throw std::invalid_argument("tgt_dm has to be height x width");
    }
    int tgt_height = tgt_dm_np.shape(0);
    int tgt_width = tgt_dm_np.shape(1);
    if (tgt_K_np.ndim() != 2 || tgt_K_np.shape(0) != 3 ||
        tgt_K_np.shape(1) != 3) {
        throw std::invalid_argument("tgt_K has to be 3 x 3");
    }
    if (tgt_R_np.ndim() != 2 || tgt_R_np.shape(0) != 3 ||
        tgt_R_np.shape(1) != 3) {
        throw std::invalid_argument("tgt_R has to be 3 x 3");
    }
    if (tgt_t_np.ndim() != 1 || tgt_R_np.shape(0) != 3) {
        throw std::invalid_argument("tgt_R has to be 3");
    }
    if (src_dms_np.ndim() != 3) {
        throw std::invalid_argument(
            "src_dms has to be n_views x height x width");
    }
    int n_views = src_dms_np.shape(0);
    int src_height = src_dms_np.shape(1);
    int src_width = src_dms_np.shape(2);
    if (src_Ks_np.ndim() != 3 || src_Ks_np.shape(0) != n_views ||
        src_Ks_np.shape(1) != 3 || src_Ks_np.shape(2) != 3) {
        throw std::invalid_argument("Ks has to be n_views x 3 x 3");
    }
    if (src_Rs_np.ndim() != 3 || src_Rs_np.shape(0) != n_views ||
        src_Rs_np.shape(1) != 3 || src_Rs_np.shape(2) != 3) {
        throw std::invalid_argument("Rs has to be n_views x 3 x 3");
    }
    if (src_ts_np.ndim() != 2 || src_ts_np.shape(0) != n_views ||
        src_ts_np.shape(1) != 3) {
        throw std::invalid_argument("ts has to be n_views x 3");
    }

    mat3_t tgt_K(tgt_K_np.data(), 3, 3);
    mat3_t tgt_R(tgt_R_np.data(), 3, 3);
    vec3_t tgt_t(tgt_t_np.data(), 3, 1);
    proj_t tgt_Pi;
    tgt_Pi.leftCols<3>() = tgt_K.inverse();
    tgt_Pi.rightCols<1>() = -tgt_t;
    tgt_Pi = tgt_R.transpose() * tgt_Pi;

    depthmap_t tgt_dm(tgt_dm_np.data(), tgt_height, tgt_width);

    std::vector<proj_t> src_Ps;
    std::vector<Eigen::Vector3f> src_Cs;
    for (int vidx = 0; vidx < n_views; ++vidx) {
        mat3_t K(src_Ks_np.data() + vidx * 3 * 3, 3, 3);
        mat3_t R(src_Rs_np.data() + vidx * 3 * 3, 3, 3);
        vec3_t t(src_ts_np.data() + vidx * 3 * 1, 3, 1);
        proj_t P;
        P.leftCols<3>() = R;
        P.rightCols<1>() = t;
        P = K * P;
        src_Ps.push_back(P);
        Eigen::Vector3f C = -R.transpose() * t;
        src_Cs.push_back(C);
    }

    std::vector<depthmap_t> src_dms;
    for (int vidx = 0; vidx < n_views; ++vidx) {
        src_dms.push_back(
            depthmap_t(src_dms_np.data() + vidx * src_height * src_width,
                       src_height, src_width));
    }

    std::vector<int> count(n_views, 0);
    for (int tgt_h = 0; tgt_h < tgt_height; ++tgt_h) {
        for (int tgt_w = 0; tgt_w < tgt_width; ++tgt_w) {
            float dt = tgt_dm(tgt_h, tgt_w);
            if (dt <= 0) {
                continue;
            }

            Eigen::Vector4f tgt_uvd(dt * (float(tgt_w) + 0.5),
                                    dt * (float(tgt_h) + 0.5), dt, 1);
            Eigen::Vector3f xyz = tgt_Pi * tgt_uvd;
            Eigen::Vector4f xyzh(xyz(0), xyz(1), xyz(2), 1);

            for (int vidx = 0; vidx < n_views; ++vidx) {
                Eigen::Vector3f src_uvd = src_Ps[vidx] * xyzh;
                float proj_d = src_uvd(2);

                float src_wf = (proj_d > 0) ? src_uvd(0) / proj_d : 0.0f;
                float src_hf = (proj_d > 0) ? src_uvd(1) / proj_d : 0.0f;

                int src_w = int(src_wf);
                int src_h = int(src_hf);

                if (is_valid_projection(src_h, src_w, proj_d, src_dms[vidx],
                                        src_height, src_width,
                                        bwd_depth_thresh)) {
                    count[vidx]++;
                }
            }
        }
    }

    return create_array1<int>(count);
}

std::tuple<py::array_t<float>, py::array_t<int>, py::array_t<float>,
           py::array_t<float>, py::array_t<int>, py::array_t<int>>
map_source_points(array_f32_d tgt_dm_np, array_f32_d tgt_K_np,
                  array_f32_d tgt_R_np, array_f32_d tgt_t_np,
                  array_i32_d tgt_count_np, array_f32_d src_dms_np,
                  array_f32_d src_Ks_np, array_f32_d src_Rs_np,
                  array_f32_d src_ts_np, float bwd_depth_thresh,
                  int n_max_sources, std::string rank_mode) {
    if (tgt_dm_np.ndim() != 2) {
        throw std::invalid_argument("tgt_dm has to be height x width");
    }
    int height = tgt_dm_np.shape(0);
    int width = tgt_dm_np.shape(1);
    if (tgt_K_np.ndim() != 2 || tgt_K_np.shape(0) != 3 ||
        tgt_K_np.shape(1) != 3) {
        throw std::invalid_argument("tgt_K has to be 3 x 3");
    }
    if (tgt_R_np.ndim() != 2 || tgt_R_np.shape(0) != 3 ||
        tgt_R_np.shape(1) != 3) {
        throw std::invalid_argument("tgt_R has to be 3 x 3");
    }
    if (tgt_t_np.ndim() != 1 || tgt_R_np.shape(0) != 3) {
        throw std::invalid_argument("tgt_R has to be 3");
    }
    if (tgt_count_np.ndim() != 1) {
        throw std::invalid_argument("tgt_R has to be n_views");
    }
    int n_views = tgt_count_np.shape(0);
    if (src_dms_np.ndim() != 3 || src_dms_np.shape(0) != n_views ||
        src_dms_np.shape(1) != height || src_dms_np.shape(2) != width) {
        throw std::invalid_argument(
            "src_dms has to be n_views x height x width");
    }
    if (src_Ks_np.ndim() != 3 || src_Ks_np.shape(0) != n_views ||
        src_Ks_np.shape(1) != 3 || src_Ks_np.shape(2) != 3) {
        throw std::invalid_argument("Ks has to be n_views x 3 x 3");
    }
    if (src_Rs_np.ndim() != 3 || src_Rs_np.shape(0) != n_views ||
        src_Rs_np.shape(1) != 3 || src_Rs_np.shape(2) != 3) {
        throw std::invalid_argument("Rs has to be n_views x 3 x 3");
    }
    if (src_ts_np.ndim() != 2 || src_ts_np.shape(0) != n_views ||
        src_ts_np.shape(1) != 3) {
        throw std::invalid_argument("ts has to be n_views x 3");
    }

    mat3_t tgt_K(tgt_K_np.data(), 3, 3);
    Eigen::Matrix3f tgt_Ki = tgt_K.inverse();
    mat3_t tgt_R(tgt_R_np.data(), 3, 3);
    vec3_t tgt_t(tgt_t_np.data(), 3, 1);
    proj_t tgt_P;
    tgt_P.leftCols<3>() = tgt_R;
    tgt_P.rightCols<1>() = tgt_t;
    tgt_P = tgt_K * tgt_P;
    Eigen::Vector3f tgt_C = -tgt_R.transpose() * tgt_t;

    depthmap_t tgt_dm(tgt_dm_np.data(), height, width);

    std::vector<proj_t> src_Ps;
    std::vector<mat3_t> src_Rs;
    std::vector<Eigen::Vector3f> src_Cs;
    for (int vidx = 0; vidx < n_views; ++vidx) {
        mat3_t K(src_Ks_np.data() + vidx * 3 * 3, 3, 3);
        mat3_t R(src_Rs_np.data() + vidx * 3 * 3, 3, 3);
        vec3_t t(src_ts_np.data() + vidx * 3 * 1, 3, 1);
        src_Rs.push_back(R);
        proj_t P;
        P.leftCols<3>() = R;
        P.rightCols<1>() = t;
        P = K * P;
        src_Ps.push_back(P);
        Eigen::Vector3f C = -R.transpose() * t;
        src_Cs.push_back(C);
    }

    std::vector<depthmap_t> src_dms;
    for (int vidx = 0; vidx < n_views; ++vidx) {
        src_dms.push_back(depthmap_t(src_dms_np.data() + vidx * height * width,
                                     height, width));
    }

    const int* tgt_count = tgt_count_np.data();

    // n_points x n_views
    // for each 3d point we have a varying number of info
    // depending on how many source images observe it
    std::vector<float> src_pos;
    std::vector<int> src_idx;
    std::vector<float> src_dirs;
    std::vector<float> tgt_dirs;
    std::vector<int> point_idx;
    point_idx.push_back(0);

    // n_points
    // target idx (h * width + w)
    std::vector<int> tgt_idx;

    for (int tgt_dm_idx = 0; tgt_dm_idx < height * width; ++tgt_dm_idx) {
        int h = tgt_dm_idx / width;
        int w = tgt_dm_idx % width;
        float d = tgt_dm(h, w);

        Eigen::Vector3f uvh(w + 0.5, h + 0.5, 1);
        Eigen::Vector3f xyz = tgt_Ki * uvh;
        xyz *= d;
        xyz = tgt_R.transpose() * (xyz - tgt_t);
        Eigen::Vector4f xyzh(xyz(0), xyz(1), xyz(2), 1);

        Eigen::Vector3f tgt_dir = xyz - tgt_C;
        tgt_dir /= tgt_dir.norm();
        Eigen::Vector3f tgt_view = tgt_R.row(2);
        tgt_view /= tgt_view.norm();

        tgt_dirs.push_back(tgt_dir(0));
        tgt_dirs.push_back(tgt_dir(1));
        tgt_dirs.push_back(tgt_dir(2));

        std::vector<std::pair<int, float>> valid_view_ind;
        for (int vidx = 0; vidx < n_views; ++vidx) {
            Eigen::Vector3f uvd = src_Ps[vidx] * xyzh;
            float proj_d = uvd(2);

            float src_wf = (proj_d > 0) ? uvd(0) / proj_d : 0.0f;
            float src_hf = (proj_d > 0) ? uvd(1) / proj_d : 0.0f;

            int src_w = int(src_wf);
            int src_h = int(src_hf);

            if (!is_valid_projection(src_h, src_w, proj_d, src_dms[vidx],
                                     height, width, bwd_depth_thresh)) {
                continue;
            }

            if (rank_mode == "pointdir") {
                Eigen::Vector3f src_dir = xyz - src_Cs[vidx];
                src_dir /= src_dir.norm();
                valid_view_ind.push_back(
                    std::make_pair(vidx, tgt_dir.dot(src_dir)));
            } else if (rank_mode == "viewdir") {
                Eigen::Vector3f src_view = src_Rs[vidx].row(2);
                src_view /= src_view.norm();
                valid_view_ind.push_back(
                    std::make_pair(vidx, tgt_view.dot(src_view)));
            } else if (rank_mode == "count") {
                valid_view_ind.push_back(
                    std::make_pair(vidx, float(tgt_count[vidx])));
            } else {
                throw std::runtime_error("invalid rank_mode");
            }
        }

        std::sort(valid_view_ind.begin(), valid_view_ind.end(),
                  [](const std::pair<int, float>& idx0,
                     const std::pair<int, float>& idx1) {
                      return idx0.second > idx1.second;
                  });
        if (n_max_sources > 0 && n_max_sources < valid_view_ind.size()) {
            valid_view_ind.resize(n_max_sources);
        }

        int n_proj_views = 0;
        for (auto const& vidx : valid_view_ind) {
            Eigen::Vector3f uvd = src_Ps[vidx.first] * xyzh;
            float proj_d = uvd(2);

            float src_wf = (proj_d > 0) ? uvd(0) / proj_d : 0.0f;
            float src_hf = (proj_d > 0) ? uvd(1) / proj_d : 0.0f;

            int src_w = int(src_wf);
            int src_h = int(src_hf);

            n_proj_views++;

            src_pos.push_back(src_hf);
            src_pos.push_back(src_wf);
            src_idx.push_back(vidx.first);

            Eigen::Vector3f src_dir = xyz - src_Cs[vidx.first];
            src_dir /= src_dir.norm();
            src_dirs.push_back(src_dir(0));
            src_dirs.push_back(src_dir(1));
            src_dirs.push_back(src_dir(2));
        }

        if (n_proj_views > 0) {
            tgt_idx.push_back(tgt_dm_idx);
            int last_point_idx = point_idx[point_idx.size() - 1];
            int new_point_idx = last_point_idx + n_proj_views;
            point_idx.push_back(new_point_idx);
        }
    }

    auto ret = std::make_tuple(
        create_array2<float>(src_pos, src_idx.size(), 2),
        create_array1<int>(src_idx),
        create_array2<float>(src_dirs, src_dirs.size() / 3, 3),
        create_array2<float>(tgt_dirs, tgt_dirs.size() / 3, 3),
        create_array1<int>(point_idx), create_array1<int>(tgt_idx));

    return ret;
}

std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<float>>
inverse_map_to_list(array_i32_d src_idx_np, array_f32_d src_pos_np,
                    int n_maps) {
    if (src_idx_np.ndim() != 1) {
        throw std::invalid_argument("src_idx should be N");
    }
    int N = src_idx_np.shape(0);
    if (src_pos_np.ndim() != 2 || src_pos_np.shape(0) != N ||
        src_pos_np.shape(1) != 2) {
        throw std::invalid_argument("src_pos should be N x 2");
    }

    std::vector<int> out_prefix_sum(n_maps, 0);
    std::vector<std::vector<int>> out_tgt_idx(n_maps);
    std::vector<std::vector<float>> out_src_pos(n_maps);

    const int* src_idx_data = src_idx_np.data();
    const float* src_pos_data = src_pos_np.data();
    for (int tgt_idx = 0; tgt_idx < N; ++tgt_idx) {
        int src_idx = src_idx_data[tgt_idx];
        float src_pos_h = src_pos_data[tgt_idx * 2 + 0];
        float src_pos_w = src_pos_data[tgt_idx * 2 + 1];

        out_prefix_sum[src_idx]++;
        out_tgt_idx[src_idx].push_back(tgt_idx);
        out_src_pos[src_idx].push_back(src_pos_h);
        out_src_pos[src_idx].push_back(src_pos_w);
    }

    std::vector<int> flat_tgt_idx(N);
    int tgt_idx = 0;
    for (const auto& vec_per_feat : out_tgt_idx) {
        for (int val : vec_per_feat) {
            flat_tgt_idx[tgt_idx] = val;
            tgt_idx++;
        }
    }

    std::vector<float> flat_src_pos(N * 2);
    tgt_idx = 0;
    for (const auto& vec_per_feat : out_src_pos) {
        for (float val : vec_per_feat) {
            flat_src_pos[tgt_idx] = val;
            tgt_idx++;
        }
    }

    return std::make_tuple(create_array1<int>(out_prefix_sum),
                           create_array1<int>(flat_tgt_idx),
                           create_array2<float>(flat_src_pos, N, 2));
}

py::array_t<int> keys_to_fc_edges(array_i32_d keys_np) {
    // Note: this method assumes that keys are sorted
    if (keys_np.ndim() != 1) {
        throw std::invalid_argument("keys has to be N");
    }
    long n = keys_np.shape(0);
    const int* keys_data = keys_np.data();

    int key = -1;
    std::vector<int> ind;
    std::vector<int> edges;
    for (int idx = 0; idx < n; ++idx) {
        int new_key = keys_data[idx];
        if (new_key == key) {
            for (int idx0 : ind) {
                edges.push_back(idx);
                edges.push_back(idx0);
                edges.push_back(idx0);
                edges.push_back(idx);
            }
            ind.push_back(idx);
        } else {
            key = new_key;
            ind.clear();
            ind.push_back(idx);
        }
    }

    return create_array2<int>(edges, edges.size() / 2, 2);
}

py::array_t<int> point_edges(array_i32_d point_keys_np, array_i32_d tgt_idx_np,
                             const std::string mode) {
    // printf("mode = %s\n", mode.c_str());
    if (point_keys_np.ndim() != 1) {
        throw std::invalid_argument("point_keys has to be N + 1");
    }
    long n_point_keys = point_keys_np.shape(0);
    const int* point_keys_data = point_keys_np.data();
    int n_points = point_keys_data[n_point_keys - 1];

    if (tgt_idx_np.ndim() != 1 || tgt_idx_np.shape(0) != n_point_keys - 1) {
        throw std::invalid_argument("tgt_idx has to be N");
    }
    long n_tgt_idx = tgt_idx_np.shape(0);
    const int* tgt_idx_data = tgt_idx_np.data();

    std::vector<int> edges;
    for (int point_key_idx = 0; point_key_idx < n_point_keys - 1;
         ++point_key_idx) {
        int point_key_from = point_keys_data[point_key_idx];
        int point_key_to = point_keys_data[point_key_idx + 1];

        // fully connected
        if (mode.find("fc") != std::string::npos) {
            for (int e0 = point_key_from; e0 < point_key_to - 1; ++e0) {
                for (int e1 = e0 + 1; e1 < point_key_to; ++e1) {
                    edges.push_back(e0);
                    edges.push_back(e1);
                    edges.push_back(e1);
                    edges.push_back(e0);
                }
            }
        }

        // star connection
        int e1 = n_points + tgt_idx_data[point_key_idx];
        for (int e0 = point_key_from; e0 < point_key_to; ++e0) {
            if (mode.find("st") != std::string::npos) {
                edges.push_back(e0);
                edges.push_back(e1);
            }
            if (mode.find("sf") != std::string::npos) {
                edges.push_back(e1);
                edges.push_back(e0);
            }
        }
    }

    return create_array2<int>(edges, edges.size() / 2, 2);
};

py::array_t<long> prefix_to_indicator(array_i64_d prefix_np) {
    if (prefix_np.ndim() != 1) {
        throw std::invalid_argument("prefix has to be N");
    }
    long prefix_length = prefix_np.shape(0);
    const long* prefix_data = prefix_np.data();
    std::vector<long> indicator(prefix_data[prefix_length - 1]);
    for (long prefix_idx = 0; prefix_idx < prefix_length - 1; ++prefix_idx) {
        for (long indicator_idx = prefix_data[prefix_idx];
             indicator_idx < prefix_data[prefix_idx + 1]; ++indicator_idx) {
            indicator[indicator_idx] = prefix_idx;
        }
    }
    return create_array1<long>(indicator);
}

