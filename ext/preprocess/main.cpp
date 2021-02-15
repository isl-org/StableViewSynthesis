#include "preprocess.h"

PYBIND11_MODULE(preprocess, m) {
    m.def("count_nbs", &count_nbs, "tgt_dm"_a, "tgt_K"_a, "tgt_R"_a, "tgt_t"_a,
          "src_dms"_a, "src_Ks"_a, "src_Rs"_a, "src_ts"_a,
          "bwd_depth_thresh"_a = 0.01);
    m.def("map_source_points", &map_source_points, "tgt_dm"_a, "tgt_K"_a,
          "tgt_R"_a, "tgt_t"_a, "tgt_count"_a, "src_dms"_a, "src_Ks"_a,
          "src_Rs"_a, "src_ts"_a, "bwd_depth_thresh"_a = 0.01,
          "n_max_sources"_a = -1, "rank_mode"_a = "pointdir");
    m.def("inverse_map_to_list", &inverse_map_to_list, "src_idx", "src_pos",
          "n_maps");
    m.def("keys_to_fc_edges", &keys_to_fc_edges, "keys"_a);
    m.def("point_edges", &point_edges, "point_keys"_a, "tgt_idx"_a, "mode"_a);
    m.def("prefix_to_indicator", &prefix_to_indicator, "prefix"_a);
}
