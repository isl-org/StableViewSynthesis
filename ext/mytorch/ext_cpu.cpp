#include <torch_common.h>

#include "generated/list_to_map_cpu.cpp"
#include "generated/map_to_list_bl_cpu.cpp"
#include "generated/map_to_list_bl_seq_cpu.cpp"
#include "generated/map_to_list_nn_cpu.cpp"

#include "point_cat_cpu.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    map_to_list_nn_pybind_init(m);
    map_to_list_bl_pybind_init(m);
    map_to_list_bl_seq_pybind_init(m);
    list_to_map_pybind_init(m);

    point_cat_pybind_init(m);
}
