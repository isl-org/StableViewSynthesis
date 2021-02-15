#pragma once

#include <vector>
#include <chrono>
#include <Eigen/Dense>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

typedef py::array_t<float, py::array::c_style | py::array::forcecast>
    array_f32_d;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> array_i32_d;
typedef py::array_t<long, py::array::c_style | py::array::forcecast> array_i64_d;

typedef Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>
    depthmap_t;
typedef Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> mat3_t;
typedef Eigen::Map<const Eigen::Matrix<float, 3, 1>> vec3_t;
typedef Eigen::Map<const Eigen::Matrix<float, -1, 3, Eigen::RowMajor>> points_t;
typedef Eigen::Matrix<float, 3, 4> proj_t;

template <typename T>
py::array_t<T> create_array1(const std::vector<T>& data) {
    T* new_data = new T[data.size()];
    std::memcpy(new_data, data.data(), data.size() * sizeof(T));
    py::capsule free_data(new_data, [](void* f) {
        float* new_data = reinterpret_cast<float*>(f);
        delete[] new_data;
    });
    return py::array_t<T>({long(data.size())}, new_data, free_data);
}

template <typename T>
py::array_t<T> create_array2(const std::vector<T>& data, int height,
                             int width) {
    if (int(data.size()) != height * width) {
        throw std::invalid_argument("invalid size in create_array2");
    }
    T* new_data = new T[data.size()];
    std::memcpy(new_data, data.data(), data.size() * sizeof(T));
    py::capsule free_data(new_data, [](void* f) {
        float* new_data = reinterpret_cast<float*>(f);
        delete[] new_data;
    });
    return py::array_t<T>({height, width}, new_data, free_data);
}

template <typename T>
py::array_t<T> create_array3(const std::vector<T>& data, int channels,
                             int height, int width) {
    if (int(data.size()) != channels * height * width) {
        throw std::invalid_argument("invalid size in create_array3");
    }
    T* new_data = new T[data.size()];
    std::memcpy(new_data, data.data(), data.size() * sizeof(T));
    py::capsule free_data(new_data, [](void* f) {
        float* new_data = reinterpret_cast<float*>(f);
        delete[] new_data;
    });
    return py::array_t<T>({channels, height, width}, new_data, free_data);
}

class StopWatch {
   public:
    void tic(const std::string& name) {
        tics[name] = std::chrono::steady_clock::now();
    }

    double toc(const std::string& name) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = now - tics[name];
        double diff_seconds = diff.count();
        if (times.count(name) == 0) {
            times[name] = 0;
        }
        times[name] += diff_seconds;
        return diff_seconds;
    }

    void print_times() const {
        for (const auto& kv : times) {
            std::cout << kv.first << " took " << kv.second << "[s]"
                      << std::endl;
        }
    }

   private:
    std::unordered_map<std::string,
                       std::chrono::time_point<std::chrono::steady_clock>>
        tics;
    std::unordered_map<std::string, double> times;
};
