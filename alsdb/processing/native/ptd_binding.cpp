// SPDX-License-Identifier: EUPL-1.2
// Python binding of progressive TIN densification (Axelsson 2000) as
// implemented in r-lidar/PTD by Jean-Romain Roussel (MIT), vendored unchanged
// under ptd/ (see ptd/README.md): the same PTD lasR and pyLASR run, on point
// arrays instead of LAS files.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "ptd/PTD.h"

namespace py = pybind11;

py::array_t<bool> classify(py::array_t<double, py::array::c_style | py::array::forcecast> xyz,
                           double seed_resolution, double angle, double distance, double spacing,
                           double buffer_size, unsigned int max_iter, unsigned int ncores) {
    if (xyz.ndim() != 2 || xyz.shape(1) != 3)
        throw std::invalid_argument("xyz must be an (n, 3) array");
    const py::ssize_t n = xyz.shape(0);
    py::array_t<bool> output(n);
    auto out = output.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < n; ++i) out(i) = false;
    if (n == 0) return output;
    auto p = xyz.unchecked<2>();

    PTD::Bbox bb;
    bb.xmin = bb.ymin = bb.zmin = std::numeric_limits<double>::max();
    bb.xmax = bb.ymax = bb.zmax = std::numeric_limits<double>::lowest();
    for (py::ssize_t i = 0; i < n; ++i) {
        if (!(std::isfinite(p(i, 0)) && std::isfinite(p(i, 1)) && std::isfinite(p(i, 2)))) continue;
        bb.xmin = std::min(bb.xmin, p(i, 0)); bb.xmax = std::max(bb.xmax, p(i, 0));
        bb.ymin = std::min(bb.ymin, p(i, 1)); bb.ymax = std::max(bb.ymax, p(i, 1));
        bb.zmin = std::min(bb.zmin, p(i, 2)); bb.zmax = std::max(bb.zmax, p(i, 2));
    }
    if (bb.xmin > bb.xmax) return output;  // no finite point

    PTD::Parameters params;
    params.seed_resolution_search = static_cast<float>(seed_resolution);
    params.max_iteration_angle = static_cast<float>(angle);
    params.max_iteration_distance = static_cast<float>(distance);
    params.spacing = static_cast<float>(spacing);
    params.buffer_size = static_cast<float>(buffer_size);
    params.max_iter = max_iter;
    params.ncores = ncores;

    std::vector<unsigned int> ground;
    {
        py::gil_scoped_release release;
        PTD::PTD ptd(params, bb);
        for (py::ssize_t i = 0; i < n; ++i) {
            if (std::isfinite(p(i, 0)) && std::isfinite(p(i, 1)) && std::isfinite(p(i, 2)))
                ptd.insert_point(p(i, 0), p(i, 1), p(i, 2), static_cast<unsigned int>(i));
        }
        ptd.run();
        ground = ptd.get_ground_fid();
    }
    for (unsigned int fid : ground)
        if (fid < static_cast<unsigned int>(n)) out(fid) = true;
    return output;
}

PYBIND11_MODULE(_ptd_native, m) {
    m.doc() = "Progressive TIN densification ground classification (r-lidar/PTD, vendored)";
    m.def("classify", &classify, py::arg("xyz"), py::arg("seed_resolution") = 10.0,
          py::arg("angle") = 30.0, py::arg("distance") = 2.0, py::arg("spacing") = 0.25,
          py::arg("buffer_size") = 30.0, py::arg("max_iter") = 50u, py::arg("ncores") = 1u);
    m.attr("upstream") = "r-lidar/PTD@ee9774b5f3613ccfac22aaf5f64107ab53fb7e03";
}
