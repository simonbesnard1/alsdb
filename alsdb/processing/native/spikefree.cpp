// SPDX-License-Identifier: EUPL-1.2
// Incremental constrained-Delaunay spike-free reconstruction, following
// Khosravipour et al. (2016). Independent implementation, not LAStools code.
// Optional per-cell freeze distances implement the locally adaptive variant of
// Fischer et al. (2024), https://doi.org/10.1111/2041-210X.14416, written from
// the paper and the authors' R workflow, not from lidR's lspikefree code.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <queue>
#include <vector>

namespace py = pybind11;
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
struct VertexInfo { double z = 0; };
using Vb = CGAL::Triangulation_vertex_base_with_info_2<VertexInfo, K>;
using Fb0 = CGAL::Constrained_triangulation_face_base_2<K>;
using Fb = CGAL::Triangulation_face_base_with_info_2<bool, K, Fb0>;
using Tds = CGAL::Triangulation_data_structure_2<Vb, Fb>;
using CDT = CGAL::Constrained_Delaunay_triangulation_2<K, Tds, CGAL::Exact_predicates_tag>;
using VH = CDT::Vertex_handle;
using FH = CDT::Face_handle;
struct Candidate {
    double level;
    std::array<VH, 3> v;
    bool operator<(const Candidate& other) const { return level < other.level; }
};

double edge2(VH a, VH b) {
    return CGAL::to_double(CGAL::squared_distance(a->point(), b->point()));
}

py::array_t<float> rasterize(py::array_t<double, py::array::c_style | py::array::forcecast> xyz,
                            double x0, double y1, double res, int nx, int ny,
                            double freeze, double buffer, double max_edge,
                            std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>> freeze_grid,
                            double fx0, double fy1, double fcell) {
    if (xyz.ndim() != 2 || xyz.shape(1) != 3 || res <= 0 || nx <= 0 || ny <= 0 ||
        freeze <= 0 || buffer < 0)
        throw std::invalid_argument("Invalid spike-free input");
    // Locally adaptive freeze distances: a north-up grid of squared distances
    // with upper-left corner (fx0, fy1) and cell size fcell, in the same
    // coordinates as xyz; a triangle uses the cell under its centroid,
    // clamped to the grid. Without a grid every triangle uses *freeze*.
    std::vector<double> local2;
    long fnx = 0, fny = 0;
    if (freeze_grid) {
        if (freeze_grid->ndim() != 2 || freeze_grid->shape(0) < 1 || freeze_grid->shape(1) < 1 ||
            !(fcell > 0))
            throw std::invalid_argument("Invalid freeze-distance grid");
        auto g = freeze_grid->unchecked<2>();
        fny = g.shape(0); fnx = g.shape(1);
        local2.resize(size_t(fnx) * fny);
        for (long r = 0; r < fny; ++r)
            for (long c = 0; c < fnx; ++c) {
                const double d = g(r, c);
                if (!std::isfinite(d) || d <= 0)
                    throw std::invalid_argument("Freeze distances must be finite and positive");
                local2[size_t(r) * fnx + c] = d * d;
            }
        fx0 -= x0; fy1 -= y1;
    }
    auto p = xyz.unchecked<2>();
    py::array_t<float> output({ny, nx});
    auto out = output.mutable_unchecked<2>();
    py::gil_scoped_release release;
    std::vector<std::array<double, 3>> points;
    points.reserve(p.shape(0));
    for (py::ssize_t i=0; i<p.shape(0); ++i) {
        if (std::isfinite(p(i,0)) && std::isfinite(p(i,1)) && std::isfinite(p(i,2)))
            points.push_back({p(i,0)-x0, p(i,1)-y1, p(i,2)});
    }
    std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) {
        if (a[2] != b[2]) return a[2] > b[2];
        if (a[0] != b[0]) return a[0] < b[0];
        return a[1] < b[1];
    });
    CDT tin;
    std::priority_queue<Candidate> pending;
    FH hint;
    // Points arrive highest first. Canopy heights are spatially uncorrelated
    // enough that consecutive points are far apart, and a locate walking from
    // the previous vertex crossed much of the TIN: on a real PNOA 1 km tile,
    // 307k vegetation returns took 9.9 s, and the same heights at random XY
    // 11 s, against 1.3 s with equal heights. A coarse grid remembering the
    // last vertex inserted per cell gives each locate a nearby start. The
    // insertion order is unchanged, so the triangulation is identical.
    auto freeze2 = [&](VH a, VH b, VH c) {
        if (local2.empty()) return freeze * freeze;
        const double x = (CGAL::to_double(a->point().x()) + CGAL::to_double(b->point().x()) +
                          CGAL::to_double(c->point().x())) / 3;
        const double y = (CGAL::to_double(a->point().y()) + CGAL::to_double(b->point().y()) +
                          CGAL::to_double(c->point().y())) / 3;
        const long col = std::clamp(long(std::floor((x - fx0) / fcell)), 0L, fnx - 1);
        const long row = std::clamp(long(std::floor((fy1 - y) / fcell)), 0L, fny - 1);
        return local2[size_t(row) * fnx + col];
    };
    double gx0 = std::numeric_limits<double>::max(), gy0 = gx0;
    double gx1 = std::numeric_limits<double>::lowest(), gy1 = gx1;
    for (const auto& point: points) {
        gx0 = std::min(gx0, point[0]); gx1 = std::max(gx1, point[0]);
        gy0 = std::min(gy0, point[1]); gy1 = std::max(gy1, point[1]);
    }
    const double cell = points.empty() ? 1.0 : std::max(
        1e-6, 2.0 * std::sqrt(std::max(gx1 - gx0, 1e-6) * std::max(gy1 - gy0, 1e-6) / points.size()));
    const int gnx = points.empty() ? 1 : int((gx1 - gx0) / cell) + 1;
    const int gny = points.empty() ? 1 : int((gy1 - gy0) / cell) + 1;
    std::vector<VH> recent(size_t(gnx) * gny, VH());
    auto cell_of = [&](double x, double y) {
        return std::make_pair(std::min(gnx - 1, int((x - gx0) / cell)),
                              std::min(gny - 1, int((y - gy0) / cell)));
    };
    auto start_face = [&](double x, double y) {
        const auto [i, j] = cell_of(x, y);
        for (int ring = 0; ring <= 3; ++ring)
            for (int dj = -ring; dj <= ring; ++dj)
                for (int di = -ring; di <= ring; ++di) {
                    if (std::max(std::abs(di), std::abs(dj)) != ring) continue;
                    const int ii = i + di, jj = j + dj;
                    if (ii < 0 || jj < 0 || ii >= gnx || jj >= gny) continue;
                    const VH v = recent[size_t(jj) * gnx + ii];
                    if (v != VH()) return v->face();
                }
        return hint;
    };
    for (const auto& point: points) {
        // Candidates are revalidated because insertion can replace an unfrozen
        // face. Vertex handles remain valid throughout this insertion-only TIN.
        while (!pending.empty() && pending.top().level > point[2]) {
            auto c = pending.top(); pending.pop();
            FH face;
            if (!tin.is_face(c.v[0], c.v[1], c.v[2], face) || face->info()) continue;
            tin.insert_constraint(c.v[0], c.v[1]);
            tin.insert_constraint(c.v[1], c.v[2]);
            tin.insert_constraint(c.v[2], c.v[0]);
            if (!tin.is_face(c.v[0], c.v[1], c.v[2], face))
                throw std::runtime_error("Frozen face was not preserved");
            face->info() = true;
        }
        K::Point_2 q(point[0], point[1]);
        CDT::Locate_type type;
        int index;
        FH face = tin.locate(q, type, index, start_face(point[0], point[1]));
        if (type == CDT::VERTEX) continue; // Highest duplicate XY already retained.
        if (tin.dimension() == 2 && !tin.is_infinite(face) && face->info()) continue;
        if (tin.dimension() == 2 && type == CDT::EDGE && tin.is_constrained({face, index})) continue;
        VH v = tin.insert(q, type, face, index);
        v->info().z = point[2];
        hint = v->face();
        const auto [ci, cj] = cell_of(point[0], point[1]);
        recent[size_t(cj) * gnx + ci] = v;
        if (tin.dimension() != 2) continue;
        auto faces = tin.incident_faces(v), start = faces;
        if (faces == 0) continue;
        do {
            FH f = faces;
            if (!tin.is_infinite(f)) {
                f->info() = false;
                auto a=f->vertex(0), b=f->vertex(1), c=f->vertex(2);
                if (std::max({edge2(a,b), edge2(b,c), edge2(c,a)}) < freeze2(a, b, c))
                    pending.push({std::min({a->info().z,b->info().z,c->info().z})-buffer, {a,b,c}});
            }
        } while (++faces != start);
    }
    hint = FH();
    for (int row=0; row<ny; ++row) {
        for (int col=0; col<nx; ++col) {
            out(row,col) = std::numeric_limits<float>::quiet_NaN();
            if (tin.dimension() != 2) continue;
            double x=(col+.5)*res, y=-(row+.5)*res;
            FH f = tin.locate(K::Point_2(x,y), hint);
            hint=f;
            if (tin.is_infinite(f)) continue;
            auto a=f->vertex(0), b=f->vertex(1), c=f->vertex(2);
            if (max_edge > 0 && std::max({edge2(a,b),edge2(b,c),edge2(c,a)}) > max_edge*max_edge) continue;
            double ax=CGAL::to_double(a->point().x()), ay=CGAL::to_double(a->point().y());
            double bx=CGAL::to_double(b->point().x()), by=CGAL::to_double(b->point().y());
            double cx=CGAL::to_double(c->point().x()), cy=CGAL::to_double(c->point().y());
            double det=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy);
            double wa=((by-cy)*(x-cx)+(cx-bx)*(y-cy))/det;
            double wb=((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/det;
            out(row,col)=float(wa*a->info().z+wb*b->info().z+(1-wa-wb)*c->info().z);
        }
    }
    return output;
}

PYBIND11_MODULE(_spikefree_native, m) {
    m.doc() = "Constrained-Delaunay spike-free surface reconstruction";
    m.def("rasterize", &rasterize, py::arg("xyz"), py::arg("x0"), py::arg("y1"), py::arg("res"),
          py::arg("nx"), py::arg("ny"), py::arg("freeze"), py::arg("buffer"), py::arg("max_edge"),
          py::arg("freeze_grid") = py::none(), py::arg("fx0") = 0.0, py::arg("fy1") = 0.0,
          py::arg("fcell") = 0.0);
    m.attr("freeze_grid") = true;
    m.attr("algorithm_version") = "1";
}
