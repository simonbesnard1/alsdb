// SPDX-License-Identifier: EUPL-1.2
// Incremental constrained-Delaunay spike-free reconstruction, following
// Khosravipour et al. (2016). Independent implementation, not LAStools code.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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
                            double freeze, double buffer, double max_edge) {
    if (xyz.ndim() != 2 || xyz.shape(1) != 3 || res <= 0 || nx <= 0 || ny <= 0 ||
        freeze <= 0 || buffer < 0)
        throw std::invalid_argument("Invalid spike-free input");
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
        FH face = tin.locate(q, type, index, hint);
        if (type == CDT::VERTEX) continue; // Highest duplicate XY already retained.
        if (tin.dimension() == 2 && !tin.is_infinite(face) && face->info()) continue;
        if (tin.dimension() == 2 && type == CDT::EDGE && tin.is_constrained({face, index})) continue;
        VH v = tin.insert(q, type, face, index);
        v->info().z = point[2];
        hint = v->face();
        if (tin.dimension() != 2) continue;
        auto faces = tin.incident_faces(v), start = faces;
        if (faces == 0) continue;
        do {
            FH f = faces;
            if (!tin.is_infinite(f)) {
                f->info() = false;
                auto a=f->vertex(0), b=f->vertex(1), c=f->vertex(2);
                if (std::max({edge2(a,b), edge2(b,c), edge2(c,a)}) < freeze*freeze)
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
    m.def("rasterize", &rasterize);
    m.attr("algorithm_version") = "1";
}
