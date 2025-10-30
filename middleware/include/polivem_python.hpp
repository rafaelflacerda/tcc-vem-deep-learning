/**
 * @file polivem_python.hpp
 * @brief Header file for Python bindings of PoliVEM library
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef POLIVEM_PYTHON_HPP
#define POLIVEM_PYTHON_HPP

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "mesh/beam.hpp"
#include "solver/beam1d.hpp"
#include "material/mat.hpp"
#include "models/enums.hpp"

namespace py = pybind11;

// Function declarations for module components
void init_beam(py::module_ &m);
void init_solver(py::module_ &m);
void init_material(py::module_ &m);
void init_enums(py::module_ &m);

#endif // POLIVEM_PYTHON_HPP