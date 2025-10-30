#include "polivem_python.hpp"
#include <iostream>

void init_beam(py::module_ &m) {
    std::cout << "Initializing beam functionality in mesh module..." << std::endl;
    
    // Binding the beam class to the mesh submodule
    py::class_<mesh::beam>(m, "Beam")
        .def(py::init<>())
        .def("horizontal_bar_disc", &mesh::beam::horizontalBarDisc,
            py::arg("bar_length"), py::arg("num_elements"),
            "Create horizontal beam mesh")
        .def_readwrite("nodes", &mesh::beam::nodes, "Node coordinates")
        .def_readwrite("elements", &mesh::beam::elements, "Element connectivity");
    
    std::cout << "Beam functionality initialized in mesh module!" << std::endl;
}