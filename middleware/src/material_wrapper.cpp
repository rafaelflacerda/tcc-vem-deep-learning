#include "polivem_python.hpp"
#include "material/mat.hpp"
#include <iostream>

void init_material(py::module_ &m){
    std::cout << "    Starting material module initialization..." << std::endl;

    try{
        py::class_<material::mat>(m, "Material")
            .def(py::init<>())
            .def("setElasticModule", &material::mat::setElasticModule,
                py::arg("elastic_module"))
            .def("setPoissonCoef", &material::mat::setPoissonCoef,
                py::arg("poisson_coef"))
            .def("setMaterialDensity", &material::mat::setMaterialDensity,
                py::arg("material_density"))
            .def("getLameParameters", &material::mat::getLameParameters)
            .def("build2DElasticity", &material::mat::build2DElasticity);
    } catch (const std::exception& e){
        std::cerr << "    Error adding Material: " << e.what() << std::endl;
    }
}