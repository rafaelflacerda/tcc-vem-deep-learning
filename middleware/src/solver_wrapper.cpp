#include "polivem_python.hpp"
#include <iostream>

#include "solver/beam1d.hpp"
#include "solver/linearElastic2d.hpp"
#include "solver/nonlinear2d.hpp"
#include "models/enums.hpp"

void init_solver(py::module_ &m) {
    std::cout << "    Starting solver module initialization..." << std::endl;
    
    // Add a simple function to test if the module is working
    std::cout << "    Adding test_function to solver module..." << std::endl;
    m.def("test_function", []() {
        return "Solver module is working!";
    });
    
    // Bind the Beam1D class - make sure the class definition exists
    try {
        std::cout << "    Adding BeamSolver class to solver module..." << std::endl;
        py::class_<solver::beam1d>(m, "BeamSolver")
            .def(py::init<Eigen::MatrixXd, Eigen::MatrixXi, int>(),
                 py::arg("nodes_coordinates"), 
                 py::arg("elements_indices"), 
                 py::arg("model_order") = 1)
            .def("setInertiaMoment", &solver::beam1d::setInertiaMoment, "Set the moment of inertia")
            .def("setArea", &solver::beam1d::setArea, "Set the cross-sectional area")
            .def("setDistributedLoad", &solver::beam1d::setDistributedLoad, "Set distributed load")
            .def("setSupp", &solver::beam1d::setSupp, "Set support conditions")
            .def("buildGlobalK", &solver::beam1d::buildGlobalK, 
                py::arg("E"),
                py::arg("type") = BeamSolverType::Beam)
            .def("buildStaticCondensation", &solver::beam1d::buildStaticCondensation,
                py::arg("K"),
                py::arg("sc_type"),
                py::arg("type") = BeamSolverType::Beam)
            .def("buildGlobalDistributedLoad", &solver::beam1d::buildGlobalDistributedLoad,
                py::arg("type") = BeamSolverType::Beam)
            .def("buildStaticDistVector", &solver::beam1d::buildStaticDistVector,
                py::arg("fb"),
                py::arg("sc_type"),
                py::arg("type") = BeamSolverType::Beam)
            .def("applyDBCMatrix", &solver::beam1d::applyDBCMatrix,
                py::arg("K"))
            .def("applyDBCVec", &solver::beam1d::applyDBCVec,
                py::arg("R"))
            .def("condense_matrix",  &solver::beam1d::condense_matrix,
                py::arg("KII"),
                py::arg("KIM"),
                py::arg("KMI"),
                py::arg("KMM"))
            .def("condense_vector", &solver::beam1d::condense_vector,
                py::arg("RI"),
                py::arg("RM"),
                py::arg("KIM"),
                py::arg("KMM"))
            .def("calculateStrain", &solver::beam1d::calculateStrain,
                py::arg("u"),
                py::arg("E"),
                py::arg("sample_points"),
                py::arg("y_top"))
            .def("calculateStress", &solver::beam1d::calculateStress,
                py::arg("u"),
                py::arg("E"),
                py::arg("sample_points"),
                py::arg("y_top"))
            .def("calculateMaxStress", &solver::beam1d::calculateMaxStress,
                py::arg("u"),
                py::arg("E"),
                py::arg("height"),
                py::arg("sample_points"))
            .def("getStrainStressAtPoint", &solver::beam1d::getStrainStressAtPoint,
                py::arg("u"),
                py::arg("E"),
                py::arg("x_global"),
                py::arg("y"));
            
        std::cout << "    Successfully added BeamSolver class" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "    Error adding BeamSolver class: " << e.what() << std::endl;
    }
    
    try{
        py::class_<solver::linearElastic2d>(m, "LinearElastic2DSolver")
            .def(py::init<Eigen::MatrixXd, Eigen::MatrixXi, int>(),
                 py::arg("nodes_coordinates"), 
                 py::arg("elements_indices"), 
                 py::arg("model_order") = 1)
            .def("buildGlobalK", &solver::linearElastic2d::buildGlobalK,
                py::arg("C"))
            .def("setSupp", &solver::linearElastic2d::setSupp,
                py::arg("dirichlet_bc"))
            .def("setLoad", &solver::linearElastic2d::setLoad,
                py::arg("load_indices"))
            .def("applyDBC", &solver::linearElastic2d::applyDBC,
                py::arg("K"))
            .def("applyNBC", &solver::linearElastic2d::applyNBC,
                py::arg("qx"),
                py::arg("qy"));
    } catch (const std::exception& e){
        std::cerr << "    Error adding LinearElastic2DSolver class: " << e.what() << std::endl;
    }


    std::cout << "    Solver module initialization complete!" << std::endl;
}