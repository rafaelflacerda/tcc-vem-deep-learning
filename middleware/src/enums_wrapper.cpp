#include "polivem_python.hpp"
#include "models/enums.hpp"
#include <iostream>

void init_enums(py::module_ &m) {
    std::cout << "    Starting enums module initialization..." << std::endl;
    
    // Bind the BeamSolverType enum
    try {
        std::cout << "    Adding BeamSolverType enum to module..." << std::endl;
        py::enum_<BeamSolverType>(m, "BeamSolverType")
            .value("BEAM", BeamSolverType::Beam)
            .value("PORTIC", BeamSolverType::Portic)
            .export_values();
        std::cout << "    Successfully added BeamSolverType enum" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "    Error adding BeamSolverType enum: " << e.what() << std::endl;
    }
    
    std::cout << "    Enums module initialization complete!" << std::endl;
}