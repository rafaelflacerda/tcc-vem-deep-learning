#include "polivem_python.hpp"
#include <iostream>

PYBIND11_MODULE(polivem_py, m) {
    try {
        m.doc() = "Python bindings for PoliVEM library";
        
        std::cout << "====== Initializing polivem_py module ======" << std::endl;
        
        // Create all submodules at the top level with verbose output
        std::cout << "Creating mesh submodule..." << std::endl;
        auto mesh_module = m.def_submodule("mesh", "Mesh module containing beam functionality");
        
        std::cout << "Creating solver submodule..." << std::endl;
        auto solver_module = m.def_submodule("solver", "Solver module");
        
        std::cout << "Creating material submodule..." << std::endl;
        auto material_module = m.def_submodule("material", "Material module");
        
        std::cout << "Creating enums submodule..." << std::endl;
        auto enums_module = m.def_submodule("enums", "Enums module");
        
        // Initialize each module with detailed error handling
        std::cout << "\nInitializing modules:" << std::endl;
        
        try {
            std::cout << "  - Initializing enums functionality..." << std::endl;
            init_enums(enums_module);
            std::cout << "  ✓ Enums initialization successful" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  ✗ Error initializing enums module: " << e.what() << std::endl;
        }
        
        try {
            std::cout << "  - Initializing beam functionality..." << std::endl;
            init_beam(mesh_module);
            std::cout << "  ✓ Beam initialization successful" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  ✗ Error initializing beam module: " << e.what() << std::endl;
        }
        
        try {
            std::cout << "  - Initializing solver functionality..." << std::endl;
            init_solver(solver_module);
            std::cout << "  ✓ Solver initialization successful" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  ✗ Error initializing solver module: " << e.what() << std::endl;
        }
        
        try {
            std::cout << "  - Initializing material functionality..." << std::endl;
            init_material(material_module);
            std::cout << "  ✓ Material initialization successful" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  ✗ Error initializing material module: " << e.what() << std::endl;
        }
        
        // Add a simple test function to the main module to verify it's working
        m.def("test_function", []() {
            return "Main module is working!";
        });
        
        std::cout << "\n====== polivem_py module initialization complete ======" << std::endl;
    } 
    catch (const std::exception& e) {
        std::cerr << "Fatal error in module initialization: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown fatal error in module initialization" << std::endl;
    }
}