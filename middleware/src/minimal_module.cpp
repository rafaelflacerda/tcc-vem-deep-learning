#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(minimal_py, m) {
    m.doc() = "Minimal test module";
    
    // Create a submodule
    auto test_module = m.def_submodule("test", "Test submodule");
    
    // Add a function to the submodule
    test_module.def("hello", []() {
        return "Hello from test submodule!";
    });
    
    // Add a function to the main module
    m.def("main_hello", []() {
        return "Hello from main module!";
    });
} 