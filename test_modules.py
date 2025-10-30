import os
import sys

from middleware.utils.formatting import print_header, print_success, print_error, print_warning, print_info

def test_imports():
    print_header("TESTING IMPORTS")
    
    # First try to import the package
    print_info("Trying to import from installed package...")
    try:
        import polivem
        print_success("Successfully imported polivem package")
        
        # Get package location
        print_info(f"Package location: {os.path.dirname(polivem.__file__)}")
        
        # Check if the module is accessible
        try:
            print_info("Trying to import polivem_py module...")
            from polivem import polivem_py
            print_success("Successfully imported polivem_py module")
            
            # Check module location
            if hasattr(polivem_py, "__file__"):
                print_info(f"Module location: {polivem_py.__file__}")
            
            # Check for submodules
            submodules = []
            for attr in dir(polivem_py):
                if not attr.startswith("__"):
                    submodules.append(attr)
            
            if submodules:
                print_success(f"Found submodules: {', '.join(submodules)}")
            else:
                print_warning("No submodules found")
            
            return True, polivem_py
            
        except ImportError as e:
            print_error(f"Error importing polivem_py module: {e}")
        except Exception as e:
            print_error(f"Unexpected error importing polivem_py module: {e}")
    
    except ImportError as e:
        print_error(f"Error importing polivem package: {e}")
    except Exception as e:
        print_error(f"Unexpected error importing polivem package: {e}")
    
    # Try direct import from build directory
    print_info("\nTrying direct import from build directory...")
    build_dir = os.path.join(os.getcwd(), 'build', 'python')
    if os.path.exists(build_dir):
        print_info(f"Build directory found: {build_dir}")
        
        # Add to path
        sys.path.insert(0, build_dir)
        print_info(f"Added build directory to sys.path")
        
        # Try import
        try:
            import polivem_py
            print_success("Successfully imported polivem_py directly")
            
            # Check for submodules
            submodules = []
            for attr in dir(polivem_py):
                if not attr.startswith("__"):
                    submodules.append(attr)
            
            if submodules:
                print_success(f"Found submodules: {', '.join(submodules)}")
            else:
                print_warning("No submodules found")
            
            return True, polivem_py
            
        except ImportError as e:
            print_error(f"Error importing polivem_py directly: {e}")
        except Exception as e:
            print_error(f"Unexpected error importing polivem_py directly: {e}")
    else:
        print_error(f"Build directory not found: {build_dir}")
    
    return False, None

def test_mesh_module(module):
    print_header("TESTING MESH MODULE")
    
    if not hasattr(module, 'mesh'):
        print_error("Mesh module not found")
        return False
    
    print_success("Mesh module found")
    
    # Check for Beam class
    if not hasattr(module.mesh, 'Beam'):
        print_error("Beam class not found in mesh module")
        return False
    
    print_success("Beam class found in mesh module")
    
    # Try to create a Beam object
    try:
        beam = module.mesh.Beam()
        print_success("Successfully created Beam object")
        
        # Check methods
        if hasattr(beam, 'horizontal_bar_disc'):
            print_success("horizontal_bar_disc method found on Beam")
        else:
            print_warning("horizontal_bar_disc method not found on Beam")
        
        return True
    except Exception as e:
        print_error(f"Error creating Beam object: {e}")
        return False

def test_solver_module(module):
    print_header("TESTING SOLVER MODULE")
    
    if not hasattr(module, 'solver'):
        print_error("Solver module not found")
        return False
    
    print_success("Solver module found")
    
    # Check for test_function
    if hasattr(module.solver, 'test_function'):
        try:
            result = module.solver.test_function()
            print_success(f"test_function works: {result}")
        except Exception as e:
            print_error(f"Error calling test_function: {e}")
    else:
        print_warning("test_function not found in solver module")
    
    # Check for BeamSolver class
    if not hasattr(module.solver, 'BeamSolver'):
        print_error("BeamSolver class not found in solver module")
        return False
    
    if not hasattr(module.solver, 'LinearElastic2DSolver'):
        print_error("LinearElastic2DSolver class not found in solver module")
        return False
    
    print_success("BeamSolver class found in solver module")
    print_success("LinearElastic2DSolver class found in solver module")
    
    # Try to create a BeamSolver object
    try:
        beam = module.mesh.Beam()
        nodes = beam.nodes
        elements = beam.elements    
        solver_beam = module.solver.BeamSolver(nodes, elements, 1)
        solver_linear = module.solver.LinearElastic2DSolver(nodes, elements, 1)
        print_success("Successfully created BeamSolver object")
        
        # Check methods
        if hasattr(solver_beam, 'setInertiaMoment'):
            print_success("setInertiaMoment method found on BeamSolver")
        else:
            print_warning("setInertiaMoment method not found on BeamSolver")

        if hasattr(solver_beam, 'setArea'):
            print_success("setArea method found on BeamSolver")
        else:
            print_warning("setArea method not found on BeamSolver")

        if hasattr(solver_beam, 'setDistributedLoad'):
            print_success("setDistributedLoad method found on BeamSolver")
        else:
            print_warning("setDistributedLoad method not found on BeamSolver")

        if hasattr(solver_beam, 'setSupp'):
            print_success("setSupp method found on BeamSolver")
        else:
            print_warning("setSupp method not found on BeamSolver")

        if hasattr(solver_beam, 'buildGlobalK'): 
            print_success("buildGlobalK method found on BeamSolver")
        else:
            print_warning("buildGlobalK method not found on BeamSolver")

        if hasattr(solver_beam, 'buildStaticCondensation'):
            print_success("buildStaticCondensation method found on BeamSolver")
        else:
            print_warning("buildStaticCondensation method not found on BeamSolver")

        if hasattr(solver_beam, 'buildGlobalDistributedLoad'):
            print_success("buildGlobalDistributedLoad method found on BeamSolver")
        else:
            print_warning("buildGlobalDistributedLoad method not found on BeamSolver")

        if hasattr(solver_beam, 'buildStaticDistVector'):
            print_success("buildStaticDistVector method found on BeamSolver")
        else:
            print_warning("buildStaticDistVector method not found on BeamSolver")

        if hasattr(solver_beam, 'buildGlobalK'):
            print_success("buildGlobalK method found on BeamSolver")
        else:
            print_warning("buildGlobalK method not found on BeamSolver")

        if hasattr(solver_beam, 'applyDBCMatrix'):
            print_success("applyDBCMatrix method found on BeamSolver")
        else:
            print_warning("applyDBCMatrix method not found on BeamSolver")

        if hasattr(solver_beam, 'applyDBCVec'):
            print_success("applyDBCVec method found on BeamSolver")
        else:
            print_warning("applyDBCVec method not found on BeamSolver")
            
        if hasattr(solver_beam, 'condense_matrix'):
            print_success("condense_matrix method found on BeamSolver")
        else:
            print_warning("condense_matrix method not found on BeamSolver")

        if hasattr(solver_beam, 'condense_vector'):
            print_success("condense_vector method found on BeamSolver")     
        else:
            print_warning("condense_vector method not found on BeamSolver")
            
        if hasattr(solver_beam, 'calculateStrain'):
            print_success("calculateStrain method found on BeamSolver")
        else:
            print_warning("calculateStrain method not found on BeamSolver")

        if hasattr(solver_beam, 'calculateStress'):
            print_success("calculateStress method found on BeamSolver")
        else:
            print_warning("calculateStress method not found on BeamSolver") 

        if hasattr(solver_beam, 'calculateMaxStress'):
            print_success("calculateMaxStress method found on BeamSolver")
        else:
            print_warning("calculateMaxStress method not found on BeamSolver")  

        if hasattr(solver_beam, 'getStrainStressAtPoint'):
            print_success("getStrainStressAtPoint method found on BeamSolver")
        else:
            print_warning("getStrainStressAtPoint method not found on BeamSolver")
            
        if hasattr(solver_linear, 'buildGlobalK'):
            print_success("buildGlobalK method found on LinearElastic2DSolver")
        else:
            print_warning("buildGlobalK method not found on LinearElastic2DSolver")

        if hasattr(solver_linear, 'setSupp'):
            print_success("setSupp method found on LinearElastic2DSolver")
        else:
            print_warning("setSupp method not found on LinearElastic2DSolver")
            
        if hasattr(solver_linear, 'setLoad'):
            print_success("setLoad method found on LinearElastic2DSolver")
            
        if hasattr(solver_linear, 'applyDBC'):
            print_success("applyDBC method found on LinearElastic2DSolver")
        else:
            print_warning("applyDBC method not found on LinearElastic2DSolver")
            
        if hasattr(solver_linear, 'applyNBC'):
            print_success("applyNBC method found on LinearElastic2DSolver")
        else:
            print_warning("applyNBC method not found on LinearElastic2DSolver") 
            
            
        return True
    except Exception as e:
        print_error(f"Error creating BeamSolver object: {e}")
        return False

def test_material_module(module):
    print_header("TESTING MATERIAL MODULE")
    
    if not hasattr(module, 'material'):
        print_error("Material module not found")
        return False
    
    print_success("Material module found")
    
    # Check for Material class
    if not hasattr(module.material, 'Material'):
        print_error("Material class not found in material module")
        return False

    print_success("Material class found in material module")
    
    # Try to create a Material object
    try:
        material = module.material.Material()
        print_success("Successfully created Material object")
        
        # Check methods
        if hasattr(material, 'setElasticModule'):
            print_success("setElasticModule method found on Material")
        else:
            print_warning("setElasticModule method not found on Material")

        if hasattr(material, 'setPoissonCoef'):
            print_success("setPoissonCoef method found on Material")
        else:
            print_warning("setPoissonCoef method not found on Material")

        if hasattr(material, 'setMaterialDensity'):
            print_success("setMaterialDensity method found on Material")
        else:
            print_warning("setMaterialDensity method not found on Material")    

        if hasattr(material, 'getLameParameters'):
            print_success("getLameParameters method found on Material")
        else:
            print_warning("getLameParameters method not found on Material")

        if hasattr(material, 'build2DElasticity'):
            print_success("build2DElasticity method found on Material")
        else:
            print_warning("build2DElasticity method not found on Material")
        
        return True
    except Exception as e:
        print_error(f"Error creating Material object: {e}")
        return False

def test_enums_module(module):
    print_header("TESTING ENUMS MODULE")
    
    if not hasattr(module, 'enums'):
        print_error("Enums module not found")
        return False
    
    print_success("Enums module found")
    
    # Check for BeamSolverType enum
    if not hasattr(module.enums, 'BeamSolverType'):
        print_error("BeamSolverType enum not found in enums module")
        return False
    
    print_success("BeamSolverType enum found in enums module")
    
    # Check enum values
    if hasattr(module.enums.BeamSolverType, 'BEAM'):
        print_success("BEAM value found in BeamSolverType enum")
    else:
        print_warning("BEAM value not found in BeamSolverType enum")
    
    if hasattr(module.enums.BeamSolverType, 'PORTIC'):
        print_success("PORTIC value found in BeamSolverType enum")
    else:
        print_warning("PORTIC value not found in BeamSolverType enum")
    
    return True

if __name__ == "__main__":
    # Check system and environment
    print_header("SYSTEM INFORMATION")
    print_info(f"Python version: {sys.version}")
    print_info(f"Current directory: {os.getcwd()}")
    
    # Test imports
    success, module = test_imports()
    
    if success and module:
        # Test each module
        mesh_success = test_mesh_module(module)
        solver_success = test_solver_module(module)
        material_success = test_material_module(module)
        enums_success = test_enums_module(module)

        # Print summary
        print_header("TEST SUMMARY")
        print_info(f"Module import: {'✅' if success else '❌'}")
        print_info(f"Mesh module: {'✅' if mesh_success else '❌'}")
        print_info(f"Solver module: {'✅' if solver_success else '❌'}")
        print_info(f"Material module: {'✅' if material_success else '❌'}")
        print_info(f"Enums module: {'✅' if enums_success else '❌'}")
        
        # Overall success
        if mesh_success and solver_success and enums_success:
            print_success("\nAll tests passed successfully!")
        else:
            print_error("\nSome tests failed.")
    else:
        print_error("\nFailed to import module. Cannot continue testing.")