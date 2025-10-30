#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "utils/operations.hpp"
#include "utils/integration.hpp"
#include "mesh/beam.hpp"
#include "solver/beam1d.hpp"
#include "material/mat.hpp"
#include "solver/linearElastic2d.hpp"
#include "solver/axisymmetric.hpp"

using Eigen::MatrixXd;

// Function declarations
bool run_patch_test_case(int strain_case, const Eigen::MatrixXd& nodes, const Eigen::MatrixXi& elements, 
                        const Eigen::MatrixXd& K_global, const std::vector<int>& boundary_nodes, 
                        int ndof, int total_nodes);
Eigen::Vector4d calculate_strain_at_centroid(const Eigen::MatrixXd& element_vertices, const Eigen::VectorXd& displacements);

void test_beam_solver(){
    std::cout<<"VEM Beam"<<std::endl;

    mesh::beam bar;

  bar.horizontalBarDisc(0.45, 20);

    Eigen::MatrixXd nodes = bar.nodes;
    Eigen::MatrixXi elements = bar.elements;

    Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
  q(0) = -1.0;
  q(1) = -1.0;

        material::mat elastic;
  elastic.setElasticModule(2.1e+11);
        double E = elastic.E;

  int order = 5;
        solver::beam1d solver(nodes, elements, order);
  
  // Set beam properties
  double height = 0.003;  // 3mm
  double width = 0.02;    // 20mm
  double I = (width * pow(height, 3)) / 12.0; // Moment of inertia
  solver.setInertiaMoment(I);

  std::cout << "\n===== GEOMETRY =====" << std::endl;
  std::cout << "Area: " << height * width << std::endl;
  std::cout << "Ix: " << I << std::endl;

        Eigen::MatrixXd K = solver.buildGlobalK(E);
  std::cout << "Global stiffness shape: " << K.size() << std::endl;

  Eigen::MatrixXd KII = solver.buildStaticCondensation(K, "KII");
  Eigen::MatrixXd KIM = solver.buildStaticCondensation(K, "KIM");
  Eigen::MatrixXd KMI = solver.buildStaticCondensation(K, "KMI");
  Eigen::MatrixXd KMM = solver.buildStaticCondensation(K, "KMM");
        
        solver.setDistributedLoad(q, elements);
        Eigen::VectorXd R = solver.buildGlobalDistributedLoad();
  Eigen::VectorXd RI = solver.buildStaticDistVector(R, "RI");
  Eigen::VectorXd RM = solver.buildStaticDistVector(R, "RM");

  Eigen::MatrixXi supp = Eigen::MatrixXi::Zero(1,4);
        supp(0,0) = 0;
        supp(0,1) = 1;
        supp(0,2) = 1;
        supp(0,3) = 0;
        solver.setSupp(supp);

            Eigen::MatrixXd K_ = KII - KIM * KMM.inverse() * KMI;
            Eigen::MatrixXd R_ = RI - KIM * KMM.inverse() * RM;
            //R_(2) = -0.225;
            //std::cout << R_ << std::endl;

            K_ = solver.applyDBCMatrix(K_);
            R_ = solver.applyDBCVec(R_);

            Eigen::VectorXd uh;
            uh = K_.ldlt().solve(R_);

  // Print first few displacement values for comparison with Python
  std::cout << "First 5 displacement values:" << std::endl;
  for (int i = 0; i < std::min(5, (int)uh.size()); i++) {
    std::cout << "uh[" << i << "] = " << uh(i) << std::endl;
  }
  
  // Original full output if needed
  // std::cout << uh << std::endl;
  std::cout<< solver.buildLocalK(nodes, E) <<std::endl;

  // ===== NEW CODE: STRAIN AND STRESS CALCULATION =====
  
  // Convert condensed displacements back to full displacement vector
  int ndof_primary = uh.size();
  int ndof_moments = RM.size();
  int ndof_total = ndof_primary + ndof_moments;
  
  std::cout << "Converting condensed solution to full solution..." << std::endl;
  std::cout << "Primary DOFs: " << ndof_primary << std::endl;
  std::cout << "Moment DOFs: " << ndof_moments << std::endl;
  
  // Reconstruct the full displacement vector
  Eigen::VectorXd u_full(ndof_total);
  
  // Copy primary displacements
  u_full.head(ndof_primary) = uh;
  
  // Calculate moment displacements: um = -inv(KMM) * (KMI * uh + RM)
  Eigen::VectorXd um = -KMM.inverse() * (KMI * uh + RM);
  u_full.tail(ndof_moments) = um;
  
  std::cout << "Full solution vector size: " << u_full.size() << std::endl;
  
  // Calculate strain and stress
  int sample_points = 10; // Number of sample points per element
  
  // Calculate strains at top fiber (y = height/2)
  std::cout << "\n===== STRAIN CALCULATION =====" << std::endl;
  Eigen::MatrixXd strains = solver.calculateStrain(u_full, E, sample_points, height/2.0);
  
  // Print strains at selected points
  std::cout << "Strains at top fiber (first 5 sample points):" << std::endl;
  std::cout << "Element | X local | Strain" << std::endl;
  for (int i = 0; i < std::min(5, (int)strains.rows()); i++) {
    std::cout << strains(i, 0) << " | " 
              << strains(i, 1) << " | " 
              << strains(i, 2) << std::endl;
  }
  
  // Calculate stresses at top fiber (y = height/2)
  std::cout << "\n===== STRESS CALCULATION =====" << std::endl;
  Eigen::MatrixXd stresses = solver.calculateStress(u_full, E, sample_points, height/2.0);
  
  // Print stresses at selected points
  std::cout << "Stresses at top fiber (first 5 sample points):" << std::endl;
  std::cout << "Element | X local | Stress (Pa)" << std::endl;
  for (int i = 0; i < std::min(5, (int)stresses.rows()); i++) {
    std::cout << stresses(i, 0) << " | " 
              << stresses(i, 1) << " | " 
              << stresses(i, 2) << std::endl;
  }
  
  // Find maximum stress
  double max_stress = stresses.col(2).cwiseAbs().maxCoeff();
  std::cout << "\nMaximum absolute stress: " << max_stress << " Pa" << std::endl;
  std::cout << "Maximum absolute stress: " << max_stress/1e6 << " MPa" << std::endl;
  
  // Get stress at specific locations
  double beam_length = 0.45; // From horizontalBarDisc call
  
  // Calculate stress at 1/4, 1/2, and 3/4 of the beam length
  std::cout << "\nStress at specific global positions:" << std::endl;
  for (double pos_ratio : {0.25, 0.5, 0.75}) {
    double x_global = pos_ratio * beam_length;
    auto [strain, stress] = solver.getStrainStressAtPoint(u_full, E, x_global, height/2.0);
    
    std::cout << "Position: " << x_global << " m (" << pos_ratio*100 << "% of length)" << std::endl;
    std::cout << "  Strain: " << strain << std::endl;
    std::cout << "  Stress: " << stress << " Pa (" << stress/1e6 << " MPa)" << std::endl;
  }
}

void test_elastic_solver(){
  std::cout<<"\n===== VEM Elastic ====="<<std::endl;
  
  int order = 1;

  // Setup geometry
  Eigen::MatrixXd nodes = (Eigen::Matrix<double,9,2>()<<0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0).finished();
  Eigen::MatrixXi elements = (Eigen::Matrix<int,4,4>()<<0,1,2,3, 1,4,5,2, 2,5,6,7, 3,2,7,8).finished();

  // Setup solver
  solver::linearElastic2d solver(nodes, elements, order);

  // Setup material
  material::mat elastic;
  elastic.setElasticModule(7000);
  elastic.setPoissonCoef(0.3);
  Eigen::MatrixXd C = elastic.build2DElasticity();

  std::cout << "Elasticity matrix C:\n" << C << std::endl;

  // Build global stiffness matrix
  Eigen::MatrixXd K = solver.buildGlobalK(C);

  // Apply boundary conditions
  Eigen::MatrixXi supp = (Eigen::Matrix<int, 5,3>()<<0,1,1, 1,0,1, 4,0,1, 3,1,0, 8,1,0).finished();
  solver.setSupp(supp); 
  Eigen::MatrixXd K_ = solver.applyDBC(K);

  Eigen::MatrixXi load = (Eigen::Matrix<int, 2,2>()<<4,5, 5,6).finished();
  solver.setLoad(load);
  double qx = 2000, qy=0;
  Eigen::VectorXd f = solver.applyNBC(qx,qy);

  // Solve the system
  Eigen::VectorXd u = K_.ldlt().solve(f);
  std::cout << "Displacement vector shape: " << u.size() << std::endl;
  std::cout << u << std::endl;
  
}

void test_displacement_function() {
  std::cout << "\n===== Testing create_displacement_function =====" << std::endl;
  
  // Create a simple edge with two vertices
  Eigen::Matrix2d edge_vertices;
  edge_vertices << 1.0, 0.0,    // Vertex 1 (r=1, z=0)
                   2.0, 1.0;    // Vertex 2 (r=2, z=1)
  
  std::pair<int, int> vertex_indices(0, 1);
  
  // Test for radial DOF at first vertex
  auto disp_func1 = solver::axisymmetric::create_displacement_function(edge_vertices, vertex_indices, 0);
  
  // Test for axial DOF at first vertex
  auto disp_func2 = solver::axisymmetric::create_displacement_function(edge_vertices, vertex_indices, 1);
  
  // Test for radial DOF at second vertex
  auto disp_func3 = solver::axisymmetric::create_displacement_function(edge_vertices, vertex_indices, 2);
  
  // Test for axial DOF at second vertex
  auto disp_func4 = solver::axisymmetric::create_displacement_function(edge_vertices, vertex_indices, 3);
  
  // Sample points along the edge
  std::vector<double> sample_points = {0.0, 0.25, 0.5, 0.75, 1.0};
  
  // Test each displacement function
  std::cout << "Radial DOF at first vertex (dof_index=0):" << std::endl;
  std::cout << "s\tv_r\tv_z" << std::endl;
  for (double s : sample_points) {
    auto [v_r, v_z] = disp_func1(s);
    std::cout << s << "\t" << v_r << "\t" << v_z << std::endl;
  }
  
  std::cout << "\nAxial DOF at first vertex (dof_index=1):" << std::endl;
  std::cout << "s\tv_r\tv_z" << std::endl;
  for (double s : sample_points) {
    auto [v_r, v_z] = disp_func2(s);
    std::cout << s << "\t" << v_r << "\t" << v_z << std::endl;
  }
  
  std::cout << "\nRadial DOF at second vertex (dof_index=2):" << std::endl;
  std::cout << "s\tv_r\tv_z" << std::endl;
  for (double s : sample_points) {
    auto [v_r, v_z] = disp_func3(s);
    std::cout << s << "\t" << v_r << "\t" << v_z << std::endl;
  }
  
  std::cout << "\nAxial DOF at second vertex (dof_index=3):" << std::endl;
  std::cout << "s\tv_r\tv_z" << std::endl;
  for (double s : sample_points) {
    auto [v_r, v_z] = disp_func4(s);
    std::cout << s << "\t" << v_r << "\t" << v_z << std::endl;
  }
}

void test_volumetric_correction() {
  std::cout << "\n===== Testing compute_volumetric_correction =====" << std::endl;
  
  // Create a quadrilateral element
  Eigen::MatrixXd quad_vertices(4, 2);
  quad_vertices << 1.0, 0.0,   // Vertex 1 (r=1, z=0)
                   2.0, 0.0,   // Vertex 2 (r=2, z=0)
                   2.0, 1.0,   // Vertex 3 (r=2, z=1)
                   1.0, 1.0;   // Vertex 4 (r=1, z=1)
  
  // Setup material
  material::mat elastic;
  elastic.setElasticModule(1.0);
  elastic.setPoissonCoef(0.3);
  Eigen::Matrix4d C = material::mat::buildAxisymmetricElasticity(1.0, 0.3);
  
  // Get base strain vectors
  std::vector<Eigen::Vector4d> base_strains = solver::axisymmetric::define_base_strain_vectors();
  
  // Compute volumetric correction
  Eigen::MatrixXd vol_correction = solver::axisymmetric::compute_volumetric_correction(
    quad_vertices, C, base_strains
  );
  
  // Print the matrix dimensions
  std::cout << "Volumetric correction matrix dimensions: " 
            << vol_correction.rows() << " × " << vol_correction.cols() << std::endl;
  
  // Print the content of the matrix
  std::cout << "Volumetric correction matrix:" << std::endl;
  std::cout << vol_correction << std::endl;
  
  // Analyze the results - calculate average correction for each strain type
  std::cout << "\nAverage corrections for each strain type:" << std::endl;
  for (int i = 0; i < vol_correction.cols(); ++i) {
    // Only consider the radial DOFs (even indices)
    double avg_correction = 0.0;
    int count = 0;
    for (int j = 0; j < vol_correction.rows(); j += 2) {
      avg_correction += vol_correction(j, i);
      count++;
    }
    avg_correction /= count;
    
    std::string strain_type;
    switch (i) {
      case 0: strain_type = "radial strain (εr)"; break;
      case 1: strain_type = "axial strain (εz)"; break;
      case 2: strain_type = "hoop strain (εθ)"; break;
      case 3: strain_type = "shear strain (γrz)"; break;
      default: strain_type = "unknown";
    }
    
    std::cout << "Strain " << i << " (" << strain_type << "): " << avg_correction << std::endl;
  }
}

void test_proj_system_matrix() {
  std::cout << "\n===== Testing compute_proj_system_matrix =====" << std::endl;
  
  // Setup material
  material::mat elastic;
  elastic.setElasticModule(1.0);  // Using unit Young's modulus for simplicity
  elastic.setPoissonCoef(0.3);
  Eigen::Matrix4d C = material::mat::buildAxisymmetricElasticity(1.0, 0.3);
  
  // Create base strain vectors
  std::vector<Eigen::Vector4d> base_strains = solver::axisymmetric::define_base_strain_vectors();
  
  // Convert base_strains to a matrix (each column is a base strain)
  Eigen::MatrixXd eps_matrix(4, 4);
  for (int i = 0; i < 4; i++) {
    eps_matrix.col(i) = base_strains[i];
  }
  
  // Define weighted volume (for testing, we'll use 1.0)
  double weighted_volume = 1.0;
  
  // Compute projection system matrix
  Eigen::MatrixXd proj_matrix = solver::axisymmetric::compute_proj_system_matrix(
    C, eps_matrix, weighted_volume
  );
  
  // Print the matrix dimensions
  std::cout << "Projection system matrix dimensions: " 
            << proj_matrix.rows() << " × " << proj_matrix.cols() << std::endl;
  
  // Print the matrix
  std::cout << "Projection system matrix:\n" << proj_matrix << std::endl;
  
  // Verify that the projection matrix is correctly computed
  // For standard elasticity with ν=0.3, we should see:
  // - Diagonal terms should be positive
  // - Off-diagonal terms (coupling between different strain types) should exist
  
  std::cout << "\nDiagonal terms:" << std::endl;
  for (int i = 0; i < 4; i++) {
    std::string strain_type;
    switch (i) {
      case 0: strain_type = "radial strain (εr)"; break;
      case 1: strain_type = "axial strain (εz)"; break;
      case 2: strain_type = "hoop strain (εθ)"; break;
      case 3: strain_type = "shear strain (γrz)"; break;
      default: strain_type = "unknown";
    }
    std::cout << strain_type << ": " << proj_matrix(i, i) << std::endl;
  }
}

void test_projection_matrix() {
  std::cout << "\n===== Testing compute_projection_matrix =====" << std::endl;
  
  // Setup material
  material::mat elastic;
  elastic.setElasticModule(1.0);  // Using unit Young's modulus for simplicity
  elastic.setPoissonCoef(0.3);
  Eigen::Matrix4d C = material::mat::buildAxisymmetricElasticity(1.0, 0.3);
  
  std::cout << "Constitutive matrix C:\n" << C << std::endl;
  
  // Create base strain vectors
  std::vector<Eigen::Vector4d> base_strains = solver::axisymmetric::define_base_strain_vectors();
  
  std::cout << "Base strain vectors:" << std::endl;
  for (size_t i = 0; i < base_strains.size(); ++i) {
    std::cout << "Strain " << i << ": " << base_strains[i].transpose() << std::endl;
  }
  
  // Define a simple square element
  Eigen::MatrixXd quad_vertices(4, 2);
  quad_vertices << 1.0, 0.0,    // Vertex 1 (r=1, z=0)
                   2.0, 0.0,    // Vertex 2 (r=2, z=0)
                   2.0, 1.0,    // Vertex 3 (r=2, z=1)
                   1.0, 1.0;    // Vertex 4 (r=1, z=1)
  
  std::cout << "Quadrilateral element vertices:\n" << quad_vertices << std::endl;
  
  // Compute projection matrix
  Eigen::MatrixXd B = solver::axisymmetric::compute_projection_matrix(
    quad_vertices, C, base_strains
  );
  
  std::cout << "\nProjection matrix B (rows: strain components, columns: DOFs):" << std::endl;
  std::cout << B << std::endl;
  
  // Test 1: Constant radial strain (u_r = r, u_z = 0)
  Eigen::VectorXd d_test1 = Eigen::VectorXd::Zero(8);
  for (int i = 0; i < 4; i++) {
    d_test1(2*i) = quad_vertices(i, 0);  // r-coordinate
  }
  
  std::cout << "\nTest 1 displacement vector (u_r = r, u_z = 0):\n" << d_test1.transpose() << std::endl;
  
  // Test 2: Constant axial strain (u_r = 0, u_z = z)
  Eigen::VectorXd d_test2 = Eigen::VectorXd::Zero(8);
  for (int i = 0; i < 4; i++) {
    d_test2(2*i+1) = quad_vertices(i, 1);  // z-coordinate
  }
  
  std::cout << "Test 2 displacement vector (u_r = 0, u_z = z):\n" << d_test2.transpose() << std::endl;
  
  // Test 3: Constant unit radial displacement (u_r = 1, u_z = 0)
  Eigen::VectorXd d_test3 = Eigen::VectorXd::Zero(8);
  for (int i = 0; i < 4; i++) {
    d_test3(2*i) = 1.0;
  }
  
  std::cout << "Test 3 displacement vector (u_r = 1, u_z = 0):\n" << d_test3.transpose() << std::endl;
  
  // Project the test fields
  Eigen::Vector4d projected_strain1 = B * d_test1;
  Eigen::Vector4d projected_strain2 = B * d_test2;
  Eigen::Vector4d projected_strain3 = B * d_test3;
  
  std::cout << "\nTest 1: Constant radial strain (u_r = r, u_z = 0)" << std::endl;
  std::cout << "Projected strains:" << std::endl;
  std::cout << "εr = " << projected_strain1(0) << " (expect 1.0)" << std::endl;
  std::cout << "εz = " << projected_strain1(1) << " (expect 0.0)" << std::endl;
  std::cout << "εθ = " << projected_strain1(2) << " (expect 1.0)" << std::endl;
  std::cout << "γrz = " << projected_strain1(3) << " (expect 0.0)" << std::endl;
  
  // Verify the expected values by direct calculation of strain components
  std::cout << "\nVerification by direct strain calculation:" << std::endl;
  double r_avg = (1.0 + 2.0 + 2.0 + 1.0) / 4.0;
  std::cout << "Average r: " << r_avg << std::endl;
  std::cout << "Direct εr calculation for u_r = r: 1.0" << std::endl;
  std::cout << "Direct εθ calculation for u_r = r: 1.0" << std::endl;
  
  std::cout << "\nTest 2: Constant axial strain (u_r = 0, u_z = z)" << std::endl;
  std::cout << "Projected strains:" << std::endl;
  std::cout << "εr = " << projected_strain2(0) << " (expect 0.0)" << std::endl;
  std::cout << "εz = " << projected_strain2(1) << " (expect 1.0)" << std::endl;
  std::cout << "εθ = " << projected_strain2(2) << " (expect 0.0)" << std::endl;
  std::cout << "γrz = " << projected_strain2(3) << " (expect 0.0)" << std::endl;
  
  std::cout << "\nTest 3: Constant unit radial displacement (u_r = 1, u_z = 0)" << std::endl;
  std::cout << "Projected strains:" << std::endl;
  std::cout << "εr = " << projected_strain3(0) << " (expect 0.0)" << std::endl;
  std::cout << "εz = " << projected_strain3(1) << " (expect 0.0)" << std::endl;
  std::cout << "εθ = " << projected_strain3(2) << " (expect ~0.33-0.5)" << std::endl;
  std::cout << "γrz = " << projected_strain3(3) << " (expect 0.0)" << std::endl;
  
  // Verify the expected values by direct calculation
  std::cout << "\nVerification by direct strain calculation:" << std::endl;
  std::cout << "Direct εθ calculation for u_r = 1: " << (1.0 / r_avg) << std::endl;
}

void test_weighted_volume_polygon() {
  std::cout << "\n===== Testing compute_weighted_volume_polygon =====" << std::endl;
  
  // Define different polygons for testing
  // Test 1: Simple square with one vertex at origin
  Eigen::MatrixXd square(4, 2);
  square << 0.0, 0.0,    // Vertex 1 (r=0, z=0)
            1.0, 0.0,    // Vertex 2 (r=1, z=0)
            1.0, 1.0,    // Vertex 3 (r=1, z=1)
            0.0, 1.0;    // Vertex 4 (r=0, z=1)
  
  // Test 2: Rectangle in positive r-space (like in the projection matrix test)
  Eigen::MatrixXd rectangle(4, 2);
  rectangle << 1.0, 0.0,    // Vertex 1 (r=1, z=0)
               2.0, 0.0,    // Vertex 2 (r=2, z=0)
               2.0, 1.0,    // Vertex 3 (r=2, z=1)
               1.0, 1.0;    // Vertex 4 (r=1, z=1)
  
  // Test 3: Triangle in positive r-space
  Eigen::MatrixXd triangle(3, 2);
  triangle << 1.0, 0.0,    // Vertex 1 (r=1, z=0)
              2.0, 0.0,    // Vertex 2 (r=2, z=0)
              1.5, 1.0;    // Vertex 3 (r=1.5, z=1)
  
  // Compute weighted volumes
  double square_wv = solver::axisymmetric::compute_weighted_volume_polygon(square);
  double rectangle_wv = solver::axisymmetric::compute_weighted_volume_polygon(rectangle);
  double triangle_wv = solver::axisymmetric::compute_weighted_volume_polygon(triangle);
  
  // Compute areas using the direct method for comparison
  utils::operations operations;
  double square_area = operations.calcArea(square);
  double rectangle_area = operations.calcArea(rectangle);
  double triangle_area = operations.calcArea(triangle);
  
  // Compute simple weighted volumes using centroid method for comparison
  double square_r_avg = square.col(0).mean();
  double rectangle_r_avg = rectangle.col(0).mean();
  double triangle_r_avg = triangle.col(0).mean();
  
  double square_simple_wv = square_area * square_r_avg;
  double rectangle_simple_wv = rectangle_area * rectangle_r_avg;
  double triangle_simple_wv = triangle_area * triangle_r_avg;
  
  // Print results
  std::cout << "Square results:" << std::endl;
  std::cout << "  Area: " << square_area << std::endl;
  std::cout << "  Average r: " << square_r_avg << std::endl;
  std::cout << "  Simple weighted volume (area * avg_r): " << square_simple_wv << std::endl;
  std::cout << "  Proper weighted volume: " << square_wv << std::endl;
  
  std::cout << "\nRectangle results:" << std::endl;
  std::cout << "  Area: " << rectangle_area << std::endl;
  std::cout << "  Average r: " << rectangle_r_avg << std::endl;
  std::cout << "  Simple weighted volume (area * avg_r): " << rectangle_simple_wv << std::endl;
  std::cout << "  Proper weighted volume: " << rectangle_wv << std::endl;
  
  std::cout << "\nTriangle results:" << std::endl;
  std::cout << "  Area: " << triangle_area << std::endl;
  std::cout << "  Average r: " << triangle_r_avg << std::endl;
  std::cout << "  Simple weighted volume (area * avg_r): " << triangle_simple_wv << std::endl;
  std::cout << "  Proper weighted volume: " << triangle_wv << std::endl;
}

void test_axisymmetric_solver(){
  std::cout<<"\n===== VEM Axisymmetric ====="<<std::endl;
  
  // Test base strain vectors
  solver::axisymmetric solver;
  std::vector<Eigen::Vector4d> base_strain_vectors = solver.define_base_strain_vectors();
  std::cout << "Base strain vectors:" << std::endl;
  for (const auto& vec : base_strain_vectors) {
    std::cout << vec.transpose() << std::endl;
  }

  // Setup material
  double E = 2.1e+11;
  double nu = 0.3;
  Eigen::Matrix4d C = material::mat::buildAxisymmetricElasticity(E, nu);

  std::cout << "Elasticity matrix C:\n" << C << std::endl;
  
  // Test displacement function
  test_displacement_function();
  
  // Test volumetric correction
  test_volumetric_correction();
  
  // Test weighted volume calculation
  test_weighted_volume_polygon();
  
  // Test projection system matrix
  test_proj_system_matrix();
  
  // Test projection matrix
  test_projection_matrix();
}

void patch_test_vem(){
  std::cout<<"\n===== VEM Patch Test ====="<<std::endl;

  // Material properties
  double E = 1.0;       // Young's modulus
  double nu = 0.3;      // Poisson's ratio

  // Create the constitutive matrix
  Eigen::Matrix4d C = material::mat::buildAxisymmetricElasticity(E, nu);

  std::cout << "\nConstitutive matrix C:" << std::endl;
  std::cout << C << std::endl;

  // Domain setup
  double r_inner = 1.0;
  double r_outer = 3.0;
  double z_min = 0.0;
  double z_max = 2.0;

  // Create a mesh of quadrilateral elements
  int n_r = 4;  // elements in r direction
  int n_z = 4;  // elements in z direction

  // Generate mesh nodes
  int nr_nodes = n_r + 1;
  int nz_nodes = n_z + 1;
  Eigen::MatrixXd grid_nodes(nr_nodes * nz_nodes, 2);  // (r, z) coordinates

  int node_idx = 0;
  for (int iz = 0; iz < nz_nodes; iz++) {
    for (int ir = 0; ir < nr_nodes; ir++) {
      double r = r_inner + (r_outer - r_inner) * ir / (nr_nodes - 1);
      double z = z_min + (z_max - z_min) * iz / (nz_nodes - 1);
      grid_nodes(node_idx, 0) = r;
      grid_nodes(node_idx, 1) = z;
      node_idx++;
    }
  }

  // Create quadrilateral elements
  int total_quads = n_r * n_z;
  Eigen::MatrixXi quads(total_quads, 4);

  int quad_idx = 0;
  for (int iz = 0; iz < n_z; iz++) {
    for (int ir = 0; ir < n_r; ir++) {
      // Indices of the four corners of the quad
      int n1 = iz * nr_nodes + ir;
      int n2 = iz * nr_nodes + (ir + 1);
      int n3 = (iz + 1) * nr_nodes + (ir + 1);
      int n4 = (iz + 1) * nr_nodes + ir;

      // Create quad (counter-clockwise ordering)
      quads(quad_idx, 0) = n1;
      quads(quad_idx, 1) = n2;
      quads(quad_idx, 2) = n3;
      quads(quad_idx, 3) = n4;
      quad_idx++;
    }
  }

  // Copy grid nodes to be our final nodes
  Eigen::MatrixXd nodes = grid_nodes;
  Eigen::MatrixXi elements = quads;

  int total_nodes = nodes.rows();
  int total_elements = elements.rows();

  std::cout << "\nTotal nodes: " << total_nodes << std::endl;
  std::cout << "Total elements: " << total_elements << std::endl;
  std::cout << "First quadrilateral nodes: " << elements.row(0) << std::endl;

  // Total degrees of freedom
  int ndof = 2 * total_nodes;  // 2 DOFs per node (r, z displacements)
  std::cout << "Total DOFs: " << ndof << std::endl;

  // Assemble global stiffness matrix
  Eigen::MatrixXd K_global = Eigen::MatrixXd::Zero(ndof, ndof);

  // Test the compute_stiffness_matrix function
  std::cout << "\nTesting compute_stiffness_matrix..." << std::endl;
  for (int e = 0; e < total_elements; e++) {
    // Get vertex indices for this element
    Eigen::VectorXi node_indices = elements.row(e);

    // Get node coordinates for this element
    Eigen::MatrixXd element_vertices(4, 2);
    for (int i = 0; i < 4; i++) {
      element_vertices.row(i) = nodes.row(node_indices[i]);
    }

    // Compute element stiffness matrix
    std::string stab_type = "boundary";  // STAB_TYPE from Python
    Eigen::MatrixXd K_elem = solver::axisymmetric::compute_stiffness_matrix(
      element_vertices, E, nu, stab_type
    );


    // Map local DOFs to global DOFs
    std::vector<int> dof_indices(8);
    for (int i = 0; i < 4; i++) {
      dof_indices[2*i] = 2 * node_indices[i];      // r-displacement
      dof_indices[2*i+1] = 2 * node_indices[i] + 1;  // z-displacement
    }

    // Assemble element matrix into global matrix
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        K_global(dof_indices[i], dof_indices[j]) += K_elem(i, j);
      }
    }
  }

  // Find boundary nodes
  std::vector<int> boundary_nodes;
  for (int i = 0; i < total_nodes; i++) {
    double r = nodes(i, 0);
    double z = nodes(i, 1);
    if (std::abs(r - r_inner) < 1e-6 || std::abs(r - r_outer) < 1e-6 ||
        std::abs(z - z_min) < 1e-6 || std::abs(z - z_max) < 1e-6) {
      boundary_nodes.push_back(i);
    }
  }

  std::cout << "\nNumber of boundary nodes: " << boundary_nodes.size() << std::endl;

  // Run the four patch tests
  bool all_tests_passed = true;

  for (int strain_case = 1; strain_case <= 4; strain_case++) {
    bool passed = run_patch_test_case(strain_case, nodes, elements, K_global, boundary_nodes, ndof, total_nodes);
    all_tests_passed = all_tests_passed && passed;
  }

  // Summary
  std::cout << "\n=== PATCH TEST SUMMARY ===" << std::endl;
  std::cout << "Overall result: " << (all_tests_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
}

bool run_patch_test_case(int strain_case, const Eigen::MatrixXd& nodes, const Eigen::MatrixXi& elements, 
                        const Eigen::MatrixXd& K_global, const std::vector<int>& boundary_nodes, 
                        int ndof, int total_nodes) {
  // Initialize force vector
  Eigen::VectorXd F = Eigen::VectorXd::Zero(ndof);

  // Set strain magnitude
  double strain_magnitude = 0.01;

  // Define strain state based on test case
  double strain_r = 0.0;
  double strain_z = 0.0;
  double strain_theta = 0.0;
  double strain_rz = 0.0;

  std::string test_name;
  if (strain_case == 1) {
    strain_r = strain_magnitude;      // Radial strain test
    test_name = "Radial Strain";
  } else if (strain_case == 2) {
    strain_z = strain_magnitude;      // Axial strain test
    test_name = "Axial Strain";
  } else if (strain_case == 3) {
    strain_theta = strain_magnitude;  // Hoop strain test
    test_name = "Hoop Strain";
  } else if (strain_case == 4) {
    strain_rz = strain_magnitude;     // Shear strain test
    test_name = "Shear Strain";
  }

  std::cout << "\n\n--- Testing Case " << strain_case << ": Constant " << test_name << " ---" << std::endl;

  // Exact displacement field based on strain state
  Eigen::VectorXd u_exact = Eigen::VectorXd::Zero(ndof);
  for (int i = 0; i < total_nodes; i++) {
    double r = nodes(i, 0);
    double z = nodes(i, 1);

    if (strain_case == 1) {  // Radial strain
      u_exact[2*i] = strain_r * r;       // Radial displacement (ur)
      u_exact[2*i+1] = 0;                // Axial displacement (uz)
    } else if (strain_case == 2) {  // Axial strain
      u_exact[2*i] = 0;                  // Radial displacement (ur)
      u_exact[2*i+1] = strain_z * z;     // Axial displacement (uz)
    } else if (strain_case == 3) {  // Hoop strain (same as radial for displacement)
      u_exact[2*i] = strain_theta * r;   // Radial displacement (ur)
      u_exact[2*i+1] = 0;                // Axial displacement (uz)
    } else if (strain_case == 4) {  // Shear strain
      u_exact[2*i] = strain_rz * z / 2;    // Radial displacement (ur)
      u_exact[2*i+1] = strain_rz * r / 2;  // Axial displacement (uz)
    }
  }

  // Set up boundary conditions
  std::vector<bool> free_dofs(ndof, true);
  std::vector<int> boundary_dofs;

  for (int i : boundary_nodes) {
    boundary_dofs.push_back(2*i);
    boundary_dofs.push_back(2*i+1);
  }

  for (int dof : boundary_dofs) {
    free_dofs[dof] = false;
  }

  // Count free DOFs
  int num_free_dofs = 0;
  for (bool is_free : free_dofs) {
    if (is_free) num_free_dofs++;
  }

  // Solve the system
  Eigen::VectorXd F_mod = F - K_global * u_exact;

  // Extract free DOFs
  Eigen::VectorXd F_reduced(num_free_dofs);
  Eigen::MatrixXd K_reduced(num_free_dofs, num_free_dofs);

  int free_idx = 0;
  for (int i = 0; i < ndof; i++) {
    if (free_dofs[i]) {
      F_reduced[free_idx] = F_mod[i];
      
      int free_jdx = 0;
      for (int j = 0; j < ndof; j++) {
        if (free_dofs[j]) {
          K_reduced(free_idx, free_jdx) = K_global(i, j);
          free_jdx++;
        }
      }
      free_idx++;
    }
  }

  // Solve reduced system
  Eigen::VectorXd u_reduced = K_reduced.ldlt().solve(F_reduced);

  // Reconstruct full solution
  Eigen::VectorXd u = u_exact;
  free_idx = 0;
  for (int i = 0; i < ndof; i++) {
    if (free_dofs[i]) {
      u[i] = u_reduced[free_idx];
      free_idx++;
    }
  }

  // Calculate strains at the centroid of each element
  int total_elements = elements.rows();
  Eigen::MatrixXd strains(total_elements, 4);  // [εr, εz, εθ, γrz]

  for (int e = 0; e < total_elements; e++) {
    Eigen::VectorXi node_indices = elements.row(e);
    Eigen::MatrixXd element_vertices(4, 2);
    for (int i = 0; i < 4; i++) {
      element_vertices.row(i) = nodes.row(node_indices[i]);
    }

    // Get element displacements
    Eigen::VectorXd elem_disps(8);
    for (int i = 0; i < 4; i++) {
      elem_disps[2*i] = u[2*node_indices[i]];       // ur
      elem_disps[2*i+1] = u[2*node_indices[i]+1];   // uz
    }

    // Calculate strain at element centroid
    Eigen::Vector4d element_strain = calculate_strain_at_centroid(element_vertices, elem_disps);
    strains.row(e) = element_strain.transpose();
  }

  // Compute average strains and errors
  Eigen::Vector4d avg_strains = strains.colwise().mean();

  std::cout << "\nAverage strains across all elements:" << std::endl;
  std::cout << "εr (Radial strain): " << avg_strains[0] << " (Expected: " << strain_r << ")" << std::endl;
  std::cout << "εz (Axial strain): " << avg_strains[1] << " (Expected: " << strain_z << ")" << std::endl;
  std::cout << "εθ (Hoop strain): " << avg_strains[2] << " (Expected: " << strain_theta << ")" << std::endl;
  std::cout << "γrz (Shear strain): " << avg_strains[3] << " (Expected: " << strain_rz << ")" << std::endl;

  // Calculate strain errors
  Eigen::Vector4d strain_errors;
  strain_errors[0] = std::abs(avg_strains[0] - strain_r);
  strain_errors[1] = std::abs(avg_strains[1] - strain_z);
  strain_errors[2] = std::abs(avg_strains[2] - strain_theta);
  strain_errors[3] = std::abs(avg_strains[3] - strain_rz);

  std::cout << "\nStrain errors:" << std::endl;
  std::cout << "Error in εr: " << strain_errors[0] << std::endl;
  std::cout << "Error in εz: " << strain_errors[1] << std::endl;
  std::cout << "Error in εθ: " << strain_errors[2] << std::endl;
  std::cout << "Error in γrz: " << strain_errors[3] << std::endl;

  // Check if patch test passed
  double tolerance = 1e-3;
  bool passed = strain_errors.maxCoeff() < tolerance;

  std::cout << "\nPatch test for " << test_name << ": " << (passed ? "PASSED" : "FAILED") 
            << " with tolerance " << tolerance << std::endl;

  return passed;
}

Eigen::Vector4d calculate_strain_at_centroid(const Eigen::MatrixXd& element_vertices, const Eigen::VectorXd& displacements) {
  // Calculate centroid
  double centroid_r = element_vertices.col(0).mean();
  double centroid_z = element_vertices.col(1).mean();

  int n_vertices = element_vertices.rows();

  // Initialize strain
  Eigen::Vector4d strain = Eigen::Vector4d::Zero();

  // Compute average displacement derivatives using a finite difference approximation
  double du_r_dr = 0;
  double du_r_dz = 0;
  double du_z_dr = 0;
  double du_z_dz = 0;

  // Count how many edges contribute to each derivative
  double count_r = 0;
  double count_z = 0;

  // Loop over all edges
  for (int i = 0; i < n_vertices; i++) {
    int j = (i + 1) % n_vertices;

    // Get the coordinates and displacements at the endpoints
    double ri = element_vertices(i, 0);
    double zi = element_vertices(i, 1);
    double rj = element_vertices(j, 0);
    double zj = element_vertices(j, 1);

    double u_ri = displacements[2*i];
    double u_zi = displacements[2*i+1];
    double u_rj = displacements[2*j];
    double u_zj = displacements[2*j+1];

    // Skip edges with zero length
    double edge_length = std::sqrt((rj-ri)*(rj-ri) + (zj-zi)*(zj-zi));
    if (edge_length < 1e-10) {
      continue;
    }

    // For vertical edges (constant r), compute z derivatives
    if (std::abs(rj - ri) < 1e-10) {
      du_r_dz += (u_rj - u_ri) / (zj - zi);
      du_z_dz += (u_zj - u_zi) / (zj - zi);
      count_z += 1;
    }
    // For horizontal edges (constant z), compute r derivatives
    else if (std::abs(zj - zi) < 1e-10) {
      du_r_dr += (u_rj - u_ri) / (rj - ri);
      du_z_dr += (u_zj - u_zi) / (rj - ri);
      count_r += 1;
    }
    // For general edges, decompose based on direction
    else {
      // Compute unit edge vector
      double dr = (rj - ri) / edge_length;
      double dz = (zj - zi) / edge_length;

      // Projection of derivatives along the edge
      double du_r_ds = (u_rj - u_ri) / edge_length;
      double du_z_ds = (u_zj - u_zi) / edge_length;

      // Decompose based on direction cosines
      du_r_dr += du_r_ds * dr * std::abs(dr);
      du_r_dz += du_r_ds * dz * std::abs(dz);
      du_z_dr += du_z_ds * dr * std::abs(dr);
      du_z_dz += du_z_ds * dz * std::abs(dz);

      count_r += std::abs(dr);
      count_z += std::abs(dz);
    }
  }

  // Average the derivatives
  du_r_dr = du_r_dr / std::max(1.0, count_r);
  du_r_dz = du_r_dz / std::max(1.0, count_z);
  du_z_dr = du_z_dr / std::max(1.0, count_r);
  du_z_dz = du_z_dz / std::max(1.0, count_z);

  // Compute strains
  strain[0] = du_r_dr;                  // εr = ∂u_r/∂r
  strain[1] = du_z_dz;                  // εz = ∂u_z/∂z

  // For εθ, compute average u_r at centroid
  double u_r_centroid = 0;
  for (int i = 0; i < n_vertices; i++) {
    u_r_centroid += displacements[2*i];
  }
  u_r_centroid /= n_vertices;

  strain[2] = u_r_centroid / centroid_r;  // εθ = u_r/r
  strain[3] = du_r_dz + du_z_dr;          // γrz = ∂u_r/∂z + ∂u_z/∂r

  return strain;
}

int main()
{
    // test_beam_solver();
    // test_elastic_solver();
    // test_axisymmetric_solver();
    patch_test_vem();
	return 0;
}