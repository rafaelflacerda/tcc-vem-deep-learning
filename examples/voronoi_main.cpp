#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include "lib/utils/operations.hpp"
#include "lib/utils/integration.hpp"
#include "lib/mesh/beam.hpp"
#include "lib/solver/beam1d.hpp"
#include "lib/material/mat.hpp"
#include "lib/solver/linearElastic2d.hpp"
#include "lib/mesh/datasource.hpp"
#include "lib/mesh/delaunay.hpp"
#include "lib/mesh/voronoi.hpp"
#include "lib/mesh/voronoiMesh.hpp"         
#include "lib/mesh/helpers.hpp"
#include "models/enums.hpp"

using Eigen::MatrixXd;
using json = nlohmann::json;
using namespace mesh;  // Add this at the top with other using statements

// Example function to test the Delaunay triangulation
void testBasicTriangulation() {
    using namespace delaunay;
    using namespace meshHelpers;
    // Create some sample points
    std::vector<Point> points = {
        Point(100, 100),
        Point(200, 50),
        Point(300, 150),
        Point(250, 250),
        Point(150, 300),
        Point(100, 200),
        Point(200, 200)
    };
    
    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create the triangulation
    DelaunayTriangulation delaunay;
    std::vector<Triangle> triangles = delaunay.triangulate(points);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    // Print results
    std::cout << "Generated " << triangles.size() << " triangles in " 
              << duration.count() << " ms:" << std::endl;
              
    for (size_t i = 0; i < triangles.size(); ++i) {
        std::cout << "Triangle " << (i + 1) << ": " << triangles[i] << std::endl;
    }
    
    std::vector<Edge> edges = delaunay.getEdges();
    std::cout << "\nGenerated " << edges.size() << " unique edges:" << std::endl;
    for (size_t i = 0; i < edges.size(); ++i) {
        std::cout << "Edge " << (i + 1) << ": " << edges[i] << std::endl;
    }
}

void testTriangulationJSONStore() {
    using namespace delaunay;
    using namespace meshHelpers;
    // Create some sample points
    std::vector<Point> points = {
        Point(100, 100),
        Point(200, 50),
        Point(300, 150),
        Point(250, 250),
        Point(150, 300),
        Point(100, 200),
        Point(200, 200)
    };
    
    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create the triangulation
    DelaunayTriangulation delaunay;
    std::vector<Triangle> triangles = delaunay.triangulate(points);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    // Print results
    std::cout << "Generated " << triangles.size() << " triangles in " 
              << duration.count() << " ms:" << std::endl;
              
    for (size_t i = 0; i < triangles.size(); ++i) {
        std::cout << "Triangle " << (i + 1) << ": " << triangles[i] << std::endl;
    }
    
    std::vector<Edge> edges = delaunay.getEdges();
    std::cout << "\nGenerated " << edges.size() << " unique edges:" << std::endl;
    for (size_t i = 0; i < edges.size(); ++i) {
        std::cout << "Edge " << (i + 1) << ": " << edges[i] << std::endl;
    }
    
    // Export to SVG
    mesh::datasource::exportTriangulationToJson(points, triangles, edges, "triangulation.json");
}

// Function to generate random points
std::vector<Point> generateRandomPoints(int count, double minX, double maxX, double minY, double maxY) {
    std::vector<Point> points;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distX(minX, maxX);
    std::uniform_real_distribution<double> distY(minY, maxY);
    
    for (int i = 0; i < count; ++i) {
        points.push_back(Point(distX(gen), distY(gen)));
    }
    
    return points;
}

// Test function for Delaunay triangulation and Voronoi diagram
void testDelaunayAndVoronoi() {
    // Create sample points
    std::vector<Point> points = {
        Point(100, 100),
        Point(200, 50),
        Point(300, 150),
        Point(250, 250),
        Point(150, 300),
        Point(100, 200),
        Point(200, 200)
    };
    
    // Alternative: generate random points
    // std::vector<Point> points = generateRandomPoints(20, 50, 450, 50, 450);
    
    // Create Delaunay triangulation
    std::cout << "Creating Delaunay triangulation..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    DelaunayTriangulation delaunay;
    std::vector<Triangle> triangles = delaunay.triangulate(points);
    std::vector<Edge> edges = delaunay.getEdges();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    std::cout << "Generated " << triangles.size() << " triangles and " 
              << edges.size() << " edges in " << duration.count() << " ms" << std::endl;
    
    // Export Delaunay triangulation to JSON
    mesh::datasource::exportTriangulationToJson(points, triangles, edges, "triangulation.json");
    std::cout << "Delaunay triangulation exported to delaunay_triangulation.json" << std::endl;
    
    // Create Voronoi diagram from Delaunay triangulation
    std::cout << "\nCreating Voronoi diagram..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    VoronoiDiagram voronoi;
    voronoi.buildFromDelaunay(points, triangles);
    
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    
    std::cout << "Generated Voronoi diagram with " 
              << voronoi.getVertices().size() << " vertices, "
              << voronoi.getEdges().size() << " edges, and "
              << voronoi.getCells().size() << " cells in "
              << duration.count() << " ms" << std::endl;

    auto vertices = voronoi.getVertices();
    auto voronoiEdges = voronoi.getEdges();
    auto cells = voronoi.getCells();
    
    // Export Voronoi diagram to JSON
    mesh::datasource::exportVoronoiToJson(vertices, voronoiEdges, cells,  "voronoi_mesh.json");
    // std::cout << "Voronoi diagram exported to voronoi_diagram.json" << std::endl;
}

void generateRectangularMesh() {
    std::cout << "Generating rectangular Voronoi mesh..." << std::endl;
    
    double minX = 0.0, minY = 0.0, maxX = 15.0, maxY = 5.0;
    RectangleGeometry rectangle(minX, minY, maxX, maxY);
    
    // Create mesh generator
    VoronoiMeshGenerator meshGenerator;
    
    int numPoints = 300;  // Make this a variable since we need a reference
    // Generate Voronoi mesh with 150 cells using grid-based distribution
    // and 2 iterations of Lloyd's relaxation
    auto cells = meshGenerator.generateMesh(rectangle, numPoints, "grid", 100);
    
    std::cout << "Generated " << cells.size() << " Voronoi cells" << std::endl;
    
    // Export the Voronoi mesh to JSON
    mesh::datasource::exportVoronoiMeshToJson(cells, rectangle, "rectangular_voronoi_v2.json");
    
    // Convert to Eigen matrices for FEM analysis
    Eigen::MatrixXd nodes;
    Eigen::MatrixXi elements;
    meshGenerator.convertToEigenMesh(cells, nodes, elements);
    
    // Export the mesh in Eigen format
    mesh::datasource::exportEigenMeshToJson(nodes, elements, "rectangular_eigen_v2.json");
    
    std::cout << "Rectangular mesh exported successfully" << std::endl;
}

void generatePolygonMesh() {
    std::cout << "Generating polygon Voronoi mesh..." << std::endl;
    
    // Define a polygon geometry (L-shape)
    std::vector<delaunay::Point> vertices = {
        delaunay::Point(0.0, 0.0),
        delaunay::Point(10.0, 0.0),
        delaunay::Point(10.0, 5.0),
        delaunay::Point(5.0, 5.0),
        delaunay::Point(5.0, 10.0),
        delaunay::Point(0.0, 10.0)
    };
    PolygonGeometry polygon(vertices);

    // Test if points are inside the polygon
    std::cout << "Testing polygon geometry..." << std::endl;
    int insideCount = 0;
    for (double x = 0.5; x < 10.0; x += 1.0) {
        for (double y = 0.5; y < 10.0; y += 1.0) {
            delaunay::Point p(x, y);
            if (polygon.contains(p)) {
                insideCount++;
                std::cout << "Point (" << x << "," << y << ") is inside" << std::endl;
            }
        }
    }
    std::cout << "Found " << insideCount << " points inside the polygon" << std::endl;
    
    // Create mesh generator
    VoronoiMeshGenerator meshGenerator;
    
    // Generate Voronoi mesh with 300 cells using Poisson disk distribution
    // and 2 iterations of Lloyd's relaxation
    auto cells = meshGenerator.generateMesh(polygon, 300, "poisson_disk", 2);
    
    std::cout << "Generated " << cells.size() << " Voronoi cells" << std::endl;
    
    // Export the Voronoi mesh to JSON
    mesh::datasource::exportVoronoiMeshToJson(cells, polygon, "polygon_voronoi_v2.json");
    
    // Convert to Eigen matrices for FEM analysis
    Eigen::MatrixXd nodes;
    Eigen::MatrixXi elements;
    meshGenerator.convertToEigenMesh(cells, nodes, elements);
    
    std::cout << "Generated mesh with " << nodes.rows() << " nodes and " 
              << elements.rows() << " elements" << std::endl;
    
    // Export the Eigen mesh to JSON
    mesh::datasource::exportEigenMeshToJson(nodes, elements, "polygon_eigen.json");
    
    std::cout << "Polygon mesh exported successfully" << std::endl;
}


 
int main()
{
    std::cout << "Running main.cpp" << std::endl;
    // testBasicTriangulation();
    // testTriangulationJSONStore();
    // testDelaunayAndVoronoi();
    generateRectangularMesh();
    // generatePolygonMesh();

	return 0;
}