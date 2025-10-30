/**
 * @file datasource.hpp
 * @brief Defines the datasource class for mesh data I/O operations
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef POLIVEM_DATASOURCE_HPP
#define POLIVEM_DATASOURCE_HPP

#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "mesh/delaunay.hpp"
#include "mesh/voronoi.hpp"
#include "mesh/voronoiMesh.hpp"
#include "mesh/helpers.hpp"
#include "utils/operations.hpp"
#include "utils/logging.hpp"
#include "models/enums.hpp"

using namespace delaunay;
using namespace meshHelpers;

namespace mesh {
    /**
     * @class datasource
     * @brief Class for handling mesh data input/output operations
     * 
     * This class provides functionality to read mesh data from files,
     * write results to files, and perform various mesh-related operations.
     */
    class datasource{
        public:
            /**
             * @brief Node coordinates matrix
             * 
             * Matrix containing the coordinates of all nodes in the mesh.
             */
            Eigen::MatrixXd nodes;

            /**
             * @brief Element connectivity matrix
             * 
             * Matrix defining the connectivity between nodes to form elements.
             */
            Eigen::MatrixXi elements;

            /**
             * @brief Dirichlet boundary conditions
             * 
             * Matrix specifying the Dirichlet boundary conditions (fixed displacements).
             */
            Eigen::MatrixXi supp;

            /**
             * @brief Neumann boundary conditions
             * 
             * Matrix specifying the Neumann boundary conditions (applied loads).
             */
            Eigen::MatrixXi load;

            /**
             * @brief Distributed load values in x and y directions
             */
            double qx, qy;

            /**
             * @brief Read mesh data from a JSON file
             * 
             * @param filepath Path to the JSON file
             */
            void readJson(std::string filepath);

            /**
             * @brief Read beam structure data from a JSON file
             * 
             * @param filepath Path to the JSON file
             */
            void readJsonBeam(std::string filepath);

            /**
             * @brief Write displacement results to an output file
             * 
             * @param u Displacement vector
             * @param filename Output filename
             */
            void writeOutput(Eigen::VectorXd u, std::string filename);

            /**
             * @brief Save beam/portic displacement data to a JSON file
             * 
             * @param u Displacement vector
             * @param E Elastic modulus
             * @param A Cross-sectional area
             * @param I Moment of inertia
             * @param filename Output filename
             */
            static void saveDisplacementsToJson(Eigen::VectorXd u, double E, double A, double I, std::string filename);

            /**
             * @brief Save beam or portic geometry to a JSON file
             * 
             * @param nodes Node coordinates
             * @param elements Element connectivity
             * @param supp Support conditions
             * @param distributed_load_elements Elements with distributed loads
             * @param loads Load values
             * @param filename Output filename
             */
            static void saveBeamGeometryToJson(Eigen::MatrixXd nodes, Eigen::MatrixXi elements, 
                                              Eigen::MatrixXi supp, Eigen::MatrixXi distributed_load_elements, 
                                              Eigen::VectorXd loads, std::string filename);
            
            /**
             * @brief Generate random samples for beam model parameters
             * 
             * @param numSamples Number of samples to generate
             * @param E_samples Output vector for elastic modulus samples
             * @param A_samples Output vector for cross-sectional area samples
             * @param I_samples Output vector for moment of inertia samples
             */
            static void generateRandomSamples(int numSamples, Eigen::VectorXd& E_samples, 
                                             Eigen::VectorXd& A_samples, Eigen::VectorXd& I_samples);

            /**
             * @brief Calculate the aspect ratio for each element in the mesh
             * 
             * @return Eigen::VectorXd Vector of aspect ratios
             */
            Eigen::VectorXd calculateAspectRatio();

            /**
             * @brief Sort node coordinates according to element indexation
             * 
             * @param u Displacement vector
             * @return Eigen::MatrixXd Sorted node coordinates
             */
            Eigen::MatrixXd sortNodes(Eigen::VectorXd u);

            /**
             * @brief Export triangulation to a JSON file
             * 
             * @param points Vector of points
             * @param triangles Vector of triangles
             * @param edges Vector of edges
             * @param filename Output filename
             */
            static void exportTriangulationToJson(const std::vector<delaunay::DelaunayPoint>& points, 
                                                 const std::vector<Triangle>& triangles, 
                                                 const std::vector<Edge>& edges,
                                                 const std::string& filename);

            /**
             * @brief Export Voronoi diagram to a JSON file
             * 
             * @param vertices Vector of Voronoi vertices
             * @param edges Vector of Voronoi edges
             * @param cells Vector of Voronoi cells
             * @param filename Output filename
             */
            static void exportVoronoiToJson(const std::vector<VoronoiVertex*>& vertices,
                                           const std::vector<VoronoiEdge*>& edges,
                                           const std::vector<VoronoiCell*>& cells,
                                           const std::string& filename);

            /**
             * @brief Export Voronoi mesh to a JSON file
             * 
             * @param cells Vector of Voronoi cells
             * @param geometry Geometry object defining the domain
             * @param filename Output filename
             */
            static void exportVoronoiMeshToJson(
                const std::vector<VoronoiCell*>& cells,
                const Geometry& geometry,
                const std::string& filename);

            /**
             * @brief Export Eigen mesh to a JSON file
             * 
             * @param nodes Node coordinates matrix
             * @param elements Element connectivity matrix
             * @param filename Output filename
             */
            static void exportEigenMeshToJson(
                const Eigen::MatrixXd& nodes,
                const Eigen::MatrixXi& elements,
                const std::string& filename);

            /**
             * @brief Constructor
             * 
             * @param filepath Path to the input file
             * @param type Type of beam solver (Beam or Portic)
             */
            datasource(std::string filepath, BeamSolverType type){
                if(type == BeamSolverType::Beam){
                    readJson(filepath);
                } else if(type == BeamSolverType::Portic){
                    readJsonBeam(filepath);
                }
            }
    };
}

#endif