/**
 * @file helpers.hpp
 * @brief Helper functions and utilities for mesh operations
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef POLIVEM_MESH_HELPERS_HPP
#define POLIVEM_MESH_HELPERS_HPP

#include <vector>
#include <random>
#include "mesh/delaunay.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>

namespace meshHelpers {

/**
 * @brief Convert a vector of points to an Eigen matrix
 * 
 * Converts a vector of 2D points to an Eigen matrix where each row
 * represents a point with x and y coordinates.
 * 
 * @param points Vector of points to convert
 * @return Eigen::MatrixXd Matrix containing the point coordinates
 */
Eigen::MatrixXd pointsToEigen(const std::vector<delaunay::DelaunayPoint>& points);

/**
 * @brief Convert a vector of triangles to an Eigen matrix
 * 
 * Converts a vector of triangles to an Eigen matrix where each row
 * represents a triangle with indices of its three vertices.
 * 
 * @param triangles Vector of triangles to convert
 * @param points Vector of points (used to find indices)
 * @return Eigen::MatrixXi Matrix containing the triangle vertex indices
 */
Eigen::MatrixXi trianglesToEigen(const std::vector<delaunay::Triangle>& triangles, 
                                const std::vector<delaunay::DelaunayPoint>& points);

/**
 * @brief Find the index of a point in a vector
 * 
 * @param point The point to find
 * @param points Vector of points to search in
 * @return int Index of the point, or -1 if not found
 */
int findPointIndex(const delaunay::DelaunayPoint& point, const std::vector<delaunay::DelaunayPoint>& points);

/**
 * @brief Check if a point is inside a polygon
 * 
 * Uses the ray casting algorithm to determine if a point is inside a polygon.
 * 
 * @param point The point to check
 * @param polygon Vector of points defining the polygon vertices
 * @return bool True if the point is inside the polygon, false otherwise
 */
bool isPointInPolygon(const delaunay::DelaunayPoint& point, const std::vector<delaunay::DelaunayPoint>& polygon);

/**
 * @brief Calculate the area of a polygon
 * 
 * Uses the shoelace formula to calculate the area of a polygon.
 * 
 * @param polygon Vector of points defining the polygon vertices
 * @return double Area of the polygon
 */
double calculatePolygonArea(const std::vector<delaunay::DelaunayPoint>& polygon);

/**
 * @brief Calculate the centroid of a polygon
 * 
 * @param polygon Vector of points defining the polygon vertices
 * @return delaunay::Point The centroid point
 */
delaunay::DelaunayPoint calculatePolygonCentroid(const std::vector<delaunay::DelaunayPoint>& polygon);

/**
 * @brief Check if a quadrilateral element has counter-clockwise orientation
 * 
 * Uses the cross product of the first three vertices to determine orientation.
 * For VEM, elements should be counter-clockwise oriented to ensure positive
 * determinant in the Jacobian and positive definite mass matrices.
 * 
 * @param nodes Matrix containing node coordinates (rows are nodes, columns are x,y)
 * @param element_vertices Array of 4 vertex indices defining the quadrilateral
 * @return bool True if counter-clockwise, false if clockwise
 */
bool isCounterClockwise(const Eigen::MatrixXd& nodes, const std::vector<int>& element_vertices);

/**
 * @brief Check if a quadrilateral element has counter-clockwise orientation
 * 
 * Overloaded version that takes vertex indices as separate parameters.
 * 
 * @param nodes Matrix containing node coordinates
 * @param v0 Index of first vertex
 * @param v1 Index of second vertex  
 * @param v2 Index of third vertex
 * @param v3 Index of fourth vertex
 * @return bool True if counter-clockwise, false if clockwise
 */
bool isCounterClockwise(const Eigen::MatrixXd& nodes, int v0, int v1, int v2, int v3);

/**
 * @brief Correct the orientation of a quadrilateral element to counter-clockwise
 * 
 * If the element is clockwise oriented, reverses the vertex order to make it
 * counter-clockwise. This ensures positive determinant and proper VEM behavior.
 * 
 * @param nodes Matrix containing node coordinates
 * @param element_vertices Vector of 4 vertex indices (modified in-place if needed)
 */
void correctElementOrientation(const Eigen::MatrixXd& nodes, std::vector<int>& element_vertices);

/**
 * @brief Correct the orientation of all elements in a mesh
 * 
 * Checks and corrects the orientation of all quadrilateral elements to ensure
 * they are counter-clockwise oriented for proper VEM behavior.
 * 
 * @param nodes Matrix containing node coordinates
 * @param elements Matrix containing element connectivity (modified in-place)
 */
void correctMeshOrientation(const Eigen::MatrixXd& nodes, Eigen::MatrixXi& elements);

    inline std::vector<delaunay::DelaunayPoint> generateRandomPoints(
        int numPoints,
        double minX, double maxX,
        double minY, double maxY
    ) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> disX(minX, maxX);
        std::uniform_real_distribution<> disY(minY, maxY);

        std::vector<delaunay::DelaunayPoint> points;
        points.reserve(numPoints);

        for (int i = 0; i < numPoints; ++i) {
            points.emplace_back(disX(gen), disY(gen));
        }

        return points;
    }
}

#endif