/**
 * @file delaunay.hpp
 * @brief Defines classes for Delaunay triangulation
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef DELAUNAY_TRINAGULATION_HPP
#define DELAUNAY_TRINAGULATION_HPP

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <iostream>


namespace delaunay {
    // Forward declarations
    class DelaunayPoint;
    class Edge;
    class Triangle;
    class DelaunayTriangulation;

    /**
     * @class DelaunayPoint
     * @brief Represents a 2D point with x and y coordinates
     */
    class DelaunayPoint {
        public:
            double x,y;

            /**
             * @brief Default constructor
             */
            DelaunayPoint(double x = 0, double y = 0) : x(x), y(y) {}

            /**
             * @brief Calculate the distance to another point
             * 
             * @param other The other point
             * @return double Distance between the points
             */
            double distanceTo(const DelaunayPoint& other) const {
                return std::sqrt(std::pow(x - other.x, 2) + std::pow(y - other.y, 2));
            }

            /**
             * @brief Equality operator
             * 
             * @param other The other point to compare with
             * @return bool True if points are equal, false otherwise
             */
            bool operator==(const DelaunayPoint& other) const {
                const double EPSILON = 1e-10;
                return std::abs(x - other.x) < EPSILON && std::abs(y - other.y) < EPSILON;
            }

            bool operator!=(const DelaunayPoint& other) const {
                return !(*this == other);
            }

            friend std::ostream& operator<<(std::ostream& os, const DelaunayPoint& point) {
                os << "(" << point.x << ", " << point.y << ")";
                return os;
            }
            
    };

    /**
     * @class Edge
     * @brief Represents an edge between two points
     */
    class Edge {
        public:
            DelaunayPoint p1, p2;

            /**
             * @brief Default constructor
             */
            Edge(const DelaunayPoint& p1, const DelaunayPoint& p2) : p1(p1), p2(p2) {}

            /**
             * @brief Equality operator
             * 
             * @param other The other edge to compare with
             * @return bool True if edges are equal, false otherwise
             */
            bool operator==(const Edge& other) const {
                return (p1 == other.p1 && p2 == other.p2) || (p1 == other.p2 && p2 == other.p1);
            }

            bool operator!=(const Edge& other) const {
                return !(*this == other);
            }

            friend std::ostream& operator<<(std::ostream& os, const Edge& edge) {
                os << edge.p1 << " -> " << edge.p2;
                return os;
            }
    };

    /**
     * @class Triangle
     * @brief Represents a triangle formed by three points
     */
    class Triangle {
        public: 
            DelaunayPoint p1, p2, p3;

            std::vector<Edge> edges;

            /**
             * @brief Default constructor
             */
            Triangle(const DelaunayPoint& p1, const DelaunayPoint& p2, const DelaunayPoint& p3)
                : p1(p1), p2(p2), p3(p3) {
                    edges.push_back(Edge(p1, p2));
                    edges.push_back(Edge(p2, p3));
                    edges.push_back(Edge(p3, p1));
                }

            /**
             * @brief Calculate the circumcenter of the triangle
             * 
             * @return DelaunayPoint The circumcenter point
             * @throws std::runtime_error if points are collinear
             */
            DelaunayPoint calculateCircumcenter() const; 

            /**
             * @brief Check if a point is inside the circumcircle of the triangle
             * 
             * @param point The point to check
             * @return bool True if the point is inside, false otherwise
             */
            bool isPointInCircumcircle(const DelaunayPoint& point) const;

            bool hasVertex(const DelaunayPoint& point) const {
                return p1 == point || p2 == point || p3 == point;
            }

            /**
             * @brief Equality operator
             * 
             * @param other The other triangle to compare with
             * @return bool True if triangles are equal, false otherwise
             */
            bool operator==(const Triangle& other) const {
                // This is a simplified check that may not work for all cases
                // A more robust check would ensure all vertices are matched
                return (p1 == other.p1 || p1 == other.p2 || p1 == other.p3) &&
                    (p2 == other.p1 || p2 == other.p2 || p2 == other.p3) &&
                    (p3 == other.p1 || p3 == other.p2 || p3 == other.p3);
            }

            bool operator!=(const Triangle& other) const {
                return !(*this == other);
            }

            friend std::ostream& operator<<(std::ostream& os, const Triangle& triangle) {
                os << "Triangle: " << triangle.p1 << ", " << triangle.p2 << ", " << triangle.p3;
                return os;
            }
    };

    /**
     * @class DelaunayTriangulation
     * @brief Implements the Delaunay triangulation algorithm
     * 
     * This class provides methods to create a Delaunay triangulation
     * from a set of points using the Bowyer-Watson algorithm.
     */
    class DelaunayTriangulation {
        private:
            std::vector<Triangle> triangles;
            Triangle superTriangle;

        public:
            /**
             * @brief Default constructor
             */
            DelaunayTriangulation() : superTriangle(DelaunayPoint(0,0), DelaunayPoint(1,0), DelaunayPoint(0,1)) {}

            /**
             * @brief Create a super triangle that contains all points
             * 
             * @param points Vector of points to be triangulated
             * @return Triangle The super triangle
             */
            Triangle createSuperTriangle(const std::vector<DelaunayPoint>& points);

            /**
             * @brief Triangulate a set of points
             * 
             * @param points Vector of points to triangulate
             * @return std::vector<Triangle> The resulting triangulation
             */
            std::vector<Triangle> triangulate(const std::vector<DelaunayPoint>& points);

            std::vector<Edge> getEdges() const;

            /**
             * @brief Get the triangles from the triangulation
             * 
             * @return const std::vector<Triangle>& Reference to the triangles
             */
            const std::vector<Triangle>& getTriangles() const {
                return triangles;
            }

            /**
             * @brief Get the super triangle
             * 
             * @return const Triangle& Reference to the super triangle
             */
            const Triangle& getSuperTriangle() const {
                return superTriangle;
            }
    };
}
#endif // DELAUNAY_TRINAGULATION_HPP