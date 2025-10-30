/**
 * @file voronoiMesh.hpp
 * @brief Defines classes for generating and manipulating Voronoi meshes
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef VORONOIMESH_HPP
#define VORONOIMESH_HPP

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <map>
#include <chrono>
#include <Eigen/Core>

#include "mesh/voronoi.hpp"
#include "mesh/delaunay.hpp"

using namespace delaunay;

namespace mesh{
    // Enum to represent geometry types
    enum class GeometryType {
        Unknown,
        Rectangle,
        Circle,
        Polygon
    };

    // Forward declaration
    class Geometry;
    class RectangleGeometry;
    class CircleGeometry;
    class PolygonGeometry;

    /**
     * @class Geometry
     * @brief Base class for geometric domains
     * 
     * This abstract class defines the interface for geometric domains
     * that can be used for mesh generation.
     */
    class Geometry {
        public:
            /**
             * @brief Virtual destructor
             */
            virtual ~Geometry() = default;

            /**
             * @brief Get the bounding box of the geometry
             * 
             * @param minX Output parameter for minimum x-coordinate
             * @param minY Output parameter for minimum y-coordinate
             * @param maxX Output parameter for maximum x-coordinate
             * @param maxY Output parameter for maximum y-coordinate
             */
            virtual void getBoundingBox(double& minX, double& minY, double& maxX, double& maxY) const;
            
            /**
             * @brief Check if a point is inside the geometry
             * 
             * @param point The point to check
             * @return bool True if the point is inside, false otherwise
             */
            virtual bool contains(const DelaunayPoint& point) const = 0;

            // Get the type of geometry
            virtual GeometryType getType() const {
                return GeometryType::Unknown;
            }
            
            // Get a string representation of the geometry type
            virtual std::string getTypeString() const {
                return "Unknown";
            }
    };

    /**
     * @class RectangularDomain
     * @brief Represents a rectangular geometric domain
     * 
     * This class defines a rectangular domain with specified dimensions.
     */
    class RectangleGeometry : public Geometry {
        private:
            double minX, minY, maxX, maxY;

        public:
            /**
             * @brief Constructor with domain boundaries
             * 
             * @param minX Minimum x-coordinate
             * @param minY Minimum y-coordinate
             * @param maxX Maximum x-coordinate
             * @param maxY Maximum y-coordinate
             */
            RectangleGeometry(const double& minX, const double& minY, const double& maxX, const double& maxY)
                : minX(minX), minY(minY), maxX(maxX), maxY(maxY) {};

            /**
             * @brief Get the bounding box of the rectangle
             * 
             * @param minX Output parameter for minimum x-coordinate
             * @param minY Output parameter for minimum y-coordinate
             * @param maxX Output parameter for maximum x-coordinate
             * @param maxY Output parameter for maximum y-coordinate
             */
            void getBoundingBox(double& outMinX, double& outMinY, double& outMaxX, double& outMaxY) const override {
                outMinX = minX;
                outMinY = minY;
                outMaxX = maxX;
                outMaxY = maxY;
            }
            
            /**
             * @brief Check if a point is inside the rectangle
             * 
             * @param point The point to check
             * @return bool True if the point is inside, false otherwise
             */
            bool contains(const DelaunayPoint& point) const override {
                return point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY;
            };
            
            GeometryType getType() const override {
                return GeometryType::Rectangle;
            }
            
            std::string getTypeString() const override {
                return "Rectangle";
            }
    };

    /**
     * @class CircularDomain
     * @brief Represents a circular geometric domain
     * 
     * This class defines a circular domain with a specified center and radius.
     */
    class CircleGeometry : public Geometry {
        private:
            DelaunayPoint center;
            double radius;

        public:
            /**
             * @brief Constructor with center and radius
             * 
             * @param centerX X-coordinate of the center
             * @param centerY Y-coordinate of the center
             * @param radius Radius of the circle
             */
            CircleGeometry(const DelaunayPoint& center, double& radius) : center(center), radius(radius) {};

            /**
             * @brief Get the bounding box of the circle
             * 
             * @param minX Output parameter for minimum x-coordinate
             * @param minY Output parameter for minimum y-coordinate
             * @param maxX Output parameter for maximum x-coordinate
             * @param maxY Output parameter for maximum y-coordinate
             */
            void getBoundingBox(double& outMinX, double& outMinY, double& outMaxX, double& outMaxY) const override{
                outMinX = center.x - radius;
                outMinY = center.y - radius;
                outMaxX = center.x + radius;
                outMaxY = center.y + radius;
            };
            
            GeometryType getType() const override {
                return GeometryType::Circle;
            }
            
            std::string getTypeString() const override {
                return "Circle";
            }
            
            // Getters for circle-specific properties
            DelaunayPoint getCenter() const { return center; }
            double getRadius() const { return radius; }
    };

    /**
     * @class PolygonalDomain
     * @brief Represents a polygonal geometric domain
     * 
     * This class defines a domain bounded by a polygon defined by a set of vertices.
     */
    class PolygonGeometry : public Geometry {
        private:
            std::vector<DelaunayPoint> vertices;

        public:
            /**
             * @brief Constructor with polygon vertices
             * 
             * @param vertices Vector of points defining the polygon vertices
             */
            PolygonGeometry(const std::vector<DelaunayPoint>& vertices) : vertices(vertices) {};

            /**
             * @brief Get the bounding box of the polygon
             * 
             * @param minX Output parameter for minimum x-coordinate
             * @param minY Output parameter for minimum y-coordinate
             * @param maxX Output parameter for maximum x-coordinate
             * @param maxY Output parameter for maximum y-coordinate
             */
            void getBoundingBox(double& outMinX, double& outMinY, double& outMaxX, double& outMaxY) const override{
                if (vertices.empty()){
                    outMinX = outMinY = outMaxX = outMaxY = 0.0;
                    return;
                }

                outMinX = outMaxX = vertices[0].x;
                outMinY = outMaxY = vertices[0].y;

                for (const auto& vertex : vertices){
                    outMinX = std::min(outMinX, vertex.x);
                    outMinY = std::min(outMinY, vertex.y);
                    outMaxX = std::max(outMaxX, vertex.x);
                    outMaxY = std::max(outMaxY, vertex.y);
                }
            };
            
            GeometryType getType() const override {
                return GeometryType::Polygon;
            }
            
            std::string getTypeString() const override {
                return "Polygon";
            }
            
            // Getter for polygon-specific properties
            const std::vector<DelaunayPoint>& getVertices() const { return vertices; }
    };

    /**
     * @brief Check if a point is inside a geometry
     * 
     * @param point The point to check
     * @param geometry The geometry to check against
     * @return bool True if the point is inside, false otherwise
     */
    bool isPointInGeometry(const DelaunayPoint& point, const Geometry& geometry);

    class VoronoiMeshGenerator {
        private:
            bool isPointInGeometry(const DelaunayPoint& point, const Geometry& geometry){
                return geometry.contains(point);
            };

            std::vector<DelaunayPoint> generatePoints(const Geometry& geometry, const int numPoints, const std::string& strategy = "random");

            std::vector<DelaunayPoint> relaxPoints(const std::vector<DelaunayPoint>& initialPoints, const Geometry& geometry, int& interations); 

            // Sutherland Hodgeman polygon clipping algorithm
            bool isInside(const DelaunayPoint& p, double linePos, int axis, bool positiveHalfPlane);
            VoronoiVertex* computeIntersection(const VoronoiVertex* v1, const VoronoiVertex* v2, double linePos, int axis);
            void clipAgainstLine(const std::vector<VoronoiVertex*>& inputVertices, std::vector<VoronoiVertex*>& outputVertices,
                                double linePos, int axis, bool positiveHalfPlane);
            void clipVoronoiCellsToGeometry(std::vector<VoronoiCell*>& cells, const Geometry& geometry);
            
            // Refined the boundary points
            void addBoundaryPoints(std::vector<DelaunayPoint>& points, double minX, double minY, double maxX, double maxY);
            void snapVerticesToBoundary(std::vector<VoronoiCell*>& cells, double minX, double minY, double maxX, double maxY);

            // Add special corner cells for unit square
            void addCornerCells(std::vector<VoronoiCell*>& cells, double minX, double minY, double maxX, double maxY);

            // Fix corners with simple square cells
            void fixCorners(std::vector<VoronoiCell*>& cells, double minX, double minY, double maxX, double maxY);

            // Polygon-specific clipping functions
            void clipAgainstPolygonEdge(const std::vector<VoronoiVertex*>& inputVertices, 
                                       std::vector<VoronoiVertex*>& outputVertices,
                                       const DelaunayPoint& p1, const DelaunayPoint& p2, double nx, double ny);
            bool isInsideHalfPlane(const DelaunayPoint& p, const DelaunayPoint& linePoint, double nx, double ny);
            VoronoiVertex* computeIntersectionWithLine(const VoronoiVertex* v1, const VoronoiVertex* v2, 
                                                         const DelaunayPoint& lineP1, const DelaunayPoint& lineP2);

            // Add boundary points for polygon geometries
            void addPolygonBoundaryPoints(std::vector<VoronoiCell*>& cells, const std::vector<DelaunayPoint>& polygonVertices);

            // Add explicit corner cells for rectangular geometries
            void addExplicitCornerCells(std::vector<VoronoiCell*>& cells, double minX, double minY, double maxX, double maxY);

            // Create a rectangular mesh directly (avoiding memory issues)
            std::vector<VoronoiCell*> createRectangularMesh(double minX, double minY, double maxX, double maxY, int numPoints);
            
            // Safe version of clipping that avoids memory issues
            void safeClipCellsToGeometry(std::vector<VoronoiCell*>& cells, const Geometry& geometry);

        public:
            VoronoiMeshGenerator() = default;

            std::vector<VoronoiCell*> generateMesh(const Geometry& geometry, const int numPoints, const std::string& pointDistribution = "random", int relaxtionIterations = 0);

            void convertToEigenMesh(const std::vector<VoronoiCell*>& cells, Eigen::MatrixXd& nodes, Eigen::MatrixXi& elements);
    };
}

#endif