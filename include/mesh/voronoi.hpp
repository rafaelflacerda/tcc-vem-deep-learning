#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

/**
 * @file voronoi.hpp
 * @brief Defines classes for Voronoi diagram generation
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#ifndef POLIVEM_VORONOI_HPP
#define POLIVEM_VORONOI_HPP

#include <set>
#include <map>
#include <vector>
#include <limits>
#include <algorithm>

#include "mesh/delaunay.hpp"

namespace delaunay {
    // Forward class declarations
    class VoronoiCell;
    class VoronoiEdge;
    class VoronoiVertex;

    /**
     * @class VoronoiVertex
     * @brief Represents a vertex in a Voronoi diagram
     * 
     * A Voronoi vertex is a point that is equidistant from three or more
     * generator points. It corresponds to the circumcenter of a Delaunay triangle.
     */
    class VoronoiVertex {
        public:
            /**
             * @brief The coordinates of the vertex
             */
            DelaunayPoint point;

            /**
             * @brief Default constructor
             */
            VoronoiVertex() = default;

            /**
             * @brief Constructor with a point
             * 
             * @param point The coordinates of the vertex
             */
            VoronoiVertex(const DelaunayPoint& point) : point(point) {}

            bool operator==(const VoronoiVertex& other) const {
                const double EPSILON = 1e-10;
                return (std::abs(point.x - other.point.x) < EPSILON &&
                        std::abs(point.y - other.point.y) < EPSILON);
            }

            bool operator<(const VoronoiVertex& other) const {
                const double EPSILON = 1e-10;
                if (std::abs(point.x - other.point.x) < EPSILON) {
                    return point.y < other.point.y;
                }
                return point.x < other.point.x;
            }
    };

    /**
     * @class VoronoiEdge
     * @brief Represents an edge in a Voronoi diagram
     * 
     * A Voronoi edge is a line segment that is equidistant from two generator points.
     * It corresponds to the perpendicular bisector of an edge in the Delaunay triangulation.
     */
    class VoronoiEdge {
        public:
            /**
             * @brief The two endpoints of the edge
             */
            VoronoiVertex *v1, *v2;
            
            /**
             * @brief The two generator points that define this edge
             * 
             * These are the two points from the original set that are closest to this edge.
             */
            DelaunayPoint generator1, generator2;
            
            /**
             * @brief Default constructor
             */
            VoronoiEdge() = default;
            
            /**
             * @brief Constructor with vertices and generators
             * 
             * @param v1 First vertex
             * @param v2 Second vertex
             * @param generator1 First generator point
             * @param generator2 Second generator point
             */
            VoronoiEdge(VoronoiVertex* v1, VoronoiVertex* v2, 
                        const DelaunayPoint& generator1, const DelaunayPoint& generator2)
                : v1(v1), v2(v2), generator1(generator1), generator2(generator2) {}

            bool operator==(const VoronoiEdge& other) const {
                return ((*v1 == *other.v1 && *v2 == *other.v2) ||
                    (*v1 == *other.v2 && *v2 == *other.v1)) &&
                    ((generator1 == other.generator1 && generator2 == other.generator2) ||
                    (generator1 == other.generator2 && generator2 == other.generator1));
            }
    };

    /**
     * @class VoronoiCell
     * @brief Represents a cell in a Voronoi diagram
     * 
     * A Voronoi cell is the region of points that are closer to a specific
     * generator point than to any other generator point.
     */
    class VoronoiCell {
        public:
            /**
             * @brief The generator point for this cell
             */
            DelaunayPoint generator;
            
            /**
             * @brief The vertices that form the boundary of the cell
             */
            std::vector<VoronoiVertex*> vertices;
            
            /**
             * @brief The edges that form the boundary of the cell
             */
            std::vector<VoronoiEdge*> edges;
            
            /**
             * @brief Default constructor
             */
            VoronoiCell() = default;
            
            /**
             * @brief Constructor with a generator point
             * 
             * @param generator The generator point for this cell
             */
            VoronoiCell(const DelaunayPoint& generator) : generator(generator) {}

            ~VoronoiCell(){
                // Note: Don't delete vertices or edges here, they're managed by VoronoiDiagram
            }
    };
    
    /**
     * @class VoronoiDiagram
     * @brief Represents a complete Voronoi diagram
     * 
     * This class provides methods to build a Voronoi diagram from a
     * Delaunay triangulation and access its components.
     */
    class VoronoiDiagram {
        private:
            /**
             * @brief Sort the vertices of a cell in counter-clockwise order
             * 
             * @param cell The cell whose vertices should be sorted
             */
            void sortVerticesCounterClockwise(VoronoiCell* cell);

        public:
            /**
             * @brief The vertices in the diagram
             */
            std::vector<VoronoiVertex*> vertices;
            
            /**
             * @brief The edges in the diagram
             */
            std::vector<VoronoiEdge*> edges;
            
            /**
             * @brief The cells in the diagram
             */
            std::vector<VoronoiCell*> cells;
            
            /**
             * @brief Default constructor
             */
            VoronoiDiagram() = default;

            /**
             * @brief Destructor
             * 
             * Cleans up all dynamically allocated vertices, edges, and cells.
             */
            ~VoronoiDiagram() {
                for (auto vertex : vertices) delete vertex;
                for (auto edge : edges) delete edge;
                for (auto cell : cells) delete cell;
            }
            
            /**
             * @brief Build a Voronoi diagram from a Delaunay triangulation
             * 
             * @param points The original set of generator points
             * @param triangles The Delaunay triangulation of the points
             */
            void buildFromDelaunay(const std::vector<DelaunayPoint>& points, const std::vector<Triangle>& triangles);

            /**
             * @brief Get the vertices of the diagram
             * 
             * @return const std::vector<VoronoiVertex*>& Reference to the vertices
             */
            const std::vector<VoronoiVertex*>& getVertices() const { return vertices; }
            
            /**
             * @brief Get the edges of the diagram
             * 
             * @return const std::vector<VoronoiEdge*>& Reference to the edges
             */
            const std::vector<VoronoiEdge*>& getEdges() const { return edges; }
            
            /**
             * @brief Get the cells of the diagram
             * 
             * @return const std::vector<VoronoiCell*>& Reference to the cells
             */
            const std::vector<VoronoiCell*>& getCells() const { return cells; }
    };
}

#endif