#include "mesh/delaunay.hpp"

namespace delaunay {
    DelaunayPoint Triangle::calculateCircumcenter() const {
        const double x1 = p1.x;
        const double y1 = p1.y;
        const double x2 = p2.x;
        const double y2 = p2.y;
        const double x3 = p3.x;
        const double y3 = p3.y;
        
        const double A = x2 - x1;
        const double B = y2 - y1;
        const double C = x3 - x1;
        const double D = y3 - y1;
        
        const double E = A * (x1 + x2) + B * (y1 + y2);
        const double F = C * (x1 + x3) + D * (y1 + y3);
        
        const double G = 2.0 * (A * (y3 - y2) - B * (x3 - x2));
        
        if (std::abs(G) < 1e-10) {
            throw std::runtime_error("The points are collinear, cannot form a triangle.");
        }
        
        const double cx = (D * E - B * F) / G;
        const double cy = (A * F - C * E) / G;
        
        return DelaunayPoint(cx, cy);
    }

    bool Triangle::isPointInCircumcircle(const DelaunayPoint& point) const {
        try {
            DelaunayPoint circumcenter = calculateCircumcenter();
            double radius = circumcenter.distanceTo(p1);
            double distance = circumcenter.distanceTo(point);
            
            return distance < radius;
        } catch (const std::runtime_error&) {
            return false;
        }
    }


    Triangle DelaunayTriangulation::createSuperTriangle(const std::vector<DelaunayPoint>& points) {
        double minX = std::numeric_limits<double>::max();
        double minY = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double maxY = std::numeric_limits<double>::lowest();

        for (const auto& point : points) {
            minX = std::min(minX, point.x);
            minY = std::min(minY, point.y);
            maxX = std::max(maxX, point.x);
            maxY = std::max(maxY, point.y);
        }

        const double dx = (maxX - minX) * 10;
        const double dy = (maxY - minY) * 10;

        DelaunayPoint p1(minX - dx, minY - dy);
        DelaunayPoint p2(maxX + dx, minY - dy);
        DelaunayPoint p3((minX + maxX) / 2, maxY + dy);

        superTriangle = Triangle(p1, p2, p3);
        return superTriangle;
    }

    std::vector<Triangle> DelaunayTriangulation::triangulate(const std::vector<DelaunayPoint>& points) {
        std::vector<DelaunayPoint> pointsCopy = points;

        // Step 1: Create a super triangle to contain all points
        Triangle superTriangle = createSuperTriangle(pointsCopy);
        triangles.clear();
        triangles.push_back(superTriangle);

        // Step 2: Add points one by one
        for (const auto& point : pointsCopy) {
            // Find all triangles that are no longer valid due to the insertions
            std::vector<Triangle> badTriangles;

            for(const auto& triangle : triangles){
                if (triangle.isPointInCircumcircle(point)) {
                    badTriangles.push_back(triangle);
                }
            }

            // Find the boundary of the polygonal hole
            std::vector<Edge> polygon;

            for (const auto& triangle : badTriangles) {
                for (const auto& edge : triangle.edges) {
                    // Check if the edge is not shared by any other bad triangle
                    bool isShared = false;

                    for (const auto& otherTriangle : badTriangles) {
                        if (triangle == otherTriangle) continue;

                        for (const auto& otherEdge : otherTriangle.edges) {
                            if (edge == otherEdge) {
                                isShared = true;
                                break;
                            }
                        }
                        if (isShared) break;    
                    }

                    // If the edge is not shared, add it to the polygon
                    if (!isShared){
                        polygon.push_back(edge);
                    }
                }
            }

            // Remove bad triangles from the triangulation
            triangles.erase(
                std::remove_if(triangles.begin(), triangles.end(), 
                    [&badTriangles](const Triangle& t) {
                        return std::find(badTriangles.begin(), badTriangles.end(), t) != badTriangles.end();
                    }
                ),
                triangles.end()
            );

           // Re-triangulate the polygonal hole
           for (const auto& edge : polygon){
                Triangle newTriangle(edge.p1, edge.p2, point);
                triangles.push_back(newTriangle);
           }
        }

        // Step 3: Remove any triangle that shares a vertex with the super triangle
        const DelaunayPoint& p1 = superTriangle.p1;
        const DelaunayPoint& p2 = superTriangle.p2;
        const DelaunayPoint& p3 = superTriangle.p3;

        triangles.erase(
            std::remove_if(triangles.begin(), triangles.end(), 
                [&p1, &p2, &p3](const Triangle& t) {
                    return t.hasVertex(p1) || t.hasVertex(p2) || t.hasVertex(p3);
                }
            ),
            triangles.end()
        );

        return triangles;
    }

    std::vector<Edge> DelaunayTriangulation::getEdges() const {
        std::vector<Edge> edges;

        for (const auto& triangle : triangles) {
            for (const auto& edge : triangle.edges) {
                // Check if the edge is not already in the list
                bool isUnique = true;

                for (const auto& otherEdge : edges) {
                    if (edge == otherEdge) {
                        isUnique = false;
                        break;
                    }
                }

                if (isUnique) {
                    edges.push_back(edge);  
                }
            }
        }

        return edges;
    }
}