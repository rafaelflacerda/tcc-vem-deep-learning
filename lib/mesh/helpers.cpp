#include "mesh/helpers.hpp"

namespace meshHelpers {

    Eigen::MatrixXd pointsToEigen(const std::vector<delaunay::DelaunayPoint>& points) {
        Eigen::MatrixXd matrix(points.size(), 2);
        
        for (size_t i = 0; i < points.size(); ++i) {
            matrix(i, 0) = points[i].x;
            matrix(i, 1) = points[i].y;
        }
        
        return matrix;
    }

    Eigen::MatrixXi trianglesToEigen(const std::vector<delaunay::Triangle>& triangles, 
                                    const std::vector<delaunay::DelaunayPoint>& points) {
        Eigen::MatrixXi matrix(triangles.size(), 3);
        
        for (size_t i = 0; i < triangles.size(); ++i) {
            const delaunay::Triangle& t = triangles[i];
            
            // Find the index of each vertex in the points array
            int idx1 = findPointIndex(t.p1, points);
            int idx2 = findPointIndex(t.p2, points);
            int idx3 = findPointIndex(t.p3, points);
            
            matrix(i, 0) = idx1;
            matrix(i, 1) = idx2;
            matrix(i, 2) = idx3;
        }
        
        return matrix;
    }

    int findPointIndex(const delaunay::DelaunayPoint& point, const std::vector<delaunay::DelaunayPoint>& points) {
        const double EPSILON = 1e-10;
        
        for (size_t i = 0; i < points.size(); ++i) {
            if (std::abs(points[i].x - point.x) < EPSILON && 
                std::abs(points[i].y - point.y) < EPSILON) {
                return static_cast<int>(i);
            }
        }
        
        return -1; // Not found
    }

    bool isPointInPolygon(const delaunay::DelaunayPoint& point, const std::vector<delaunay::DelaunayPoint>& polygon) {
        if (polygon.size() < 3) {
            return false;
        }
        
        bool inside = false;
        size_t j = polygon.size() - 1;
        
        for (size_t i = 0; i < polygon.size(); i++) {
            const delaunay::DelaunayPoint& pi = polygon[i];
            const delaunay::DelaunayPoint& pj = polygon[j];
            
            if (((pi.y > point.y) != (pj.y > point.y)) &&
                (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x)) {
                inside = !inside;
            }
            j = i;
        }
        
        return inside;
    }

    double calculatePolygonArea(const std::vector<delaunay::DelaunayPoint>& polygon) {
        if (polygon.size() < 3) {
            return 0.0;
        }
        
        double area = 0.0;
        size_t n = polygon.size();
        
        for (size_t i = 0; i < n; ++i) {
            size_t j = (i + 1) % n;
            area += polygon[i].x * polygon[j].y;
            area -= polygon[j].x * polygon[i].y;
        }
        
        return std::abs(area) / 2.0;
    }

    delaunay::DelaunayPoint calculatePolygonCentroid(const std::vector<delaunay::DelaunayPoint>& polygon) {
        if (polygon.empty()) {
            return delaunay::DelaunayPoint(0.0, 0.0);
        }
        
        if (polygon.size() == 1) {
            return polygon[0];
        }
        
        if (polygon.size() == 2) {
            return delaunay::DelaunayPoint(
                (polygon[0].x + polygon[1].x) / 2.0,
                (polygon[0].y + polygon[1].y) / 2.0
            );
        }
        
        double area = calculatePolygonArea(polygon);
        if (area == 0.0) {
            // Degenerate polygon, return average of vertices
            double sumX = 0.0, sumY = 0.0;
            for (const auto& point : polygon) {
                sumX += point.x;
                sumY += point.y;
            }
            return delaunay::DelaunayPoint(sumX / polygon.size(), sumY / polygon.size());
        }
        
        double cx = 0.0, cy = 0.0;
        size_t n = polygon.size();
        
        for (size_t i = 0; i < n; ++i) {
            size_t j = (i + 1) % n;
            double cross = polygon[i].x * polygon[j].y - polygon[j].x * polygon[i].y;
            cx += (polygon[i].x + polygon[j].x) * cross;
            cy += (polygon[i].y + polygon[j].y) * cross;
        }
        
        cx /= (6.0 * area);
        cy /= (6.0 * area);
        
        return delaunay::DelaunayPoint(cx, cy);
    }

    bool isCounterClockwise(const Eigen::MatrixXd& nodes, const std::vector<int>& element_vertices) {
        if (element_vertices.size() < 3) {
            return false; // Cannot determine orientation with less than 3 vertices
        }
        
        // Use first three vertices to determine orientation
        int v0 = element_vertices[0];
        int v1 = element_vertices[1]; 
        int v2 = element_vertices[2];
        
        return isCounterClockwise(nodes, v0, v1, v2, -1);
    }

    bool isCounterClockwise(const Eigen::MatrixXd& nodes, int v0, int v1, int v2, int v3) {
        // Get coordinates of first three vertices
        double x0 = nodes(v0, 0), y0 = nodes(v0, 1);
        double x1 = nodes(v1, 0), y1 = nodes(v1, 1);
        double x2 = nodes(v2, 0), y2 = nodes(v2, 1);
        
        // Calculate cross product: (v1 - v0) × (v2 - v0)
        // If positive, vertices are counter-clockwise
        // If negative, vertices are clockwise
        double cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
        
        return cross > 0.0;
    }

    void correctElementOrientation(const Eigen::MatrixXd& nodes, std::vector<int>& element_vertices) {
        // Check orientation using signed area (shoelace formula)
        double signed_area = 0.0;
        int n = element_vertices.size();
        
        for (int i = 0; i < n; ++i) {
            int j = (i + 1) % n;
            int v_i = element_vertices[i];
            int v_j = element_vertices[j];
            
            signed_area += (nodes(v_i, 0) * nodes(v_j, 1) - nodes(v_j, 0) * nodes(v_i, 1));
        }
        
        // If signed area is negative, polygon is clockwise - reverse it
        if (signed_area < 0) {
            // Reverse the vertex order (keep first vertex, reverse the rest)
            // [v0, v1, v2, v3, ...] -> [v0, v_{n-1}, v_{n-2}, ..., v_2, v_1]
            std::reverse(element_vertices.begin() + 1, element_vertices.end());
        }
    }

    void correctMeshOrientation(const Eigen::MatrixXd& nodes, Eigen::MatrixXi& elements) {
        int corrected_count = 0;
        
        for (int elem_id = 0; elem_id < elements.rows(); ++elem_id) {
            // Count actual vertices for this element (skip -1 padding if present)
            int n_vertices = 0;
            for (int i = 0; i < elements.cols(); ++i) {
                if (elements(elem_id, i) >= 0) {
                    n_vertices++;
                } else {
                    break;
                }
            }
            
            // Extract vertex indices for this element
            std::vector<int> vertices(n_vertices);
            for (int i = 0; i < n_vertices; ++i) {
                vertices[i] = elements(elem_id, i);
            }
            
            // Compute signed area before correction
            double signed_area_before = 0.0;
            for (int i = 0; i < n_vertices; ++i) {
                int j = (i + 1) % n_vertices;
                int v_i = vertices[i];
                int v_j = vertices[j];
                signed_area_before += (nodes(v_i, 0) * nodes(v_j, 1) - nodes(v_j, 0) * nodes(v_i, 1));
            }
            
            // Correct orientation if needed
            correctElementOrientation(nodes, vertices);
            
            // Update the elements matrix
            for (int i = 0; i < n_vertices; ++i) {
                elements(elem_id, i) = vertices[i];
            }
            
            // Count if we corrected this element
            if (signed_area_before < 0) {
                corrected_count++;
            }
        }
        
        if (corrected_count > 0) {
            std::cout << "Corrected orientation of " << corrected_count << " elements (from clockwise to counter-clockwise)" << std::endl;
        }
    }

}

