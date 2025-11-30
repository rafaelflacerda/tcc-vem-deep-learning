#include "mesh/voronoi.hpp"
#include <cmath>
#include <vector>   // se usar std::vector aqui também

namespace delaunay {
    void VoronoiDiagram::sortVerticesCounterClockwise(VoronoiCell* cell) {
        // Find the centroid of the cell
        double centroidX = 0.0, centroidY = 0.0;
        for (const VoronoiVertex* vertex : cell->vertices) {
            centroidX += vertex->point.x;
            centroidY += vertex->point.y;
        }
        centroidX /= cell->vertices.size();
        centroidY /= cell->vertices.size();
        
        // Sort vertices counter-clockwise around the centroid
        std::sort(cell->vertices.begin(), cell->vertices.end(), 
            [&centroidX, &centroidY](const VoronoiVertex* v1, const VoronoiVertex* v2) {
                // Calculate the angle from the centroid to each vertex
                double angle1 = std::atan2(v1->point.y - centroidY, v1->point.x - centroidX);
                double angle2 = std::atan2(v2->point.y - centroidY, v2->point.x - centroidX);
                return angle1 < angle2;
            }
        );
    }

    void VoronoiDiagram::buildFromDelaunay(const std::vector<delaunay::DelaunayPoint>& points, const std::vector<Triangle>& triangles) {
        // Clear existing data
        for (auto vertex : vertices) delete vertex;
        for (auto edge : edges) delete edge;
        for (auto cell : cells) delete cell;

        vertices.clear();
        edges.clear();
        cells.clear();
        
        // Step 1: Create a map to store vertices by their coordinates
        std::map<std::pair<double, double>, VoronoiVertex*> vertexMap;
        
        // Step 2: For each triangle, compute its circumcenter (Voronoi vertex)
        for (const Triangle& triangle : triangles){
            try{
                delaunay::DelaunayPoint circumcenter = triangle.calculateCircumcenter();

                // Use map to avoid duplicate vertices
                std::pair<double, double> key(circumcenter.x, circumcenter.y);
                if (vertexMap.find(key) == vertexMap.end()){
                    // Create new vertex
                    VoronoiVertex* vertex = new VoronoiVertex(circumcenter);
                    vertices.push_back(vertex);
                    vertexMap[key] = vertex;
                }
            } catch (const std::exception& e){
                // Skip triangles with no circumcenter (degenerate case)
                continue;
            }
        }
        
        // Step 3: Create a map of all the Delaunay edges
        std::map<std::pair<std::pair<double,double>, std::pair<double, double>>, std::vector<VoronoiVertex*>> delaunayEdgeMap;

        // For each triangle, register its edges in the map
        for (const Triangle& triangle: triangles){
            try{
                delaunay::DelaunayPoint circumcenter = triangle.calculateCircumcenter();
                VoronoiVertex* vertex = vertexMap[{circumcenter.x, circumcenter.y}];

                // For each edge of the triangle, add this circumcenter to the list
                for (const Edge& edge : triangle.edges){
                    // Create a canonial key for the edhe (ordered by coordinates)
                    std::pair<double, double> p1(edge.p1.x, edge.p1.y);
                    std::pair<double, double> p2(edge.p2.x, edge.p2.y);

                    std::pair<std::pair<double, double>, std::pair<double, double>> key;
                    if (p1<p2){
                        key = {p1, p2};
                    } else {
                        key = {p2, p1};
                    }

                    delaunayEdgeMap[key].push_back(vertex);
                }
            } catch (const std::runtime_error&){
                // Skip colinear triangles
                continue;
            }
        }

        // Step 4: Create Voronoi edges from the Delaunay edge map
        for (const auto& entry : delaunayEdgeMap) {
            // The Voronoi edge is a perpendicular bisector of the Delaunay edge
            if (entry.second.size() == 2){
                // Voronoi vertices
                VoronoiVertex* v1 = entry.second[0];
                VoronoiVertex* v2 = entry.second[1];

                // Create the two generator points corresponding to the Delaunay edges
                delaunay::DelaunayPoint generator1(entry.first.first.first, entry.first.first.second);
                delaunay::DelaunayPoint generator2(entry.first.second.first, entry.first.second.second);

                VoronoiEdge* edge = new VoronoiEdge(v1, v2, generator1, generator2);
                edges.push_back(edge);
            }
            // If size == 1, this is a boundary edge
            // We could extend the Voronoi diagram to infinity here, but for simplicity, we'll skip it
        }

        // Step 5: Create Voronoi cells for each input point
        for (const delaunay::DelaunayPoint& point : points) {
            VoronoiCell* cell = new VoronoiCell(point);
            // Explicitly set the generator point to ensure it's properly initialized
            cell->generator = point;
            
            // Debug output
            std::cout << "Creating cell for generator point (" << point.x << ", " << point.y << ")" << std::endl;
            
            for (VoronoiEdge* edge : edges) {
                // Use distance comparison instead of exact equality
                if (edge->generator1.distanceTo(point) < 1e-10 || edge->generator2.distanceTo(point) < 1e-10) {
                    cell->edges.push_back(edge);
                    
                    if (std::find(cell->vertices.begin(), cell->vertices.end(), edge->v1) == cell->vertices.end()) {
                        cell->vertices.push_back(edge->v1);
                    }
                    if (std::find(cell->vertices.begin(), cell->vertices.end(), edge->v2) == cell->vertices.end()) {
                        cell->vertices.push_back(edge->v2);
                    }
                }
            }
            
            // Debug output
            std::cout << "Cell has " << cell->vertices.size() << " vertices" << std::endl;
            
            sortVerticesCounterClockwise(cell);
            cells.push_back(cell);
        }
    }
}
