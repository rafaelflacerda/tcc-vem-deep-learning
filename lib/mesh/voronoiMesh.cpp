#include "mesh/voronoiMesh.hpp"

namespace mesh {
    void Geometry::getBoundingBox(double& minX, double& minY, double& maxX, double& maxY) const {
        // Default implementation (should be overridden by derived classes)
        minX = minY = maxX = maxY = 0.0;
    }

    std::vector<DelaunayPoint> VoronoiMeshGenerator::generatePoints(const Geometry& geometry, 
    const int numPoints, 
    const std::string& strategy){
        std::vector<delaunay::DelaunayPoint> points;
    
        // Get the bounding box of the geometry
        double minX, minY, maxX, maxY;
        geometry.getBoundingBox(minX, minY, maxX, maxY);
        
        // Debug output
        std::cout << "Geometry bounding box: (" << minX << ", " << minY << ") to (" 
                  << maxX << ", " << maxY << ")" << std::endl;
        
        // Set up random number generator
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 generator(seed);
        std::uniform_real_distribution<double> distX(minX, maxX);
        std::uniform_real_distribution<double> distY(minY, maxY);
        
        if (strategy == "random") {
            // Generate random points within the geometry
            while (points.size() < numPoints) {
                delaunay::DelaunayPoint candidate(distX(generator), distY(generator));
                if (isPointInGeometry(candidate, geometry)) {
                    // Debug output for first few points
                    if (points.size() < 5) {
                        std::cout << "Generated point: (" << candidate.x << ", " << candidate.y << ")" << std::endl;
                    }
                    points.push_back(candidate);
                }
            }
        } else if (strategy == "grid") {
            // Generate a grid of points with random perturbation
            int gridSize = static_cast<int>(std::sqrt(numPoints));
            double stepX = (maxX - minX) / gridSize;
            double stepY = (maxY - minY) / gridSize;
            
            std::normal_distribution<double> perturbation(0.0, 0.2 * std::min(stepX, stepY));
            
            for (int i = 0; i <= gridSize; i++) {
                for (int j = 0; j <= gridSize; j++) {
                    double x = minX + i * stepX + perturbation(generator);
                    double y = minY + j * stepY + perturbation(generator);
                    delaunay::DelaunayPoint candidate(x, y);
                    
                    if (isPointInGeometry(candidate, geometry)) {
                        points.push_back(candidate);
                        if (points.size() >= numPoints) {
                            return points;
                        }
                    }
                }
            }
        } else if (strategy == "poisson_disk") {
            // Poisson disk sampling implementation
            // This is a more complex algorithm that generates evenly spaced points
            
            // Simplified implementation for demonstration
            // In a real implementation, you'd use a proper Poisson disk sampling algorithm
            double r = std::sqrt((maxX - minX) * (maxY - minY) / (numPoints * 0.5));
            
            // First point is random
            {
                delaunay::DelaunayPoint candidate(distX(generator), distY(generator));
                if (isPointInGeometry(candidate, geometry)) {
                    points.push_back(candidate);
                }
            }
            
            // Generate more points using rejection sampling
            int attempts = 0;
            const int maxAttempts = numPoints * 10;
            
            while (points.size() < numPoints && attempts < maxAttempts) {
                delaunay::DelaunayPoint candidate(distX(generator), distY(generator));
                
                // Check if the point is far enough from all existing points
                bool valid = true;
                for (const auto& p : points) {
                    if (p.distanceTo(candidate) < r) {
                        valid = false;
                        break;
                    }
                }
                
                if (valid && isPointInGeometry(candidate, geometry)) {
                    points.push_back(candidate);
                }
                
                attempts++;
            }
            
            // If we couldn't generate enough points, fall back to random sampling
            if (points.size() < numPoints) {
                while (points.size() < numPoints) {
                    delaunay::DelaunayPoint candidate(distX(generator), distY(generator));
                    if (isPointInGeometry(candidate, geometry)) {
                        points.push_back(candidate);
                    }
                }
            }
        }
        
        // Number of points to add on each boundary edge
        int boundaryPoints = std::sqrt(numPoints) / 2;
        
        // Add points along the boundary
        double stepX = (maxX - minX) / boundaryPoints;
        double stepY = (maxY - minY) / boundaryPoints;
        
        for (int i = 0; i <= boundaryPoints; i++) {
            // Bottom and top edges
            points.push_back(DelaunayPoint(minX + i * stepX, minY));
            points.push_back(DelaunayPoint(minX + i * stepX, maxY));
            
            // Left and right edges
            points.push_back(DelaunayPoint(minX, minY + i * stepY));
            points.push_back(DelaunayPoint(maxX, minY + i * stepY));
        }
        
        return points;
    }

    bool VoronoiMeshGenerator::isInside(const DelaunayPoint& p, double linePos, int axis, bool positiveHalfPlane) {
        double coord = (axis == 0) ? p.x : p.y;
        return positiveHalfPlane ? (coord >= linePos) : (coord <= linePos);
    }

    VoronoiVertex* VoronoiMeshGenerator::computeIntersection(
        const VoronoiVertex* v1, 
        const VoronoiVertex* v2, 
        double linePos, 
        int axis) {
        
        // axis: 0 for vertical boundary (x=const), 1 for horizontal boundary (y=const)
        double p1Coord = (axis == 0) ? v1->point.x : v1->point.y;
        double p2Coord = (axis == 0) ? v2->point.x : v2->point.y;
        
        // If line is parallel to boundary, no intersection
        if (std::abs(p1Coord - p2Coord) < 1e-10) {
            return nullptr;
        }
        
        // Compute the intersection point
        double t = (linePos - p1Coord) / (p2Coord - p1Coord);
        double xIntersect = v1->point.x + t * (v2->point.x - v1->point.x);
        double yIntersect = v1->point.y + t * (v2->point.y - v1->point.y);
        
        return new VoronoiVertex(DelaunayPoint(xIntersect, yIntersect));
    }

    void VoronoiMeshGenerator::clipAgainstLine(
        const std::vector<VoronoiVertex*>& inputVertices, 
        std::vector<VoronoiVertex*>& outputVertices,
        double linePos,
        int axis,  // 0 for x (vertical line), 1 for y (horizontal line)
        bool positiveHalfPlane) {
        
        outputVertices.clear();
        
        if (inputVertices.empty()) {
            return;
        }
        
        // Last vertex in the input list
        const VoronoiVertex* s = inputVertices.back();
        
        // Process each edge of the polygon
        for (const VoronoiVertex* e : inputVertices) {
            bool sInside = isInside(s->point, linePos, axis, positiveHalfPlane);
            bool eInside = isInside(e->point, linePos, axis, positiveHalfPlane);
            
            // Case 1: Both endpoints inside - add only the second endpoint
            if (sInside && eInside) {
                outputVertices.push_back(new VoronoiVertex(e->point));
            }
            // Case 2: First outside, second inside - add intersection and second endpoint
            else if (!sInside && eInside) {
                VoronoiVertex* intersection = computeIntersection(s, e, linePos, axis);
                if (intersection) {
                    outputVertices.push_back(intersection);
                }
                outputVertices.push_back(new VoronoiVertex(e->point));
            }
            // Case 3: First inside, second outside - add only intersection
            else if (sInside && !eInside) {
                VoronoiVertex* intersection = computeIntersection(s, e, linePos, axis);
                if (intersection) {
                    outputVertices.push_back(intersection);
                }
            }
            // Case 4: Both outside - add nothing
            
            // Current endpoint becomes the starting point for the next edge
            s = e;
        }
    }

    void VoronoiMeshGenerator::clipVoronoiCellsToGeometry(std::vector<VoronoiCell*>& cells, const Geometry& geometry) {
        // Get geometry bounds
        double minX, minY, maxX, maxY;
        geometry.getBoundingBox(minX, minY, maxX, maxY);
        
        // For rectangle and circle, use the existing clipping logic
        for (auto* cell : cells) {
            if (!cell || cell->vertices.empty()) {
                continue;
            }
            
            // Original vertices
            std::vector<VoronoiVertex*> originalVertices;
            for (auto* v : cell->vertices) {
                if (v) originalVertices.push_back(v);
            }
            
            // Skip if no valid vertices
            if (originalVertices.empty()) {
                continue;
            }
            
            // Clear the cell's vertices
            for (auto* v : cell->vertices) {
                delete v;
            }
            cell->vertices.clear();
            
            // Clip against left boundary (x = minX)
            std::vector<VoronoiVertex*> clippedVertices;
            clipAgainstLine(originalVertices, clippedVertices, minX, 0, true);
            
            // Clean up original vertices
            for (auto* v : originalVertices) {
                delete v;
            }
            originalVertices.clear();
            
            // Clip against right boundary (x = maxX)
            std::vector<VoronoiVertex*> temp;
            clipAgainstLine(clippedVertices, temp, maxX, 0, false);
            
            // Clean up clipped vertices
            for (auto* v : clippedVertices) {
                delete v;
            }
            clippedVertices.clear();
            
            // Clip against bottom boundary (y = minY)
            clipAgainstLine(temp, clippedVertices, minY, 1, true);
            
            // Clean up temp vertices
            for (auto* v : temp) {
                delete v;
            }
            temp.clear();
            
            // Clip against top boundary (y = maxY)
            clipAgainstLine(clippedVertices, temp, maxY, 1, false);
            
            // Clean up clipped vertices again
            for (auto* v : clippedVertices) {
                delete v;
            }
            
            // Update cell vertices with clipped result
            cell->vertices = temp;
        }
    }

    void VoronoiMeshGenerator::addPolygonBoundaryPoints(std::vector<VoronoiCell*>& cells, const std::vector<DelaunayPoint>& polygonVertices) {
        if (polygonVertices.size() < 3) {
            return;
        }
        
        // Add cells for each polygon vertex
        const double CORNER_SIZE = 0.1;
        
        for (size_t i = 0; i < polygonVertices.size(); i++) {
            const DelaunayPoint& vertex = polygonVertices[i];
            
            // Check if this vertex is already covered by an existing cell
            bool vertexCovered = false;
            for (const auto* cell : cells) {
                for (const auto* cellVertex : cell->vertices) {
                    if (cellVertex && std::abs(cellVertex->point.x - vertex.x) < 1e-6 && 
                        std::abs(cellVertex->point.y - vertex.y) < 1e-6) {
                        vertexCovered = true;
                        break;
                    }
                }
                if (vertexCovered) break;
            }
            
            if (!vertexCovered) {
                // Create a small cell around this vertex
                VoronoiCell* cornerCell = new VoronoiCell(vertex);
                
                // Get adjacent vertices
                size_t prev = (i + polygonVertices.size() - 1) % polygonVertices.size();
                size_t next = (i + 1) % polygonVertices.size();
                
                // Compute unit vectors along the edges
                double dx1 = polygonVertices[prev].x - vertex.x;
                double dy1 = polygonVertices[prev].y - vertex.y;
                double len1 = std::sqrt(dx1*dx1 + dy1*dy1);
                dx1 /= len1;
                dy1 /= len1;
                
                double dx2 = polygonVertices[next].x - vertex.x;
                double dy2 = polygonVertices[next].y - vertex.y;
                double len2 = std::sqrt(dx2*dx2 + dy2*dy2);
                dx2 /= len2;
                dy2 /= len2;
                
                // Compute inward normal vectors
                double nx1 = -dy1;
                double ny1 = dx1;
                double nx2 = dy2;
                double ny2 = -dx2;
                
                // Create a small quadrilateral
                VoronoiVertex* v1 = new VoronoiVertex(vertex);
                VoronoiVertex* v2 = new VoronoiVertex(DelaunayPoint(
                    vertex.x + CORNER_SIZE * (-dx1 + nx1),
                    vertex.y + CORNER_SIZE * (-dy1 + ny1)
                ));
                VoronoiVertex* v3 = new VoronoiVertex(DelaunayPoint(
                    vertex.x + CORNER_SIZE * (nx1 + nx2) * 0.5,
                    vertex.y + CORNER_SIZE * (ny1 + ny2) * 0.5
                ));
                VoronoiVertex* v4 = new VoronoiVertex(DelaunayPoint(
                    vertex.x + CORNER_SIZE * (-dx2 + nx2),
                    vertex.y + CORNER_SIZE * (-dy2 + ny2)
                ));
                
                cornerCell->vertices.push_back(v1);
                cornerCell->vertices.push_back(v2);
                cornerCell->vertices.push_back(v3);
                cornerCell->vertices.push_back(v4);
                
                cells.push_back(cornerCell);
                std::cout << "Added corner cell at vertex " << i << std::endl;
            }
        }
        
        // Add cells along each edge of the polygon
        for (size_t i = 0; i < polygonVertices.size(); i++) {
            size_t j = (i + 1) % polygonVertices.size();
            const DelaunayPoint& p1 = polygonVertices[i];
            const DelaunayPoint& p2 = polygonVertices[j];
            
            // Compute edge length
            double dx = p2.x - p1.x;
            double dy = p2.y - p1.y;
            double edgeLength = std::sqrt(dx*dx + dy*dy);
            
            // Compute number of edge cells to add
            int numEdgeCells = std::max(1, static_cast<int>(edgeLength / CORNER_SIZE));
            
            // Compute inward normal
            double nx = -dy / edgeLength;
            double ny = dx / edgeLength;
            
            // Add cells along the edge
            for (int k = 1; k < numEdgeCells; k++) {
                double t = static_cast<double>(k) / numEdgeCells;
                DelaunayPoint edgePoint(
                    p1.x + t * dx,
                    p1.y + t * dy
                );
                
                // Check if this point is already covered
                bool pointCovered = false;
                for (const auto* cell : cells) {
                    for (const auto* vertex : cell->vertices) {
                        if (vertex && vertex->point.distanceTo(edgePoint) < CORNER_SIZE * 0.5) {
                            pointCovered = true;
                            break;
                        }
                    }
                    if (pointCovered) break;
                }
                
                if (!pointCovered) {
                    // Create a small cell at this edge point
                    VoronoiCell* edgeCell = new VoronoiCell(edgePoint);
                    
                    // Create a small quadrilateral
                    VoronoiVertex* v1 = new VoronoiVertex(edgePoint);
                    VoronoiVertex* v2 = new VoronoiVertex(DelaunayPoint(
                        edgePoint.x - CORNER_SIZE * dx / edgeLength,
                        edgePoint.y - CORNER_SIZE * dy / edgeLength
                    ));
                    VoronoiVertex* v3 = new VoronoiVertex(DelaunayPoint(
                        edgePoint.x - CORNER_SIZE * dx / edgeLength + CORNER_SIZE * nx,
                        edgePoint.y - CORNER_SIZE * dy / edgeLength + CORNER_SIZE * ny
                    ));
                    VoronoiVertex* v4 = new VoronoiVertex(DelaunayPoint(
                        edgePoint.x + CORNER_SIZE * nx,
                        edgePoint.y + CORNER_SIZE * ny
                    ));
                    
                    edgeCell->vertices.push_back(v1);
                    edgeCell->vertices.push_back(v2);
                    edgeCell->vertices.push_back(v3);
                    edgeCell->vertices.push_back(v4);
                    
                    cells.push_back(edgeCell);
                    std::cout << "Added edge cell on edge " << i << " at position " << t << std::endl;
                }
            }
        }
    }

    void VoronoiMeshGenerator::clipAgainstPolygonEdge(
        const std::vector<VoronoiVertex*>& inputVertices, 
        std::vector<VoronoiVertex*>& outputVertices,
        const DelaunayPoint& p1, const DelaunayPoint& p2, double nx, double ny) {
        
        if (inputVertices.empty()) {
            return;
        }
        
        // Last vertex
        const VoronoiVertex* s = inputVertices.back();
        
        // Check if last vertex is inside the half-plane
        bool sInside = isInsideHalfPlane(s->point, p1, nx, ny);
        
        // Process all vertices
        for (const auto* e : inputVertices) {
            // Check if current vertex is inside
            bool eInside = isInsideHalfPlane(e->point, p1, nx, ny);
            
            // If there's a transition (one inside, one outside)
            if (sInside != eInside) {
                // Compute intersection point
                VoronoiVertex* intersection = computeIntersectionWithLine(s, e, p1, p2);
                if (intersection) {
                    outputVertices.push_back(intersection);
                }
            }
            
            // If current vertex is inside, add it to output
            if (eInside) {
                outputVertices.push_back(new VoronoiVertex(e->point));
            }
            
            // Current becomes previous for next iteration
            s = e;
            sInside = eInside;
        }
    }

    bool VoronoiMeshGenerator::isInsideHalfPlane(const DelaunayPoint& p, const DelaunayPoint& linePoint, double nx, double ny) {
        // Check if point p is on the inside of the half-plane defined by linePoint and normal (nx,ny)
        double dx = p.x - linePoint.x;
        double dy = p.y - linePoint.y;
        
        // Dot product with normal
        return (dx * nx + dy * ny) >= 0;
    }

    VoronoiVertex* VoronoiMeshGenerator::computeIntersectionWithLine(
        const VoronoiVertex* v1, const VoronoiVertex* v2, 
        const DelaunayPoint& lineP1, const DelaunayPoint& lineP2) {
        
        // Line segment from v1 to v2
        double x1 = v1->point.x, y1 = v1->point.y;
        double x2 = v2->point.x, y2 = v2->point.y;
        
        // Line segment from lineP1 to lineP2
        double x3 = lineP1.x, y3 = lineP1.y;
        double x4 = lineP2.x, y4 = lineP2.y;
        
        // Compute intersection
        double den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
        
        // Check if lines are parallel
        if (std::abs(den) < 1e-10) {
            return nullptr;
        }
        
        double ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den;
        
        // Compute intersection point
        double x = x1 + ua * (x2 - x1);
        double y = y1 + ua * (y2 - y1);
        
        return new VoronoiVertex(DelaunayPoint(x, y));
    }

    void VoronoiMeshGenerator::addBoundaryPoints(std::vector<DelaunayPoint>& points, double minX, double minY, double maxX, double maxY) {
        int boundaryPoints = std::sqrt(points.size()) / 2;
        double stepX = (maxX - minX) / boundaryPoints;
        double stepY = (maxY - minY) / boundaryPoints;
        
        // Place points exactly on the boundaries
        for (int i = 0; i <= boundaryPoints; i++) {
            // Bottom and top edges - exact alignment
            points.push_back(DelaunayPoint(minX + i * stepX, minY));
            points.push_back(DelaunayPoint(minX + i * stepX, maxY));
            
            // Left and right edges - exact alignment
            points.push_back(DelaunayPoint(minX, minY + i * stepY));
            points.push_back(DelaunayPoint(maxX, minY + i * stepY));
        }
        
        // Add corners explicitly to ensure they're captured
        points.push_back(DelaunayPoint(minX, minY));
        points.push_back(DelaunayPoint(maxX, minY));
        points.push_back(DelaunayPoint(minX, maxY));
        points.push_back(DelaunayPoint(maxX, maxY));
        
        // Add points slightly inside the corners to improve corner cell quality
        double offset = std::min(stepX, stepY) * 0.25;
        points.push_back(DelaunayPoint(minX + offset, minY + offset));
        points.push_back(DelaunayPoint(maxX - offset, minY + offset));
        points.push_back(DelaunayPoint(minX + offset, maxY - offset));
        points.push_back(DelaunayPoint(maxX - offset, maxY - offset));
        
        // Add extra points near problematic corners (top-right and bottom-right)
        double smallOffset = offset * 0.5;
        points.push_back(DelaunayPoint(maxX - smallOffset, maxY - smallOffset)); // Top-right
        points.push_back(DelaunayPoint(maxX - smallOffset, minY + smallOffset)); // Bottom-right
        
        std::cout << "Added " << (4 * boundaryPoints + 14) << " boundary points" << std::endl;
    }

    void VoronoiMeshGenerator::snapVerticesToBoundary(std::vector<VoronoiCell*>& cells, double minX, double minY, double maxX, double maxY) {
        const double EPSILON = 1e-6;
        
        // First pass: snap vertices to exact boundary positions
        for (auto* cell : cells) {
            for (auto* vertex : cell->vertices) {
                if (!vertex) continue;
                
                // Snap to exact boundary values
                if (std::abs(vertex->point.x - minX) < EPSILON) {
                    vertex->point.x = minX;
                }
                if (std::abs(vertex->point.x - maxX) < EPSILON) {
                    vertex->point.x = maxX;
                }
                if (std::abs(vertex->point.y - minY) < EPSILON) {
                    vertex->point.y = minY;
                }
                if (std::abs(vertex->point.y - maxY) < EPSILON) {
                    vertex->point.y = maxY;
                }
            }
        }
        
        // Second pass: sort boundary vertices in counterclockwise order
        for (auto* cell : cells) {
            // Skip cells with too few vertices
            if (cell->vertices.size() < 3) continue;
            
            // Check if this cell has any vertices on the boundary
            bool hasLeftBoundary = false;
            bool hasRightBoundary = false;
            bool hasBottomBoundary = false;
            bool hasTopBoundary = false;
            
            for (auto* vertex : cell->vertices) {
                if (!vertex) continue;
                
                if (std::abs(vertex->point.x - minX) < EPSILON) hasLeftBoundary = true;
                if (std::abs(vertex->point.x - maxX) < EPSILON) hasRightBoundary = true;
                if (std::abs(vertex->point.y - minY) < EPSILON) hasBottomBoundary = true;
                if (std::abs(vertex->point.y - maxY) < EPSILON) hasTopBoundary = true;
            }
            
            // For cells that touch the boundary, ensure they have at least two vertices on that boundary
            if (hasLeftBoundary || hasRightBoundary || hasBottomBoundary || hasTopBoundary) {
                // Sort vertices in counterclockwise order
                std::vector<VoronoiVertex*> sortedVertices = cell->vertices;
                
                // Find the centroid of the cell
                double cx = 0, cy = 0;
                for (auto* v : sortedVertices) {
                    cx += v->point.x;
                    cy += v->point.y;
                }
                cx /= sortedVertices.size();
                cy /= sortedVertices.size();
                
                // Sort vertices counterclockwise around the centroid
                std::sort(sortedVertices.begin(), sortedVertices.end(), [cx, cy](VoronoiVertex* a, VoronoiVertex* b) {
                    return std::atan2(a->point.y - cy, a->point.x - cx) < 
                           std::atan2(b->point.y - cy, b->point.x - cx);
                });
                
                // Replace the cell's vertices with the sorted ones
                cell->vertices = sortedVertices;
            }
        }
    }

    std::vector<delaunay::VoronoiCell*> VoronoiMeshGenerator::generateMesh(
        const Geometry& geometry,
        const int numPoints, 
        const std::string& pointDistribution,
        int relaxationIterations) {
        
        // Get geometry bounds
        double minX, minY, maxX, maxY;
        geometry.getBoundingBox(minX, minY, maxX, maxY);
        
        // For rectangular geometries, use a simpler approach to avoid memory issues
        if (geometry.getType() == GeometryType::Rectangle) {
            return createRectangularMesh(minX, minY, maxX, maxY, numPoints);
        }
        
        // For non-rectangular geometries, use the original approach
        // Generate initial points
        std::vector<DelaunayPoint> points = generatePoints(geometry, numPoints, pointDistribution);
        
        // Apply Lloyd's relaxation if requested
        if (relaxationIterations > 0) {
            int iterations = relaxationIterations;
            points = relaxPoints(points, geometry, iterations);
        }
        
        // Create Delaunay triangulation
        DelaunayTriangulation delaunay;
        std::vector<Triangle> triangles = delaunay.triangulate(points);
        
        // Create Voronoi diagram
        VoronoiDiagram voronoi;
        voronoi.buildFromDelaunay(points, triangles);
        
        // Create our own copies of the Voronoi cells
        std::vector<VoronoiCell*> cells;
        const std::vector<VoronoiCell*>& voronoiCells = voronoi.getCells();
        
        for (const auto* voronoiCell : voronoiCells) {
            if (!voronoiCell || voronoiCell->vertices.size() < 3) continue;
            
            // Only include cells whose generator is inside the geometry
            if (!geometry.contains(voronoiCell->generator)) continue;
            
            // Create a new cell
            VoronoiCell* cell = new VoronoiCell(voronoiCell->generator);
            
            // Copy vertices
            for (const auto* vertex : voronoiCell->vertices) {
                if (!vertex) continue;
                cell->vertices.push_back(new VoronoiVertex(vertex->point));
            }
            
            cells.push_back(cell);
        }
        
        // Clip cells to geometry using the safer approach
        safeClipCellsToGeometry(cells, geometry);
        
        return cells;
    }

    std::vector<VoronoiCell*> VoronoiMeshGenerator::createRectangularMesh(
        double minX, double minY, double maxX, double maxY, int numPoints) {
        
        std::vector<VoronoiCell*> cells;
        
        // Calculate grid dimensions based on aspect ratio
        double width = maxX - minX;
        double height = maxY - minY;
        double aspectRatio = width / height;
        
        int numX = std::round(std::sqrt(numPoints * aspectRatio));
        int numY = std::round(std::sqrt(numPoints / aspectRatio));
        
        // Ensure we have at least 2 cells in each direction
        numX = std::max(2, numX);
        numY = std::max(2, numY);
        
        std::cout << "Creating rectangular Voronoi mesh with " << numX << "x" << numY << " cells" << std::endl;
        
        // Create a grid of points with slight perturbation
        std::vector<DelaunayPoint> gridPoints;
        
        // Set up random number generator for perturbation
        std::random_device rd;
        std::mt19937 gen(rd());
        double cellWidth = width / numX;
        double cellHeight = height / numY;
        double jitter = std::min(cellWidth, cellHeight) * 0.2; // 20% jitter
        std::uniform_real_distribution<> dist(-jitter, jitter);
        
        // Create grid points with jitter
        for (int i = 0; i < numX; i++) {
            for (int j = 0; j < numY; j++) {
                double x = minX + (i + 0.5) * cellWidth + dist(gen);
                double y = minY + (j + 0.5) * cellHeight + dist(gen);
                
                // Ensure points stay within bounds
                x = std::max(minX + 0.01 * cellWidth, std::min(maxX - 0.01 * cellWidth, x));
                y = std::max(minY + 0.01 * cellHeight, std::min(maxY - 0.01 * cellHeight, y));
                
                gridPoints.push_back(DelaunayPoint(x, y));
            }
        }
        
        // Add boundary points to ensure proper coverage
        for (int i = 0; i <= numX; i++) {
            double x = minX + i * cellWidth;
            gridPoints.push_back(DelaunayPoint(x, minY));
            gridPoints.push_back(DelaunayPoint(x, maxY));
        }
        
        for (int j = 0; j <= numY; j++) {
            double y = minY + j * cellHeight;
            gridPoints.push_back(DelaunayPoint(minX, y));
            gridPoints.push_back(DelaunayPoint(maxX, y));
        }
        
        // Create Delaunay triangulation
        DelaunayTriangulation delaunay;
        std::vector<Triangle> triangles = delaunay.triangulate(gridPoints);
        
        // Create Voronoi diagram
        VoronoiDiagram voronoi;
        voronoi.buildFromDelaunay(gridPoints, triangles);
        
        // Create our own copies of the Voronoi cells
        const std::vector<VoronoiCell*>& voronoiCells = voronoi.getCells();
        
        for (const auto* voronoiCell : voronoiCells) {
            if (!voronoiCell || voronoiCell->vertices.size() < 3) continue;
            
            // Only include cells whose generator is inside the rectangle
            if (voronoiCell->generator.x < minX || voronoiCell->generator.x > maxX ||
                voronoiCell->generator.y < minY || voronoiCell->generator.y > maxY) {
                continue;
            }
            
            // Create a new cell
            VoronoiCell* cell = new VoronoiCell(voronoiCell->generator);
            
            // Copy vertices
            for (const auto* vertex : voronoiCell->vertices) {
                if (!vertex) continue;
                
                // Clip vertex to rectangle boundaries
                double x = vertex->point.x;
                double y = vertex->point.y;
                
                x = std::max(minX, std::min(maxX, x));
                y = std::max(minY, std::min(maxY, y));
                
                cell->vertices.push_back(new VoronoiVertex(DelaunayPoint(x, y)));
            }
            
            // Only add cells with at least 3 vertices
            if (cell->vertices.size() >= 3) {
                cells.push_back(cell);
            } else {
                // Clean up if we're not adding this cell
                for (auto* vertex : cell->vertices) {
                    delete vertex;
                }
                delete cell;
            }
        }
        
        // Add corner cells to ensure complete coverage
        addCornerCells(cells, minX, minY, maxX, maxY);
        
        return cells;
    }

    void VoronoiMeshGenerator::safeClipCellsToGeometry(std::vector<VoronoiCell*>& cells, const Geometry& geometry) {
        // Get geometry bounds
        double minX, minY, maxX, maxY;
        geometry.getBoundingBox(minX, minY, maxX, maxY);
        
        // Process each cell
        for (auto* cell : cells) {
            if (!cell || cell->vertices.size() < 3) continue;
            
            // Create a new set of vertices for the clipped cell
            std::vector<VoronoiVertex*> clippedVertices;
            
            // First, clip against the bounding box
            // This is a simpler operation that avoids complex memory management
            
            // For each vertex in the cell
            for (auto* vertex : cell->vertices) {
                if (!vertex) continue;
                
                // If the vertex is inside the geometry, keep it
                if (geometry.contains(vertex->point)) {
                    clippedVertices.push_back(new VoronoiVertex(vertex->point));
                }
            }
            
            // If we have at least 3 vertices inside, use those
            if (clippedVertices.size() >= 3) {
                // Clear original vertices and replace with clipped ones
                for (auto* vertex : cell->vertices) {
                    delete vertex;
                }
                cell->vertices = clippedVertices;
            } else {
                // Otherwise, this cell is problematic and should be removed later
                for (auto* vertex : clippedVertices) {
                    delete vertex;
                }
            }
        }
        
        // Remove cells with fewer than 3 vertices
        std::vector<VoronoiCell*> validCells;
        for (auto* cell : cells) {
            if (cell && cell->vertices.size() >= 3) {
                validCells.push_back(cell);
            } else if (cell) {
                for (auto* vertex : cell->vertices) {
                    delete vertex;
                }
                delete cell;
            }
        }
        
        cells = validCells;
    }

    void VoronoiMeshGenerator::addCornerCells(std::vector<VoronoiCell*>& cells, double minX, double minY, double maxX, double maxY) {
        // Calculate corner cell size based on domain dimensions
        double width = maxX - minX;
        double height = maxY - minY;
        double cornerSize = std::min(width, height) * 0.03; // 3% of the smaller dimension
        
        std::cout << "Adding corner cells with size " << cornerSize << std::endl;
        
        // Top-right corner
        {
            VoronoiCell* cell = new VoronoiCell(DelaunayPoint(maxX - cornerSize/2, maxY - cornerSize/2));
            
            VoronoiVertex* v1 = new VoronoiVertex(DelaunayPoint(maxX - cornerSize, maxY));
            VoronoiVertex* v2 = new VoronoiVertex(DelaunayPoint(maxX, maxY));
            VoronoiVertex* v3 = new VoronoiVertex(DelaunayPoint(maxX, maxY - cornerSize));
            VoronoiVertex* v4 = new VoronoiVertex(DelaunayPoint(maxX - cornerSize, maxY - cornerSize));
            
            cell->vertices.push_back(v1);
            cell->vertices.push_back(v2);
            cell->vertices.push_back(v3);
            cell->vertices.push_back(v4);
            
            cells.push_back(cell);
            std::cout << "Added top-right corner cell" << std::endl;
        }
        
        // Bottom-right corner
        {
            VoronoiCell* cell = new VoronoiCell(DelaunayPoint(maxX - cornerSize/2, minY + cornerSize/2));
            
            VoronoiVertex* v1 = new VoronoiVertex(DelaunayPoint(maxX - cornerSize, minY));
            VoronoiVertex* v2 = new VoronoiVertex(DelaunayPoint(maxX, minY));
            VoronoiVertex* v3 = new VoronoiVertex(DelaunayPoint(maxX, minY + cornerSize));
            VoronoiVertex* v4 = new VoronoiVertex(DelaunayPoint(maxX - cornerSize, minY + cornerSize));
            
            cell->vertices.push_back(v1);
            cell->vertices.push_back(v2);
            cell->vertices.push_back(v3);
            cell->vertices.push_back(v4);
            
            cells.push_back(cell);
            std::cout << "Added bottom-right corner cell" << std::endl;
        }
        
        // Top-left corner
        {
            VoronoiCell* cell = new VoronoiCell(DelaunayPoint(minX + cornerSize/2, maxY - cornerSize/2));
            
            VoronoiVertex* v1 = new VoronoiVertex(DelaunayPoint(minX, maxY - cornerSize));
            VoronoiVertex* v2 = new VoronoiVertex(DelaunayPoint(minX, maxY));
            VoronoiVertex* v3 = new VoronoiVertex(DelaunayPoint(minX + cornerSize, maxY));
            VoronoiVertex* v4 = new VoronoiVertex(DelaunayPoint(minX + cornerSize, maxY - cornerSize));
            
            cell->vertices.push_back(v1);
            cell->vertices.push_back(v2);
            cell->vertices.push_back(v3);
            cell->vertices.push_back(v4);
            
            cells.push_back(cell);
            std::cout << "Added top-left corner cell" << std::endl;
        }
        
        // Bottom-left corner
        {
            VoronoiCell* cell = new VoronoiCell(DelaunayPoint(minX + cornerSize/2, minY + cornerSize/2));
            
            VoronoiVertex* v1 = new VoronoiVertex(DelaunayPoint(minX, minY));
            VoronoiVertex* v2 = new VoronoiVertex(DelaunayPoint(minX + cornerSize, minY));
            VoronoiVertex* v3 = new VoronoiVertex(DelaunayPoint(minX + cornerSize, minY + cornerSize));
            VoronoiVertex* v4 = new VoronoiVertex(DelaunayPoint(minX, minY + cornerSize));
            
            cell->vertices.push_back(v1);
            cell->vertices.push_back(v2);
            cell->vertices.push_back(v3);
            cell->vertices.push_back(v4);
            
            cells.push_back(cell);
            std::cout << "Added bottom-left corner cell" << std::endl;
        }
    }

    void VoronoiMeshGenerator::convertToEigenMesh(
        const std::vector<delaunay::VoronoiCell*>& cells,
        Eigen::MatrixXd& nodes,
        Eigen::MatrixXi& elements) {
        // Map to store unique vertices
        std::map<std::pair<double, double>, int> vertexMap;
        std::vector<std::pair<double, double>> uniqueVertices;
        
        // Debug output
        std::cout << "Converting " << cells.size() << " cells to Eigen mesh" << std::endl;
        
        // First pass: collect unique vertices
        for (const auto* cell : cells) {
            for (const auto* vertex : cell->vertices) {
                // Skip null vertices
                if (!vertex) {
                    continue;
                }
                
                // Add the vertex to the map
                std::pair<double, double> key(vertex->point.x, vertex->point.y);
                if (vertexMap.find(key) == vertexMap.end()) {
                    vertexMap[key] = uniqueVertices.size();
                    uniqueVertices.push_back(key);
                }
            }
        }
        
        std::cout << "Collected " << uniqueVertices.size() << " unique vertices" << std::endl;
        
        // Allocate nodes matrix
        nodes.resize(uniqueVertices.size(), 2);
        
        // Fill nodes matrix
        for (size_t i = 0; i < uniqueVertices.size(); i++) {
            nodes(i, 0) = uniqueVertices[i].first;
            nodes(i, 1) = uniqueVertices[i].second;
            
            // Debug: Print the first 20 nodes
            if (i < 20) {
                std::cout << "Node " << i << ": (" << nodes(i, 0) << ", " << nodes(i, 1) << ")" << std::endl;
            }
        }
        
        // Find maximum number of vertices per cell
        int maxVertices = 0;
        for (const auto* cell : cells) {
            maxVertices = std::max(maxVertices, static_cast<int>(cell->vertices.size()));
        }
        
        std::cout << "Maximum vertices per cell: " << maxVertices << std::endl;
        
        // Allocate elements matrix
        elements.resize(cells.size(), maxVertices);
        elements.setConstant(-1); // Fill with -1 for unused vertices
        
        // Fill elements matrix
        for (size_t i = 0; i < cells.size(); i++) {
            const auto* cell = cells[i];
            int validVertexCount = 0;
            
            for (size_t j = 0; j < cell->vertices.size(); j++) {
                const auto* vertex = cell->vertices[j];
                
                // Skip null vertices
                if (!vertex) {
                    continue;
                }
                
                std::pair<double, double> key(vertex->point.x, vertex->point.y);
                elements(i, validVertexCount++) = vertexMap[key];
            }
            
            // Debug: Print the first 10 elements
            if (i < 10) {
                std::cout << "Element " << i << " has " << validVertexCount << " vertices: ";
                for (int j = 0; j < validVertexCount; j++) {
                    std::cout << elements(i, j) << " ";
                }
                std::cout << std::endl;
                
                // Print the actual coordinates of each vertex in this element
                std::cout << "  Vertex coordinates for element " << i << ":" << std::endl;
                for (int j = 0; j < validVertexCount; j++) {
                    int nodeIdx = elements(i, j);
                    std::cout << "    Vertex " << j << " (node " << nodeIdx << "): (" 
                              << nodes(nodeIdx, 0) << ", " << nodes(nodeIdx, 1) << ")" << std::endl;
                }
            }
        }
    }

    void VoronoiMeshGenerator::fixCorners(std::vector<VoronoiCell*>& cells, double minX, double minY, double maxX, double maxY) {
        const double CORNER_SIZE = 0.05; // Increased size to ensure coverage
        
        // Check if corners are covered
        bool hasTopRight = false;
        bool hasBottomRight = false;
        bool hasTopLeft = false;
        bool hasBottomLeft = false;
        
        // Check if any cell contains each corner point
        for (const auto* cell : cells) {
            for (const auto* vertex : cell->vertices) {
                if (!vertex) continue;
                
                // Check for corner vertices
                if (std::abs(vertex->point.x - maxX) < 1e-6 && std::abs(vertex->point.y - maxY) < 1e-6) {
                    hasTopRight = true;
                }
                if (std::abs(vertex->point.x - maxX) < 1e-6 && std::abs(vertex->point.y - minY) < 1e-6) {
                    hasBottomRight = true;
                }
                if (std::abs(vertex->point.x - minX) < 1e-6 && std::abs(vertex->point.y - maxY) < 1e-6) {
                    hasTopLeft = true;
                }
                if (std::abs(vertex->point.x - minX) < 1e-6 && std::abs(vertex->point.y - minY) < 1e-6) {
                    hasBottomLeft = true;
                }
            }
        }
        
        // Also check if there are cells that have edges along both boundaries at each corner
        for (const auto* cell : cells) {
            if (cell->vertices.size() < 3) continue;
            
            bool hasRightEdge = false;
            bool hasTopEdge = false;
            bool hasLeftEdge = false;
            bool hasBottomEdge = false;
            
            for (const auto* vertex : cell->vertices) {
                if (!vertex) continue;
                
                if (std::abs(vertex->point.x - maxX) < 1e-6) hasRightEdge = true;
                if (std::abs(vertex->point.y - maxY) < 1e-6) hasTopEdge = true;
                if (std::abs(vertex->point.x - minX) < 1e-6) hasLeftEdge = true;
                if (std::abs(vertex->point.y - minY) < 1e-6) hasBottomEdge = true;
            }
            
            // If a cell has both edges at a corner, mark that corner as covered
            if (hasRightEdge && hasTopEdge) hasTopRight = true;
            if (hasRightEdge && hasBottomEdge) hasBottomRight = true;
            if (hasLeftEdge && hasTopEdge) hasTopLeft = true;
            if (hasLeftEdge && hasBottomEdge) hasBottomLeft = true;
        }
        
        // Add missing corner cells
        if (!hasTopRight) {
            VoronoiCell* cell = new VoronoiCell(DelaunayPoint(maxX - CORNER_SIZE/2, maxY - CORNER_SIZE/2));
            
            // Create a slightly larger square to ensure coverage
            VoronoiVertex* v1 = new VoronoiVertex(DelaunayPoint(maxX - CORNER_SIZE, maxY));
            VoronoiVertex* v2 = new VoronoiVertex(DelaunayPoint(maxX, maxY));
            VoronoiVertex* v3 = new VoronoiVertex(DelaunayPoint(maxX, maxY - CORNER_SIZE));
            VoronoiVertex* v4 = new VoronoiVertex(DelaunayPoint(maxX - CORNER_SIZE, maxY - CORNER_SIZE));
            
            cell->vertices.push_back(v1);
            cell->vertices.push_back(v2);
            cell->vertices.push_back(v3);
            cell->vertices.push_back(v4);
            
            cells.push_back(cell);
            std::cout << "Added top-right corner cell" << std::endl;
        }
        
        if (!hasBottomRight) {
            VoronoiCell* cell = new VoronoiCell(DelaunayPoint(maxX - CORNER_SIZE/2, minY + CORNER_SIZE/2));
            
            VoronoiVertex* v1 = new VoronoiVertex(DelaunayPoint(maxX - CORNER_SIZE, minY));
            VoronoiVertex* v2 = new VoronoiVertex(DelaunayPoint(maxX, minY));
            VoronoiVertex* v3 = new VoronoiVertex(DelaunayPoint(maxX, minY + CORNER_SIZE));
            VoronoiVertex* v4 = new VoronoiVertex(DelaunayPoint(maxX - CORNER_SIZE, minY + CORNER_SIZE));
            
            cell->vertices.push_back(v1);
            cell->vertices.push_back(v2);
            cell->vertices.push_back(v3);
            cell->vertices.push_back(v4);
            
            cells.push_back(cell);
            std::cout << "Added bottom-right corner cell" << std::endl;
        }
        
        if (!hasTopLeft) {
            VoronoiCell* cell = new VoronoiCell(DelaunayPoint(minX + CORNER_SIZE/2, maxY - CORNER_SIZE/2));
            
            VoronoiVertex* v1 = new VoronoiVertex(DelaunayPoint(minX, maxY - CORNER_SIZE));
            VoronoiVertex* v2 = new VoronoiVertex(DelaunayPoint(minX, maxY));
            VoronoiVertex* v3 = new VoronoiVertex(DelaunayPoint(minX + CORNER_SIZE, maxY));
            VoronoiVertex* v4 = new VoronoiVertex(DelaunayPoint(minX + CORNER_SIZE, maxY - CORNER_SIZE));
            
            cell->vertices.push_back(v1);
            cell->vertices.push_back(v2);
            cell->vertices.push_back(v3);
            cell->vertices.push_back(v4);
            
            cells.push_back(cell);
            std::cout << "Added top-left corner cell" << std::endl;
        }
        
        if (!hasBottomLeft) {
            VoronoiCell* cell = new VoronoiCell(DelaunayPoint(minX + CORNER_SIZE/2, minY + CORNER_SIZE/2));
            
            VoronoiVertex* v1 = new VoronoiVertex(DelaunayPoint(minX, minY));
            VoronoiVertex* v2 = new VoronoiVertex(DelaunayPoint(minX + CORNER_SIZE, minY));
            VoronoiVertex* v3 = new VoronoiVertex(DelaunayPoint(minX + CORNER_SIZE, minY + CORNER_SIZE));
            VoronoiVertex* v4 = new VoronoiVertex(DelaunayPoint(minX, minY + CORNER_SIZE));
            
            cell->vertices.push_back(v1);
            cell->vertices.push_back(v2);
            cell->vertices.push_back(v3);
            cell->vertices.push_back(v4);
            
            cells.push_back(cell);
            std::cout << "Added bottom-left corner cell" << std::endl;
        }
        
        // Add extra corner cells if needed
        // This ensures complete coverage by adding slightly offset corner cells
        if (!hasTopRight) {
            VoronoiCell* cell = new VoronoiCell(DelaunayPoint(maxX - CORNER_SIZE/4, maxY - CORNER_SIZE/4));
            
            VoronoiVertex* v1 = new VoronoiVertex(DelaunayPoint(maxX - CORNER_SIZE/2, maxY));
            VoronoiVertex* v2 = new VoronoiVertex(DelaunayPoint(maxX, maxY));
            VoronoiVertex* v3 = new VoronoiVertex(DelaunayPoint(maxX, maxY - CORNER_SIZE/2));
            VoronoiVertex* v4 = new VoronoiVertex(DelaunayPoint(maxX - CORNER_SIZE/2, maxY - CORNER_SIZE/2));
            
            cell->vertices.push_back(v1);
            cell->vertices.push_back(v2);
            cell->vertices.push_back(v3);
            cell->vertices.push_back(v4);
            
            cells.push_back(cell);
            std::cout << "Added extra top-right corner cell" << std::endl;
        }
    }

    std::vector<DelaunayPoint> VoronoiMeshGenerator::relaxPoints(
        const std::vector<DelaunayPoint>& initialPoints, 
        const Geometry& geometry, 
        int& iterations) {
        
        std::cout << "Applying Lloyd's relaxation for " << iterations << " iterations..." << std::endl;
        
        std::vector<DelaunayPoint> points = initialPoints;
        
        // Get geometry bounds
        double minX, minY, maxX, maxY;
        geometry.getBoundingBox(minX, minY, maxX, maxY);
        
        // Apply Lloyd's relaxation algorithm
        for (int iter = 0; iter < iterations; iter++) {
            // Create Delaunay triangulation
            DelaunayTriangulation delaunay;
            std::vector<Triangle> triangles = delaunay.triangulate(points);
            
            // Create Voronoi diagram
            VoronoiDiagram voronoi;
            voronoi.buildFromDelaunay(points, triangles);
            
            // Get Voronoi cells - these are owned by the VoronoiDiagram
            // and should NOT be deleted by us
            const std::vector<VoronoiCell*>& cells = voronoi.getCells();
            
            // Compute new points as centroids of Voronoi cells
            std::vector<DelaunayPoint> newPoints;
            for (size_t i = 0; i < cells.size() && i < points.size(); i++) {
                const auto* cell = cells[i];
                
                // Skip cells with too few vertices
                if (!cell || cell->vertices.size() < 3) {
                    newPoints.push_back(points[i]);
                    continue;
                }
                
                // Compute centroid
                double cx = 0.0, cy = 0.0;
                double area = 0.0;
                
                // Use shoelace formula to compute area and centroid
                for (size_t j = 0; j < cell->vertices.size(); j++) {
                    size_t k = (j + 1) % cell->vertices.size();
                    
                    if (!cell->vertices[j] || !cell->vertices[k]) {
                        continue;
                    }
                    
                    double x1 = cell->vertices[j]->point.x;
                    double y1 = cell->vertices[j]->point.y;
                    double x2 = cell->vertices[k]->point.x;
                    double y2 = cell->vertices[k]->point.y;
                    
                    double cross = x1 * y2 - x2 * y1;
                    area += cross;
                    
                    cx += (x1 + x2) * cross;
                    cy += (y1 + y2) * cross;
                }
                
                area *= 0.5;
                
                // If area is too small, keep original point
                if (std::abs(area) < 1e-10) {
                    newPoints.push_back(points[i]);
                    continue;
                }
                
                // Compute centroid
                cx /= (6.0 * area);
                cy /= (6.0 * area);
                
                // Ensure the new point is inside the geometry
                DelaunayPoint newPoint(cx, cy);
                if (isPointInGeometry(newPoint, geometry)) {
                    newPoints.push_back(newPoint);
                } else {
                    // If centroid is outside, keep original point
                    newPoints.push_back(points[i]);
                }
                
                // DO NOT delete the cell here - it's owned by the VoronoiDiagram
                // delete cell; <-- This was causing the error
            }
            
            // Update points
            points = newPoints;
            
            std::cout << "Completed relaxation iteration " << (iter + 1) << std::endl;
        }
        
        return points;
    }
}