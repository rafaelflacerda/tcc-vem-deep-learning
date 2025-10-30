#include "mesh/datasource.hpp"
#include "mesh/helpers.hpp"

void mesh::datasource::readJson(std::string filepath){
    using json = nlohmann::json;

    // load json data
    std::ifstream input_file(filepath);
    json j;
    input_file >> j;

    auto nodes_values = j["Nodes"];
    auto elements_values = j["Elements"];
    auto supp_values = j["DirichletBCs"];
    auto load_values = j["NeumannBCs"];

    // compatibilize matrices to the correct size
    nodes.resize(nodes_values.size(), nodes_values[0].size());
    elements.resize(elements_values.size(), elements_values[0].size());
    supp.resize(supp_values.size(), 3);
    load.resize(load_values.size()-1, 2);

    // fill node matrix with values
    for(int i = 0; i < nodes_values.size(); i++){
        for(int j = 0; j < nodes_values[0].size(); j++){
            nodes(i,j) = nodes_values[i][j];
        }
    }

    // fill element matrix with valiues
    for(int i = 0; i < elements_values.size(); i++){
        for(int j = 0; j < elements_values[0].size(); j++){
            elements(i,j) = int(elements_values[i][j])-1;
        }
    }

    // fill support matrix with values (node, isXFixed, isYFixed)
    for(int i = 0; i < supp_values.size(); i++){
        supp(i,0) = int(supp_values[i]["Node"])-1;

        if (supp_values[i]["uBound"] == 0.0){
            supp(i,1) = 1;
        } else {
            supp(i,1) = 0;
        }

        if(supp_values[i]["vBound"] == 0.0){
            supp(i,2) = 1;
        } else {
            supp(i,2) = 0;
        }
    }

    // fill load matrix with values (startNode, endNode)
    for(int i = 0; i<load_values.size()-1; i++){
        load(i,0) = int(load_values[i]["Node"])-1;
        load(i,1) = int(load_values[i+1]["Node"])-1;
    }

    // get load values (assuming only uniform load)
    qx = load_values[0]["q0x"];
    qy = load_values[0]["q0y"];
}

void mesh::datasource::readJsonBeam(std::string filepath){
    using json = nlohmann::json;

    // load json data
    std::ifstream input_file(filepath);
    json j;
    input_file >> j;

    auto nodes_values = j["nodes"];
    auto elements_values = j["elements"];
    auto supp_values = j["dbc"];
    auto load_values = j["nbc"];

    // compatibilize matrices to the correct size
    nodes.resize(nodes_values.size(), nodes_values[0].size());
    elements.resize(elements_values.size(), elements_values[0].size());
    supp.resize(supp_values.size(), 4);
    load.resize(load_values.size(), 2);

    std::cout << "------------------------------------" << std::endl;
    std::cout << "Load data information: " << std::endl;
    std::cout << "Nodes: " << nodes_values.size() << std::endl;
    std::cout << "Elements: " << elements_values.size() << std::endl;
    std::cout << "DBC: " << supp_values.size() << std::endl;
    std::cout << "NBC: " << load_values.size() << std::endl;
    std::cout << "------------------------------------" << std::endl;

    // fill node matrix with values
    for(int i = 0; i < nodes_values.size(); i++){
        for(int j = 0; j < nodes_values[0].size(); j++){
            nodes(i,j) = nodes_values[i][j];
        }
    }

    // fill element matrix with valiues
    for(int i = 0; i < elements_values.size(); i++){
        for(int j = 0; j < elements_values[0].size(); j++){
            elements(i,j) = int(elements_values[i][j]);
        }
    }

    // fill support matrix with values (node, isXFixed, isYFixed)
    for(int i = 0; i < supp_values.size(); i++){
        supp(i,0) = int(supp_values[i]["node"]);

        if (supp_values[i]["uBound"] == 0.0){
            supp(i,1) = 1;
        } else {
            supp(i,1) = 0;
        }

        if(supp_values[i]["vBound"] == 0.0){
            supp(i,2) = 1;
        } else {
            supp(i,2) = 0;
        }

        if(supp_values[i]["rotationBound"] == 0.0){
            supp(i,3) = 1;
        } else {
            supp(i,3) = 0;
        }
    }

    // fill load matrix with values (startNode, endNode)
    for(int i = 0; i<load_values.size()-1; i++){
        load(i,0) = int(load_values[i]["element"][0]);
        load(i,1) = int(load_values[i]["element"][1]);
    }

    // get load values (assuming only uniform load)
    qx = load_values[0]["q0x"];
    qy = load_values[0]["q0y"];

   
}

void mesh::datasource::writeOutput(Eigen::VectorXd u, std::string filename){

    utils::logging log;

    nlohmann::json outputJson;

    // Assuming each node has two displacement components (u_x, u_y)
    for (int i = 0; i < u.size(); i += 2) {
        int nodeIndex = i / 2;

        // Create a JSON object for each node
        nlohmann::json nodeJson;
        nodeJson["displacement_x"] = u[i];
        nodeJson["displacement_y"] = u[i + 1];
        nodeJson["coordinate_x"] = nodes(nodeIndex, 0);
        nodeJson["coordinate_y"] = nodes(nodeIndex, 1);

        // Add the node JSON object to the output JSON
        outputJson["nodes"].push_back(nodeJson);
    }

    // Write the JSON object to a file
    std::string timestamp = log.generateTimestamp();
    std::string filename_ = "data/_output/" + filename + "_" + timestamp + ".json";
    std::ofstream outputFile(filename_);
    if (outputFile.is_open()) {
        outputFile << std::setw(4) << outputJson << std::endl;
        outputFile.close();
        std::cout << "Data written to " << filename_ << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename_ << std::endl;
    }
}

void mesh::datasource::saveDisplacementsToJson(Eigen::VectorXd u, double E, double A, double I, std::string filename) {
    utils::logging log;

    nlohmann::json outputJson;

    // Add the material properties to the output JSON
    outputJson["material"]["E"] = E;
    outputJson["material"]["A"] = A;
    outputJson["material"]["I"] = I;

    // Save the displacement values
    std::vector<double> dispVec(u.data(), u.data() + u.size());
    outputJson["displacements"] = dispVec;

    // Generate timestamp and prepare the file path
    std::string timestamp = log.generateTimestamp();
    std::string fullPath = "data/_output/" + filename + "_" + timestamp + ".json";

    // Ensure the directory exists
    std::filesystem::path dir = std::filesystem::path(fullPath).parent_path();
    std::filesystem::create_directories(dir);

    // Write the JSON object to the file
    std::ofstream outFile(fullPath);
    if (outFile.is_open()) {
        outFile << outputJson.dump(4);  // Pretty-print with 4 spaces
        std::cout << "Displacements saved to " << fullPath << std::endl;
    } else {
        std::cerr << "Failed to open file: " << fullPath << std::endl;
    }
}

void mesh::datasource::saveBeamGeometryToJson(Eigen::MatrixXd nodes, Eigen::MatrixXi elements, Eigen::MatrixXi supp, Eigen::MatrixXi distributed_load_elements, Eigen::VectorXd loads, std::string filename) {
    nlohmann::json geometryJson;

    // Convert nodes matrix to JSON array of arrays
    nlohmann::json nodesJson = nlohmann::json::array();
    for (int i = 0; i < nodes.rows(); i++) {
        std::vector<double> nodeCoords = {nodes(i, 0), nodes(i, 1)};
        nodesJson.push_back(nodeCoords);
    }
    geometryJson["nodes"] = nodesJson;

    // Convert elements matrix to JSON array of arrays
    nlohmann::json elementsJson = nlohmann::json::array();
    for (int i = 0; i < elements.rows(); i++) {
        std::vector<int> elementNodes = {elements(i, 0), elements(i, 1)};
        elementsJson.push_back(elementNodes);
    }
    geometryJson["elements"] = elementsJson;

    // Convert support conditions to JSON array of objects
    nlohmann::json dbcJson = nlohmann::json::array();
    for (int i = 0; i < supp.rows(); i++) {
        nlohmann::json dbcEntry;
        int nodeIndex = supp(i, 0);
        
        dbcEntry["node"] = nodeIndex;
        dbcEntry["uBound"] = supp(i, 1) == 1 ? 0.0 : 1.0;      // Convert from fixed (1) to bound (0.0)
        dbcEntry["vBound"] = supp(i, 2) == 1 ? 0.0 : 1.0;      // Convert from fixed (1) to bound (0.0)
        dbcEntry["rotationBound"] = supp(i, 3) == 1 ? 0.0 : 1.0; // Convert from fixed (1) to bound (0.0)
        dbcEntry["xCoord"] = nodes(nodeIndex, 0);
        dbcEntry["yCoord"] = nodes(nodeIndex, 1);
        
        dbcJson.push_back(dbcEntry);
    }
    geometryJson["dbc"] = dbcJson;

    // Convert Neumann boundary conditions to JSON array of objects
    nlohmann::json nbcJson = nlohmann::json::array();
    for (int i = 0; i < distributed_load_elements.rows(); i++) {
        nlohmann::json nbcEntry;
        
        // Create array for element nodes
        std::vector<int> elementNodes = {
            distributed_load_elements(i, 0),
            distributed_load_elements(i, 1)
        };
        nbcEntry["element"] = elementNodes;
        
        // Add load values
        // Assuming loads vector contains alternating x and y components
        // If loads is organized differently, adjust these indices accordingly
        nbcEntry["q0x"] = loads(0);  // First component for x
        nbcEntry["q0y"] = 0.0;  // Second component for y
        
        nbcJson.push_back(nbcEntry);
    }
    geometryJson["nbc"] = nbcJson;

    // Prepare the filepath
    std::string fullPath = "data/" + filename + ".json";
    
    // Ensure that the directory exists
    std::filesystem::path dir = std::filesystem::path(fullPath).parent_path();
    std::filesystem::create_directories(dir);

    // Write the JSON object to the file
    std::ofstream outFile(fullPath);
    if (outFile.is_open()) {
        outFile << geometryJson.dump(4);  // Pretty-print with 4 spaces
        std::cout << "Displacements saved to " << fullPath << std::endl;
    } else {
        std::cerr << "Failed to open file: " << fullPath << std::endl;
    }
}

void mesh::datasource::generateRandomSamples(int numSamples, Eigen::VectorXd& E_samples, Eigen::VectorXd& A_samples, Eigen::VectorXd& I_samples) {
    // Define ranges
    double E_min = 1e6, E_max = 210e9;
    // double E_min = 10e6, E_max = 40e7;
    double A_min = 1, A_max = 1.5;
    double I_min = 1e-6, I_max = 1e-3;
    // double I_min = 10, I_max = 100;

    // Random number generator setup
    std::random_device rd; // Seed
    std::mt19937 gen(rd()); // Mersenne Twister RNG

    // Define the distributions
    std::uniform_real_distribution<double> E_dist(E_min, E_max);
    std::uniform_real_distribution<double> A_dist(A_min, A_max);
    std::uniform_real_distribution<double> I_dist(I_min, I_max);

    // Resize Eigen vectors to store the samples
    E_samples.resize(numSamples);
    A_samples.resize(numSamples);
    I_samples.resize(numSamples);

    // Generate the samples
    for (int i = 0; i < numSamples; ++i) {
        E_samples(i) = E_dist(gen);
        A_samples(i) = A_dist(gen);
        I_samples(i) = I_dist(gen);
    }
}

Eigen::VectorXd mesh::datasource::calculateAspectRatio(){
    // define support functions
    utils::operations op;

    // aspect ratio vector
    Eigen::VectorXd aspectRatios(elements.rows());

    // coordinates and elements
    Eigen::MatrixXd coords;
    Eigen::MatrixXi e;

    // element centroid
    Eigen::VectorXd centroid;

    for(int i = 0; i < elements.rows(); i++){
        e = elements.row(i);
        coords = op.getCoordinatesPlane(e, nodes);

        centroid = op.calcCentroid(coords);

        // Convert Eigen matrix to std::vector<cv::Point2f> and add the centroid
        std::vector<cv::Point2f> extendedPoints;
        for (int i = 0; i < coords.rows(); ++i) {
            extendedPoints.push_back(cv::Point2f(coords(i, 0), coords(i, 1)));
        }
        extendedPoints.push_back(cv::Point2f(centroid(0), centroid(1)));
        
        // Fit an ellipse to the points
        cv::RotatedRect ellipse;
        try {
        ellipse = cv::fitEllipse(extendedPoints);
        } catch(const cv::Exception& e) {
            std::cerr << "OpenCV Error: " << e.what() << std::endl;
        }

        // Extract the major and minor axes
        double majorAxis = std::max(ellipse.size.width, ellipse.size.height) / 2.0;
        double minorAxis = std::min(ellipse.size.width, ellipse.size.height) / 2.0;
        double aspect_ratio = majorAxis/minorAxis;

        // Add to the aspect ratio vector
        aspectRatios(i) = aspect_ratio;
    }

    // std::cout << aspectRatios << std::endl;

    return aspectRatios;
}

Eigen::MatrixXd mesh::datasource::sortNodes(Eigen::VectorXd u){
    Eigen::MatrixXd nodeCoords(u.size() / 2, nodes.cols());

    for (int i = 0; i < u.size(); i += 2) {
        int nodeIndex = i / 2;
        nodeCoords.row(nodeIndex) = nodes.row(nodeIndex);
    }

    return nodeCoords;
}

void mesh::datasource::exportTriangulationToJson(const std::vector<delaunay::DelaunayPoint>& points, 
                  const std::vector<Triangle>& triangles, 
                  const std::vector<Edge>& edges,
                  const std::string& filename){
    using json = nlohmann::json;
    json j;
    
    // Export points
    json jsonPoints = json::array();
    for (size_t i = 0; i < points.size(); ++i) {
        const delaunay::DelaunayPoint& p = points[i];
        jsonPoints.push_back({
            {"id", i},
            {"x", p.x},
            {"y", p.y}
        });
    }
    j["points"] = jsonPoints;
    
    // Create a map of points to their indices for easy reference
    std::map<std::pair<double, double>, size_t> pointIndices;
    for (size_t i = 0; i < points.size(); ++i) {
        pointIndices[{points[i].x, points[i].y}] = i;
    }
    
    // Export triangles
    json jsonTriangles = json::array();
    for (size_t i = 0; i < triangles.size(); ++i) {
        const Triangle& t = triangles[i];
        
        // Find the index of each vertex in the points array
        size_t idx1 = meshHelpers::findPointIndex(t.p1, points);
        size_t idx2 = meshHelpers::findPointIndex(t.p2, points);
        size_t idx3 = meshHelpers::findPointIndex(t.p3, points);
        
        jsonTriangles.push_back({
            {"id", i},
            {"vertices", {idx1, idx2, idx3}},
            {"points", {
                {{"x", t.p1.x}, {"y", t.p1.y}},
                {{"x", t.p2.x}, {"y", t.p2.y}},
                {{"x", t.p3.x}, {"y", t.p3.y}}
            }}
        });
    }
    j["triangles"] = jsonTriangles;
    
    // Export edges
    json jsonEdges = json::array();
    for (size_t i = 0; i < edges.size(); ++i) {
        const Edge& e = edges[i];
        
        // Find the index of each endpoint in the points array
        size_t idx1 = meshHelpers::findPointIndex(e.p1, points);
        size_t idx2 = meshHelpers::findPointIndex(e.p2, points);
        
        jsonEdges.push_back({
            {"id", i},
            {"vertices", {idx1, idx2}},
            {"points", {
                {{"x", e.p1.x}, {"y", e.p1.y}},
                {{"x", e.p2.x}, {"y", e.p2.y}}
            }}
        });
    }
    j["edges"] = jsonEdges;
    
    // Add metadata
    j["metadata"] = {
        {"pointCount", points.size()},
        {"triangleCount", triangles.size()},
        {"edgeCount", edges.size()}
    };
    
    // Write to file
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    file << j.dump(4); // 4-space indentation for pretty formatting
    file.close();
}

void mesh::datasource::exportVoronoiToJson(
    const std::vector<VoronoiVertex*>& vertices,
    const std::vector<VoronoiEdge*>& edges,
    const std::vector<VoronoiCell*>& cells,
    const std::string& filename) 
{
    using json = nlohmann::json;
    json j;
    
    // Export Voronoi vertices
    json jsonVertices = json::array();
    for (size_t i = 0; i < vertices.size(); ++i) {
        jsonVertices.push_back({
            {"id", i},
            {"x", vertices[i]->point.x},
            {"y", vertices[i]->point.y}
        });
    }
    j["vertices"] = jsonVertices;
    
    // Create a map of vertices to their indices for easy reference
    std::map<const VoronoiVertex*, size_t> vertexIndices;
    for (size_t i = 0; i < vertices.size(); ++i) {
        vertexIndices[vertices[i]] = i;
    }
    
    // Export Voronoi edges
    json jsonEdges = json::array();
    for (size_t i = 0; i < edges.size(); ++i) {
        const VoronoiEdge* edge = edges[i];
        jsonEdges.push_back({
            {"id", i},
            {"v1", vertexIndices[edge->v1]},
            {"v2", vertexIndices[edge->v2]},
            {"generator1", {{"x", edge->generator1.x}, {"y", edge->generator1.y}}},
            {"generator2", {{"x", edge->generator2.x}, {"y", edge->generator2.y}}}
        });
    }
    j["edges"] = jsonEdges;
    
    // Export Voronoi cells
    json jsonCells = json::array();
    for (size_t i = 0; i < cells.size(); ++i) {
        const VoronoiCell* cell = cells[i];
        
        // Collect vertex indices for this cell
        std::vector<size_t> cellVertexIndices;
        for (const VoronoiVertex* vertex : cell->vertices) {
            cellVertexIndices.push_back(vertexIndices[vertex]);
        }
        
        // Collect edge indices for this cell
        std::vector<size_t> cellEdgeIndices;
        for (const VoronoiEdge* edge : cell->edges) {
            auto it = std::find(edges.begin(), edges.end(), edge);
            if (it != edges.end()) {
                cellEdgeIndices.push_back(std::distance(edges.begin(), it));
            }
        }
        
        jsonCells.push_back({
            {"id", i},
            {"generator", {{"x", cell->generator.x}, {"y", cell->generator.y}}},
            {"vertices", cellVertexIndices},
            {"edges", cellEdgeIndices}
        });
    }
    j["cells"] = jsonCells;
    
    // Add metadata
    j["metadata"] = {
        {"vertexCount", vertices.size()},
        {"edgeCount", edges.size()},
        {"cellCount", cells.size()}
    };
    
    // Write to file
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    file << j.dump(4); // 4-space indentation for pretty formatting
    file.close();
}

void mesh::datasource::exportVoronoiMeshToJson(
    const std::vector<VoronoiCell*>& cells,
    const Geometry& geometry,
    const std::string& filename)
{
    using json = nlohmann::json;
    json j;
    
    // Debug the first few cells
    for (int i = 0; i < std::min(5, (int)cells.size()); i++) {
        std::cout << "Cell " << i << " generator: (" 
                  << cells[i]->generator.x << ", " 
                  << cells[i]->generator.y << ")" << std::endl;
    }
    
    // Map to store unique vertices
    std::map<std::pair<double, double>, int> vertexMap;
    std::vector<std::pair<double, double>> uniqueVertices;
    
    // First pass: collect unique vertices
    for (const auto* cell : cells) {
        for (const auto* vertex : cell->vertices) {
            std::pair<double, double> key(vertex->point.x, vertex->point.y);
            if (vertexMap.find(key) == vertexMap.end()) {
                vertexMap[key] = uniqueVertices.size();
                uniqueVertices.push_back(key);
            }
        }
    }
    
    // Create nodes array
    json jsonNodes = json::array();
    for (size_t i = 0; i < uniqueVertices.size(); i++) {
        jsonNodes.push_back({
            {"id", i},
            {"x", uniqueVertices[i].first},
            {"y", uniqueVertices[i].second}
        });
    }
    j["nodes"] = jsonNodes;
    
    // Create elements array - each element contains the indices of its vertices
    json jsonElements = json::array();
    for (size_t i = 0; i < cells.size(); i++) {
        const auto* cell = cells[i];
        
        // Add cell data to JSON
        json cellJson;
        cellJson["id"] = i;
        cellJson["generator"] = {{"x", cell->generator.x}, {"y", cell->generator.y}};
        
        // Get vertices for this cell
        std::vector<int> vertexIndices;
        for (const auto* vertex : cell->vertices) {
            std::pair<double, double> key(vertex->point.x, vertex->point.y);
            vertexIndices.push_back(vertexMap[key]);
        }
        
        cellJson["vertices"] = vertexIndices;
        jsonElements.push_back(cellJson);
    }
    j["elements"] = jsonElements;
    
    // Add metadata
    j["metadata"] = {
        {"nodeCount", uniqueVertices.size()},
        {"elementCount", cells.size()},
        {"meshType", "voronoi"}
    };
    
    // Write to file
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    file << j.dump(4); // 4-space indentation for pretty formatting
    file.close();
    
    std::cout << "Mesh exported to " << filename << std::endl;
}

void mesh::datasource::exportEigenMeshToJson(const Eigen::MatrixXd& nodes, const Eigen::MatrixXi& elements, const std::string& filename) {
    using json = nlohmann::json;
    json j;
    
    // Add metadata
    j["metadata"]["meshType"] = "voronoi_eigen";
    j["metadata"]["nodeCount"] = nodes.rows();
    j["metadata"]["elementCount"] = elements.rows();
    
    // Export nodes
    json jsonNodes = json::array();
    for (int i = 0; i < nodes.rows(); i++) {
        json node;
        node["id"] = i;
        node["x"] = nodes(i, 0);
        node["y"] = nodes(i, 1);
        jsonNodes.push_back(node);
    }
    j["nodes"] = jsonNodes;
    
    // Export elements
    json jsonElements = json::array();
    for (int i = 0; i < elements.rows(); i++) {
        json element;
        element["id"] = i;
        
        // Create array of vertex indices for this element
        json vertices = json::array();
        for (int j = 0; j < elements.cols(); j++) {
            if (elements(i, j) >= 0) {  // Only include valid vertex indices
                vertices.push_back(elements(i, j));
            }
        }
        element["vertices"] = vertices;
        jsonElements.push_back(element);
    }
    j["elements"] = jsonElements;
    
    // Write to file
    std::string fullPath = "data/" + filename;
    std::ofstream file(fullPath);
    if (file.is_open()) {
        file << std::setw(4) << j << std::endl;
        file.close();
        std::cout << "Mesh exported to " << fullPath << std::endl;
    } else {
        std::cerr << "Error: Could not open file " << fullPath << " for writing" << std::endl;
    }
}


