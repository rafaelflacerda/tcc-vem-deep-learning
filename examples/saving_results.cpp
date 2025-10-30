#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <cmath>
#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "lib/utils/operations.hpp"
#include "lib/utils/integration.hpp"
#include "lib/mesh/beam.hpp"
#include "lib/mesh/datasource.hpp"
#include "lib/solver/beam1d.hpp"
#include "lib/material/mat.hpp"
#include "lib/solver/linearElastic2d.hpp"
#include "lib/solver/nonlinear2d.hpp"

 
using Eigen::MatrixXd;

using json = nlohmann::json;

std::pair<double, double> calculateAspectRatio(const std::vector<cv::Point2f>& points) {
    if (points.size() < 5) {
        std::cerr << "Not enough points to fit an ellipse. Number of points: " << points.size() << std::endl;
        return {0.0, 0.0}; // Return a default value
    }

    // Fit an ellipse to the points
    cv::RotatedRect ellipse;
    try {
        ellipse = cv::fitEllipse(points);
    } catch(const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
        return {0.0, 0.0}; // Return a default value
    }

    // Extract the major and minor axes
    double majorAxis = std::max(ellipse.size.width, ellipse.size.height) / 2.0;
    double minorAxis = std::min(ellipse.size.width, ellipse.size.height) / 2.0;

    return {majorAxis, minorAxis};
}

int main() {
    // std::vector<cv::Point2f> points = {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}}; // Ensure at least 5 points
    // std::cout << "Number of points provided: " << points.size() << std::endl;
    // std::pair<double, double> aspectRatios = calculateAspectRatio(points);
    // double majorAxis = aspectRatios.first;
    // double minorAxis = aspectRatios.second;
    // std::cout << "Major Axis: " << majorAxis << std::endl;
    // std::cout << "Minor Axis: " << minorAxis << std::endl;

    // mesh::datasource ds("data/unit_square_16.json");
    mesh::datasource ds("data/unit_square_triang_16.json");
    // Eigen::VectorXd aspect_ratios =  ds.calculateAspectRatio();

    Eigen::MatrixXd nodes = ds.nodes;
    Eigen::MatrixXi elements = ds.elements;
    Eigen::MatrixXi supp = ds.supp;
    Eigen::MatrixXi load = ds.load;
    double qx = ds.qx;
    double qy = ds.qy;

    material::mat mat;
    mat.setElasticModule(200.0);
    mat.setPoissonCoef(0.3);
    mat.getLameParameters();

    double La = mat.La;
    double Mu = mat.Mu;
    double nu = mat.nu;

    Eigen::MatrixXi element = ds.elements.row(0);
    Eigen::MatrixXd coords(elements.cols(), 2);

    for(int i=0; i<element.cols(); i++){
        coords.row(i) << nodes.row(element(i));
    }

    //std::cout << aspect_ratios << std::endl;

    solver::nonlinear2d s(nodes, elements, supp, load, qx, qy, Mu, La);

    // double taylor = s.calculateTaylor5(nu);
    // std::cout << "5-th Taylor Expansion :: " << taylor << std::endl;

    int numVertex = element.cols();
    std::cout << "Number of vertices per element :: " << numVertex << std::endl;


    // Local terms
    Eigen::VectorXd ue = Eigen::VectorXd::Zero(2*numVertex);
    Eigen::MatrixXd Kloc = s.buildLocalK(coords, ue);
    Eigen::VectorXd Rt = Eigen::VectorXd::Zero(4);
    Eigen::VectorXd Rc = s.buildLocalRc(coords, ue);
    // s.localLoadVector(1.0, 10.0, 0.0, u, Rt);


    
    // Global terms
    Eigen::VectorXd u = Eigen::VectorXd::Zero(2*nodes.rows());
    Eigen::MatrixXd K = s.buildGlobalK(u);
    Eigen::MatrixXd K_ = s.applyDBC(K);
    Eigen::VectorXd R = s.buildGlobalR(u);
    Eigen::VectorXd F = s.applyNBC(u);

    // Solver
    int ndof = 2*nodes.rows();
    Eigen::VectorXd u0 = Eigen::VectorXd::Constant(ndof, 0.1);
    // Eigen::VectorXd u1 = s.newtonRaphson(u0, 0.1, 0.0);
    // Eigen::VectorXd uh = s.solve(u0, 0.1);

    // Eigen::MatrixXd sortedNodes = ds.sortNodes(u0);
    ds.writeOutput(u0, "debug_results");

    return 0;
}
