#include "material/mat.hpp"


void material::mat::setElasticModule(double elastic_module){
    E = elastic_module;
}

void material::mat::setPoissonCoef(double poisson_coef){
    nu = poisson_coef;
}

void material::mat::setMaterialDensity(double material_density){
    rho = material_density;
}

void material::mat::getLameParameters(){
    Mu = E/(2.0*(1.0+nu));
    La = E*nu/((1.0+nu)*(1.0-2.0*nu));
}

Eigen::MatrixXd material::mat::build2DElasticity(){
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(3,3);
    C(0,0) = E/(1.0-nu*nu);
    C(0,1) = C(0,0) * nu;
    C(1,0) = C(0,1);
    C(1,1) = C(0,0);
    C(2,2) = E/(2.0*(1.0+nu));

    return C;
}

Eigen::MatrixXd material::mat::buildAxisymmetricElasticity(double E, double nu){
    double factor = E / ((1 + nu) * (1 - 2 * nu));
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(4,4);

    C(0,0) = factor * (1 - nu);
    C(0,1) = factor * nu;
    C(0,2) = factor * nu;

    C(1,0) = factor * nu;
    C(1,1) = factor * (1 - nu);
    C(1,2) = factor * nu;

    C(2,0) = factor * nu;
    C(2,1) = factor * nu;
    C(2,2) = factor * (1 - nu);

    C(3,3) = factor * (1 - 2*nu) / 2;

    return C;
}

double material::mat::compute_stabilization_parameter(double E, double nu){
    // Compute shear modulus
    double mu = E / (2.0 * (1.0 + nu));

    // Base scaling factor relative to shear modulus
    double base_scaling = 1e-3;

    // Initialize tau
    double tau = 0.0;

    // Scale τ based on material properties
    if (nu < 0.4){
        // Standard scaling for normal materials
        tau = base_scaling * mu;
    } else if (nu < 0.45){
        // Smooth linear increase for transition region
        double t = (nu - 0.4) / 0.05;
        tau = base_scaling * mu * (1.0 + t);
    } else {
        // Enhanced scaling for nearly incompressible materials
        double t = (nu - 0.45) / 0.05;
        tau = base_scaling * mu * (2.0 + 3.0 * t);
    }

    return tau; 
}