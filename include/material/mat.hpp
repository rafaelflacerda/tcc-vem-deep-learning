/**
 * @file mat.hpp
 * @brief Defines the material properties class
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef POLIVEM_MATERIAL_MAT_HPP
#define POLIVEM_MATERIAL_MAT_HPP


#include<iostream>
#include<Eigen/Dense>

namespace material{
    /**
     * @class mat
     * @brief Class for defining material properties
     * 
     * This class encapsulates material properties used in structural analysis,
     * such as elastic modulus, Poisson's ratio, etc.
     */
    class mat{
        public:
            /**
             * @brief Elastic modulus
             */
            double E;

            /**
             * @brief Poisson's ratio
             */
            double nu;

            // material density
            double rho;

            // Lamé parameters
            double Mu, La;

            // set elastic module
            void setElasticModule(double elastic_module);

            // set poisson coefficient
            void setPoissonCoef(double poisson_coef);

            // set material density
            void setMaterialDensity(double material_desnity);

            // get Lamé parameters
            void getLameParameters();

            // build elasticity matrix
            Eigen::MatrixXd build2DElasticity();

            // build axisymmetric elasticity matrix
            static Eigen::MatrixXd buildAxisymmetricElasticity(double E, double nu);

            // Compute stabilization parameter τ for axisymmetric elements
            static double compute_stabilization_parameter(double E, double nu);

            /**
             * @brief Default constructor
             */
            mat() = default;

    };
}

#endif