#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef POLIVEM_UNIFORM_HPP
#define POLIVEM_UNIFORM_HPP

#include <cmath>
#include <iomanip>
#include <iostream>

#include <Eigen/Dense>

namespace mesh {

class uniform{
    public:

        uniform(bool debug = false) : debug_(debug) {}
        ~uniform() = default;

        /**
         * @brief Create n×n structured mesh on unit square (general version)
         * @param nodes Output matrix for node coordinates  
         * @param elements Output matrix for element connectivity
         * @param n Number of elements in each direction (creates n×n elements)
        */
        void create_square_nxn_mesh(Eigen::MatrixXd& nodes, Eigen::MatrixXi& elements, int n);

        double get_diameter() { return diameter_; }

    private:
        bool debug_ = false;
        double diameter_ = 0.0;
};

}

#endif
