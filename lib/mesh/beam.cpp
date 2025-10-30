#include "mesh/beam.hpp"

/**
 * Discretize a horizontal bar/beam/cable by receiving the length of 
 * the geometry and the number of desired elements. It is assumed only
 * the one-dimensional case. And the mesh is uniform.
 *
 * @param values length of the bar and number of elements in the final mesh.
 */
void mesh::beam::horizontalBarDisc(double bar_length, int num_elements){

    // initialize nodes arrays with zeros
    nodes = Eigen::MatrixXd::Zero(num_elements+1, 2);

    // initialize elements array with zeros
    elements = Eigen::MatrixXi::Zero(num_elements,2);

    // mesh increment
    double inc = double(bar_length/num_elements);
    double coord = inc;

    // initialize index controller
    int i = 1;

    while (coord <= bar_length){
        nodes(i,0) = coord;
        i += 1;
        coord += inc;
    }

    for(int i=0; i<num_elements; i++){
        elements(i,0) = i;
        elements(i,1) = i+1;
    }
}