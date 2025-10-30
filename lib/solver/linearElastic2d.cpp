#include "solver/linearElastic2d.hpp"

/**
 * Set the restriced nodes array.
 * Eigen::MatrixXi supp = (Eigen::Matrix<int, 1,3>()<<node_index,restriction_x,restriction_y).finished();
 * retriction_x -> 1 if it the displacement is restricted in horizontal direction; 0 otherwise.
 * retriction_y -> 1 if it the displacement is restricted in vertical; 0 otherwise.
 * 
 * @param MatrixXi information about the restricted nodes
 * @return None.
 */
void solver::linearElastic2d::setSupp(Eigen::MatrixXi dirichlet_bc){
    supp = dirichlet_bc;
}

/**
 * Set the edges where the distributed load is applied.
 * Eigen::MatrixXi load = (Eigen::Matrix<int, 1,2>()<<startNode,endNode).finished()
 * 
 * @param MatrixXi nodes forming an edge where the distributed load is applied.
 * @return None.
 */
void solver::linearElastic2d::setLoad(Eigen::MatrixXi load_indices){
    load = load_indices;
}

/**
 * Build the Np matrix.
 * 
 * @param None
 * @return Np matrix.
 */
Eigen::MatrixXd solver::linearElastic2d::buildNp(){
    Eigen::MatrixXd Np;
    if(order==1){
        Np = Eigen::Matrix<double,3,3>::Identity();
    }
    return Np;
}


/**
 * Build the normal vector matrix NE. This matrix contains the components of the normal vector
 * regarding an edge.
 * 
 * @param MatrixXd coord: coordinates of an edge.
 * @return NE matrix.
 */
Eigen::MatrixXd solver::linearElastic2d::buildNE(Eigen::MatrixXd coord){
    Eigen::MatrixXd NE;

    utils::operations op;

    if(order==1){
        Eigen::Vector2d n = op.computerNormalVector(coord);
        NE = Eigen::MatrixXd::Zero(2,3);
        NE(0,0) = n(0);
        NE(0,2) = n(1);
        NE(1,1) = n(1);
        NE(1,2) = n(0);
        // std::cout << n << std::endl;
        // std::cout << "-------" << std::endl;
    }
    return NE;
}

/**
 * Build auxiliary Nv matrix for the linear case. This matrix is responsible to organize the contributions
 * of each node in the correct position (similar to the interpolation matrix in the FEM).
 * 
 * @param int startIndex: start node index of the edge.
 * @param int endIndex: end node index of the edge.
 * @return positioning auxiliary matrix
 */
Eigen::MatrixXd solver::linearElastic2d::buildNv(int startIndex, int endIndex){
    Eigen::MatrixXd Nv;
    if(order==1){
        Nv = Eigen::MatrixXd::Zero(2,2*elements.cols());
        Nv(0,2*startIndex) = 1;
        Nv(1,2*startIndex+1) = 1;
        Nv(0,2*endIndex) = 1;
        Nv(1,2*endIndex+1)=1;
    }
    return Nv;
}

/**
 * Build the G matrix for the consistency term regarding the implmentation order.
 * 
 * @param MatrixXd coords: coordinates of the element.
 * @return G matrix.
 */
Eigen::MatrixXd solver::linearElastic2d::buildG(Eigen::MatrixXd coords){
    Eigen::MatrixXd G;
    utils::operations op;
    if(order==1){
        double area = op.calcArea(coords);
        Eigen::MatrixXd Np = buildNp();
        G = area*Np.transpose()*Np;
    }
    return G;
}

/**
 * Build the B matrix for the consistency term regarding the implmentation order.
 * The matrix is build considering each edge of an element.
 * 
 * @param MatrixXd coords: coordinates of the element.
 * @return B matrix.
 */
Eigen::MatrixXd solver::linearElastic2d::buildB(Eigen::MatrixXd coords){
    Eigen::MatrixXd B;
    utils::operations op;
    Eigen::MatrixXd Np = buildNp();
    Eigen::MatrixXd Nv;
    // extended coordinate vector (last row consists of the first coordinate)
    Eigen::MatrixXd extCoords(coords.rows()+1, coords.cols());
    extCoords << coords, coords.row(0);
    double w = 1;
    if(order==1){
        B = Eigen::MatrixXd::Zero(3, 2*elements.cols());
        for(int i = 0; i<coords.rows(); i++){
            int startIndex = i;
            int endIndex = i + 1;
            Eigen::MatrixXd startNode = extCoords.row(startIndex);
            Eigen::MatrixXd endNode = extCoords.row(endIndex);
            Eigen::MatrixXd edge = op.buildEdge(startNode, endNode);
            
            double length = op.calcLength(edge);
            Eigen::MatrixXd NE = buildNE(edge);
            // std::cout << edge << std::endl;
            // std::cout << "-------" << std::endl;
            if(i==coords.rows()-1){
                Nv = buildNv(coords.rows()-1,0);
            } else {
                Nv = buildNv(startIndex, endIndex);
            }
            B += 0.5*length*w*(NE*Np).transpose()*Nv;
        }
    }
    return B;
}

/**
 * Build the D matrix associated to the stability matrix. Columns are related to interpolation
 * function and rows are related to the degrees of freedom of each node.
 * 
 * @param MatrixXd coords: coordinates of the element.
 * @return D matrix.
 */
Eigen::MatrixXd solver::linearElastic2d::buildD(Eigen::MatrixXd coords){
    Eigen::MatrixXd D;
    utils::operations op;
    if(order==1){
        D = Eigen::MatrixXd::Zero(2*elements.cols(),6);
        Eigen::Vector2d c = op.calcCentroid(coords);
        double h = op.calcPolygonalDiam(coords, elements.cols());
        Eigen::Vector2d xi;
        int control = 0;
        for(int i=0;i<2*coords.rows()-1;i++){
            if(i!=0){
                i++;
            }
            xi = op.computeScaledCoord(coords.row(control), c, h);
            D(i,0) = 1;
            D(i,2) = xi(0);
            D(i,4) = xi(1);
            D(i+1,1) = 1;
            D(i+1,3) = xi(0);
            D(i+1,5) = xi(1);
            control++;
        }
    }
    return D;
}

/**
 * Build the consistency matrix.
 * 
 * @param MatrixXd coords: coordinates of the element.
 * @param MatrixXd C: material matrix.
 * @return Kc matrix.
 */
Eigen::MatrixXd solver::linearElastic2d::buildConsistency(Eigen::MatrixXd coords, Eigen::MatrixXd C){
    Eigen::MatrixXd Kc;
    Eigen::MatrixXd B = buildB(coords);
    Eigen::MatrixXd G = buildG(coords);
    Eigen::MatrixXd Np = buildNp();

    utils::operations op;

    double area = op.calcArea(coords);

    Kc = area * B.transpose()*G.inverse().transpose()*(Np.transpose()*C*Np)*G.inverse()*B;

    return Kc;
}

/**
 * Build the consistency matrix.
 * 
 * @param MatrixXd coords: coordinates of the element.
 * @param MatrixXd Kc: consistency matrix
 * @return Ks matrix.
 */
Eigen::MatrixXd solver::linearElastic2d::buildStability(Eigen::MatrixXd coords, Eigen::MatrixXd Kc){
    double tau = 0.5;
    int rowsKc = Kc.rows(), colsKc = Kc.cols();
    Eigen::MatrixXd Ks;
    Eigen::MatrixXd D = buildD(coords);
    Eigen::MatrixXd invDD =(D.transpose()*D).inverse();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(rowsKc, colsKc);
    Ks = tau*Kc.trace()*(I-D*invDD*D.transpose());

    return Ks;
}

/**
 * Build the local stiffness matrix.
 * 
 * @param MatrixXd coords: coordinates of the element.
 * @param MatrixXd C: material matrix.
 * @return Kloc matrix.
 */
Eigen::MatrixXd solver::linearElastic2d::buildLocalK(Eigen::MatrixXd coords, Eigen::MatrixXd C){
    Eigen::MatrixXd Kc = buildConsistency(coords, C);
    Eigen::MatrixXd Ks = buildStability(coords, Kc);
    return Kc + Ks;
}

/**
 * Assemble the global stiffness matrix
 * 
 * @param MatrixXd C: material matrix.
 * @return K matrix.
 */
Eigen::MatrixXd solver::linearElastic2d::buildGlobalK(Eigen::MatrixXd C){
    // number of degrees of freedom and number of elements
    int ndof = 2*nodes.rows();
    int ne = elements.rows();

    std::cout << "Number of dofs: " << ndof << std::endl;
    std::cout << "Number of elements: " << ne << std::endl;

    // stiffness matrices
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(ndof,ndof);
    Eigen::MatrixXd Kloc;
    
    // elements and coordinates
    Eigen::MatrixXi e; // store individual element for calculations
    Eigen::VectorXi e_dofs;
    Eigen::MatrixXd coords;

    utils::operations op;


    for(int i=0; i<ne; i++){
        e = elements.row(i);
        // std::cout << "STEP 1" << std::endl;
        e_dofs = op.getOrder1Indices(e);
        // std::cout << "STEP 2" << std::endl;
        coords = op.getCoordinatesPlane(e, nodes);
        // std::cout << "STEP 3" << std::endl;
        Kloc = buildLocalK(coords, C);
        // std::cout << "STEP 4" << std::endl;
        K = op.assembleMatrix(K, Kloc, e_dofs);
        // std::cout << i <<  std::endl;
    }
    std::cout << "Global stiffness matrix assembled" << std::endl;
    return K;
}


/**
 * Apply Dirichlet boundary conditions in the specified nodes.
 * 
 * @param MatrixXd K: global stiffness matrix.
 * @return modified K matrix.
 */
Eigen::MatrixXd solver::linearElastic2d::applyDBC(Eigen::MatrixXd K){
    int n_supp = supp.rows();
    int index; // dof index

    for(int i=0; i <n_supp; i++){
        if(supp(i,1)==1){
            index = 2*supp(i,0);
            K.row(index).setZero();
            K.col(index).setZero();
            K(index,index) = 1;
        }

        if(supp(i,2)==1){
            index = 2*supp(i,0)+1;
            K.row(index).setZero();
            K.col(index).setZero();
            K(index,index) = 1;
        }
        
    }
    return K;
}

/**
 * Apply the Neumann boundary conditions (especifically the distributed load).
 * 
 * @param double qx: distributed horizontal load.
 * @param double qy: distributed vertical load.
 * @return load vector f.
 */
Eigen::VectorXd solver::linearElastic2d::applyNBC(double qx, double qy){
    int n = load.rows();
    int ndof = 2*nodes.rows(); // number of degrees of freedom
    int startIndex, endIndex;
    double length;
    Eigen::MatrixXd startNode, endNode;
    Eigen::Vector4d floc;
    Eigen::VectorXd f = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXi e_dofs;
    Eigen::MatrixXi e;
    Eigen::MatrixXd edge;
    utils::operations op;

    for(int i=0; i<n; i++){
        e = load.row(i);
        startIndex = load(i, 0);
        endIndex = load(i, 1);
        startNode = nodes(startIndex, Eigen::all);
        endNode = nodes(endIndex, Eigen::all);
        edge = op.buildEdge(startNode, endNode);
        length = op.calcLength(edge);
        floc = (Eigen::Matrix<double, 4, 1>()<<qx*length/2.0, qy*length/2.0, qx*length/2.0, qy*length/2.0).finished();
        e_dofs = op.getOrder1Indices(e);
        f = op.assembleVector(f, floc, e_dofs);
    }
    return f;
}