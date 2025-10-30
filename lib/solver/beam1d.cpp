#include "solver/beam1d.hpp"

void solver::beam1d::setInertiaMoment(double inertia_moment){
    I = inertia_moment;
}

void solver::beam1d::setArea(double cross_section_area){
    area = cross_section_area;
}

void solver::beam1d::setDistributedLoad(Eigen::VectorXd load_values, Eigen::MatrixXi load_indices){
    q = load_values;
    load = load_indices;
}

void solver::beam1d::setSupp(Eigen::MatrixXi dirichlet_bc){
    supp = dirichlet_bc;
}

Eigen::MatrixXd solver::beam1d::buildLocalK(Eigen::MatrixXd coord, double E){

    Eigen::MatrixXd K;
    utils::operations op;

    double EI = E * I;

    double L = op.calcLength(coord);

    if (order == 3){
        K = Eigen::MatrixXd::Zero(4,4);

        K(0,0) = 12;
        K(0,1) = 6*L;
        K(0,2) = -12;
        K(0,3) = 6*L;

        K(1,0) = K(0,1);
        K(1,1) = 4*pow(L,2);
        K(1,2) = -6*L;
        K(1,3) = 2*pow(L,2);

        K(2,0) = K(0,2);
        K(2,1) = K(1,2);
        K(2,2) = 12;
        K(2,3) = -6*L;

        K(3,0) = K(0,3);
        K(3,1) = K(1,3);
        K(3,2) = K(2,3);
        K(3,3) = 4*pow(L,2);

        K = K * EI / pow(L,3);
    }

    if(order == 4){
        K = Eigen::MatrixXd::Zero(5,5);

        K(0,0) = 192 * EI / pow(L,3);
        K(0,1) = 36 * EI / pow(L,2);
        K(0,2) = 168 * EI / pow(L,3);
        K(0,3) = - 24 * EI / pow(L,2);
        K(0,4) = -360 * EI / pow(L,3);

        K(1,0) = K(0,1);
        K(1,1) = 9 * EI / L;
        K(1,2) = 24 * EI / pow(L,2);
        K(1,3) = -3 * EI / L;
        K(1,4) = -60 * EI / pow(L,2);

        K(2,0) = K(0,2);
        K(2,1) = K(1,2);
        K(2,2) = 192 * EI / pow(L,3);
        K(2,3) = -36 * EI/ pow(L,2);
        K(2,4) = -360 * EI / pow(L,3);

        K(3,0) = K(0,3);
        K(3,1) = K(1,3);
        K(3,2) = K(2,3);
        K(3,3) = 9 * EI / L;
        K(3,4) = 60 * EI / pow(L,2);

        K(4,0) = K(0,4);
        K(4,1) = K(1,4);
        K(4,2) = K(2,4);
        K(4,3) = K(3,4);
        K(4,4) = 720 * EI / pow(L,3);
    }

    if(order == 5){
        K = Eigen::MatrixXd::Zero(6,6);

        K(0,0) = 1200*EI/pow(L,3);
        K(0,1) = 120*EI/pow(L,2);
        K(0,2) = -840*EI/pow(L,3);
        K(0,3) = 60*EI/pow(L,2);
        K(0,4) = -5400*EI/pow(L,3);
        K(0,5) = 10080*EI/pow(L,3);

        K(1,0) = K(0,1);
        K(1,1) = 16*EI/L;
        K(1,2) = -60*EI/pow(L,2);
        K(1,3) = 4*EI/L;
        K(1,4) = -480*EI/pow(L,2);
        K(1,5) = 840*EI/pow(L,2);

        K(2,0) = K(0,2);
        K(2,1) = K(1,2);
        K(2,2) = 1200*EI/pow(L,3);
        K(2,3) = -120*EI/pow(L,2);
        K(2,4) = 4680*EI/pow(L,3);
        K(2,5) = -10080*EI/pow(L,3);

        K(3,0) = K(0,3);
        K(3,1) = K(1,3);
        K(3,2) = K(2,3);
        K(3,3) = 16*EI/L;
        K(3,4) = -360*EI/pow(L,2);
        K(3,5) = 840*EI/pow(L,2);

        K(4,0) = K(0,4);
        K(4,1) = K(1,4);
        K(4,2) = K(2,4);
        K(4,3) = K(3,4);
        K(4,4) = 25920*EI/pow(L,3);
        K(4,5) = -50400*EI/pow(L,3);

        K(5,0) = K(0,5);
        K(5,1) = K(1,5);
        K(5,2) = K(2,5);
        K(5,3) = K(3,5);
        K(5,4) = K(4,5);
        K(5,5) = 100800*EI/pow(L,3);
    
    }

    // std::cout<<K<<std::endl;
    // std::cout<<"-----"<<std::endl;

    return K;
}

Eigen::MatrixXd solver::beam1d::buildLocalKPortic(Eigen::MatrixXd coord, double E){
    Eigen::MatrixXd K;
    K = Eigen::MatrixXd::Zero(7,7);

    // Setup support methods
    utils::operations op;

    // Parameters
    double EI = E * I;
    double L = op.calcLength(coord);

    if (order == 4){
        K = Eigen::MatrixXd::Zero(7,7);

        K(0,0) = E*area/L;
        K(3,0) = -E*area/L;
        K(0,3) = -E*area/L;
        K(3,3) = E*area/L;

        K(1,1) = 192 * EI / pow(L,3);
        K(1,2) = 36 * EI / pow(L,2);
        K(1,4) = 168 * EI / pow(L,3);
        K(1,5) = - 24 * EI / pow(L,2);
        K(1,6) = -360 * EI / pow(L,3);

        K(2,1) = K(1,2);
        K(2,2) = 9 * EI / L;
        K(2,4) = 24 * EI / pow(L,2);
        K(2,5) = -3 * EI / L;
        K(2,6) = -60 * EI / pow(L,2);

        K(4,1) = K(1,4);
        K(4,2) = K(2,4);
        K(4,4) = 192 * EI / pow(L,3);
        K(4,5) = -36 * EI/ pow(L,2);
        K(4,6) = -360 * EI / pow(L,3);

        K(5,1) = K(1,5);
        K(5,2) = K(2,5);
        K(5,4) = K(4,5);
        K(5,5) = 9 * EI / L;
        K(5,6) = 60 * EI / pow(L,2);

        K(6,1) = K(1,6);
        K(6,2) = K(2,6);
        K(6,4) = K(4,6);
        K(6,5) = K(5,6);
        K(6,6) = 720 * EI / pow(L,3);
        
    }

    if (order == 5){
        K = Eigen::MatrixXd::Zero(8,8);

        // Axial stiffness
        K(0,0) = E*area/L;
        K(3,0) = -E*area/L;
        K(0,3) = -E*area/L;
        K(3,3) = E*area/L;

        // Bending stiffness
        K(1,1) = 1200*EI/pow(L,3);
        K(1,2) = 120*EI/pow(L,2);
        K(1,4) = -840*EI/pow(L,3);
        K(1,5) = 60*EI/pow(L,2);
        K(1,6) = -5400*EI/pow(L,3);
        K(1,7) = 10080*EI/pow(L,3);

        K(2,1) = K(1,2);
        K(2,2) = 16*EI/L;
        K(2,4) = -60*EI/pow(L,2);
        K(2,5) = 4*EI/L;
        K(2,6) = -480*EI/pow(L,2);
        K(2,7) = 840*EI/pow(L,2);

        K(4,1) = K(1,4);
        K(4,2) = K(2,4);
        K(4,4) = 1200*EI/pow(L,3);
        K(4,5) = -120*EI/pow(L,2);
        K(4,6) = 4680*EI/pow(L,3);
        K(4,7) = -10080*EI/pow(L,3);

        K(5,1) = K(1,5);
        K(5,2) = K(2,5);
        K(5,4) = K(4,5);
        K(5,5) = 16*EI/L;
        K(5,6) = -360*EI/pow(L,2);
        K(5,7) = 840*EI/pow(L,2);

        K(6,1) = K(1,6);
        K(6,2) = K(2,6);
        K(6,4) = K(4,6);
        K(6,5) = K(5,6);
        K(6,6) = 25920*EI/pow(L,3);
        K(6,7) = -50400*EI/pow(L,3);

        K(7,1) = K(1,7);
        K(7,2) = K(2,7);
        K(7,4) = K(4,7);
        K(7,5) = K(5,7);
        K(7,6) = K(6,7);
        K(7,7) = 100800*EI/pow(L,3);
    }

    Eigen::MatrixXd Q = buildRotationOperator(coord);

    K = Q.transpose() * K * Q;

    return K;
}

Eigen::MatrixXd solver::beam1d::buildRotationOperator(Eigen::MatrixXd coord){
    Eigen::MatrixXd Q;
    utils::operations op;

    double angle = op.calcAngle(coord);

    if(order == 4){
        Q = Eigen::MatrixXd::Zero(7,7);

        Q(0,0) = cos(angle);
        Q(0,1) = sin(angle);
        Q(1,0) = -sin(angle);
        Q(1,1) = cos(angle);

        Q(2,2) = 1;

        Q(3,3) = cos(angle);
        Q(3,4) = sin(angle);
        Q(4,3) = -sin(angle);
        Q(4,4) = cos(angle);

        Q(5,5) = 1;

        Q(6,6) = 1;
    }

    if (order == 5){
        Q = Eigen::MatrixXd::Zero(8,8);

        Q(0,0) = cos(angle);
        Q(0,1) = sin(angle);
        Q(1,0) = -sin(angle);
        Q(1,1) = cos(angle);

        Q(2,2) = 1;

        Q(3,3) = cos(angle);
        Q(3,4) = sin(angle);
        Q(4,3) = -sin(angle);
        Q(4,4) = cos(angle);

        Q(5,5) = 1;

        Q(6,6) = 1;
        Q(7,7) = 1;
    }

    return Q;
}

Eigen::MatrixXd solver::beam1d::buildLocalM(Eigen::MatrixXd coord, double rho){
    Eigen::MatrixXd M;
    utils::operations op;

    double m = rho * area;

    double L = op.calcLength(coord);

    if(order == 4){
        M = Eigen::MatrixXd::Zero(5,5);

        M(0,0) = 8 * L * m / 35;
        M(0,1) = pow(L, 2) * m / 60;
        M(0,2) = -L * m / 70;
        M(0,3) = pow(L,2) * m / 210;
        M(0,4) = -3 * L * m / 14;

        M(1,0) = M(0,1);
        M(1,1) = pow(L,3) * m / 630;
        M(1,2) = -pow(L,2) * m / 210;
        M(1,3) = pow(L,3) * m / 1260;
        M(1,4) = -pow(L,2) * m / 84;

        M(2,0) = M(0,2);
        M(2,1) = M(1,2);
        M(2,2) = 8 * L * m / 35;
        M(2,3) = -pow(L,2) * m / 60;
        M(2,4) = -3 * L * m / 14;

        M(3,0) = M(0,3);
        M(3,1) = M(1,3);
        M(3,2) = M(2,3);
        M(3,3) = pow(L,3) * m / 630;
        M(3,4) = pow(L,2) * m /84;

        M(4,0) = M(0,4);
        M(4,1) = M(1,4);
        M(4,2) = M(2,4);
        M(4,3) = M(3,4);
        M(4,4) = 10 * L * m / 7;
    }

    return M;
}

Eigen::MatrixXd solver::beam1d::buildGlobalK(double E, BeamSolverType type){
    int ne = elements.rows();
    utils::operations op;
    // Number of moments
    int nm;
    if (type == BeamSolverType::Portic){
        int ndof = 3*nodes.rows() + ne * (order - 3);
        Eigen::MatrixXd K = Eigen::MatrixXd::Zero(ndof, ndof);
        std::cout << "Number of degrees of freedom: " << ndof << std::endl;
        int control = ne * (order - 3);

        // degrees of freedom
        Eigen::VectorXi e_dofs;

        for(int i=0; i<ne; i++){
            Eigen::MatrixXi e = elements.row(i);
            int momentInd = ndof - control;
            if(order == 4){
                nm = 1;
                e_dofs = op.getOrder2Indices(e, momentInd, type);
            } else if(order == 5){
                nm = 2;
                e_dofs = op.getOrder5Indices(e, momentInd, type);
            }
            
            Eigen::MatrixXd coord = op.getCooridanteBeam(e, nodes);
            Eigen::MatrixXd Kloc = buildLocalKPortic(coord, E);
            K = op.assembleMatrix(K, Kloc, e_dofs);
            control-=nm;
        }
        return K;
    }
    int ndof = 2*nodes.rows() + ne * (order - 3);
    std::cout << "Number of degrees of freedom: " << ndof << std::endl;
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(ndof, ndof);
    int control = ne * (order - 3);

    // degrees of freedom
    Eigen::VectorXi e_dofs;

    for(int i=0; i<ne; i++){
        Eigen::MatrixXi e = elements.row(i);
        int momentInd = ndof - control;
        if(order == 3 ){
            nm = 0;
            e_dofs = op.getOrder1Indices(e);
        } else if(order == 4){
            nm = 1;
            e_dofs = op.getOrder2Indices(e, momentInd);;
        } else if(order == 5){
            nm = 2;
            e_dofs = op.getOrder5Indices(e, momentInd);;
        }
        
        Eigen::MatrixXd coord = op.getCooridanteBeam(e, nodes);
        // Eigen::MatrixXd Kloc = buildLocalK(coord, E);
        Eigen::MatrixXd Kloc = buildLocalK(coord, E);
        K = op.assembleMatrix(K, Kloc, e_dofs);
        control-=nm;
    }


    // std::cout << K << std::endl;
    // std::cout << "-----" << std::endl;

    return K;
}

Eigen::MatrixXd solver::beam1d::buildGlobalM(double rho){
    int ne = elements.rows();
    int ndof = 2*nodes.rows() + ne;
    std::cout << "Number of degrees of freedom: " << ndof << std::endl;
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(ndof, ndof);
    int control = ne;

    utils::operations op;

    for(int i=0; i<ne; i++){
        Eigen::MatrixXi e = elements.row(i);
        int momentInd = ndof - control;
        Eigen::VectorXi e_dofs = op.getOrder2Indices(e, momentInd);
        Eigen::MatrixXd coord = op.getCooridanteBeam(e, nodes);
        Eigen::MatrixXd Mloc = buildLocalK(coord, rho);
        M = op.assembleMatrix(M, Mloc, e_dofs);
        control-=1;
    }

    // std::cout << M << std::endl;
    // std::cout << "-----" << std::endl;

    return M;
}

Eigen::MatrixXd solver::beam1d::buildStaticCondensation(Eigen::MatrixXd K, std::string sc_type, BeamSolverType type){
    int ne = elements.rows();
    int ndof_ = 2*nodes.rows(); // does not include the moments
    std::cout << "ndof_: " << ndof_ << std::endl;
    if(type == BeamSolverType::Portic){
        // std::cout << "ndof_: " << ndof_ << std::endl;
        ndof_ = 3*nodes.rows();
    }
    Eigen::MatrixXd K_; 
    if(sc_type.compare("KII")==0 || sc_type.compare("MII")==0){
        K_ = K(Eigen::seqN(0,ndof_), Eigen::seqN(0,ndof_));
    } else if(sc_type.compare("KIM")==0 || sc_type.compare("MIM")==0){
        K_ = K(Eigen::seqN(0,ndof_), Eigen::seq(ndof_,ndof_+ne * (order - 3)-1));
    } else if(sc_type.compare("KMI")==0 || sc_type.compare("MMI")==0){
        K_ = K(Eigen::seq(ndof_,ndof_+ne * (order - 3)-1), Eigen::seqN(0,ndof_));
    } else if(sc_type.compare("KMM")==0 || sc_type.compare("MMM")==0){
        K_ = K(Eigen::seq(ndof_,ndof_+ne * (order - 3)-1), Eigen::seq(ndof_,ndof_+ne * (order - 3)-1));
    }
    
    return K_;
}

Eigen::VectorXd solver::beam1d::buildLocalDistributedLoad(Eigen::MatrixXd coord){
    Eigen::VectorXd fb;
    utils::operations op;

    double L = op.calcLength(coord);

    if (order == 3){
        fb = Eigen::VectorXd::Zero(4);

        fb(0) = L*q(0)/2;
        fb(1) = pow(L,2)*q(0)/12;
        fb(2) = L*q(1)/2;
        fb(3) = -pow(L,2)*q(1)/12;

    }

    if(order == 4){
        fb = Eigen::VectorXd::Zero(5);
        
        fb(0) = (L*q(0)-L*q(1))/10;
        fb(1) = (pow(L,2)*q(0)-pow(L,2)*q(1))/120;
        fb(2) = (-L*q(0)+L*q(1))/10;
        fb(3) = (pow(L,2)*q(0)-pow(L,2)*q(1))/120;
        fb(4) = (L*q(0)+L*q(1))/2;
        
    }

    if(order == 5){
        fb = Eigen::VectorXd(6);

        fb(4) = L*q(0);
        fb(5) = -L*q(0) + L*q(1);
    }

    // std::cout << fb << std::endl;
    // std::cout << "-----" << std::endl;
    return fb;
}

Eigen::VectorXd solver::beam1d::buildLocalDistributedLoadPortic(Eigen::MatrixXd coord){
    Eigen::VectorXd fb;
    utils::operations op;

    double L = op.calcLength(coord);

    if(order == 4){
        fb = Eigen::VectorXd::Zero(7);
        
        fb(0) = 0;
        fb(1) = (L*q(0)-L*q(1))/10;
        fb(2) = (pow(L,2)*q(0)-pow(L,2)*q(1))/120;
        fb(3) = 0;
        fb(4) = (-L*q(0)+L*q(1))/10;
        fb(5) = (pow(L,2)*q(0)-pow(L,2)*q(1))/120;
        fb(6) = (L*q(0)+L*q(1))/2;

        Eigen::MatrixXd Q = buildRotationOperator(coord);
        fb = Q.transpose() * fb;
    }

    if(order == 5){
        fb = Eigen::VectorXd::Zero(8);
        
        fb(0) = 0;
        fb(1) = (L*q(0)-L*q(1))/10;
        fb(2) = (pow(L,2)*q(0)-pow(L,2)*q(1))/120;
        fb(3) = 0;
        fb(4) = (-L*q(0)+L*q(1))/10;
        fb(5) = (pow(L,2)*q(0)-pow(L,2)*q(1))/120;
        fb(6) = L*q(0);
        fb(7) = -L*q(0) + L*q(1);

        Eigen::MatrixXd Q = buildRotationOperator(coord);
        fb = Q.transpose() * fb;
    }

    // std::cout << fb << std::endl;
    // std::cout << "-----" << std::endl;
    return fb;
}

Eigen::VectorXd solver::beam1d::buildGlobalDistributedLoad(BeamSolverType type){
    if(type == BeamSolverType::Portic){
        std::cout << "Portic Load" << std::endl;
        int ne = elements.rows();
        int ndof = 3*nodes.rows() + ne * (order - 3);
        std::cout << "Number of degrees of freedom: " << ndof << std::endl;
        std::cout << "Number of elements: " << ne << std::endl;
        int control = ne * (order - 3);
        int load_size = load.rows();

        Eigen::VectorXd fb = Eigen::VectorXd::Zero(ndof);

        utils::operations op;

        // number of moments
        int nm;

        // degrees of freedom
        Eigen::VectorXi e_dofs;
        
        std::cout << "Elements rows: " << elements.rows() << std::endl;
        std::cout << "Load rows: " << load.rows() << std::endl;

        for(int i=0; i<load_size; i++){
            Eigen::MatrixXi e = load.row(i);
            int momentInd = ndof - control;
            if(order == 4){
                nm = 1;
                e_dofs = op.getOrder2Indices(e, momentInd, type);
            } else if(order == 5){
                nm = 2;
                e_dofs = op.getOrder5Indices(e, momentInd, type);
            }
            Eigen::MatrixXd coord = op.getCooridanteBeam(e, nodes);
            Eigen::VectorXd floc = buildLocalDistributedLoadPortic(coord);
            fb = op.assembleVector(fb, floc, e_dofs);
            control-=nm;
        }

        // std::cout << fb << std::endl;
        // std::cout << "-----" << std::endl;

        return fb;
    }
    int ne = elements.rows();
    int ndof = 2*nodes.rows() + ne * (order - 3);
    int control = ne * (order - 3);

    Eigen::VectorXd fb = Eigen::VectorXd::Zero(ndof);

    utils::operations op;

    // number of moments
    int nm;

    // degrees of freedom
    Eigen::VectorXi e_dofs;

    for(int i=0; i<ne; i++){
        Eigen::MatrixXi e = load.row(i);
        int momentInd = ndof - control;

        if(order == 3){
            nm = 0;
            e_dofs = op.getOrder1Indices(e);
        } else if(order == 4){
            nm = 1;
            e_dofs = op.getOrder2Indices(e, momentInd);;
        } else if(order == 5){
            nm = 2;
            e_dofs = op.getOrder5Indices(e, momentInd);;
        }
        Eigen::MatrixXd coord = op.getCooridanteBeam(e, nodes);
        Eigen::VectorXd floc = buildLocalDistributedLoad(coord);
        fb = op.assembleVector(fb, floc, e_dofs);
        control-=nm;
    }

    // std::cout << fb << std::endl;
    // std::cout << "-----" << std::endl;

    return fb;
}

Eigen::VectorXd solver::beam1d::buildStaticDistVector(Eigen::VectorXd fb, std::string sc_type, BeamSolverType type){
    int ne = elements.rows();
    int ndof_ = 2*nodes.rows();
    if(type == BeamSolverType::Portic){
        ndof_ = 3*nodes.rows();
    }
    Eigen::VectorXd R_;
    if(sc_type=="RI"){
        R_ = fb(Eigen::seqN(0, ndof_));
    } else if(sc_type=="RM"){
        R_ = fb(Eigen::seq(ndof_, ndof_+ne * (order - 3)-1));
    }

    return R_;
}

Eigen::MatrixXd solver::beam1d::applyDBCMatrix(Eigen::MatrixXd K){
    int n_supp = supp.rows();
    int ind; // dof index
    
    std::cout << "SUPP :: " << supp(0,1) << std::endl;
    
    for(int i=0; i<n_supp; i++){
        if(supp(i, 1) == 1){
            ind = 3*supp(i,0);
            std::cout << "Ind: " << ind << std::endl;
            K.row(ind).setZero();
            K.col(ind).setZero();
            K(ind,ind) = 1;
        }

        if(supp(i,2) == 1){
            ind = 3*supp(i,0) + 1;
            std::cout << "Ind: " << ind << std::endl;
            K.row(ind).setZero();
            K.col(ind).setZero();
            K(ind,ind) = 1;
        }

        if(supp(i,3) == 1){
            ind = 3*supp(i,0) + 2;
            std::cout << "Ind: " << ind << std::endl;
            K.row(ind).setZero();
            K.col(ind).setZero();
            K(ind,ind) = 1;
        }
    }
    
    return K;
}

Eigen::VectorXd solver::beam1d::applyDBCVec(Eigen::VectorXd R){
    int n_supp = supp.rows();
    int ind; // dof index
    for(int i=0; i<n_supp; i++){
        if(supp(i, 1) == 1){
            ind = 3*supp(i,0);
            R(ind) = 0;
        }

        if(supp(i,2) == 1){
            ind = 3*supp(i,0) + 1;
            R(ind) = 0;
        }

        if(supp(i,3) == 1){
            ind = 3*supp(i,0) + 2;
            R(ind) = 0;
        }
    }
    return R;
}

Eigen::MatrixXd solver::beam1d::calculateStrain(const Eigen::VectorXd& u, double E, int sample_points, double y_top) {
    int ne = elements.rows();
    utils::operations op;
    
    // Initialize output matrix: [element_id, x_local, strain]
    Eigen::MatrixXd strain_results(ne * sample_points, 3);

    int control = ne * (order - 3);
    int ndof = 2 * nodes.rows() + ne * (order - 3);
    
    // For each element
    for (int i = 0; i < ne; i++) {
        Eigen::MatrixXi e = elements.row(i);
        Eigen::VectorXi e_dofs;
        int momentInd = ndof - control;
        
        // Get element DOFs based on order
        if (order == 3) {
            e_dofs = op.getOrder1Indices(e);
        } else if (order == 4) {
            e_dofs = op.getOrder2Indices(e, momentInd);
        } else if (order == 5) {
            e_dofs = op.getOrder5Indices(e, momentInd);
        }
        
        // Extract element displacement vector
        Eigen::VectorXd u_elem(e_dofs.size());
        for (int j = 0; j < e_dofs.size(); j++) {
            u_elem(j) = u(e_dofs(j));
        }
        
        // Get element coordinates
        Eigen::MatrixXd coord = op.getCooridanteBeam(e, nodes);
        double L = op.calcLength(coord);
        
        // Calculate strains at sample points along the element
        for (int j = 0; j < sample_points; j++) {
            double xi = j * L / (sample_points - 1); // Local coordinate (0 to L)
            double strain = calculateElementStrain(u_elem, xi, L, y_top);
            
            // Store results: [element_id, x_local, strain]
            strain_results(i * sample_points + j, 0) = i;
            strain_results(i * sample_points + j, 1) = xi;
            strain_results(i * sample_points + j, 2) = strain;
        }
        
        control -= (order - 3);
    }
    
    return strain_results;
}

Eigen::MatrixXd solver::beam1d::calculateStress(const Eigen::VectorXd& u, double E, int sample_points, double y_top) {
    // Get strain results
    Eigen::MatrixXd strain_results = calculateStrain(u, E, sample_points, y_top);
    
    // Convert strain to stress using Hooke's law: σ = E * ε
    Eigen::MatrixXd stress_results = strain_results;
    stress_results.col(2) = E * strain_results.col(2);
    
    return stress_results;
}

double solver::beam1d::calculateElementStrain(const Eigen::VectorXd& u_elem, double x, double L, double y) {
    // Normalize coordinate to range [0,1] for shape functions
    double xi = x / L;
    
    // Calculate second derivative of displacement (curvature)
    double curvature = 0.0;
    
    // For standard Euler-Bernoulli beam elements (order 3)
    if (order == 3) {
        // Hermite cubic shape functions' second derivatives
        double d2N1_dx2 = 6 * (1 - 2*xi) / (L*L);
        double d2N2_dx2 = (6*xi - 4) / L;
        double d2N3_dx2 = 6 * (2*xi - 1) / (L*L);
        double d2N4_dx2 = (6*xi - 2) / L;
        
        // w = N1*u1 + N2*theta1 + N3*u2 + N4*theta2
        // where u1,u2 are displacements and theta1,theta2 are rotations
        curvature = d2N1_dx2 * u_elem(0) + d2N2_dx2 * u_elem(1) + 
                    d2N3_dx2 * u_elem(2) + d2N4_dx2 * u_elem(3);
    }
    // For higher order elements (order 4)
    else if (order == 4) {
        // For order 4, we have additional internal moment DOF
        // Implement shape functions' second derivatives for order 4
        double d2N1_dx2 = 12 * (10*xi*xi - 10*xi + 1) / (L*L);
        double d2N2_dx2 = 6 * (10*xi*xi - 8*xi + 1) / L;
        double d2N3_dx2 = 12 * (10*xi*xi - 10*xi + 1) / (L*L);
        double d2N4_dx2 = 6 * (10*xi*xi - 12*xi + 3) / L;
        double d2N5_dx2 = 30 * (2*xi - 1) / (L*L);
        
        curvature = d2N1_dx2 * u_elem(0) + d2N2_dx2 * u_elem(1) + 
                    d2N3_dx2 * u_elem(2) + d2N4_dx2 * u_elem(3) +
                    d2N5_dx2 * u_elem(4);
    }
    // For order 5 elements
    else if (order == 5) {
        // Implement shape functions' second derivatives for order 5
        // These would need to be derived based on your VEM implementation
        // Placeholder implementation:
        double d2N1_dx2 = 30 * (1 - 3*xi + 2*xi*xi) / (L*L);
        double d2N2_dx2 = 6 * (5 - 30*xi + 30*xi*xi) / L;
        double d2N3_dx2 = 30 * (1 - 3*xi + 2*xi*xi) / (L*L);
        double d2N4_dx2 = 6 * (5 - 30*xi + 30*xi*xi) / L;
        double d2N5_dx2 = 60 * (3*xi - 1) / (L*L);
        double d2N6_dx2 = 60 * (3*xi - 2) / (L*L);
        
        curvature = d2N1_dx2 * u_elem(0) + d2N2_dx2 * u_elem(1) + 
                    d2N3_dx2 * u_elem(2) + d2N4_dx2 * u_elem(3) +
                    d2N5_dx2 * u_elem(4) + d2N6_dx2 * u_elem(5);
    }
    
    // Strain is: ε = -y * curvature
    return -y * curvature;
}

Eigen::MatrixXd solver::beam1d::calculateMaxStress(const Eigen::VectorXd& u, double E, double height, int sample_points) {
    // Calculate stress at the top/bottom fiber (y = height/2)
    double y_top = height / 2.0;
    return calculateStress(u, E, sample_points, y_top);
}

std::pair<double, double> solver::beam1d::getStrainStressAtPoint(const Eigen::VectorXd& u, double E, double x_global, double y) {
    // Find which element contains the global coordinate
    int ne = elements.rows();
    utils::operations op;
    
    int element_id = -1;
    double x_local = 0.0;
    
    for (int i = 0; i < ne; i++) {
        Eigen::MatrixXi e = elements.row(i);
        Eigen::MatrixXd coord = op.getCooridanteBeam(e, nodes);
        
        double x_start = coord(0, 0);
        double x_end = coord(1, 0);
        
        // Check if x_global is within this element
        if (x_global >= x_start && x_global <= x_end) {
            element_id = i;
            x_local = x_global - x_start;
            break;
        }
    }
    
    if (element_id == -1) {
        // Point not found in any element
        return std::make_pair(0.0, 0.0);
    }
    
    // Get element displacement vector
    int ndof = 2 * nodes.rows() + ne * (order - 3);
    int control = ne * (order - 3) - element_id * (order - 3);
    int momentInd = ndof - control;
    
    Eigen::MatrixXi e = elements.row(element_id);
    Eigen::VectorXi e_dofs;
    
    if (order == 3) {
        e_dofs = op.getOrder1Indices(e);
    } else if (order == 4) {
        e_dofs = op.getOrder2Indices(e, momentInd);
    } else if (order == 5) {
        e_dofs = op.getOrder5Indices(e, momentInd);
    }
    
    Eigen::VectorXd u_elem(e_dofs.size());
    for (int j = 0; j < e_dofs.size(); j++) {
        u_elem(j) = u(e_dofs(j));
    }
    
    // Get element length
    Eigen::MatrixXd coord = op.getCooridanteBeam(e, nodes);
    double L = op.calcLength(coord);
    
    // Calculate strain and stress
    double strain = calculateElementStrain(u_elem, x_local, L, y);
    double stress = E * strain;
    
    return std::make_pair(strain, stress);
}


Eigen::MatrixXd solver::beam1d::condense_matrix(Eigen::MatrixXd& KII, Eigen::MatrixXd& KIM, Eigen::MatrixXd& KMI, Eigen::MatrixXd& KMM){
    Eigen::MatrixXd K_;
    K_ = KII - KIM * KMM.inverse() * KMI;
    return K_;
}

Eigen::VectorXd solver::beam1d::condense_vector(Eigen::VectorXd& RI, Eigen::VectorXd& RM, Eigen::MatrixXd& KIM, Eigen::MatrixXd& KMM){
    Eigen::VectorXd R_;
    R_ = RI - KIM * KMM.inverse() * RM;
    return R_;
}