/**
 * @file enums.hpp
 * @brief Defines enumerations used throughout the library
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#ifndef POLIVEM_MODELS_ENUMS_HPP
#define POLIVEM_MODELS_ENUMS_HPP

/**
 * @enum BeamSolverType
 * @brief Enumeration of beam solver types
 */
enum class BeamSolverType {
    /**
     * @brief Simple beam solver
     */
    Beam,
    
    /**
     * @brief Portic (frame) solver
     */
    Portic
};

/**
 * @enum MeshType
 * @brief Enumeration of mesh types
 */
enum class MeshType {
    /**
     * @brief Triangular mesh
     */
    Triangular,
    
    /**
     * @brief Voronoi mesh
     */
    Voronoi,
    
    /**
     * @brief Quadrilateral mesh
     */
    Quad
};

// Add other enumerations here with similar documentation

#endif // POLIVEM_MODELS_ENUMS_HPP