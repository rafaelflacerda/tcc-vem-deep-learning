
import numpy as np
import math
from scipy.special import comb

def compute_polygon_area_shoelace(vertices):
    """Compute polygon area using shoelace formula for verification."""
    vertices = np.array(vertices)
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0

def compute_polygon_centroid(vertices):
    """Compute polygon centroid using standard formula."""
    vertices = np.array(vertices)
    n = len(vertices)
    area = compute_polygon_area_shoelace(vertices)
    
    if abs(area) < 1e-14:
        return np.mean(vertices, axis=0)
    
    cx = cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = vertices[i, 0] * vertices[j, 1] - vertices[j, 0] * vertices[i, 1]
        cx += (vertices[i, 0] + vertices[j, 0]) * cross
        cy += (vertices[i, 1] + vertices[j, 1]) * cross
    
    factor = 1.0 / (6.0 * area)
    return np.array([cx * factor, cy * factor])

def compute_moment_Ipq(vertices, p, q, centroid=None):
    """
    Compute exact moment I_pq = ∫_E (x-x_c)^p * (y-y_c)^q dx
    using the corrected closed algebraic formula.
    
    Note: This computes the UNSCALED moment. For VEM, you need to divide by h_e^(p+q).
    """
    vertices = np.array(vertices)
    N = len(vertices)
    
    if centroid is None:
        centroid = compute_polygon_centroid(vertices)
    else:
        centroid = np.array(centroid)
    
    x_c, y_c = centroid
    total_moment = 0.0
    
    # Loop over all edges
    for r in range(N):
        r_next = (r + 1) % N
        
        # Edge vertices
        x_r, y_r = vertices[r]
        x_next, y_next = vertices[r_next]
        
        # Edge vectors
        Delta_x_r = x_next - x_r
        Delta_y_r = y_next - y_r
        
        # Shifted coordinates (relative to centroid, NOT scaled)
        A_r = x_r - x_c
        B_r = x_next - x_c  
        C_r = y_r - y_c
        D_r = y_next - y_c
        
        # First term: Delta_y_r * sum
        first_term = 0.0
        for i in range(p + 2):  # i from 0 to p+1
            for j in range(q + 1):  # j from 0 to q
                # Binomial coefficients
                binom_p1_i = comb(p + 1, i, exact=True)
                binom_q_j = comb(q, j, exact=True)
                
                # Exponents
                exp_A = p + 1 - i
                exp_B = i
                exp_C = q - j
                exp_D = j
                
                # Only proceed if exponents are non-negative
                if exp_A >= 0 and exp_C >= 0:
                    # Compute powers (handle 0^0 = 1)
                    A_power = A_r**exp_A if exp_A > 0 else 1.0
                    B_power = B_r**exp_B if exp_B > 0 else 1.0
                    C_power = C_r**exp_C if exp_C > 0 else 1.0
                    D_power = D_r**exp_D if exp_D > 0 else 1.0
                    
                    # Beta coefficient: (a! * b!) / (p+q+2)!
                    a = p + 1 - i + q - j
                    b = i + j
                    
                    if a >= 0 and b >= 0:
                        try:
                            # Direct computation for small factorials
                            if p + q + 2 <= 20:
                                beta_coeff = (math.factorial(a) * math.factorial(b)) / math.factorial(p + q + 2)
                            else:
                                # Use log-gamma for large factorials
                                log_beta = (math.lgamma(a + 1) + math.lgamma(b + 1) - math.lgamma(p + q + 3))
                                beta_coeff = math.exp(log_beta)
                        except (OverflowError, ValueError):
                            beta_coeff = 0.0
                        
                        term = binom_p1_i * binom_q_j * A_power * B_power * C_power * D_power * beta_coeff
                        first_term += term
        
        first_term *= Delta_y_r
        
        # Second term: Delta_x_r * sum
        second_term = 0.0
        for i in range(p + 1):  # i from 0 to p
            for j in range(q + 2):  # j from 0 to q+1
                # Binomial coefficients
                binom_p_i = comb(p, i, exact=True)
                binom_q1_j = comb(q + 1, j, exact=True)
                
                # Exponents
                exp_A = p - i
                exp_B = i
                exp_C = q + 1 - j
                exp_D = j
                
                # Only proceed if exponents are non-negative
                if exp_A >= 0 and exp_C >= 0:
                    # Compute powers (handle 0^0 = 1)
                    A_power = A_r**exp_A if exp_A > 0 else 1.0
                    B_power = B_r**exp_B if exp_B > 0 else 1.0
                    C_power = C_r**exp_C if exp_C > 0 else 1.0
                    D_power = D_r**exp_D if exp_D > 0 else 1.0
                    
                    # Beta coefficient: (a! * b!) / (p+q+2)!
                    a = p - i + q + 1 - j
                    b = i + j
                    
                    if a >= 0 and b >= 0:
                        try:
                            # Direct computation for small factorials
                            if p + q + 2 <= 20:
                                beta_coeff = (math.factorial(a) * math.factorial(b)) / math.factorial(p + q + 2)
                            else:
                                # Use log-gamma for large factorials
                                log_beta = (math.lgamma(a + 1) + math.lgamma(b + 1) - math.lgamma(p + q + 3))
                                beta_coeff = math.exp(log_beta)
                        except (OverflowError, ValueError):
                            beta_coeff = 0.0
                        
                        term = binom_p_i * binom_q1_j * A_power * B_power * C_power * D_power * beta_coeff
                        second_term += term
        
        second_term *= Delta_x_r
        
        # Add edge contribution
        edge_contribution = first_term - second_term
        total_moment += edge_contribution
    
    # Apply prefactor
    prefactor = 1.0 / (p + q + 2)
    result = prefactor * total_moment
    
    return result

def compute_scaled_moment_Ipq(vertices, p, q, h_e=None, centroid=None):
    """
    Compute scaled moment I_pq = ∫_E ((x-x_c)/h_e)^p * ((y-y_c)/h_e)^q dx
    This is what VEM actually needs.
    """
    # First compute unscaled moment
    unscaled = compute_moment_Ipq(vertices, p, q, centroid)
    
    # Compute h_e if not provided
    if h_e is None:
        vertices = np.array(vertices)
        n = len(vertices)
        max_dist = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(vertices[i] - vertices[j])
                max_dist = max(max_dist, dist)
        h_e = max_dist
    
    # Apply scaling: divide by h_e^(p+q)
    if p + q == 0:
        return unscaled  # No scaling needed for area
    else:
        return unscaled / (h_e**(p + q))

def test_unit_square():
    """Test on unit square with known analytical results."""
    print("=== Unit Square Test ===")
    
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    centroid = np.array([0.5, 0.5])
    
    # Test area (should be 1.0)
    area = compute_moment_Ipq(vertices, 0, 0, centroid)
    print(f"I_00 (area): {area:.6f} (expected: 1.000000)")
    
    # Test first moments (should be 0 due to symmetry)
    I_10 = compute_moment_Ipq(vertices, 1, 0, centroid)  
    I_01 = compute_moment_Ipq(vertices, 0, 1, centroid)
    print(f"I_10: {I_10:.6f} (expected: 0.000000)")
    print(f"I_01: {I_01:.6f} (expected: 0.000000)")
    
    # Test second moments 
    I_20 = compute_moment_Ipq(vertices, 2, 0, centroid)
    I_02 = compute_moment_Ipq(vertices, 0, 2, centroid)
    I_11 = compute_moment_Ipq(vertices, 1, 1, centroid)
    I_04 = compute_moment_Ipq(vertices, 0, 4, centroid)
    I_40 = compute_moment_Ipq(vertices, 4, 0, centroid)
    
    print(f"I_20: {I_20:.6f} (expected: 0.083333 = 1/12)")
    print(f"I_02: {I_02:.6f} (expected: 0.083333 = 1/12)")
    print(f"I_11: {I_11:.6f} (expected: 0.000000)")
    print(f"I_04: {I_04:.6f} (expected: 0.0125 = 1/80)")
    print(f"I_40: {I_40:.6f} (expected: 0.0125 = 1/80)")
    
    # Verify with analytical computation
    # For unit square with centroid (0.5, 0.5):
    # I_20 = ∫₀¹ ∫₀¹ (x-0.5)² dx dy = ∫₀¹ (x-0.5)² dx = 1/12
    analytical_I20 = 1.0/12.0
    print(f"Analytical I_20: {analytical_I20:.6f}")
    
    return abs(area - 1.0) < 1e-10 and abs(I_10) < 1e-10 and abs(I_01) < 1e-10 and abs(I_11) < 1e-10

if __name__ == "__main__":
    test_unit_square()