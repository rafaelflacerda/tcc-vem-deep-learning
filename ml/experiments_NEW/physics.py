import numpy as np
import torch

def analytical_solution(E, I, q, x, L=1.0):
    """
    Calcula a solução analítica (Deflexão e Rotação) para viga cantilever.
    
    Args:
        E, I, q, x: Podem ser floats, numpy arrays ou torch tensors.
        L: Comprimento da viga.
        
    Returns:
        w (deflexão), theta (rotação)
    """
    # Garante consistência de sinal e formato
    # As fórmulas assumem q constante. 
    # Nota: No seu dataset q é negativo (para baixo), a fórmula abaixo
    # deve ser consistente com a convenção de sinais do seu solver.
    
    # Deflexão: -[q * x^2 / (24*E*I)] * [(6*L^2) - (4*L*x) + x^2]
    # Rotação (Derivada): -[q * x / (6*E*I)] * [(3*L^2) - (3*L*x) + x^2] 
    
    # Termo comum de rigidez
    EI = E * I
    
    # Deflexão (w)
    term_w1 = (q * (x ** 2)) / (24 * EI)
    term_w2 = (6 * (L ** 2)) - (4 * L * x) + (x ** 2)
    w = -term_w1 * term_w2
    
    # Rotação (theta)
    term_t1 = (q * x) / (6 * EI)
    term_t2 = (3 * (L ** 2)) - (3 * L * x) + (x ** 2)
    theta = -term_t1 * term_t2
    
    return w, theta