import math
from typing import Dict, Any, Tuple, Union, Optional

def calculate_moment_of_inertia(
    geometry_type: str, 
    params: Dict[str, Any]
) -> Dict[str, Union[float, Tuple[float, float]]]:
    """
    Calculate the moment of inertia for different cross-section geometries.
    
    Parameters:
        geometry_type: Type of cross-section (rectangle, circle, I-beam, etc.)
        params: Dictionary containing the geometric parameters
        
    Returns:
        Dictionary containing the calculated moment(s) of inertia:
        - For symmetric sections: {'Ix': value, 'Iy': value}
        - For asymmetric sections: {'Ix': value, 'Iy': value, 'Ixy': value}
        
    Raises:
        ValueError: If the geometry type is not supported or parameters are invalid
    """
    geometry_type = geometry_type.lower()
    
    # Validate required parameters
    _validate_parameters(geometry_type, params)
    
    if geometry_type == "rectangle":
        b = params["width"]
        h = params["height"]
        Ix = (b * h**3) / 12
        Iy = (h * b**3) / 12
        return {"Ix": Ix, "Iy": Iy}
    
    elif geometry_type == "circle":
        r = params["radius"]
        I = (math.pi * r**4) / 4
        return {"Ix": I, "Iy": I}
    
    elif geometry_type == "hollow_circle" or geometry_type == "tube":
        r_outer = params["outer_radius"]
        r_inner = params["inner_radius"]
        I = (math.pi / 4) * (r_outer**4 - r_inner**4)
        return {"Ix": I, "Iy": I}
    
    elif geometry_type == "i_beam" or geometry_type == "i_section":
        h = params["height"]
        b_f = params["flange_width"]
        t_f = params["flange_thickness"]
        t_w = params["web_thickness"]
        
        # Moment of inertia about x-axis (strong axis)
        Ix = (b_f * h**3) / 12 - (b_f - t_w) * (h - 2*t_f)**3 / 12
        
        # Moment of inertia about y-axis (weak axis)
        Iy = (2 * (b_f**3 * t_f) / 12) + (t_w**3 * (h - 2*t_f) / 12)
        
        return {"Ix": Ix, "Iy": Iy}
    
    elif geometry_type == "t_section":
        h = params["height"]
        b_f = params["flange_width"]
        t_f = params["flange_thickness"]
        t_w = params["web_thickness"]
        
        # Calculate centroid position from the bottom
        A_f = b_f * t_f  # Area of flange
        A_w = t_w * (h - t_f)  # Area of web
        A_total = A_f + A_w
        
        y_f = h - t_f/2  # Centroid of flange from bottom
        y_w = (h - t_f)/2  # Centroid of web from bottom
        
        y_c = (A_f * y_f + A_w * y_w) / A_total
        
        # Moment of inertia about x-axis through centroid
        I_f = (b_f * t_f**3) / 12 + A_f * (y_f - y_c)**2
        I_w = (t_w * (h - t_f)**3) / 12 + A_w * (y_w - y_c)**2
        Ix = I_f + I_w
        
        # Moment of inertia about y-axis through centroid
        I_f_y = (t_f * b_f**3) / 12
        I_w_y = ((h - t_f) * t_w**3) / 12
        Iy = I_f_y + I_w_y
        
        return {"Ix": Ix, "Iy": Iy}
    
    elif geometry_type == "channel" or geometry_type == "c_section":
        h = params["height"]
        b = params["flange_width"]
        t_f = params["flange_thickness"]
        t_w = params["web_thickness"]
        
        # Moment of inertia about x-axis (strong axis)
        Ix = (t_w * h**3) / 12 + (2 * b * t_f**3) / 12 + 2 * b * t_f * (h/2 - t_f/2)**2
        
        # Moment of inertia about y-axis (weak axis)
        Iy = (2 * b**3 * t_f) / 12 + (h * t_w**3) / 12
        
        return {"Ix": Ix, "Iy": Iy}
    
    elif geometry_type == "angle" or geometry_type == "l_section":
        h = params["height"]
        b = params["width"]
        t_h = params["height_thickness"]
        t_b = params["width_thickness"]
        
        # Calculate centroid position
        A_h = t_h * h
        A_b = t_b * (b - t_h)
        A_total = A_h + A_b
        
        x_c = (A_h * t_h/2 + A_b * (t_h + (b - t_h)/2)) / A_total
        y_c = (A_h * h/2 + A_b * t_b/2) / A_total
        
        # Moment of inertia about centroidal x-axis
        I_h_x = (t_h * h**3) / 12 + A_h * (h/2 - y_c)**2
        I_b_x = (b - t_h) * t_b**3 / 12 + A_b * (t_b/2 - y_c)**2
        Ix = I_h_x + I_b_x
        
        # Moment of inertia about centroidal y-axis
        I_h_y = (h * t_h**3) / 12 + A_h * (t_h/2 - x_c)**2
        I_b_y = ((b - t_h)**3 * t_b) / 12 + A_b * (t_h + (b - t_h)/2 - x_c)**2
        Iy = I_h_y + I_b_y
        
        # Product of inertia
        Ixy = A_h * (t_h/2 - x_c) * (h/2 - y_c) + A_b * ((t_h + (b - t_h)/2) - x_c) * (t_b/2 - y_c)
        
        return {"Ix": Ix, "Iy": Iy, "Ixy": Ixy}
    
    elif geometry_type == "hollow_rectangle" or geometry_type == "box_section":
        h_outer = params["outer_height"]
        b_outer = params["outer_width"]
        h_inner = params["inner_height"]
        b_inner = params["inner_width"]
        
        Ix = (b_outer * h_outer**3) / 12 - (b_inner * h_inner**3) / 12
        Iy = (h_outer * b_outer**3) / 12 - (h_inner * b_inner**3) / 12
        
        return {"Ix": Ix, "Iy": Iy}
    
    elif geometry_type == "ellipse":
        a = params["semi_major_axis"]
        b = params["semi_minor_axis"]
        
        Ix = (math.pi * a * b**3) / 4
        Iy = (math.pi * a**3 * b) / 4
        
        return {"Ix": Ix, "Iy": Iy}
    
    elif geometry_type == "triangle":
        b = params["base"]
        h = params["height"]
        
        # For a triangle with base parallel to x-axis
        Ix = (b * h**3) / 36
        Iy = (h * b**3) / 36
        
        return {"Ix": Ix, "Iy": Iy}
    
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")

def calculate_area(
    geometry_type: str, 
    params: Dict[str, Any]
) -> float:
    """
    Calculate the area for different cross-section geometries.
    
    Parameters:
        geometry_type: Type of cross-section (rectangle, circle, I-beam, etc.)
        params: Dictionary containing the geometric parameters
        
    Returns:
        The calculated cross-sectional area
        
    Raises:
        ValueError: If the geometry type is not supported or parameters are invalid
    """
    geometry_type = geometry_type.lower()
    
    # Validate required parameters
    _validate_parameters(geometry_type, params)
    
    if geometry_type == "rectangle":
        return params["width"] * params["height"]
    
    elif geometry_type == "circle":
        return math.pi * params["radius"]**2
    
    elif geometry_type == "hollow_circle" or geometry_type == "tube":
        return math.pi * (params["outer_radius"]**2 - params["inner_radius"]**2)
    
    elif geometry_type == "i_beam" or geometry_type == "i_section":
        h = params["height"]
        b_f = params["flange_width"]
        t_f = params["flange_thickness"]
        t_w = params["web_thickness"]
        
        return 2 * (b_f * t_f) + t_w * (h - 2 * t_f)
    
    elif geometry_type == "t_section":
        h = params["height"]
        b_f = params["flange_width"]
        t_f = params["flange_thickness"]
        t_w = params["web_thickness"]
        
        return b_f * t_f + t_w * (h - t_f)
    
    elif geometry_type == "channel" or geometry_type == "c_section":
        h = params["height"]
        b = params["flange_width"]
        t_f = params["flange_thickness"]
        t_w = params["web_thickness"]
        
        return t_w * h + 2 * b * t_f - t_f * t_w
    
    elif geometry_type == "angle" or geometry_type == "l_section":
        h = params["height"]
        b = params["width"]
        t_h = params["height_thickness"]
        t_b = params["width_thickness"]
        
        return t_h * h + t_b * (b - t_h)
    
    elif geometry_type == "hollow_rectangle" or geometry_type == "box_section":
        h_outer = params["outer_height"]
        b_outer = params["outer_width"]
        h_inner = params["inner_height"]
        b_inner = params["inner_width"]
        
        return b_outer * h_outer - b_inner * h_inner
    
    elif geometry_type == "ellipse":
        a = params["semi_major_axis"]
        b = params["semi_minor_axis"]
        
        return math.pi * a * b
    
    elif geometry_type == "triangle":
        b = params["base"]
        h = params["height"]
        
        return 0.5 * b * h
    
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")

def _validate_parameters(geometry_type: str, params: Dict[str, Any]) -> None:
    """
    Validate that all required parameters for a given geometry type are present.
    
    Args:
        geometry_type: Type of cross-section
        params: Dictionary containing the geometric parameters
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    required_params = {
        "rectangle": ["width", "height"],
        "circle": ["radius"],
        "hollow_circle": ["outer_radius", "inner_radius"],
        "tube": ["outer_radius", "inner_radius"],
        "i_beam": ["height", "flange_width", "flange_thickness", "web_thickness"],
        "i_section": ["height", "flange_width", "flange_thickness", "web_thickness"],
        "t_section": ["height", "flange_width", "flange_thickness", "web_thickness"],
        "channel": ["height", "flange_width", "flange_thickness", "web_thickness"],
        "c_section": ["height", "flange_width", "flange_thickness", "web_thickness"],
        "angle": ["height", "width", "height_thickness", "width_thickness"],
        "l_section": ["height", "width", "height_thickness", "width_thickness"],
        "hollow_rectangle": ["outer_height", "outer_width", "inner_height", "inner_width"],
        "box_section": ["outer_height", "outer_width", "inner_height", "inner_width"],
        "ellipse": ["semi_major_axis", "semi_minor_axis"],
        "triangle": ["base", "height"]
    }
    
    if geometry_type not in required_params:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")
    
    for param in required_params[geometry_type]:
        if param not in params:
            raise ValueError(f"Missing required parameter '{param}' for {geometry_type}")
        
        if not isinstance(params[param], (int, float)) or params[param] <= 0:
            raise ValueError(f"Parameter '{param}' must be a positive number")
    
    # Additional validation for specific geometries
    if geometry_type in ["hollow_circle", "tube"] and params["inner_radius"] >= params["outer_radius"]:
        raise ValueError("Inner radius must be less than outer radius")
    
    if geometry_type in ["hollow_rectangle", "box_section"]:
        if params["inner_height"] >= params["outer_height"] or params["inner_width"] >= params["outer_width"]:
            raise ValueError("Inner dimensions must be less than outer dimensions")

def calculate_extreme_fiber_distances(
    geometry_type: str, 
    params: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate the distances from the neutral axis to the extreme fibers of the cross-section.
    
    Parameters:
        geometry_type: Type of cross-section (rectangle, circle, I-beam, etc.)
        params: Dictionary containing the geometric parameters
        
    Returns:
        Dictionary containing the distances to extreme fibers:
        - For symmetric sections about x-axis: {'y_top': value, 'y_bottom': value}
        - For symmetric sections about y-axis: {'x_left': value, 'x_right': value}
        - For asymmetric sections, includes centroid position
        
    Raises:
        ValueError: If the geometry type is not supported or parameters are invalid
    """
    geometry_type = geometry_type.lower()
    
    # Validate required parameters
    _validate_parameters(geometry_type, params)
    
    if geometry_type == "rectangle":
        h = params["height"]
        return {"y_top": h/2, "y_bottom": -h/2, "x_left": -params["width"]/2, "x_right": params["width"]/2}
    
    elif geometry_type == "circle":
        r = params["radius"]
        return {"y_top": r, "y_bottom": -r, "x_left": -r, "x_right": r}
    
    elif geometry_type == "hollow_circle" or geometry_type == "tube":
        r_outer = params["outer_radius"]
        return {"y_top": r_outer, "y_bottom": -r_outer, "x_left": -r_outer, "x_right": r_outer}
    
    elif geometry_type == "i_beam" or geometry_type == "i_section":
        h = params["height"]
        return {"y_top": h/2, "y_bottom": -h/2, "x_left": -params["flange_width"]/2, "x_right": params["flange_width"]/2}
    
    elif geometry_type == "t_section":
        h = params["height"]
        b_f = params["flange_width"]
        t_f = params["flange_thickness"]
        t_w = params["web_thickness"]
        
        # Calculate centroid position from the bottom
        A_f = b_f * t_f  # Area of flange
        A_w = t_w * (h - t_f)  # Area of web
        A_total = A_f + A_w
        
        y_f = h - t_f/2  # Centroid of flange from bottom
        y_w = (h - t_f)/2  # Centroid of web from bottom
        
        y_c = (A_f * y_f + A_w * y_w) / A_total
        
        return {
            "centroid_y": y_c,  # Distance from bottom to centroid
            "y_top": h - y_c,   # Distance from centroid to top
            "y_bottom": -y_c,   # Distance from centroid to bottom
            "x_left": -b_f/2,   # Distance from centroid to left extreme
            "x_right": b_f/2    # Distance from centroid to right extreme
        }
    
    elif geometry_type == "channel" or geometry_type == "c_section":
        h = params["height"]
        b = params["flange_width"]
        
        # For a channel section, the centroid is not at the geometric center
        # We need to calculate it, but for simplicity, we'll assume it's symmetric about x-axis
        return {"y_top": h/2, "y_bottom": -h/2, "x_left": -b, "x_right": 0}
    
    elif geometry_type == "angle" or geometry_type == "l_section":
        h = params["height"]
        b = params["width"]
        t_h = params["height_thickness"]
        t_b = params["width_thickness"]
        
        # Calculate centroid position
        A_h = t_h * h
        A_b = t_b * (b - t_h)
        A_total = A_h + A_b
        
        x_c = (A_h * t_h/2 + A_b * (t_h + (b - t_h)/2)) / A_total
        y_c = (A_h * h/2 + A_b * t_b/2) / A_total
        
        return {
            "centroid_x": x_c,  # Distance from left edge to centroid
            "centroid_y": y_c,  # Distance from bottom to centroid
            "y_top": h - y_c,   # Distance from centroid to top
            "y_bottom": -y_c,   # Distance from centroid to bottom
            "x_left": -x_c,     # Distance from centroid to left extreme
            "x_right": b - x_c  # Distance from centroid to right extreme
        }
    
    elif geometry_type == "hollow_rectangle" or geometry_type == "box_section":
        h_outer = params["outer_height"]
        return {"y_top": h_outer/2, "y_bottom": -h_outer/2, 
                "x_left": -params["outer_width"]/2, "x_right": params["outer_width"]/2}
    
    elif geometry_type == "ellipse":
        a = params["semi_major_axis"]
        b = params["semi_minor_axis"]
        
        return {"y_top": b, "y_bottom": -b, "x_left": -a, "x_right": a}
    
    elif geometry_type == "triangle":
        h = params["height"]
        b = params["base"]
        
        # For a triangle with base at the bottom, centroid is at h/3 from the bottom
        y_c = h/3
        
        return {
            "centroid_y": y_c,    # Distance from bottom to centroid
            "y_top": h - y_c,     # Distance from centroid to top
            "y_bottom": -y_c,     # Distance from centroid to bottom
            "x_left": -b/2,       # Distance from centroid to left extreme
            "x_right": b/2        # Distance from centroid to right extreme
        }
    
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")
