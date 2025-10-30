import numpy as np
from pydantic import BaseModel, Field, field_serializer
from typing import List, Any, Dict

class Beam1DMeshModel(BaseModel):
    num_nodes: int = Field(..., description="Number of nodes in the mesh")
    length: float = Field(..., description="Length of the beam")
    nodes: Any = Field(..., description="Nodes of the beam")
    elements: Any = Field(..., description="Elements of the beam")
    model_order: int = Field(..., description="Order of the model")
    cross_section: str = Field(..., description="Cross-section of the beam")
    cross_section_params: Dict[str, Any] = Field(..., description="Parameters of the cross-section")
    young_modulus: float = Field(..., description="Young's modulus of the beam")
    post_processing_sample_points: int = Field(10, description="Sample points for post-processing")
    
    class Config:
        arbitrary_types_allowed = True
    
    @field_serializer('nodes', 'elements')
    def serialize_numpy(self, value: np.ndarray) -> List[Any]:
        return value.tolist()
    
class Beam1DModel(BaseModel):
    displacements: Any = Field(..., description="Displacements of the beam")
    geometry: Beam1DMeshModel = Field(..., description="Geometry of the beam")
    strain: Any = Field(..., description="Strain of the beam")
    stress: Any = Field(..., description="Stress of the beam")

    class Config:
        arbitrary_types_allowed = True
    
    @field_serializer('displacements', 'strain', 'stress')
    def serialize_numpy(self, value: np.ndarray) -> List[Any]:
        return value.tolist()