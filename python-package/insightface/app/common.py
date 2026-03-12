# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : Face data structure

from typing import Any, Dict, Optional, List, Union, Tuple
import numpy as np
from numpy.linalg import norm as l2norm

__all__ = ['Face']


class Face(dict):
    """Face data container that stores all face-related information.
    
    This class extends dict to provide both dictionary-like and attribute-like
    access to face data. It can store various face attributes such as:
    - bbox: Bounding box coordinates [x1, y1, x2, y2]
    - det_score: Detection confidence score
    - kps: Facial keypoints
    - embedding: Face feature vector
    - gender: Gender prediction (0=female, 1=male)
    - age: Age prediction
    - pose: Head pose angles [pitch, yaw, roll]
    - Various landmark predictions
    
    Attributes are dynamically set and accessed. Missing attributes return None
    instead of raising AttributeError.
    
    Example:
        >>> face = Face(bbox=np.array([100, 100, 200, 200]), det_score=0.95)
        >>> print(face.bbox)  # Attribute access
        >>> print(face['bbox'])  # Dictionary access
        >>> face.gender = 1  # Set attribute
        >>> print(face.sex)  # Computed property
    """

    def __init__(self, d: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize a Face object.
        
        Args:
            d: Dictionary of initial face attributes.
            **kwargs: Additional face attributes as keyword arguments.
        """
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute with automatic conversion of nested dicts to Face objects."""
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                    if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name: str) -> Any:
        """Get attribute, return None if not found."""
        return None

    @property
    def embedding_norm(self) -> Optional[float]:
        """Compute L2 norm of the face embedding.
        
        Returns:
            L2 norm of embedding, or None if embedding is not set.
        """
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property 
    def normed_embedding(self) -> Optional[np.ndarray]:
        """Get L2-normalized embedding vector.
        
        Returns:
            Normalized embedding vector, or None if embedding is not set.
        """
        if self.embedding is None:
            return None
        norm_val = self.embedding_norm
        if norm_val is None or norm_val == 0:
            return None
        return self.embedding / norm_val

    @property 
    def sex(self) -> Optional[str]:
        """Get gender as string representation.
        
        Returns:
            'M' for male (gender=1), 'F' for female (gender=0), or None if not set.
        """
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Face object to a regular dictionary.
        
        Returns:
            Dictionary containing all face attributes.
        """
        result = {}
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                result[key] = value.copy()
            elif isinstance(value, Face):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    v.to_dict() if isinstance(v, Face) else v 
                    for v in value
                ]
            else:
                result[key] = value
        return result
