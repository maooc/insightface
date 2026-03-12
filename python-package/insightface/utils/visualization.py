# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : Visualization utilities

import cv2
import numpy as np
from typing import List, Optional, Dict, Any

class FaceVisualizer:
    """
    Face visualization class with configurable options
    """
    def __init__(self, 
                 draw_bboxes: bool = True, 
                 draw_keypoints: bool = True, 
                 draw_attributes: bool = True,
                 bbox_color: tuple = (0, 0, 255),
                 keypoint_color: tuple = (0, 0, 255),
                 special_keypoint_color: tuple = (0, 255, 0),
                 text_color: tuple = (0, 255, 0),
                 bbox_thickness: int = 2,
                 keypoint_radius: int = 1,
                 keypoint_thickness: int = 2,
                 font_scale: float = 0.7,
                 font_thickness: int = 1):
        """
        Initialize FaceVisualizer
        
        Args:
            draw_bboxes: Whether to draw bounding boxes
            draw_keypoints: Whether to draw keypoints
            draw_attributes: Whether to draw attributes (gender, age)
            bbox_color: Bounding box color (B, G, R)
            keypoint_color: Keypoint color (B, G, R)
            special_keypoint_color: Special keypoint color (B, G, R)
            text_color: Text color (B, G, R)
            bbox_thickness: Bounding box thickness
            keypoint_radius: Keypoint radius
            keypoint_thickness: Keypoint thickness
            font_scale: Font scale
            font_thickness: Font thickness
        """
        self.draw_bboxes = draw_bboxes
        self.draw_keypoints = draw_keypoints
        self.draw_attributes = draw_attributes
        self.bbox_color = bbox_color
        self.keypoint_color = keypoint_color
        self.special_keypoint_color = special_keypoint_color
        self.text_color = text_color
        self.bbox_thickness = bbox_thickness
        self.keypoint_radius = keypoint_radius
        self.keypoint_thickness = keypoint_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
    
    def draw(self, img: np.ndarray, faces: List[Any]) -> np.ndarray:
        """
        Draw faces on the image
        
        Args:
            img: Input image (BGR format)
            faces: List of Face objects
        
        Returns:
            Image with faces drawn
        """
        dimg = img.copy()
        for i, face in enumerate(faces):
            if not hasattr(face, 'bbox') or face.bbox is None:
                continue
            
            try:
                box = face.bbox.astype(int)
                
                # Draw bounding box
                if self.draw_bboxes:
                    cv2.rectangle(dimg, 
                                 (box[0], box[1]), 
                                 (box[2], box[3]), 
                                 self.bbox_color, 
                                 self.bbox_thickness)
                
                # Draw keypoints
                if self.draw_keypoints and hasattr(face, 'kps') and face.kps is not None:
                    try:
                        kps = face.kps.astype(int)
                        for l in range(kps.shape[0]):
                            color = self.special_keypoint_color if l == 0 or l == 3 else self.keypoint_color
                            cv2.circle(dimg, 
                                       (kps[l][0], kps[l][1]), 
                                       self.keypoint_radius, 
                                       color, 
                                       self.keypoint_thickness)
                    except (ValueError, AttributeError):
                        pass
                
                # Draw attributes
                if self.draw_attributes and hasattr(face, 'gender') and face.gender is not None and hasattr(face, 'age') and face.age is not None:
                    try:
                        text = f'{face.sex},{face.age}'
                        cv2.putText(dimg, 
                                   text, 
                                   (box[0]-1, box[1]-4),
                                   cv2.FONT_HERSHEY_COMPLEX, 
                                   self.font_scale, 
                                   self.text_color, 
                                   self.font_thickness)
                    except (ValueError, AttributeError):
                        pass
                
                # Draw confidence score if available
                if hasattr(face, 'det_score') and face.det_score is not None:
                    try:
                        score_text = f'{face.det_score:.2f}'
                        cv2.putText(dimg, 
                                   score_text, 
                                   (box[0], box[3]+20),
                                   cv2.FONT_HERSHEY_COMPLEX, 
                                   self.font_scale * 0.8, 
                                   self.text_color, 
                                   self.font_thickness)
                    except (ValueError, AttributeError):
                        pass
                        
            except (ValueError, AttributeError) as e:
                # Ignore errors to ensure visualization doesn't break
                pass
        
        return dimg

# Default visualizer instance
default_visualizer = FaceVisualizer()

def draw_faces(img: np.ndarray, faces: List[Any], **kwargs) -> np.ndarray:
    """
    Draw faces on the image with default settings
    
    Args:
        img: Input image (BGR format)
        faces: List of Face objects
        **kwargs: Visualization options
    
    Returns:
        Image with faces drawn
    """
    # Create a temporary visualizer with provided options
    visualizer = FaceVisualizer(**kwargs)
    return visualizer.draw(img, faces)
