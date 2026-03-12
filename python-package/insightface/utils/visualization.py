# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : Visualization utilities for face analysis results

from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class FaceVisualizer:
    """Visualizer for face detection and analysis results.
    
    This class provides methods to draw face detection results on images,
    including bounding boxes, keypoints, and attributes.
    
    Attributes:
        bbox_color: Color for bounding boxes in BGR format. Default is red (0, 0, 255).
        keypoint_color: Default color for keypoints in BGR format. Default is red (0, 0, 255).
        keypoint_special_color: Special color for specific keypoints in BGR format. Default is green (0, 255, 0).
        text_color: Color for text annotations in BGR format. Default is green (0, 255, 0).
        bbox_thickness: Thickness of bounding box lines. Default is 2.
        keypoint_thickness: Thickness of keypoint circles. Default is 2.
        keypoint_radius: Radius of keypoint circles. Default is 1.
        font: Font type for text. Default is FONT_HERSHEY_COMPLEX.
        font_scale: Scale factor for font. Default is 0.7.
        font_thickness: Thickness of font. Default is 1.
    """
    
    def __init__(
        self,
        bbox_color: Tuple[int, int, int] = (0, 0, 255),
        keypoint_color: Tuple[int, int, int] = (0, 0, 255),
        keypoint_special_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (0, 255, 0),
        bbox_thickness: int = 2,
        keypoint_thickness: int = 2,
        keypoint_radius: int = 1,
        font: int = None,  # Will use FONT_HERSHEY_COMPLEX as default
        font_scale: float = 0.7,
        font_thickness: int = 1
    ):
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for visualization. "
                            "Please install it with: pip install opencv-python")
        
        self.bbox_color = bbox_color
        self.keypoint_color = keypoint_color
        self.keypoint_special_color = keypoint_special_color
        self.text_color = text_color
        self.bbox_thickness = bbox_thickness
        self.keypoint_thickness = keypoint_thickness
        self.keypoint_radius = keypoint_radius
        self.font = font if font is not None else cv2.FONT_HERSHEY_COMPLEX
        self.font_scale = font_scale
        self.font_thickness = font_thickness
    
    def draw_on(
        self,
        img: np.ndarray,
        faces: List,
        draw_bbox: bool = True,
        draw_keypoints: bool = True,
        draw_attributes: bool = True
    ) -> np.ndarray:
        """Draw face analysis results on an image.
        
        Args:
            img: Input image in BGR format with shape (H, W, 3).
                 Expected dtype is uint8 with values in range [0, 255].
            faces: List of Face objects to draw.
            draw_bbox: Whether to draw bounding boxes. Default is True.
            draw_keypoints: Whether to draw facial keypoints. Default is True.
            draw_attributes: Whether to draw attributes (gender/age). Default is True.
            
        Returns:
            A copy of the input image with drawings.
            
        Raises:
            ValueError: If img is not a valid numpy array.
            ImportError: If OpenCV is not available.
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for visualization.")
        
        if not isinstance(img, np.ndarray):
            raise ValueError(f"img must be a numpy array, got {type(img)}")
        
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"img must have shape (H, W, 3), got {img.shape}")
        
        dimg = img.copy()
        
        for face in faces:
            if face is None:
                continue
                
            if draw_bbox and face.bbox is not None:
                dimg = self._draw_bbox(dimg, face)
            
            if draw_keypoints and face.kps is not None:
                dimg = self._draw_keypoints(dimg, face)
            
            if draw_attributes:
                dimg = self._draw_attributes(dimg, face)
        
        return dimg
    
    def _draw_bbox(self, img: np.ndarray, face) -> np.ndarray:
        """Draw bounding box for a face.
        
        Args:
            img: Image to draw on.
            face: Face object containing bbox.
            
        Returns:
            Image with bounding box drawn.
        """
        if face.bbox is None or len(face.bbox) < 4:
            return img
        
        try:
            box = face.bbox.astype(int)
            # Ensure box coordinates are valid
            if len(box) >= 4:
                cv2.rectangle(
                    img,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    self.bbox_color,
                    self.bbox_thickness
                )
        except (ValueError, TypeError, IndexError):
            # Skip drawing if bbox is invalid
            pass
        
        return img
    
    def _draw_keypoints(self, img: np.ndarray, face) -> np.ndarray:
        """Draw facial keypoints for a face.
        
        Args:
            img: Image to draw on.
            face: Face object containing kps.
            
        Returns:
            Image with keypoints drawn.
        """
        if face.kps is None:
            return img
        
        try:
            kps = face.kps.astype(int)
            if kps.ndim != 2 or kps.shape[1] < 2:
                return img
            
            for idx in range(kps.shape[0]):
                # Use special color for specific keypoints (indices 0 and 3)
                color = self.keypoint_special_color if idx in (0, 3) else self.keypoint_color
                cv2.circle(
                    img,
                    (int(kps[idx][0]), int(kps[idx][1])),
                    self.keypoint_radius,
                    color,
                    self.keypoint_thickness
                )
        except (ValueError, TypeError, IndexError):
            # Skip drawing if keypoints are invalid
            pass
        
        return img
    
    def _draw_attributes(self, img: np.ndarray, face) -> np.ndarray:
        """Draw face attributes (gender and age) for a face.
        
        Args:
            img: Image to draw on.
            face: Face object containing gender and age.
            
        Returns:
            Image with attributes drawn.
        """
        if face.gender is None or face.age is None or face.bbox is None:
            return img
        
        try:
            box = face.bbox.astype(int)
            if len(box) < 4:
                return img
            
            sex_label = face.sex if face.sex is not None else 'N/A'
            text = f"{sex_label},{face.age}"
            
            # Position text above the bounding box
            text_pos = (int(box[0]) - 1, int(box[1]) - 4)
            
            cv2.putText(
                img,
                text,
                text_pos,
                self.font,
                self.font_scale,
                self.text_color,
                self.font_thickness
            )
        except (ValueError, TypeError, IndexError):
            # Skip drawing if attributes are invalid
            pass
        
        return img


def draw_on(
    img: np.ndarray,
    faces: List,
    **kwargs
) -> np.ndarray:
    """Convenience function to draw face analysis results on an image.
    
    This is a wrapper around FaceVisualizer for quick usage.
    
    Args:
        img: Input image in BGR format with shape (H, W, 3).
             Expected dtype is uint8 with values in range [0, 255].
        faces: List of Face objects to draw.
        **kwargs: Additional arguments passed to FaceVisualizer.
        
    Returns:
        A copy of the input image with drawings.
    """
    visualizer = FaceVisualizer(**kwargs)
    return visualizer.draw_on(img, faces)
