# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : Face Analysis Pipeline

from __future__ import division

import glob
import json
import os
import os.path as osp
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import onnxruntime
from numpy.linalg import norm

from ..model_zoo import model_zoo
from ..utils import DEFAULT_MP_NAME, ensure_available
from ..utils.visualization import FaceVisualizer
from .common import Face

__all__ = ['FaceAnalysis']


# Type aliases for better readability
ImageArray = np.ndarray  # Expected: uint8 array with shape (H, W, 3), BGR format, values in [0, 255]
BoundingBox = np.ndarray  # Shape: (4,), format: [x1, y1, x2, y2]
Keypoints = np.ndarray  # Shape: (N, 2), format: [[x, y], ...]


def _get_onnx_model_info(onnx_file: str) -> Optional[Dict[str, Any]]:
    """Get model information from ONNX file without creating InferenceSession.
    
    This function reads ONNX file metadata directly to determine model type
    and extract configuration, avoiding the overhead of creating an InferenceSession.
    
    Args:
        onnx_file: Path to the ONNX model file.
        
    Returns:
        Dictionary containing model info, or None if model is not recognized.
    """
    try:
        # Try to use onnx to read model metadata (lightweight)
        import onnx
        model = onnx.load(onnx_file, load_external_data=False)
        graph = model.graph
        
        # Get input shape from the first input
        if len(graph.input) == 0:
            return None
        
        input_tensor = graph.input[0]
        input_shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                input_shape.append(dim.dim_value)
            else:
                input_shape.append(None)
        
        # Get number of outputs
        num_outputs = len(graph.output)
        num_inputs = len(graph.input)
        
        # Determine model type based on input shape and output count
        # This logic mirrors the ModelRouter in model_zoo.py
        taskname = None
        input_mean = 127.5
        input_std = 128.0
        
        # Check for RetinaFace/SCRFD (detection models have many outputs)
        if num_outputs >= 5:
            taskname = 'detection'
            # Detection models use 127.5/128.0 by default
        
        # Check input shape for other model types
        elif len(input_shape) >= 4:
            h, w = input_shape[2], input_shape[3]
            
            if h == 192 and w == 192:
                taskname = 'landmark'
                # Landmark model
            elif h == 96 and w == 96:
                taskname = 'genderage'
                # Attribute model
            elif num_inputs == 2 and h == 128 and w == 128:
                taskname = 'inswapper'
                # INSwapper model
            elif h == w and h >= 112 and h % 16 == 0:
                taskname = 'recognition'
                # Recognition model - check for mxnet style
                # Check first few nodes for Sub/Mul (mxnet style)
                find_sub = False
                find_mul = False
                for nid, node in enumerate(graph.node[:8]):
                    if node.name.startswith('Sub') or node.name.startswith('_minus'):
                        find_sub = True
                    if node.name.startswith('Mul') or node.name.startswith('_mul'):
                        find_mul = True
                
                if find_sub and find_mul:
                    input_mean = 0.0
                    input_std = 1.0
        
        if taskname is None:
            return None
        
        return {
            'taskname': taskname,
            'input_shape': input_shape,
            'input_mean': input_mean,
            'input_std': input_std,
            'num_outputs': num_outputs,
            'num_inputs': num_inputs,
        }
        
    except ImportError:
        # Fallback: onnx not available, return minimal info
        # This should not happen in normal usage
        return None
    except Exception as e:
        # If ONNX parsing fails, return None
        return None


class FaceAnalysis:
    """Face analysis pipeline that combines detection, alignment, and feature extraction.
    
    This class provides a high-level interface for face analysis tasks including:
    - Face detection
    - Facial landmark detection
    - Face recognition/embedding extraction
    - Attribute prediction (gender, age)
    
    The class uses lazy loading for models, meaning models are only loaded into memory
    when they are first needed, reducing initial memory footprint.
    
    Attributes:
        model_dir: Directory containing ONNX model files.
        det_thresh: Detection threshold for face detection.
        det_size: Input size for face detection model (width, height).
        
    Example:
        >>> import cv2
        >>> from insightface.app import FaceAnalysis
        >>> 
        >>> # Initialize the app
        >>> app = FaceAnalysis(name='buffalo_l')
        >>> app.prepare(ctx_id=0, det_size=(640, 640))
        >>> 
        >>> # Process an image
        >>> img = cv2.imread('image.jpg')  # BGR format, uint8, [0, 255]
        >>> faces = app.get(img)
        >>> 
        >>> # Access results
        >>> for face in faces:
        ...     print(f"BBox: {face.bbox}")
        ...     print(f"Detection score: {face.det_score}")
        ...     print(f"Embedding shape: {face.embedding.shape if face.embedding is not None else None}")
    """
    
    def __init__(
        self,
        name: str = DEFAULT_MP_NAME,
        root: str = '~/.insightface',
        allowed_modules: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize the FaceAnalysis pipeline.
        
        Args:
            name: Name of the model package to use. Default is DEFAULT_MP_NAME.
            root: Root directory for model storage. Default is '~/.insightface'.
            allowed_modules: List of module names to load. If None, loads all available modules.
                Common values: 'detection', 'recognition', 'genderage', 'landmark_2d_106', etc.
            **kwargs: Additional arguments passed to model initialization.
        """
        onnxruntime.set_default_logger_severity(3)
        
        # Store model configurations for lazy loading
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        self._loaded_models: Dict[str, Any] = {}
        self._model_kwargs = kwargs
        
        self.model_dir = ensure_available('models', name, root=root)
        
        # Scan and register available models
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        
        for onnx_file in onnx_files:
            model_info = self._get_model_info(onnx_file)
            if model_info is None:
                print('model not recognized:', onnx_file)
                continue
            
            taskname = model_info['taskname']
            
            if allowed_modules is not None and taskname not in allowed_modules:
                print('model ignore:', onnx_file, taskname)
                continue
            
            if taskname in self._model_configs:
                print('duplicated model task type, ignore:', onnx_file, taskname)
                continue
            
            print('find model:', onnx_file, taskname, 
                  model_info.get('input_shape'), 
                  model_info.get('input_mean'), 
                  model_info.get('input_std'))
            
            self._model_configs[taskname] = {
                'onnx_file': onnx_file,
                'kwargs': kwargs
            }
        
        # Ensure detection model is available
        if 'detection' not in self._model_configs:
            raise ValueError("No detection model found. FaceAnalysis requires a detection model.")
    
    def _get_model_info(self, onnx_file: str) -> Optional[Dict[str, Any]]:
        """Get model information without fully loading the model.
        
        This method uses a fast path to read ONNX metadata without creating
        an InferenceSession, significantly improving initialization speed.
        
        Args:
            onnx_file: Path to the ONNX model file.
            
        Returns:
            Dictionary containing model info, or None if model is not recognized.
        """
        # First try the fast path using onnx directly
        info = _get_onnx_model_info(onnx_file)
        if info is not None:
            return info
        
        # Fallback: use model_zoo if onnx parsing fails
        # This is slower but more robust
        try:
            model = model_zoo.get_model(onnx_file, **self._model_kwargs)
            if model is None:
                return None
            
            # Extract relevant info and delete model
            info = {
                'taskname': model.taskname,
                'input_shape': getattr(model, 'input_shape', None),
                'input_mean': getattr(model, 'input_mean', None),
                'input_std': getattr(model, 'input_std', None),
            }
            del model
            return info
        except Exception as e:
            print(f'Error loading model info for {onnx_file}: {e}')
            return None
    
    def _load_model(self, taskname: str) -> Any:
        """Lazy load a model by taskname.
        
        Args:
            taskname: Name of the task/model to load.
            
        Returns:
            The loaded model instance.
            
        Raises:
            KeyError: If the taskname is not registered.
        """
        if taskname in self._loaded_models:
            return self._loaded_models[taskname]
        
        if taskname not in self._model_configs:
            raise KeyError(f"Model for task '{taskname}' not found")
        
        config = self._model_configs[taskname]
        model = model_zoo.get_model(config['onnx_file'], **config['kwargs'])
        
        # Prepare model if ctx_id was set
        if hasattr(self, '_ctx_id'):
            if taskname == 'detection':
                model.prepare(
                    self._ctx_id,
                    input_size=getattr(self, 'det_size', None),
                    det_thresh=getattr(self, 'det_thresh', 0.5)
                )
            else:
                model.prepare(self._ctx_id)
        
        self._loaded_models[taskname] = model
        return model
    
    @property
    def det_model(self) -> Any:
        """Get the detection model (lazy loaded)."""
        return self._load_model('detection')
    
    @property
    def models(self) -> Dict[str, Any]:
        """Get all loaded models."""
        # Ensure detection model is loaded
        self._load_model('detection')
        return self._loaded_models
    
    def prepare(
        self,
        ctx_id: int,
        det_thresh: float = 0.5,
        det_size: Tuple[int, int] = (640, 640)
    ) -> None:
        """Prepare the pipeline with execution context and detection parameters.
        
        Args:
            ctx_id: Execution provider ID. Use >= 0 for GPU, -1 for CPU.
            det_thresh: Detection threshold. Faces with scores below this value are filtered out.
                Default is 0.5.
            det_size: Input size for detection model as (width, height). Default is (640, 640).
                Larger sizes detect smaller faces but are slower.
        """
        self.det_thresh = det_thresh
        self.det_size = det_size
        self._ctx_id = ctx_id
        
        assert det_size is not None, "det_size cannot be None"
        print('set det-size:', det_size)
        
        # Prepare already loaded models
        for taskname, model in self._loaded_models.items():
            if taskname == 'detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)
    
    def get(
        self,
        img: ImageArray,
        max_num: int = 0,
        det_metric: str = 'default'
    ) -> List[Face]:
        """Detect and analyze faces in an image.
        
        Args:
            img: Input image as a numpy array.
                - Format: BGR (OpenCV default)
                - Shape: (height, width, 3)
                - Dtype: uint8
                - Value range: [0, 255]
            max_num: Maximum number of faces to return. If 0, returns all detected faces.
                Default is 0.
            det_metric: Metric for selecting faces when max_num is specified.
                - 'default': Uses area minus distance from center (prefers larger, centered faces)
                - 'max': Uses only face area
                Default is 'default'.
        
        Returns:
            List of Face objects containing detection and analysis results.
            Each Face object may contain:
                - bbox: Bounding box [x1, y1, x2, y2]
                - det_score: Detection confidence score
                - kps: Facial keypoints
                - embedding: Face feature vector
                - gender: Gender prediction (0=female, 1=male)
                - age: Age prediction
                - Various landmark predictions depending on loaded models
        
        Raises:
            ValueError: If img is not a valid numpy array.
        """
        # Validate input
        if not isinstance(img, np.ndarray):
            raise ValueError(f"img must be a numpy array, got {type(img)}")
        
        if img.ndim != 3:
            raise ValueError(f"img must have 3 dimensions (H, W, C), got shape {img.shape}")
        
        if img.shape[2] != 3:
            raise ValueError(f"img must have 3 channels (BGR), got shape {img.shape}")
        
        # Detect faces
        bboxes, kpss = self.det_model.detect(
            img,
            max_num=max_num,
            metric=det_metric
        )
        
        # Handle empty detection
        if bboxes is None or bboxes.shape[0] == 0:
            return []
        
        # Process each detected face
        ret = []
        for i in range(bboxes.shape[0]):
            # Safely extract bounding box with bounds checking
            bbox = self._safe_extract_bbox(bboxes, i)
            det_score = self._safe_extract_score(bboxes, i)
            
            # Safely extract keypoints if available
            kps = None
            if kpss is not None and i < kpss.shape[0]:
                kps = kpss[i]
            
            # Create Face object
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            
            # Apply other models (lazy loaded)
            for taskname in self._model_configs:
                if taskname == 'detection':
                    continue
                try:
                    model = self._load_model(taskname)
                    model.get(img, face)
                except Exception as e:
                    # Log error but continue processing other models
                    print(f"Error applying model '{taskname}': {e}")
            
            ret.append(face)
        
        return ret
    
    def _safe_extract_bbox(self, bboxes: np.ndarray, index: int) -> Optional[np.ndarray]:
        """Safely extract bounding box from detection results.
        
        Args:
            bboxes: Array of bounding boxes with shape (N, 5+) where each row is [x1, y1, x2, y2, score, ...]
            index: Index of the bounding box to extract.
            
        Returns:
            Bounding box array [x1, y1, x2, y2] or None if extraction fails.
        """
        try:
            if index < 0 or index >= bboxes.shape[0]:
                return None
            if bboxes.shape[1] < 4:
                return None
            return bboxes[index, :4].copy()
        except (IndexError, ValueError):
            return None
    
    def _safe_extract_score(self, bboxes: np.ndarray, index: int) -> Optional[float]:
        """Safely extract detection score from detection results.
        
        Args:
            bboxes: Array of bounding boxes with shape (N, 5+) where each row is [x1, y1, x2, y2, score, ...]
            index: Index of the bounding box to extract.
            
        Returns:
            Detection score as float or None if extraction fails.
        """
        try:
            if index < 0 or index >= bboxes.shape[0]:
                return None
            if bboxes.shape[1] < 5:
                return None
            return float(bboxes[index, 4])
        except (IndexError, ValueError, TypeError):
            return None
    
    def draw_on(
        self,
        img: ImageArray,
        faces: List[Face],
        **kwargs
    ) -> np.ndarray:
        """Draw face analysis results on an image.
        
        .. deprecated::
            This method is deprecated. Use insightface.utils.visualization.draw_on() 
            or insightface.utils.visualization.FaceVisualizer instead.
        
        Args:
            img: Input image in BGR format with shape (H, W, 3).
                 Expected dtype is uint8 with values in range [0, 255].
            faces: List of Face objects to draw.
            **kwargs: Additional arguments passed to FaceVisualizer.
            
        Returns:
            A copy of the input image with drawings.
        """
        import warnings
        warnings.warn(
            "FaceAnalysis.draw_on() is deprecated. "
            "Use insightface.utils.visualization.draw_on() or FaceVisualizer instead.",
            DeprecationWarning,
            stacklevel=2
        )
        visualizer = FaceVisualizer(**kwargs)
        return visualizer.draw_on(img, faces)
