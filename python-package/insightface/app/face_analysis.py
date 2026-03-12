# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 


from __future__ import division

import glob
import os.path as osp
from typing import List, Optional, Tuple, Union

import numpy as np
import onnxruntime
from numpy.linalg import norm

from ..model_zoo import model_zoo
from ..utils import DEFAULT_MP_NAME, ensure_available
from ..utils.visualization import draw_faces
from .common import Face

__all__ = ['FaceAnalysis']

class FaceAnalysis:
    def __init__(self, name=DEFAULT_MP_NAME, root='~/.insightface', allowed_modules=None, **kwargs):
        """
        Initialize FaceAnalysis
        
        Args:
            name: Model name
            root: Root directory for models
            allowed_modules: List of allowed modules (e.g., ['detection', 'recognition'])
            **kwargs: Additional parameters for model loading
        """
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        self.model_files = []  # Store all ONNX files
        self.allowed_modules = allowed_modules
        self.model_dir = ensure_available('models', name, root=root)
        self.kwargs = kwargs
        
        # Only collect model files, don't load any models
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        
        for onnx_file in onnx_files:
            self.model_files.append(onnx_file)
            print('found model file:', onnx_file)
        
        # Ensure at least one model is found
        assert len(self.model_files) > 0, 'No models found'

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        """
        Prepare models for inference
        
        Args:
            ctx_id: Context ID (0 for GPU, -1 for CPU)
            det_thresh: Detection threshold
            det_size: Detection input size
        """
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        
        # Clear existing models
        self.models = {}
        
        # Load all models and determine their task types
        for onnx_file in self.model_files:
            model = model_zoo.get_model(onnx_file, **self.kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
                continue
            
            taskname = model.taskname
            
            # Check if module is allowed
            if self.allowed_modules is not None and taskname not in self.allowed_modules:
                print('model ignore:', onnx_file, taskname)
                del model
                continue
            
            if taskname not in self.models:
                print('find model:', onnx_file, taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, taskname)
                del model
        
        # Ensure detection model is available
        assert 'detection' in self.models, 'Detection model not found'
        self.det_model = self.models['detection']
        
        # Prepare all models
        for taskname, model in self.models.items():
            if taskname == 'detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img: np.ndarray, max_num: int = 0, det_metric: str = 'default') -> List[Face]:
        """
        Get faces from image
        
        Args:
            img: Input image (BGR format, 0-255 range)
            max_num: Maximum number of faces to detect
            det_metric: Detection metric
        
        Returns:
            List of Face objects
        """
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric=det_metric)
        
        if bboxes.shape[0] == 0:
            return []
        
        ret = []
        for i in range(bboxes.shape[0]):
            try:
                # Safe slicing with bounds checking
                bbox = bboxes[i, :4] if bboxes.shape[1] >= 4 else np.array([0, 0, 0, 0])
                det_score = bboxes[i, 4] if bboxes.shape[1] >= 5 else 0.0
                
                kps = None
                if kpss is not None and i < kpss.shape[0]:
                    kps = kpss[i]
                
                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                
                # Process with other models
                for taskname, model in self.models.items():
                    if taskname == 'detection':
                        continue
                    try:
                        model.get(img, face)
                    except Exception as e:
                        print(f'Error processing {taskname}: {e}')
                        pass
                
                ret.append(face)
            except Exception as e:
                print(f'Error processing face {i}: {e}')
                pass
        
        return ret

    def draw_on(self, img: np.ndarray, faces: List[Face]) -> np.ndarray:
        """
        Draw faces on image
        
        Args:
            img: Input image (BGR format)
            faces: List of Face objects
        
        Returns:
            Image with faces drawn
        """
        return draw_faces(img, faces)


