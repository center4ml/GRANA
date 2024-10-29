import torch
import numpy as np
from typing import Union, Optional
from PIL import Image
from mmdet.apis import DetInferencer
from ultralytics.engine.results import Results
import warnings

class MMDetector(DetInferencer):
    def __call__(
            self,
            inputs,
            ) -> Results:
        """Call the inferencer as in DetInferencer but for single image.

        Args:
            inputs (np.ndarray | str): Inputs for the inferencer.

        Returns:
            Result: yolo-like result
        """

        ori_inputs = self._inputs_to_list(inputs)

        data = list(self.preprocess(
            ori_inputs, batch_size=1))[0][1]

        preds = self.forward(data)[0]
        
        yolo_result = Results(
            orig_img=ori_inputs[0], path="", names=[""],
            boxes=torch.cat((preds.pred_instances.bboxes, preds.pred_instances.scores.unsqueeze(-1), preds.pred_instances.labels.unsqueeze(-1)), dim=1),
            masks=preds.pred_instances.masks
        )
            
        return yolo_result
    
    def predict(self, source: Image.Image, conf=None):
        """yolo interface"""
        if conf is not None:
            warnings.warn(f"confidence value {conf} ignored")
        return [self.__call__(np.array(source.convert("RGB")))]