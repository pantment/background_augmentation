import cv2
import numpy as np
import albumentations as A
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class ImageTransformationHandler(ABC):

    def __init__(self) -> None:
        super().__init__()

        self.background_transformation = None
        self.object_transformation = None

    @abstractmethod
    def _configure_transformations(self,transformation_list):
        pass

    
    def _resize_mask_to_match_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Resize the mask to the size of the image.
        """
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            raise TypeError("Both image and mask must be numpy arrays.")
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    @abstractmethod
    def _transform(self):
        pass

    def apply_background_transformations(self, background: np.ndarray) -> np.ndarray:
        """
        Apply the background transformation.
        """
        if self.background_transformation is None:
            raise ValueError("Background transformation is not defined.")
        return self.background_transformation(image=background)['image']

    def apply_object_transformations(self,image: np.ndarray, mask: np.ndarray, input_annotation: Optional[list] = None)-> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.object_transformation is None:
            raise ValueError("Object transformation is not defined.")

        # Resize the mask to match the image size
        mask = self._resize_mask_to_match_image(image, mask)
        return self._transform(image, mask, input_annotation)


class ImageTransformationBinarySegmentationHandler(ImageTransformationHandler):

    def __init__(self, background_transformations=None, object_transformations=None):
        super().__init__()
        self.background_transformation =  self._configure_transformations(background_transformations)
        self.object_transformation = self._configure_transformations(object_transformations)

    def _configure_transformations(self,transformation_list):
        if transformation_list is None:
            return A.Compose([])  # Identity transformation if none provided

        if not isinstance(transformation_list, list):
            raise TypeError("Transformations must be a list of albumentations.")
        
        for transformation in transformation_list:
            if not isinstance(transformation, A.BasicTransform):
                raise TypeError("Each transformation must be an instance of A.BasicTransform.")

        return A.Compose(transformation_list)
    def _transform(self, src_image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, None]:
        """
        Transform the image and mask.
        """
        transformed = self.object_transformation(image=src_image, mask=mask)
        augmented_image = transformed['image']
        transformed_mask = transformed['mask']
        return augmented_image, transformed_mask, None
    
class ImageTransformationObjectDetectionHandler(ImageTransformationHandler):

    def __init__(self, background_transformations=None, object_transformations=None, annotation_format: Optional[str]=None):
        super().__init__()
        self.annotation_format = annotation_format
        self.background_transformation =  self._configure_transformations(background_transformations)
        self.object_transformation = self._configure_transformations(object_transformations,annotations=True)

    def _configure_transformations(self,transformation_list,annotations=False):
        if transformation_list is None:
            return A.Compose([])  # Identity transformation if none provided

        if not isinstance(transformation_list, list):
            raise TypeError("Transformations must be a list of albumentations.")
        
        for transformation in transformation_list:
            if not isinstance(transformation, A.BasicTransform):
                raise TypeError("Each transformation must be an instance of A.BasicTransform.")
            if annotations:
                if self.annotation_format is None:
                    raise ValueError("Annotation format must be defined for object detection.")
                return A.Compose(transformation_list, bbox_params=A.BboxParams(format=self.annotation_format, label_fields=['class_labels'], min_visibility=0.1))

        return A.Compose(transformation_list)
    def _transform(self, src_image: np.ndarray, mask: np.ndarray, annotation: Optional [np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform the image, mask, and annotations.
        """
        transformed = self.object_transformation(image=src_image, mask=mask, bboxes=annotation[:, 1:], class_labels=annotation[:, 0])
        augmented_image = transformed['image']
        transformed_mask = transformed['mask']
        transformed_labels = np.array([[c, *b] for c, b in zip(transformed['class_labels'], transformed['bboxes'])])
        return augmented_image, transformed_mask, transformed_labels

