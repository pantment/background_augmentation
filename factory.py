from transformation import ImageTransformationObjectDetectionHandler, ImageTransformationBinarySegmentationHandler
from object_placement import ObjectPlacementBinarySegmentationHandler, ObjectPlacementObjectDetectionHandler
from augmentation import BinarySegmantationAugmentor, ObjectDetectionAugmentor
from abc import ABC,abstractmethod
import cv2

class AugmentationFactory(ABC):

    @abstractmethod
    def get_transformation_handler(self):
        pass
    @abstractmethod
    def get_placement_handler(self):
        pass


class BinarySegmentationAugmentationFactory(AugmentationFactory):

    def get_transformation_handler(self, background_transformations=None, object_transformations=None, annotation_format=None):
        if background_transformations is None or object_transformations is None:
            raise ValueError("Background and object transformations must be defined.")
        
        return ImageTransformationBinarySegmentationHandler(background_transformations=background_transformations, object_transformations=object_transformations)

    def get_placement_handler(self, interpolation=cv2.INTER_NEAREST, enable_size_variance=False):
        return ObjectPlacementBinarySegmentationHandler(interpolation=interpolation, enable_size_variance=enable_size_variance)
    def get_augmentror(self, image_combiner, augmentation_handler):
        return BinarySegmantationAugmentor(image_combiner=image_combiner, augmentation_handler=augmentation_handler)
    
class ObjectDetectionAugmentationFactory(AugmentationFactory):

    def get_transformation_handler(self, background_transformations=None, object_transformations=None, annotation_format=None):
        if background_transformations is None or object_transformations is None or annotation_format is None:
            raise ValueError("Background, object transformations and annotation format must be defined.")
        
        return ImageTransformationObjectDetectionHandler(background_transformations, object_transformations, annotation_format)
    
    def get_placement_handler(self, interpolation=cv2.INTER_NEAREST, enable_size_variance=False):
        return ObjectPlacementObjectDetectionHandler(interpolation=interpolation, enable_size_variance=enable_size_variance)
    
    def get_augmentror(self, image_combiner, augmentation_handler):
        return ObjectDetectionAugmentor(image_combiner=image_combiner, augmentation_handler=augmentation_handler)
    
