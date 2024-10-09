import cv2
import yaml
from enum import Enum
from io_handler import InputHandler
from transformer import ImageCombiner, ImageAugmentor
from abc import ABC, abstractmethod


class AugmentationCategory(Enum):
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    OBJECT_DETECTION = "object_detection"


class ImageBackgroundAugmentor(ABC):
    def __init__(self, image_combiner, augmentation_handler, dataloader):
        self.image_combiner = image_combiner
        self.augmentation_handler = augmentation_handler
        self.dataloader = dataloader

    @abstractmethod
    def apply_augmentation(self, src_img, src_mask, src_backgrounds, src_annotation=None):
        """
        Apply the augmentation. This method should be implemented by subclasses.
        """
        pass


class SemanticSegmantationAugmentor(ImageBackgroundAugmentor):
    def apply_augmentation(self, src_img, src_mask, src_backgrounds, src_annotation=None):
        return self._augment_for_segmentation(src_img, src_mask, src_backgrounds)

    def _augment_for_segmentation(self, src_img, src_mask, src_backgrounds):

        transformed_background = self.augmentation_handler.apply_background_transformations(src_backgrounds)
        # Transform the input data using the provided augmentation handler and image combiner
        transformed_image, transformed_mask, transformed_background, _ = self.augmentation_handler.apply_object_transformations(
            src_img, src_mask)

        # Combine the transformed image, mask, and background
        augmented_image, _ = self.image_combiner.extract_and_place_object_on_background(
            transformed_image, transformed_mask, transformed_background)

        return augmented_image, None


class ObjectDetectionAugmentor(ImageBackgroundAugmentor):
    def apply_augmentation(self, src_img, src_mask, src_backgrounds, src_annotation=None):
        if src_annotation is None:
            raise ValueError(
                "Annotations are required for object detection augmentation.")
        return self._augment_for_detection(src_img, src_mask, src_backgrounds, src_annotation)

    def _augment_for_detection(self, src_img, src_mask, src_backgrounds, src_annotation):
        # Transform the input data using the provided augmentation handler and image combiner
        transformed_background = self.augmentation_handler.apply_background_transformations(src_backgrounds)

        transformed_image, transformed_mask, transformed_background, transformed_annotation = self.augmentation_handler.apply_object_transformations(
            src_img, src_mask, src_annotation)

        # Combine the transformed image, mask, and background, considering annotations
        augmented_image, augmented_annotation = self.image_combiner.extract_and_place_object_on_background(
            transformed_image, transformed_mask, transformed_background, transformed_annotation,
            annotation_format=self.augmentation_handler.annotation_format)

        return augmented_image, augmented_annotation


class ImageAugmentorBuilder:
    def build_segmantation_augmentor(self, config, background_augmentations, object_augmentations, interpolation, enable_size_variance):
        image_combiner = ImageCombiner(interpolation, enable_size_variance,
                                          augmentation_type=AugmentationCategory.SEMANTIC_SEGMENTATION)
        augmentation_handler = ImageAugmentor(
            background_augmentations, object_augmentations, annotation_format=None ,augmentation_type=AugmentationCategory.SEMANTIC_SEGMENTATION)

        dataloader = InputHandler(
            config['images_dir'], config['masks_dir'], config['background_dir'])
        return SemanticSegmantationAugmentor(image_combiner, augmentation_handler, dataloader)

    def build_detection_augmentor(self, config, background_augmentations, object_augmentations, annotation_format, interpolation_type, enable_size_variance):
        image_combiner = ImageCombiner(
            interpolation=interpolation_type, enable_size_variance=enable_size_variance, augmentation_type=AugmentationCategory.OBJECT_DETECTION)
        augmentation_handler = ImageAugmentor(
            background_augmentations, object_augmentations, annotation_format ,augmentation_type=AugmentationCategory.OBJECT_DETECTION)

        dataloader = InputHandler(
            config['images_dir'], config['masks_dir'], config['background_dir'], config['annotations_dir'])
        return ObjectDetectionAugmentor(image_combiner, augmentation_handler, dataloader)


class AugmentationDirector:
    REQUIRED_FIELDS = {
        AugmentationCategory.SEMANTIC_SEGMENTATION: ['images_dir', 'masks_dir', 'background_dir'],
        AugmentationCategory.OBJECT_DETECTION: [
            'images_dir', 'masks_dir', 'background_dir', 'annotations_dir']
    }

    def __init__(self, builder, augmentation_type):
        self.builder = builder
        self.configuration = None
        self.augmentation_type = augmentation_type

    def load_configuration(self, config_path):
        with open(config_path, 'r') as file:
            self.configuration = yaml.safe_load(file)

    def validate_configuration(self):
        required_fields = self.REQUIRED_FIELDS.get(self.augmentation_type)
        if not required_fields:
            raise ValueError(
                f"Unknown augmentation type: {self.augmentation_type}")
        for field in required_fields:
            if field not in self.configuration:
                raise ValueError(f"Missing required field: {field}")

    def create_augmentor(self, background_augmentations=None, object_augmentations=None, annotation_format='yolo', interpolation_type = cv2.INTER_AREA ,enable_size_variance=True):
        self.validate_configuration(self.augmentation_type)
        if self.augmentation_type == AugmentationCategory.SEMANTIC_SEGMENTATION:
            return self.builder.build_segmantation_augmentor(self.configuration, background_augmentations, object_augmentations, interpolation_type, enable_size_variance)
        elif self.augmentation_type == AugmentationCategory.OBJECT_DETECTION:
            return self.builder.build_detection_augmentor(self.configuration, background_augmentations, object_augmentations, annotation_format, interpolation_type, enable_size_variance)
        else:
            raise ValueError(
                f"Unknown augmentation type: {self.augmentation_type}")
