from abc import ABC, abstractmethod



class ImageBackgroundAugmentor(ABC):
    def __init__(self, image_combiner, augmentation_handler):
        self.image_combiner = image_combiner
        self.augmentation_handler = augmentation_handler

    @abstractmethod
    def apply_augmentation(self, src_img, src_mask, src_backgrounds, src_annotation=None):
        """
        Apply the augmentation. This method should be implemented by subclasses.
        """
        pass


class BinarySegmantationAugmentor(ImageBackgroundAugmentor):
    def __init__(self, image_combiner, augmentation_handler):
        super().__init__(image_combiner, augmentation_handler)

    def apply_augmentation(self, src_img, src_mask, src_backgrounds, src_annotation=None):
        return self._augment_for_segmentation(src_img, src_mask, src_backgrounds)

    def _augment_for_segmentation(self, src_img, src_mask, src_backgrounds):

        transformed_background = self.augmentation_handler.apply_background_transformations(
            src_backgrounds)
        # Transform the input data using the provided augmentation handler and image combiner
        transformed_image, transformed_mask, transformed_background, _ = self.augmentation_handler.apply_object_transformations(
            src_img, src_mask)

        # Combine the transformed image, mask, and background
        augmented_image, transformed_mask = self.image_combiner.extract_and_place_object_on_background(
            transformed_image, transformed_mask, transformed_background)

        return augmented_image, transformed_mask


class ObjectDetectionAugmentor(ImageBackgroundAugmentor):

    def __init__(self, image_combiner, augmentation_handler):
        super().__init__(image_combiner, augmentation_handler)

    def apply_augmentation(self, src_img, src_mask, src_backgrounds, src_annotation=None):
        if src_annotation is None:
            raise ValueError(
                "Annotations are required for object detection augmentation.")
        return self._augment_for_detection(src_img, src_mask, src_backgrounds, src_annotation)

    def _augment_for_detection(self, src_img, src_mask, src_backgrounds, src_annotation):
        # Transform the input data using the provided augmentation handler and image combiner
        transformed_background = self.augmentation_handler.apply_background_transformations(
            src_backgrounds)

        transformed_image, transformed_mask, transformed_background, transformed_annotation = self.augmentation_handler.apply_object_transformations(
            src_img, src_mask, src_annotation)

        # Combine the transformed image, mask, and background, considering annotations
        augmented_image, augmented_annotation = self.image_combiner.extract_and_place_object_on_background(
            transformed_image, transformed_mask, transformed_background, transformed_annotation,
            annotation_format=self.augmentation_handler.annotation_format)

        return augmented_image, augmented_annotation



