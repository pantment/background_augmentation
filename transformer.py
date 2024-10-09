from typing import Tuple, Optional
import albumentations as A
import cv2
import numpy as np
import random

class ImageAugmentor:
    def __init__(self, background_transformations=None, object_transformations=None, annotation_format: Optional[str] = None, augmentation_type: Optional[str] = None) -> None:
        '''
        Args:
            background_transformations (list): A list of albumentations to apply to the background.
            object_transformations (list): A list of albumentations to apply to the object.
            annotation_format (string): The format of the labels.
            augmentation_type (string): The type of augmentation ('semantic_segmentation' or 'object_detection').
        '''
        if augmentation_type not in {"semantic_segmentation", "object_detection"}:
            raise ValueError("augmentation_type must be 'semantic_segmentation' or 'object_detection'.")

        self.augmentation_type = augmentation_type
        self.annotation_format = annotation_format

        # Initialize transformations
        self.background_transformation = self._configure_transformations(background_transformations)
        self.object_transformation = self._configure_transformations(object_transformations, annotations=(augmentation_type == "object_detection"))

    def _configure_transformations(self, transformation_list, annotations=False):
        '''
        Set the transformations to be applied to the background or the object.

        Args:
            transformations (list): A list of albumentations to apply to the background or the object.
            annotations (bool): A flag to indicate if the transformations are for the object (with bounding boxes).
        '''
        if transformation_list is None:
            return A.Compose([])  # Identity transformation if none provided

        if not isinstance(transformation_list, list):
            raise TypeError("Transformations must be a list of albumentations.")
        
        for transformation in transformation_list:
            if not isinstance(transformation, A.BasicTransform):
                raise TypeError("Each transformation must be an instance of A.BasicTransform.")

        # Set up bbox parameters for object detection if annotations are True
        if annotations:
            if self.annotation_format is None:
                raise ValueError("Annotation format must be defined for object detection.")
            return A.Compose(transformation_list, bbox_params=A.BboxParams(format=self.annotation_format, label_fields=['class_labels'], min_visibility=0.1))

        return A.Compose(transformation_list)

    def apply_background_transformations(self, background: np.ndarray) -> np.ndarray:
        """
        Apply the background transformation.
        """
        if self.background_transformation is None:
            raise ValueError("Background transformation is not defined.")
        return self.background_transformation(image=background)['image']

    def resize_mask_to_match_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Resize the mask to the size of the image.
        """
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            raise TypeError("Both image and mask must be numpy arrays.")
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def apply_object_transformations(self, src_image: np.ndarray, src_mask: np.ndarray, annotation: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Apply the object transformation to the image and mask.
        If annotations are provided, transform them as well.
        """
        if self.object_transformation is None:
            raise ValueError("Object transformation is not defined.")

        # Resize the mask to match the image size
        mask = self.resize_mask_to_match_image(src_image, src_mask)

        if annotation is not None:
            return self._transform_with_annotations(src_image, mask, annotation)
        else:
            return self._transform_without_annotations(src_image, mask)

    def _transform_with_annotations(self, src_image: np.ndarray, mask: np.ndarray, annotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform the image, mask, and annotations.
        """
        transformed = self.object_transformation(image=src_image, mask=mask, bboxes=annotation[:, 1:], class_labels=annotation[:, 0])
        augmented_image = transformed['image']
        transformed_mask = transformed['mask']
        transformed_labels = np.array([[c, *b] for c, b in zip(transformed['class_labels'], transformed['bboxes'])])
        return augmented_image, transformed_mask, transformed_labels

    def _transform_without_annotations(self, src_image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, None]:
        """
        Transform the image and mask without considering annotations.
        """
        transformed = self.object_transformation(image=src_image, mask=mask)
        augmented_image = transformed['image']
        transformed_mask = transformed['mask']
        return augmented_image, transformed_mask, None

class ImageCombiner:
    def __init__(self, interpolation: int, enable_size_variance: bool = False, augmentation_type: Optional[str] = None) -> None:
        '''
        Args:
            interpolation (int): Interpolation method for resizing.
            enable_size_variance (bool): Whether to enable size variance during resizing.
            augmentation_type (str): The type of augmentation ('semantic_segmentation' or 'object_detection').
        '''
        if not isinstance(interpolation, int):
            raise TypeError("Interpolation must be an integer from cv2.InterpolationFlags.")
        
        if augmentation_type not in {"semantic_segmentation", "object_detection"}:
            raise ValueError("augmentation_type must be 'semantic_segmentation' or 'object_detection'.")

        self.interpolation = interpolation
        self.enable_size_variance = enable_size_variance
        self.augmentation_type = augmentation_type

    def resize_image_mask_and_annotations(self, image: np.ndarray, mask: np.ndarray, background_width: int, background_height: int, 
                                          annotation_format: Optional[str] = None, annotations: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        '''
        Resize the image, mask, and optionally annotations to fit the background size. Supports both semantic segmentation and object detection.

        Args:
            image (np.ndarray): The source image.
            mask (np.ndarray): The source mask.
            background_width (int): The width of the background.
            background_height (int): The height of the background.
            annotation_format (Optional[str]): The format of the labels (required for object detection).
            annotations (Optional[np.ndarray]): Annotations for object detection.

        Returns:
            Tuple containing the resized image, mask, and optionally the transformed annotations.
        '''
        resized_height = random.randint(background_height // 2, background_height) if self.enable_size_variance else background_height
        resized_width = random.randint(background_width // 2, background_width) if self.enable_size_variance else background_width

        transform = A.Resize(resized_height, resized_width)

        if self.augmentation_type == "object_detection" and annotations is not None:
            if annotation_format is None:
                raise ValueError("annotation_format is required for object detection.")
            transform = A.Compose([transform], bbox_params=A.BboxParams(format=annotation_format, label_fields=['class_labels'], min_visibility=0.1))
            transformed = transform(image=image, mask=mask, bboxes=annotations[:, 1:], class_labels=annotations[:, 0])
            augmented_image = transformed['image']
            transformed_mask = transformed['mask']
            transformed_annotations = np.array([[c, *b] for c, b in zip(transformed['class_labels'], transformed['bboxes'])])
            return augmented_image, transformed_mask, transformed_annotations
        else:
            transformed = A.Compose([transform])(image=image, mask=mask)
            augmented_image = transformed['image']
            transformed_mask = transformed['mask']
            return augmented_image, transformed_mask, None

    def pad_image_and_adjust_annotations(self, image: np.ndarray, mask: np.ndarray, annotations: Optional[np.ndarray], 
                                            background_width: int, background_height: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        '''
        Pad the image and mask to the background size and adjust annotations if provided.

        Args:
            image (np.ndarray): Image to pad.
            mask (np.ndarray): Mask to pad.
            annotations (Optional[np.ndarray]): Annotations for object detection.
            background_width (int): The width of the background.
            background_height (int): The height of the background.

        Returns:
            Tuple containing the padded image, mask, and optionally the adjusted annotations.
        '''
        image_height, image_width = image.shape[:2]

        if annotations is not None:
            annotations[:, 2::2] *= image_height
            annotations[:, 1::2] *= image_width

        top, bottom, left, right = self._calculate_padding(image_width, image_height, background_width, background_height)

        # Apply padding
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        if annotations is not None:
            annotations[:, 2::2] /= background_height
            annotations[:, 1::2] /= background_width

        return image, mask, annotations

    def _calculate_padding(self, image_width: int, image_height: int, background_width: int, background_height: int) -> Tuple[int, int, int, int]:
        '''
        Calculate padding needed to fit the background.

        Args:
            image_width (int): The width of the image.
            image_height (int): The height of the image.
            background_width (int): The width of the background.
            background_height (int): The height of the background.

        Returns:
            Tuple of top, bottom, left, right padding values.
        '''
        padding_choice = random.randint(1, 5)

        if padding_choice == 1:  # Pad bottom and right
            top, bottom = 0, background_height - image_height
            left, right = 0, background_width - image_width
        elif padding_choice == 2:  # Pad top and left
            top, bottom = background_height - image_height, 0
            left, right = background_width - image_width, 0
        elif padding_choice == 3:  # Pad bottom and left
            top, bottom = 0, background_height - image_height
            left, right = background_width - image_width, 0
        elif padding_choice == 4:  # Pad top and right
            top, bottom = background_height - image_height, 0
            left, right = 0, background_width - image_width
        else:  # Pad all sides
            top = (background_height - image_height) // 2
            bottom = background_height - image_height - top
            left = (background_width - image_width) // 2
            right = background_width - image_width - left

        return top, bottom, left, right

    def extract_and_place_object_on_background(self, augmented_image: np.ndarray, transformed_mask: np.ndarray, 
                                               transformed_background: np.ndarray, transformed_annotations: Optional[np.ndarray], 
                                               annotation_format: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract the object from the transformed image and place it on the transformed background.

        Args:
            augmented_image (np.ndarray): The transformed image.
            transformed_mask (np.ndarray): The transformed mask.
            transformed_background (np.ndarray): The transformed background.
            transformed_annotations (Optional[np.ndarray]): The annotations for object detection.
            annotation_format (Optional[str]): The format of the annotations.

        Returns:
            The combined image and the transformed annotations if available.
        """
        if not isinstance(augmented_image, np.ndarray) or not isinstance(transformed_mask, np.ndarray) or not isinstance(transformed_background, np.ndarray):
            raise TypeError("Inputs must be numpy arrays.")

        bgrnd_height, bgrnd_width = transformed_background.shape[:2]

        if augmented_image.shape[0] != bgrnd_height or augmented_image.shape[1] != bgrnd_width:
            augmented_image, transformed_mask, transformed_annotations = self.resize_image_mask_and_annotations(
                augmented_image, transformed_mask, bgrnd_width, bgrnd_height, annotation_format, transformed_annotations)

            augmented_image, transformed_mask, transformed_annotations = self.pad_image_and_adjust_annotations(
                augmented_image, transformed_mask, transformed_annotations, bgrnd_width, bgrnd_height)

        # Create an inverse mask and apply it to the background
        inverse_mask = cv2.bitwise_not(transformed_mask * 255)
        transformed_background = cv2.bitwise_and(transformed_background, transformed_background, mask=inverse_mask)

        # Combine the masked transformed image with the background
        augmented_image = cv2.bitwise_and(augmented_image, augmented_image, mask=transformed_mask)
        combined_image = cv2.add(augmented_image, transformed_background)

        return combined_image, transformed_annotations