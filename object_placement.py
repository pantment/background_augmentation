import cv2
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import albumentations as A
import random


class ObjectPlacementHandler(ABC):

    def __init__(self,interpolation = cv2.INTER_NEAREST, enable_size_variance: bool = False) -> None:
        '''
        Args:
            interpolation (int): Interpolation method for resizing.
            enable_size_variance (bool): Whether to enable size variance during resizing.
            augmentation_type (str): The type of augmentation ('semantic_segmentation' or 'object_detection').
        '''
        super().__init__()
        if not isinstance(interpolation, int):
            raise TypeError("Interpolation must be an integer from cv2.InterpolationFlags.")

        self.interpolation = interpolation
        self.enable_size_variance = enable_size_variance

    @abstractmethod
    def _resize_image_mask_and_annotations(self, image: np.ndarray, mask: np.ndarray, background_width: int, background_height: int, 
                                          annotation_format: Optional[str] = None, annotations: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        pass

    def _pad_image_and_adjust_annotations(self, image: np.ndarray, mask: np.ndarray, annotations: Optional[np.ndarray], 
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
            augmented_image, transformed_mask, transformed_annotations = self._resize_image_mask_and_annotations(
                augmented_image, transformed_mask, bgrnd_width, bgrnd_height, annotation_format, transformed_annotations)

            augmented_image, transformed_mask, transformed_annotations = self._pad_image_and_adjust_annotations(
                augmented_image, transformed_mask, transformed_annotations, bgrnd_width, bgrnd_height)

        # Create an inverse mask and apply it to the background
        inverse_mask = cv2.bitwise_not(transformed_mask * 255)
        transformed_background = cv2.bitwise_and(transformed_background, transformed_background, mask=inverse_mask)

        # Combine the masked transformed image with the background
        augmented_image = cv2.bitwise_and(augmented_image, augmented_image, mask=transformed_mask)
        combined_image = cv2.add(augmented_image, transformed_background)

        return combined_image, transformed_mask


class ObjectPlacementBinarySegmentationHandler(ObjectPlacementHandler):

    def __init__(self,interpolation = cv2.INTER_NEAREST, enable_size_variance: bool = False) -> None:
        super().__init__(interpolation, enable_size_variance)


    def _resize_image_mask_and_annotations(self, image: np.ndarray, mask: np.ndarray, background_width: int, background_height: int, 
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
        transformed = A.Compose([transform])(image=image, mask=mask)
        augmented_image = transformed['image']
        transformed_mask = transformed['mask']
        return augmented_image, transformed_mask, None
    
class ObjectPlacementObjectDetectionHandler(ObjectPlacementHandler):

    def __init__(self,interpolation = cv2.INTER_NEAREST, enable_size_variance: bool = False) -> None:
        super().__init__(interpolation, enable_size_variance)


    def _resize_image_mask_and_annotations(self, image: np.ndarray, mask: np.ndarray, background_width: int, background_height: int, 
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
        if annotations is not None:

            if annotation_format is None:
                raise ValueError("annotation_format is required for object detection.")
            transform = A.Compose([transform], bbox_params=A.BboxParams(format=annotation_format, label_fields=['class_labels'], min_visibility=0.1))
            transformed = transform(image=image, mask=mask, bboxes=annotations[:, 1:], class_labels=annotations[:, 0])
            augmented_image = transformed['image']
            transformed_mask = transformed['mask']
            transformed_annotations = np.array([[c, *b] for c, b in zip(transformed['class_labels'], transformed['bboxes'])])
            return augmented_image, transformed_mask, transformed_annotations