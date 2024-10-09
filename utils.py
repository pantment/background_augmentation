import os
from PIL import Image
import cv2
import numpy as np
import random
import logging
from typing import Optional, Tuple

def process_image(src_image: str, src_mask: str, src_backgrounds: list, annotation: Optional[str], 
                  dataloader, image_augmentor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Process the input image, mask, and background using the provided augmentor and combiner.

    Args:
        src_image (str): Path to the source image.
        src_mask (str): Path to the source mask.
        src_backgrounds (list): List of background image paths.
        annotation (Optional[str]): Path to the annotation file (if available).
        dataloader: The dataloader used for loading the images, masks, and annotations.
        image_augmentor (ImageAugmentor): The augmentor to apply transformations.
        image_combiner (ImageCombiner): The combiner for resizing and combining images.
    
    Returns:
        Tuple containing the processed image, mask, transformed background, and transformed annotation (if available).
    """
    bgrnd_id = random.randint(0, len(src_backgrounds) - 1)

    try:
        # Load the image, mask, and background
        input_image = dataloader.load_image(src_image)
        input_mask = dataloader.load_mask(src_mask)
        background = dataloader.load_image(src_backgrounds[bgrnd_id])
        bounding_boxes = None

        # Load annotations if provided (only for object detection)
        if annotation is not None:
            bounding_boxes = dataloader.load_annotation(annotation)

        # Transform the background
        transformed_background = image_augmentor.apply_background_transformations(background)

        # Transform the image, mask, and optionally the annotations
        augmented_image, augmented_mask, transformed_bounding_boxes = image_augmentor.apply_object_transformations(
            input_image, input_mask, bounding_boxes)

        return augmented_image, augmented_mask, transformed_background, transformed_bounding_boxes

    except Exception as e:
        logging.error(f"\nError processing image: {src_image}, mask: {src_mask}, annotation: {annotation} with error: {e}")
        return None, None, None, None

def visualize_yolo_annotations(image: np.ndarray, annotations: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Return the image with mask (if provided) and bboxes.

    Args:
        image (np.ndarray): The image to visualize mask and bboxes.
        annotations (np.ndarray): The bounding boxes to visualize in YOLO format.
        mask (np.ndarray, optional): The mask to visualize. Defaults to None.
    """
    # Copy the image to avoid modifying the original one
    image_copy = image.copy()

    # If a mask is provided, visualize it
    if mask is not None:
        # Ensure the mask is binary
        mask = (mask > 0).astype(np.uint8) * 255

        # Create a color version of the mask
        color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        # Blend the color mask and the image
        blended_image = cv2.addWeighted(image_copy, 0.7, color_mask, 0.3, 0)
    else:
        blended_image = image_copy

    # Get the image dimensions
    img_height, img_width = image.shape[:2]

    # Draw each bounding box
    for annotation in annotations:
        # Convert the normalized bounding box coordinates to pixel coordinates
        x, y, width, height = map(
            int, (annotation[1:] * np.array([img_width, img_height, img_width, img_height])).tolist())
        x_min = int(x - width / 2)
        y_min = int(y - height / 2)
        x_max = int(x + width / 2)
        y_max = int(y + height / 2)
        # Draw the bounding box
        cv2.rectangle(blended_image, (x_min, y_min),
                      (x_max, y_max), (0, 255, 0), 2)

    return blended_image


def convert_to_jpg(image_path):
    '''
    Convert to jpg
    '''
    images = os.listdir(image_path)
    for image in images:
        if image.endswith('.webp'):
            im = Image.open(image_path + '/' + image).convert('RGB')
            im.save(image_path + '/' + image.split('.')[0] + '.jpg', 'jpeg')
        if image.endswith('.avif'):
            im = Image.open(image_path + '/' + image)
            im.save(image_path + '/' + image.split('.')[0] + '.jpg', 'jpeg')


def debug_image_bboxes(image_name, image, annotations):
    '''
    Debug image with bboxes
    '''
    # Draw the bounding boxes
    image_with_bboxes = visualize_yolo_annotations(image, annotations)
    # Save the image with the bounding boxes
    cv2.imwrite(f'{image_name}_bboxes.jpg', image_with_bboxes)
