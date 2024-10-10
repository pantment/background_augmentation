import os
from PIL import Image
import cv2
import numpy as np
from factory import BinarySegmentationAugmentationFactory, ObjectDetectionAugmentationFactory

def get_augmentation_factory(augmentation_type):

    factories = {
        'binary_segmentation': BinarySegmentationAugmentationFactory(),
        'object_detection': ObjectDetectionAugmentationFactory()
    }
    if augmentation_type in factories:
        return factories[augmentation_type]
    else:
        raise ValueError(f"Invalid augmentation type: {augmentation_type}")
    
def create_background_augmentor(augmentation_type, background_transformations=None, object_transformations=None, annotation_format=None, interpolation=cv2.INTER_NEAREST, enable_size_variance=False):
    factory = get_augmentation_factory(augmentation_type)
    transformation_handler = factory.get_transformation_handler(background_transformations, object_transformations, annotation_format)
    placement_handler = factory.get_placement_handler(interpolation, enable_size_variance)
    return factory.get_augmentror(transformation_handler, placement_handler)

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
