import cv2
import albumentations as A
import os
import tqdm
from io_handler import InputHandler, OutputHandler
from transformer import ImageCombination, ImageAugmentation
from utils import visualize_yolo_annotations, process_image


def main(config):
    dataloader = InputHandler(config['images_dir'], config['masks_dir'],
                            config['background_dir'], config['annotations_dir'])

    background_transformations = [
        A.RandomBrightnessContrast(p=0.2),
    ]

    object_transformations = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ]

    image_combination = ImageCombination(
        interpolation=cv2.INTER_AREA, enable_size_variance=True)
    image_augmentation = ImageAugmentation(
        background_transformations, object_transformations)

    src_images = sorted(list(map(lambda x: os.path.join(
        dataloader.images_dir, x), os.listdir(dataloader.images_dir))))
    src_masks = sorted(list(map(lambda x: os.path.join(
        dataloader.masks_dir, x), os.listdir(dataloader.masks_dir))))
    src_backgrounds = list(map(lambda x: os.path.join(
        dataloader.background_dir, x), os.listdir(dataloader.background_dir)))
    annotations = sorted(list(map(lambda x: os.path.join(
        dataloader.annotations_dir, x), os.listdir(dataloader.annotations_dir))))

    file_manager = OutputHandler(parent_dir=config['parent_dir'], num_images_to_save=config['num_images_to_save'],
                               augmentation_dataset_name=config['augmentation_dataset_name'])

    for src_image, src_mask, annotation in tqdm.tqdm(zip(src_images, src_masks, annotations)):
        transformed_image, transformed_mask, transformed_background, transformed_annotation = process_image(src_image, src_mask, annotation, dataloader,
                                                                                                            image_augmentation, image_combination, file_manager, src_backgrounds)

        augmented_image, augmented_annotation = image_combination.extract_and_place_object_on_background(
            transformed_image, transformed_mask, transformed_background, transformed_annotation, annotation_format=image_augmentation.annotation_format)
        if file_manager.image_counter <= file_manager.num_images_to_save:
            transformed_gt = visualize_yolo_annotations(
                transformed_image, transformed_annotation, transformed_mask)
            file_manager.save_transformed_files(
                transformed_image, transformed_mask, transformed_background, transformed_gt)
        file_manager.save_augmented_files(
            src_image, augmented_image, augmented_annotation)


if __name__ == "__main__":
    config = {
        'images_dir': '/home/pantelis/Projects/Simar/background-augmentation/test_images',
        'masks_dir': '/home/pantelis/Projects/Simar/background-augmentation/test_masks',
        'background_dir': '/home/pantelis/Projects/Simar/background-augmentation/test_backgrounds',
        'annotations_dir': '/home/pantelis/Projects/Simar/background-augmentation/test_annotations',
        'parent_dir': 'output',
        'num_images_to_save': 15,
        'augmentation_dataset_name': 'augmented'
    }
    main(config)
