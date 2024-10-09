import cv2
import numpy as np
import cv2
import os


class InputHandler:
    def __init__(self, images_dir=None, masks_dir=None, background_dir=None, annotations_dir=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.annotations_dir = annotations_dir
        self.background_dir = background_dir

    def load_image(self, image_path):
        """
        Load an image given its path.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_mask(self, mask_path):
        """
        Load a mask given its path.
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask

    def format_annotation(self, annotation):
        """
        Format the annotation.
        """
        formatted_annotation = []
        for line in annotation:
            formatted_line = list(map(float, line.split()))
            formatted_annotation.append(formatted_line)
        return np.array(formatted_annotation)
    
    def load_annotation(self, annotation_path):
        """
        Load an annotation given its path.
        """
        with open(annotation_path, 'r') as file:
            annotation = file.read().splitlines()
        return self.format_annotation(annotation)

class OutputHandler:
    def __init__(self, parent_dir='output', num_images_to_save=15, augmentation_dataset_name='augmented') -> None:
        self.num_images_to_save = num_images_to_save
        self.parent_dir = parent_dir
        self.augmentation_dataset_name = augmentation_dataset_name
        self.output_dirs = ['transformed/images', 'transformed/masks',
                            'transformed/backgrounds', 'transformed/gt', 'augmented/images', 'augmented/labels']
        self.image_counter = 0
        self.create_directory_ouput_tree()

    def create_directory_ouput_tree(self) -> None:
        '''
        Create the output directory tree.
        '''
        dir_path = os.path.join(
            self.parent_dir, self.augmentation_dataset_name)
        if os.path.exists(dir_path):
            counter = 1
            while os.path.exists(f"{dir_path}_{counter}"):
                counter += 1
            dir_path = f"{dir_path}_{counter}"
            self.augmentation_dataset_name = f"{self.augmentation_dataset_name}_{counter}"
        # Create each directory
        for dir_name in self.output_dirs:
            # Create the full path to the directory
            dir_to_create = os.path.join(dir_path, dir_name)

            # Create the directory
            os.makedirs(dir_to_create, exist_ok=True)

    def save_transformed_files(self, transformed_image, transformed_mask, transformed_background, transformed_gt) -> None:
        """
        Save the transformed image, mask, background, and annotation to the output/transformed directory.

        Parameters:
        transformed_image (np.array): The transformed image.
        transformed_mask (np.array): The transformed mask.
        transformed_background (np.array): The transformed background.
        transformed_gt (np.array): The transformed ground truth.
        """

        try:

            cv2.imwrite(os.path.join(self.parent_dir, self.augmentation_dataset_name, 'transformed/images',
                        f'transformed_image_{self.image_counter}.jpg'), transformed_image)
            cv2.imwrite(os.path.join(self.parent_dir, self.augmentation_dataset_name, 'transformed/masks',
                        f'transformed_mask_{self.image_counter}.png'), transformed_mask)
            cv2.imwrite(os.path.join(self.parent_dir, self.augmentation_dataset_name, 'transformed/backgrounds',
                        f'transformed_background_{self.image_counter}.jpg'), transformed_background)
            cv2.imwrite(os.path.join(self.parent_dir, self.augmentation_dataset_name, 'transformed/gt',
                        f'transformed_annotation_{self.image_counter}.jpg'), transformed_gt)
            self.image_counter += 1
        except Exception as e:
            print(f"Error saving transformed files: {e}")

    def save_augmented_files(self, file_name, augmented_image, augmented_annotation) -> None:
        """
        Save the augmented image and annotation to the output/augmented directory.

        Parameters:
        file_name (str): The base name for the output files.
        augmented_image (np.array): The augmented image.
        augmented_annotation (str): The augmented annotation.
        """
        file_name = file_name.split('/')[-1].split('.')[0]

        try:
            cv2.imwrite(os.path.join(self.parent_dir, self.augmentation_dataset_name, 'augmented/images',
                        f'augmented_{file_name}.jpg'), augmented_image)
            with open(os.path.join(self.parent_dir, self.augmentation_dataset_name, 'augmented/labels', f'augmented_{file_name}.txt'), 'w') as file:
                for row in augmented_annotation:

                    file.write(' '.join([str(int(row[0]))] + list(map(str, row[1:]))) + '\n')
        except Exception as e:
            print(f"Error saving augmented files: {e}")
