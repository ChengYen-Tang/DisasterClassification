from PIL import Image
import numpy as np
import cv2
import os, sys
sys.path.append('./')

def get_files(path):
    images_path = []
    labels = []

    for label in os.listdir(path):
        for image_path in os.listdir(path+label):
            images_path.append(path + '/' + label + '/' + image_path)
            labels.append(label)

    # group = np.array([images_path, labels])
    # group = group.transpose()
    # rng = np.random.default_rng()
    # rng.shuffle(group)

    # images_path = np.array(group[:, 0])
    # labels = np.array(group[:, 1])

    images_path = np.array(images_path)
    labels = np.array(labels)

    for index, label in enumerate(os.listdir(path)):
        labels[labels == label] = index

    labels = labels.astype('int32')
    return images_path, labels 

def load_files(path):
    images_path, labels = get_files(path)

    images = []
    for image_path in images_path:
        image = Image.open(image_path)
        # Update orientation based on EXIF tags, if the file has orientation info.
        image = update_orientation(image)
        # Convert to OpenCV format
        image = convert_to_opencv(image)
        # If the image has either w or h greater than 1600 we resize it down respecting
        # aspect ratio such that the largest dimension is 1600
        image = resize_down_to_1600_max_dim(image)
        # We next get the largest center square
        h, w = image.shape[:2]
        min_dim = min(w,h)
        max_square_image = crop_center(image, min_dim, min_dim)
        # Resize that square down to 256x256
        augmented_image = resize_to_256_square(max_square_image)
        # Crop the center for the specified network_input_Size
        augmented_image = crop_center(augmented_image, 224, 224)
        images.append(augmented_image)

    images = np.array(images)

    return images, labels

def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

if __name__ == '__main__':
    load_files('./dataset/train/')
