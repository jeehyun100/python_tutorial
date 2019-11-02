from keras.preprocessing.image import ImageDataGenerator
import glob
import cv2
import numpy as np
import os


def image_data_augmentation():
    if not os.path.exists("./augmentation/train"):
        os.makedirs("augmentation/train")

    batch = []
    for file in glob.glob("./output/test/" + '*.png'):
        try:
            _img = cv2.imread(file)
            img = _img[..., ::-1].copy()
        except FileExistsError as e:
            None
        batch.append(img)
    all_images = np.array(batch)

    # # data augmentation
    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images
    # # (std, mean, and principal components if ZCA whitening is applied).

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.1,
        height_shift_range=0.001,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    total = 0
    datagen.fit(all_images)
    print("[INFO] generating images...")
    imageGen = datagen.flow(all_images, batch_size=1, save_to_dir="./augmentation/train",
                        save_prefix="class_11", save_format="png")
    total = 0
    # loop over examples from our image data augmentation generator
    for image in imageGen:
        # increment our counter
        total += 1
        print(total)
        # if we have reached the specified number of examples, break
        # from the loop
        if total == 2:
            break

if __name__ == '__main__':
    image_data_augmentation()