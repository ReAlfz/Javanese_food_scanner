import random
import shutil
import numpy as np

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras.losses import CategoricalCrossentropy
from keras.applications import MobileNet
import cv2 as cv2
import os


def make_to_jpeg(src_dir):
    output = 'Dataset/food_city'

    if not os.path.exists(output):
        os.mkdir(output)

    for class_dir in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_dir)
        dest_class_path = os.path.join(output, class_dir)

        if not os.path.exists(dest_class_path):
            os.mkdir(dest_class_path)

        for item in os.listdir(class_path):
            item_path = os.path.join(class_path, item)
            dest_item = os.path.join(dest_class_path, item)

            if not os.path.exists(dest_item):
                os.mkdir(dest_item)

            for i, img in enumerate(os.listdir(item_path)):
                image = cv2.imread(os.path.join(item_path, img))
                dest_path = os.path.join(dest_item, f'{class_dir}_{item}_{i}.jpg')
                image = cv2.resize(image, (128, 128))
                if not os.path.exists(dest_path):
                    cv2.imwrite(dest_path, image)
    print('Convert and resize done')


def split_dataset(dir):
    out = 'Dataset/final'
    train_size = 0.7
    val_size = 0.25

    if not os.path.exists(out):
        os.mkdir(out)

    for class_dir in os.listdir(dir):
        class_path = f'{dir}/{class_dir}'

        for item in os.listdir(class_path):
            path = f'{class_path}/{item}'
            images = os.listdir(path)
            random.shuffle(images)

            number_images = len(images)
            number_train = int(train_size * number_images)
            number_val = int(val_size * number_images)

            train_images = images[:number_train]
            val_images = images[number_train:number_train + number_val]
            test_images = images[number_train + number_val:]

            if not os.path.exists(f'{out}/train'):
                os.mkdir(f'{out}/train')
            for img in train_images:
                src = f'{path}/{img}'
                dst = f'{out}/train/{class_dir}'

                if not os.path.exists(dst):
                    os.mkdir(dst)
                if not os.path.exists(f'{dst}/{img}'):
                    shutil.copy(src, f'{dst}/{img}')

            if not os.path.exists(f'{out}/validation'):
                os.mkdir(f'{out}/validation')
            for img in val_images:
                src = f'{path}/{img}'
                dst = f'{out}/validation/{class_dir}'

                if not os.path.exists(dst):
                    os.mkdir(dst)
                if not os.path.exists(f'{dst}/{img}'):
                    shutil.copy(src, f'{dst}/{img}')

            if not os.path.exists(f'{out}/test'):
                os.mkdir(f'{out}/test')
            for img in test_images:
                src = f'{path}/{img}'
                dst = f'{out}/test/{class_dir}'

                if not os.path.exists(dst):
                    os.mkdir(dst)
                if not os.path.exists(f'{dst}/{img}'):
                    shutil.copy(src, f'{dst}/{img}')
    print('Split Dataset Done')


def preprocessing():
    dir = 'Dataset/final'

    # ** check images **
    fig, axs = plt.subplots(1, 3)
    for i, class_name in enumerate(os.listdir(f'{dir}/train')):
        folder_path = f'{dir}/train/{class_name}'
        img_path = f'{folder_path}/{os.listdir(folder_path)[0]}'
        img = plt.imread(img_path)
        axs[i].imshow(img)
        axs[i].set_title(class_name)
    plt.show()

    datagen = ImageDataGenerator(rescale=1/255)

    train_generator = datagen.flow_from_directory(
        directory=f'{dir}/train',
        target_size=(128, 128),
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        directory=f'{dir}/validation',
        target_size=(128, 128),
        color_mode='rgb',
        class_mode='categorical',
        shuffle=False
    )

    test_generator = datagen.flow_from_directory(
        directory=f'{dir}/test',
        target_size=(128, 128),
        color_mode='rgb',
        class_mode=None,
        shuffle=False
    )
    return train_generator, val_generator, test_generator


def modeling(train_generator, val_generator):
    pre_train = MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    pre_train.trainable = False
    model = Sequential([
        pre_train,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dense(15, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=['acc']
    )

    model.summary()
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
    )

    model.save('model.h5')
    return history


def predict(test_generator):
    model = load_model('model.h5')
    predictions = model.predict(test_generator)
    class_label = test_generator.class_indices
    predicted_label = [list(class_label.keys())[np.argmax(pred)] for pred in predictions]

    for i, (image_path, true_label) in enumerate(zip(test_generator.filepaths, test_generator.filenames)):
        img = image.load_img(image_path, target_size=(128, 128))
        img = image.img_to_array(img)
        img = img / 255.0  # Normalize the image data

        print(f"Image {i+1}:")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {predicted_label[i]}\n")

        plt.imshow(img)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    # ** convert all image to jpeg and resize **
    # dir_ = 'Dataset/data/'
    # make_to_jpeg(dir_)
    #
    # ** split dataset to train, val, and test **
    # dir2_ = 'Dataset/food_city'
    # split_dataset(dir2_)

    train_generator_, val_generator_, test_generator_ = preprocessing()
    # # history_ = modeling(train_generator_, val_generator_)
    # predict(test_generator_)