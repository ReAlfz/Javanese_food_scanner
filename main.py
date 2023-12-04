import deeplearning, pretrain, classic
import random, shutil, os
from sklearn.model_selection import train_test_split
import cv2 as cv2
from keras.preprocessing.image import ImageDataGenerator


def make_to_jpeg(src_dir):
    output = 'Dataset/food_city'
    x, y = [], []

    if not os.path.exists(output):
        os.mkdir(output)

    label_mapping = {}  # Mapping from class names to numerical labels
    label_counter = 0

    for class_dir in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_dir)
        dest_class_path = os.path.join(output, class_dir)

        if not os.path.exists(dest_class_path):
            os.mkdir(dest_class_path)

        label_mapping[class_dir] = label_counter
        label_counter += 1

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

                x.append(image.flatten())
                y.append(class_dir)
    print('Convert and resize done')
    return x, y, label_mapping


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


def preprocessing(label_mapping):
    dir = 'Dataset/final'

    # ** check images **
    # fig, axs = plt.subplots(1, 4)
    # for i, class_name in enumerate(os.listdir(f'{dir}/train')):
    #     folder_path = f'{dir}/train/{class_name}'
    #     img_path = f'{folder_path}/{os.listdir(folder_path)[0]}'
    #     img = plt.imread(img_path)
    #     axs[i].imshow(img)
    #     axs[i].set_title(class_name)
    # plt.show()

    datagen = ImageDataGenerator(rescale=1/255)

    train_generator = datagen.flow_from_directory(
        directory=f'{dir}/train',
        target_size=(128, 128),
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True,
        classes=list(label_mapping.keys())
    )

    val_generator = datagen.flow_from_directory(
        directory=f'{dir}/validation',
        target_size=(128, 128),
        color_mode='rgb',
        class_mode='categorical',
        shuffle=False,
        classes=list(label_mapping.keys())
    )

    test_generator = datagen.flow_from_directory(
        directory=f'{dir}/test',
        target_size=(128, 128),
        color_mode='rgb',
        class_mode=None,
        shuffle=False
    )
    return train_generator, val_generator, test_generator


if __name__ == '__main__':
    dir_ = 'Dataset/data'
    x, y, label_map = make_to_jpeg(dir_)

    # dir2_ = 'Dataset/food_city'
    # split_dataset(dir2_)

    train_generator_, val_generator_, test_generator_ = preprocessing(label_map)

    # DeepLearning metode
    # pretrain.modeling(train_generator_, val_generator_)
    # deeplearning.modeling(train_generator_, val_generator_)

    # Classic metode
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    svc = classic.modelling(x_train, y_train)
    classic.evaluate(x_test, y_test, svc)

