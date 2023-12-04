from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from keras.models import Sequential, load_model
from keras.applications import MobileNet
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.losses import CategoricalCrossentropy


def modeling(train_generator, val_generator):
    pre_train = MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    pre_train.trainable = False
    model = Sequential([
        pre_train,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dense(4, activation='softmax')
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

    model.compile()
    model.save('pretrain_model.h5')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()


def evaluate(test_generator):
    model = load_model('pretrain_model.h5')
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=-1)

    true_labels = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    print("Classification Report:")
    print(classification_report(true_labels, y_pred, target_names=class_names))

    cm = confusion_matrix(true_labels, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def predict(test_generator):
    model = load_model('pretrain_model.h5')
    predictions = model.predict(test_generator)
    class_label = test_generator.class_indices
    predicted_label = [list(class_label.keys())[np.argmax(pred)] for pred in predictions]

    # for i, (image_path, true_label) in enumerate(zip(test_generator.filepaths, test_generator.filenames)):
    #     img = image.load_img(image_path, target_size=(128, 128))
    #     img = image.img_to_array(img)
    #     img = img / 255.0  # Normalize the image data
    #
    #     print(f"Image {i+1}:")
    #     print(f"True Label: {true_label}")
    #     print(f"Predicted Label: {predicted_label[i]}\n")
    #
    #     plt.imshow(img)
    #     plt.axis('off')
    #     plt.show()
