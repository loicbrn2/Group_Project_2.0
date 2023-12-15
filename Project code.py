import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def charger_images(dossier):
    images = []
    labels = []
    label_dict = {}

    for i, classe in enumerate(os.listdir(dossier)):
        label_dict[i] = classe

        for fichier in os.listdir(os.path.join(dossier, classe)):
            chemin_image = os.path.join(dossier, classe, fichier)
            image = cv2.imread(chemin_image)
            image = cv2.resize(image, (64, 224))  
            images.append(image)
            labels.append(i)
        print(len(images))
    return np.array(images), np.array(labels), label_dict

dossier_donnees = "flower_images"
X, y, label_dict = charger_images(dossier_donnees)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = models.Sequential() 
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 64, 3))) #removable 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32 * 2, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(label_dict), activation='softmax'))


model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model1=model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

print(f"Accuracy : {model1.history['accuracy'][-1]*100:.2f}")
print(f"Val Accuracy : {model1.history['val_accuracy'][-1]*100:.2f}")

y_pred = np.argmax(model.predict(X), axis=1)


cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_dict.values(), yticklabels=label_dict.values())
plt.xlabel('Prédiction')
plt.ylabel('Vraie classe')
plt.title('Matrice de confusion')
plt.show()


print("Rapport de classification :")
print(classification_report(y, y_pred, target_names=label_dict.values()))

dossier_test = "flower_images2"
X2, y2, label_dict2 = charger_images(dossier_test)

loss, accuracy = model.evaluate(X2, y2)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

dossier_test = "Testr"
X2, y2, label_dict2 = charger_images(dossier_test)

loss, accuracy = model.evaluate(X2, y2)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

y2_pred = model.predict(X2)
y2_pred_classes = np.argmax(y2_pred, axis=1)

conf_matrix2 = confusion_matrix(y2, y2_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Blues', xticklabels=label_dict2.values(), yticklabels=label_dict2.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


#WARNING : the rest of the code take a long time to compile

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# all_acc = []
# all_val_acc = []


# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 64, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(32 * 2, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(32 * 2, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(len(label_dict), activation='softmax'))


# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

  

# history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))


# train_accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']


# for epoch, (train_acc, val_acc) in enumerate(zip(train_accuracy, val_accuracy), 1):
#     print(f'Epoch {epoch}: Train Accuracy = {train_acc}, Validation Accuracy = {val_acc}')
#     all_acc.append(train_acc * 100) 
#     all_val_acc.append(val_acc * 100) 

# epoch_list = range(1, 31)  
# plt.plot(epoch_list, all_acc, label='Accuracy')
# plt.plot(epoch_list, all_val_acc, label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# all_acc2 = []
# all_val_acc2 = []


# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 64, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(32 * 2, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(32 * 2, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(len(label_dict), activation='softmax'))


# model.compile(optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy'])

# datagen = ImageDataGenerator(
#     rotation_range=10,  
#     width_shift_range=0.1,  
#     height_shift_range=0.1, 
#     shear_range=0.1,  
#     zoom_range=0.1,  
#     horizontal_flip=True,
#     fill_mode='nearest'
# )


# batch_size_values=[10,25,50,100,250,500]
# for x in batch_size_values :
#     m=model.fit(datagen.flow(X_train, y_train, batch_size=x), epochs=5, validation_data=(X_test, y_test))
#     all_acc2.append(m.history['accuracy'][-1] * 100)  # Ajout des valeurs d'accuracy à la liste
#     all_val_acc2.append(m.history['val_accuracy'][-1] * 100)  # Ajout des valeurs de val_accuracy à la liste

# plt.plot(batch_size_values, all_acc2, label='Accuracy')
# plt.plot(batch_size_values, all_val_acc2, label='Validation Accuracy')
# plt.xlabel('batch size')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# optimizer_options = ['adam', 'sgd', 'rmsprop']
# learning_rate_options = [0.001, 0.01, 0.1]
# num_filters_options = [32, 64]
# num_dense_units_options = [64, 128]

# for opt in optimizer_options:
#     for lr in learning_rate_options:
#         for nf in num_filters_options:
#             for ndu in num_dense_units_options:
                
#                 model = models.Sequential()
#                 model.add(layers.Conv2D(nf, (3, 3), activation='relu', input_shape=(224, 64, 3)))
#                 model.add(layers.MaxPooling2D((2, 2)))
#                 model.add(layers.Conv2D(nf * 2, (3, 3), activation='relu'))
#                 model.add(layers.MaxPooling2D((2, 2)))
#                 model.add(layers.Conv2D(nf * 2, (3, 3), activation='relu'))
#                 model.add(layers.Flatten())
#                 model.add(layers.Dense(ndu, activation='relu'))
#                 model.add(layers.Dense(len(label_dict), activation='softmax'))

                
#                 model.compile(optimizer=opt,
#                               loss='sparse_categorical_crossentropy',
#                               metrics=['accuracy'])

               
#                 model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

              
#                 val_loss, val_accuracy = model.evaluate(X_test, y_test)
#                 print(f"Optimiseur: {opt}, Taux d'apprentissage: {lr}, "
#                       f"Nombre de filtres: {nf}, Nombre d'unités dans la couche dense: {ndu}")
#                 print(f"Précision sur l'ensemble de validation: {val_accuracy * 100:.2f}%\n")



