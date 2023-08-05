import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dir = 'Datasets'
imageFile = 'Datas/imagesData.pickle';
modelFile = 'Models/imagesModel.sav';

categories = ['cat', 'dog'];
data = [];

#for category in categories:
#    path = os.path.join(dir, category);
#    label = categories.index(category);
#    for img in os.listdir(path):
#        imgPath = os.path.join(path, img);
#        petImage = cv2.imread(imgPath, 0);
#        try:
#            petImage = cv2.resize(petImage, (50, 50));
#            image = np.array(petImage).flatten();   
#            data.append([image, label]);
#        except Exception as e:
#            pass

#print('Quantidade de imagens: ' + str(len(data)));

#print('As imagens foram processadas e salvas no diretório ' + imageFile)

# Escrever Arquivo
#pickIn = open(imageFile, 'wb');
#pickle.dump(data, pickIn);
#pickIn.close();

#print('Imagens salva com sucesso!');

# =====================================================================

# Ler Arquivo
pickIn = open(imageFile, 'rb');
data = pickle.load(pickIn);
pickIn.close();

print('Quantidade de imagens: ' + str(len(data)));

random.shuffle(data);
features = [];
labels = [];

for feature, label in data:
    features.append(feature);
    labels.append(label);

# Separando os dados para treinamento e teste do modelo    
X_train, X_test, y_train, y_test  = train_test_split(features, labels, test_size=0.25);

# =====================================================================

kernel = 'poly';

print('Criando modelo utilizando o kernel ' + kernel);

# Criando meu modelo
model = SVC(kernel=kernel, C=1, gamma='auto');
model.fit(X_train, y_train);

print('Salvando o modelo em ' + modelFile);

pick = open(modelFile, 'wb');
pickle.dump(model, pick);
pick.close();

print('Modelo salvo com sucesso!');

# =====================================================================

# Ler arquivo do modelo
pick = open(modelFile, 'rb');
model = pickle.load(pick);
pick.close();

# =====================================================================

#Fazendo uma predição de uma imagem específica
#file = 'Datas/coidei.jpeg';
#petImage = cv2.imread(file, 0);
#try:
#    petImage = cv2.resize(petImage, (50, 50));
#    image = np.array(petImage).flatten();
#except Exception as e:
#    pass

#prediction = model.predict([image]);
#myPet = image.reshape(50, 50);

# =====================================================================

#Fazendo uma predição do dataset
prediction = model.predict(X_test);
myPet = X_test[0].reshape(50, 50);

# =====================================================================

acc = round(model.score(X_test, y_test) * 100, 2);

print('Predição: ' + categories[prediction[0]]);
print('Acurácia: ' + str(acc) + '%');

plt.imshow(myPet, cmap='gray');
plt.show();