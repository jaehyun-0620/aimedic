import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 가정: images와 labels에 이미지 데이터와 레이블이 각각 numpy 배열로 준비됨
# 데이터 전처리
num_classes = 2  # 이진 분류일 경우, 클래스 수에 따라 변경
labels = to_categorical(labels, num_classes)  # 레이블 원-핫 인코딩
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# ResNet50 모델 불러오기 (사전 학습된 가중치 사용, 분류층 제외)
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# ResNet50 모델에 새로운 분류층 추가
model = Sequential()
model.add(resnet_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # 클래스 수에 따라 조정

# 사전 학습된 층을 고정 (ResNet50 가중치를 고정하여 학습되지 않도록 설정)
resnet_base.trainable = False

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 데이터 증강 (선택 사항)
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

# 모델 학습
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=10,
                    validation_data=(X_val, y_val))

# 학습 결과 시각화
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
