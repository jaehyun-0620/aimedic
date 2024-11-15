import pandas as pd
import os
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


num_images_to_load = 100  # 원하는 개수로 설정

metadata_path = 'train-metadata.csv'

print("Loading metadata and images...")

metadata = pd.read_csv(metadata_path, low_memory=False)

print("Metadata loaded. Preparing images...")

image_folder = 'train-image/image'  # 이미지 파일이 저장된 폴더 이름

images = []
labels = []

# 이미지와 라벨 준비
for _, row in metadata.head(num_images_to_load).iterrows():
    filename = row['isic_id'] + ".jpg"
    label = row['target']
    img_path = os.path.join(image_folder, filename)
    if os.path.exists(img_path):
        image = Image.open(img_path).resize((128, 128))
        image = np.array(image, dtype=np.float32) / 255.0
        images.append(image)
        labels.append(label)
print("Images and labels prepared. Total images:", len(images))

images = np.array(images)
labels = np.array(labels)

# 데이터 확인 - 첫 번째 이미지 출력
plt.imshow(images[0])
plt.title(f"Label: {labels[0]}")
plt.axis('off')
plt.show()

# 데이터셋 나누기 단계
print("Splitting dataset...")
labels = to_categorical(labels)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
print("Dataset split complete. Starting model setup...")

# 모델 구성 및 컴파일
print("Building the model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # 출력 노드를 1로 변경하고 활성화 함수를 sigmoid로 변경
])
print("Model built successfully. Compiling model...")

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("Model compiled. Starting training...")

# 모델 학습
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))
print("Training complete.")

loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")


num_samples = 5  # 예측할 샘플 개수
samples = random.sample(range(len(X_val)), num_samples)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(samples):
    # 예측 수행
    img = X_val[idx]
    actual_label = np.argmax(y_val[idx])  # 실제 레이블
    predicted_label = int(model.predict(np.expand_dims(img, axis=0)) > 0.5)  # 예측 레이블

    # 이미지 시각화
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(img)
    plt.title(f"Actual: {actual_label}\nPredicted: {predicted_label}")
    plt.axis('off')
plt.show()
