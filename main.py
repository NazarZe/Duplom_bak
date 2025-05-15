import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import pickle
import scipy.io

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set up logging
logging.basicConfig(filename="pipeline_log.txt", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PARAMETERS
IMG_SIZE = (224, 224)
NUM_CLASSES = 10
BATCH_LIMIT_PER_CLASS = 100
DATA_DIR = "images"

# Load images and labels
def load_images_and_labels(base_dir, img_size, max_per_class):
    X, y, class_names = [], [], []
    class_dirs = sorted(os.listdir(base_dir))[:NUM_CLASSES]
    for label, class_name in enumerate(class_dirs):
        class_path = os.path.join(base_dir, class_name)
        images = glob.glob(os.path.join(class_path, "*.jpg"))[:max_per_class]
        class_names.append(class_name)
        for img_path in images:
            img = load_img(img_path, target_size=img_size)
            img = img_to_array(img)
            X.append(img)
            y.append(label)
    logging.info(f"Завантажено {len(X)} зображень з {len(class_names)} класів")
    return np.array(X), np.array(y), class_names

# Extract features
def extract_features(model, preprocess_fn, X):
    X_prep = preprocess_fn(X.copy())
    return model.predict(X_prep, verbose=0)

# Train classifiers
def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_name):
    results = []
    models = {
        "SVM": SVC(kernel='linear'),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    with open(f"scaler_{feature_name}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    for name, model in models.items():
        model_path = f"model_{feature_name}_{name}.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"[INFO] Модель {name} завантажено з диску.")
        else:
            model.fit(X_train, y_train)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        start = time.time()
        y_pred = model.predict(X_test)
        duration = time.time() - start
        acc = accuracy_score(y_test, y_pred)
        print(f"{name}: Accuracy = {acc:.4f}, Time = {duration:.2f}s")
        print("Звіт:")
        print(classification_report(y_test, y_pred))
        logging.info(f"{feature_name}-{name}: Accuracy={acc:.4f}, Time={duration:.2f}s")
        results.append((name, acc, duration))
    return results

# Predict all models
def predict_all_models(image_path, class_names):
    print(f"\n Класифікація зображення {image_path} усіма моделями...\n")
    img = load_img(image_path, target_size=IMG_SIZE)
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)

    for cnn_name, (cnn_model, preprocess_fn) in cnn_models.items():
        try:
            print(f" {cnn_name}")
            features = cnn_model.predict(preprocess_fn(img_arr), verbose=0)
            scaler_path = f"scaler_{cnn_name}.pkl"
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            features_scaled = scaler.transform(features)

            for model_name in ["SVM", "MLP", "RandomForest"]:
                model_path = f"model_{cnn_name}_{model_name}.pkl"
                if not os.path.exists(model_path):
                    print(f"⚠️ Модель {model_name} не знайдена.")
                    continue
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                prediction = model.predict(features_scaled)
                predicted_class = class_names[int(prediction[0])]
                print(f"🔹 {model_name}: {predicted_class}")
        except Exception as e:
            print(f"❗ Помилка при класифікації з {cnn_name}-{model_name}: {e}")

    try:
        print("\n Stanford-MAT модель (ExtraTrees)")
        with open("model_MAT_ExtraTrees.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler_mat.pkl", "rb") as f:
            scaler = pickle.load(f)

        flat_img = img_to_array(load_img(image_path, target_size=(224, 224))).flatten().reshape(1, -1)
        flat_img_scaled = scaler.transform(flat_img)
        prediction = model.predict(flat_img_scaled)
        print(f"🔹 MAT-ExtraTrees: Клас {int(prediction[0])}")
    except Exception as e:
        print(f"❗ Помилка у MAT-класифікації: {e}")

# MAIN
X_train_img = X_test_img = y_train = y_test = class_names = None
if os.path.exists(DATA_DIR):
    X, y, class_names = load_images_and_labels(DATA_DIR, IMG_SIZE, BATCH_LIMIT_PER_CLASS)
    X_train_img, X_test_img, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
else:
    print(f"[WARNING] Папка '{DATA_DIR}' не знайдена. Завантаження зображень пропущено.")
    class_names = [f"Class {i}" for i in range(NUM_CLASSES)]

cnn_models = {
    "ResNet50": (ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=IMG_SIZE + (3,)), resnet_preprocess),
    "MobileNetV2": (MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=IMG_SIZE + (3,)), mobilenet_preprocess),
    "EfficientNetB0": (EfficientNetB0(weights="imagenet", include_top=False, pooling="avg", input_shape=IMG_SIZE + (3,)), efficientnet_preprocess),
}

all_results = []
if X_train_img is not None:
    for cnn_name, (cnn_model, preprocess_fn) in cnn_models.items():
        print(f"\n[INFO] Extracting features with {cnn_name}...")
        start_time = time.time()
        X_train_feat = extract_features(cnn_model, preprocess_fn, X_train_img)
        X_test_feat = extract_features(cnn_model, preprocess_fn, X_test_img)
        feature_time = time.time() - start_time
        logging.info(f"{cnn_name}: Feature extraction time: {feature_time:.2f}s")

        results = train_and_evaluate_models(X_train_feat, X_test_feat, y_train, y_test, cnn_name)
        for model_name, acc, model_time in results:
            all_results.append({
                "FeatureExtractor": cnn_name,
                "Model": model_name,
                "Accuracy": acc,
                "TrainTime": model_time,
                "FeatureGenTime": feature_time
            })

try:
    if all(os.path.exists(f) for f in ['train_data.mat', 'test_data.mat', 'train_list.mat', 'test_list.mat']):
        print("\n[INFO] Завантаження попередньо обчислених ознак з .mat")
        train_raw = scipy.io.loadmat('train_data.mat')['train_data']
        test_raw = scipy.io.loadmat('test_data.mat')['test_data']
        train_labels = scipy.io.loadmat('train_list.mat')['labels'].flatten()
        test_labels = scipy.io.loadmat('test_list.mat')['labels'].flatten()

        #  Фільтрація лише 10 перших класів
        train_mask = train_labels < 10
        test_mask = test_labels < 10

        train_data = train_raw[train_mask][:3000]
        test_data = test_raw[test_mask][:1000]
        train_labels = train_labels[train_mask][:3000]
        test_labels = test_labels[test_mask][:1000]

        print("\n[INFO] Навчання моделей на .mat ознаках (класи 0–9)")

        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        with open("scaler_mat.pkl", "wb") as f:
            pickle.dump(scaler, f)

        models = {
            "LinearSVC": SVC(kernel='linear'),
            "LogisticRegression": MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42),
            "ExtraTrees": RandomForestClassifier(n_estimators=50, random_state=42)
        }

        for name, model in models.items():
            model_path = f"model_MAT_{name}.pkl"
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                print(f"[INFO] Модель (MAT) {name} завантажено з диску.")
            else:
                print(f"\n Навчання  моделі (MAT): {name}")
                model.fit(train_data, train_labels)
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            predictions = model.predict(test_data)
            acc = accuracy_score(test_labels, predictions)
            print(f" Точність (MAT): {acc:.4f}")
            print(" Звіт (MAT):")
            print(classification_report(test_labels, predictions, zero_division=0))
            logging.info(f"MAT-{name}: Accuracy={acc:.4f}")
            all_results.append({
                "FeatureExtractor": "Stanford-MAT",
                "Model": name,
                "Accuracy": acc,
                "TrainTime": 0,
                "FeatureGenTime": 0
            })
    else:
        print("[WARNING] Один або кілька .mat файлів не знайдено. Блок MAT пропущено.")
        logging.warning("Пропущено обробку .mat файлів: деякі з них відсутні.")
except Exception as e:
    print("[WARNING] Не вдалося обробити .mat файли:", e)
    logging.warning("Помилка при роботі з .mat файлами: " + str(e))


# Save results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv("model_comparison_results.csv", index=False)

# Plot accuracy results
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="FeatureExtractor", y="Accuracy", hue="Model")
plt.title("Model Accuracy by Feature Extractor")
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()

# Plot train time
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="FeatureExtractor", y="TrainTime", hue="Model")
plt.title("Training Time by Feature Extractor")
plt.ylabel("Time (seconds)")
plt.tight_layout()
plt.savefig("train_time_comparison.png")
plt.show()

# Plot feature generation time
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="FeatureExtractor", y="FeatureGenTime")
plt.title("Feature Extraction Time by Network")
plt.ylabel("Time (seconds)")
plt.tight_layout()
plt.savefig("feature_time_comparison.png")
plt.show()

predict_all_models("MyDog.jpg", class_names)
