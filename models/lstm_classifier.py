"""
LSTM 기반 딥러닝 분류기
CSI 시계열 데이터를 직접 입력받아 행동 분류 수행
"""
import sys
import os
import logging
import json

import numpy as np
import h5py

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TF 경고 억제
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NUM_CLASSES, ACTIVITY_CLASSES, MODEL_DIR,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    LEARNING_RATE, BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE,
    WINDOW_SIZE, CSI_NUM_SUBCARRIERS
)

logger = logging.getLogger(__name__)


class LSTMClassifier:
    """LSTM 기반 시계열 행동 분류 모델"""

    def __init__(self, input_shape=None, num_classes=None):
        """
        Args:
            input_shape: (window_size, n_subcarriers) 튜플
            num_classes: 분류 클래스 수
        """
        self.input_shape = input_shape or (WINDOW_SIZE, CSI_NUM_SUBCARRIERS)
        self.num_classes = num_classes or NUM_CLASSES
        self.model = None
        self.history = None

    def build_model(self):
        """LSTM 모델 구축"""
        model = keras.Sequential([
            # 입력 레이어
            layers.Input(shape=self.input_shape),

            # LSTM 레이어 1
            layers.LSTM(
                LSTM_HIDDEN_SIZE,
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.2
            ),
            layers.BatchNormalization(),

            # LSTM 레이어 2
            layers.LSTM(
                LSTM_HIDDEN_SIZE // 2,
                return_sequences=False,
                dropout=0.3,
                recurrent_dropout=0.2
            ),
            layers.BatchNormalization(),

            # 분류 헤드
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation="softmax")
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model = model
        logger.info("LSTM 모델 구축 완료")
        model.summary(print_fn=logger.info)
        return model

    def build_cnn_lstm_model(self):
        """CNN + LSTM 하이브리드 모델 (더 높은 성능)"""
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),

            # 1D CNN으로 로컬 패턴 추출
            layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),

            # LSTM으로 시계열 관계 학습
            layers.LSTM(LSTM_HIDDEN_SIZE, return_sequences=True, dropout=0.3),
            layers.LSTM(LSTM_HIDDEN_SIZE // 2, return_sequences=False, dropout=0.3),

            # 분류 헤드
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation="softmax")
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model = model
        logger.info("CNN-LSTM 하이브리드 모델 구축 완료")
        model.summary(print_fn=logger.info)
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=None, batch_size=None):
        """
        모델 학습
        
        Args:
            X_train: (n_samples, window_size, n_subcarriers)
            y_train: (n_samples,)
        """
        if self.model is None:
            self.build_model()

        epochs = epochs or EPOCHS
        batch_size = batch_size or BATCH_SIZE

        # 콜백 설정
        callback_list = [
            callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # 모델 체크포인트
        ckpt_path = os.path.join(MODEL_DIR, "lstm_best.h5")
        os.makedirs(MODEL_DIR, exist_ok=True)
        callback_list.append(
            callbacks.ModelCheckpoint(
                ckpt_path,
                monitor="val_accuracy" if X_val is not None else "accuracy",
                save_best_only=True,
                verbose=1
            )
        )

        # 클래스 가중치 계산 (불균형 처리)
        unique, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        class_weight = {int(c): total / (len(unique) * n) for c, n in zip(unique, counts)}

        validation_data = (X_val, y_val) if X_val is not None else None

        logger.info(f"학습 시작: epochs={epochs}, batch_size={batch_size}")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            class_weight=class_weight,
            callbacks=callback_list,
            verbose=1
        )

        return self.history

    def predict(self, X):
        """예측 (클래스 인덱스)"""
        proba = self.model.predict(X, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """확률 예측"""
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test, y_test, verbose=True):
        """
        모델 평가
        
        Returns:
            accuracy: float
        """
        from sklearn.metrics import classification_report, confusion_matrix

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.predict(X_test)

        if verbose:
            print(f"\n{'='*50}")
            print(f"  LSTM 분류 결과")
            print(f"{'='*50}")
            print(f"  손실: {loss:.4f}")
            print(f"  정확도: {accuracy:.4f}")
            print(f"\n{classification_report(y_test, y_pred, target_names=ACTIVITY_CLASSES)}")
            print(f"혼동 행렬:")
            print(confusion_matrix(y_test, y_pred))

        return accuracy

    def save(self, filename="lstm_classifier.h5"):
        """모델 저장"""
        filepath = os.path.join(MODEL_DIR, filename)
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model.save(filepath, save_format='h5')
        logger.info(f"모델 저장: {filepath}")

        # 학습 이력 저장
        if self.history:
            hist_path = os.path.join(MODEL_DIR, "lstm_history.json")
            hist_dict = {k: [float(v) for v in vals]
                         for k, vals in self.history.history.items()}
            with open(hist_path, "w") as f:
                json.dump(hist_dict, f)

    def load(self, filename="lstm_classifier.h5"):
        """모델 로드"""
        filepath = os.path.join(MODEL_DIR, filename)
        try:
            self.model = keras.models.load_model(filepath)
        except UnicodeDecodeError:
            # TensorFlow on Windows can fail on non-ASCII paths. Loading through
            # an already-opened HDF5 handle avoids that path decoding step.
            with h5py.File(filepath, "r") as model_file:
                self.model = keras.models.load_model(model_file)
        self.input_shape = self.model.input_shape[1:]
        logger.info(f"모델 로드: {filepath}")
