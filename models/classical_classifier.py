"""
전통적 ML 분류기
SVM, Random Forest 기반 행동 분류
"""
import sys
import os
import logging
import pickle

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACTIVITY_CLASSES, MODEL_DIR, NUM_CLASSES

logger = logging.getLogger(__name__)


class ClassicalClassifier:
    """전통 ML 분류기 (SVM, Random Forest)"""

    def __init__(self, model_type="svm"):
        """
        Args:
            model_type: "svm" 또는 "rf" (Random Forest)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def _create_model(self):
        """모델 생성"""
        if self.model_type == "svm":
            self.model = SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=42
            )
        elif self.model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"지원하지 않는 모델: {self.model_type}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        모델 학습
        
        Args:
            X_train: (n_samples, n_features)
            y_train: (n_samples,)
        """
        self._create_model()

        # 표준화
        X_scaled = self.scaler.fit_transform(X_train)

        # 학습
        logger.info(f"{self.model_type.upper()} 학습 시작...")
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

        # 훈련 정확도
        train_acc = self.model.score(X_scaled, y_train)
        logger.info(f"훈련 정확도: {train_acc:.4f}")

        # 검증
        if X_val is not None and y_val is not None:
            val_acc = self.evaluate(X_val, y_val)
            logger.info(f"검증 정확도: {val_acc:.4f}")
            return train_acc, val_acc

        return train_acc, None

    def predict(self, X):
        """예측"""
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """확률 예측"""
        if not self.is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X_test, y_test, verbose=True):
        """
        모델 평가
        
        Returns:
            accuracy: float
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        if verbose:
            print(f"\n{'='*50}")
            print(f"  {self.model_type.upper()} 분류 결과")
            print(f"{'='*50}")
            print(f"  정확도: {accuracy:.4f}")
            print(f"  F1 스코어 (가중): {f1:.4f}")
            print(f"\n{classification_report(y_test, y_pred, target_names=ACTIVITY_CLASSES)}")
            print(f"혼동 행렬:")
            print(confusion_matrix(y_test, y_pred))

        return accuracy

    def cross_validate(self, X, y, cv=5):
        """교차 검증"""
        X_scaled = self.scaler.fit_transform(X)
        self._create_model()
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="accuracy")
        logger.info(f"교차 검증 정확도: {scores.mean():.4f} ± {scores.std():.4f}")
        return scores

    def hyperparameter_search(self, X_train, y_train):
        """하이퍼파라미터 그리드 서치"""
        X_scaled = self.scaler.fit_transform(X_train)

        if self.model_type == "svm":
            param_grid = {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.01, 0.001],
                "kernel": ["rbf", "linear"]
            }
            base_model = SVC(probability=True, class_weight="balanced", random_state=42)
        else:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10]
            }
            base_model = RandomForestClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1
            )

        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring="accuracy",
            verbose=1, n_jobs=-1
        )
        grid_search.fit(X_scaled, y_train)

        logger.info(f"최적 파라미터: {grid_search.best_params_}")
        logger.info(f"최적 점수: {grid_search.best_score_:.4f}")

        self.model = grid_search.best_estimator_
        self.is_trained = True
        return grid_search.best_params_

    def save(self, filename=None):
        """모델 저장"""
        if filename is None:
            filename = f"{self.model_type}_classifier.pkl"
        filepath = os.path.join(MODEL_DIR, filename)
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        logger.info(f"모델 저장: {filepath}")

    def load(self, filename=None):
        """모델 로드"""
        if filename is None:
            filename = f"{self.model_type}_classifier.pkl"
        filepath = os.path.join(MODEL_DIR, filename)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        self.is_trained = True
        logger.info(f"모델 로드: {filepath}")
