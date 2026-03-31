"""
모델 학습 원스탑 스크립트
데이터 소스: 시뮬레이션 / Widar 3.0 / ESP32 수집 데이터
전처리: 기본 전처리 / RuView 기반 고급 전처리
"""
import sys
import os
import argparse
import logging
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ACTIVITY_CLASSES, RAW_DATA_DIR, MODEL_DIR
)
from data_collection.csi_simulator import CSISimulator
from preprocessing.dataset import CSIDataset
from models.classical_classifier import ClassicalClassifier
from models.lstm_classifier import LSTMClassifier
from evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


def evaluate_trained_model(model, X_test, y_test, model_name):
    """Run the shared evaluation suite and persist artifacts."""
    evaluator = ModelEvaluator()

    if hasattr(model, "model") and hasattr(model.model, "n_jobs"):
        model.model.n_jobs = 1

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = evaluator.evaluate(y_test, y_pred, y_proba=y_proba, model_name=model_name)
    metrics["inference_time"] = evaluator.measure_inference_time(model, X_test[:1], n_iterations=20)
    return metrics


def generate_data(samples_per_class=200, duration=3.0, seed=42):
    """시뮬레이션 데이터 생성"""
    print("=" * 60)
    print("  1단계: CSI 시뮬레이션 데이터 생성")
    print("=" * 60)

    sim = CSISimulator(seed=seed)
    dataset = sim.generate_dataset(
        samples_per_class=samples_per_class,
        duration_per_sample=duration
    )
    sim.save_dataset(dataset)
    print(f"  [완료] {len(dataset)}개 샘플 생성 완료\n")


def load_widar3_data(widar_dir, max_per_class=200):
    """Widar 3.0 데이터셋 로드 및 프로젝트 형식으로 변환"""
    from data_collection.widar3_loader import Widar3Loader

    print("=" * 60)
    print("  1단계: Widar 3.0 데이터셋 로드")
    print("=" * 60)

    loader = Widar3Loader(dataset_dir=widar_dir)
    samples, labels = loader.prepare_dataset(max_per_class=max_per_class)

    if samples:
        loader.save_as_project_format(samples, labels)
        print(f"  [완료] Widar 3.0 데이터 변환: {len(samples)}개\n")
    else:
        print("  [경고] Widar 3.0 데이터 없음. 시뮬레이션으로 전환\n")
        generate_data()


def preprocess_data(samples, use_advanced=False):
    """데이터 전처리 (기본 또는 고급)"""
    if use_advanced:
        from preprocessing.advanced_signal_processor import AdvancedSignalProcessor
        processor = AdvancedSignalProcessor()
        processed = []
        for sample in samples:
            result = processor.advanced_pipeline(sample)
            processed.append(result["cleaned"])
        logger.info("RuView 고급 전처리 적용 완료")
        return processed
    else:
        ds = CSIDataset()
        return ds.preprocess_all(samples)


def train_ml_model(model_type="svm", use_advanced=False):
    """전통 ML 모델 학습"""
    print("=" * 60)
    print(f"  {model_type.upper()} 모델 학습")
    print("=" * 60)

    ds = CSIDataset()
    samples, labels = ds.load_raw_data()
    processed = preprocess_data(samples, use_advanced)
    X, y = ds.prepare_ml_dataset(processed, labels, mode="all")

    X_train, X_val, X_test, y_train, y_val, y_test = ds.split_dataset(X, y)

    clf = ClassicalClassifier(model_type=model_type)
    start = time.time()
    clf.train(X_train, y_train, X_val, y_val)
    elapsed = time.time() - start
    print(f"  학습 시간: {elapsed:.2f}초")

    metrics = evaluate_trained_model(clf, X_test, y_test, model_type.upper())
    clf.save()
    print(f"\n  [완료] {model_type.upper()} 모델 저장 완료")
    return metrics


def train_lstm_model(model_arch="lstm", use_advanced=False, epochs=None):
    """LSTM 모델 학습"""
    print("=" * 60)
    print(f"  {'CNN-LSTM' if model_arch == 'cnn_lstm' else 'LSTM'} 모델 학습")
    print("=" * 60)

    ds = CSIDataset()
    samples, labels = ds.load_raw_data()
    processed = preprocess_data(samples, use_advanced)
    X, y = ds.prepare_dl_dataset(processed, labels)

    X_train, X_val, X_test, y_train, y_val, y_test = ds.split_dataset(X, y)

    lstm = LSTMClassifier(input_shape=X_train.shape[1:])
    if model_arch == "cnn_lstm":
        lstm.build_cnn_lstm_model()
    else:
        lstm.build_model()

    start = time.time()
    lstm.train(X_train, y_train, X_val, y_val, epochs=epochs)
    elapsed = time.time() - start
    print(f"  학습 시간: {elapsed:.2f}초")

    model_name = "CNN-LSTM" if model_arch == "cnn_lstm" else "LSTM"
    metrics = evaluate_trained_model(lstm, X_test, y_test, model_name)
    lstm.save()
    print(f"\n  [완료] LSTM 모델 저장 완료")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="CSI 행동 분류 모델 학습")
    parser.add_argument("--model", type=str, default="all",
                        choices=["svm", "rf", "lstm", "cnn_lstm", "all"],
                        help="학습할 모델 (기본: all)")
    parser.add_argument("--samples", type=int, default=200,
                        help="클래스당 시뮬레이션 샘플 수 (기본: 200)")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="샘플당 시간(초) (기본: 3.0)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="데이터 생성 건너뛰기 (기존 데이터 사용)")
    parser.add_argument("--widar-dir", type=str, default=None,
                        help="Widar 3.0 데이터셋 디렉토리 (BVP/CSI 포함)")
    parser.add_argument("--advanced", action="store_true",
                        help="RuView 기반 고급 전처리 사용 (Hampel, Fresnel, BVP)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override epochs for LSTM/CNN-LSTM training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    results = {}

    # 데이터 준비
    if args.widar_dir:
        load_widar3_data(args.widar_dir, max_per_class=args.samples)
    elif not args.skip_generate:
        generate_data(args.samples, args.duration, args.seed)

    # 모델 학습
    if args.model in ("svm", "all"):
        results["SVM"] = train_ml_model("svm", args.advanced)

    if args.model in ("rf", "all"):
        results["Random Forest"] = train_ml_model("rf", args.advanced)

    if args.model in ("lstm", "all"):
        results["LSTM"] = train_lstm_model("lstm", args.advanced, epochs=args.epochs)

    if args.model in ("cnn_lstm", "all"):
        results["CNN-LSTM"] = train_lstm_model("cnn_lstm", args.advanced, epochs=args.epochs)

    # 최종 결과 요약
    print("\n" + "=" * 60)
    print("  최종 결과 요약")
    print("=" * 60)
    for model_name, metrics in results.items():
        acc = metrics["accuracy"]
        status = "PASS" if acc >= 0.85 else "WARN"
        print(f"  [{status}] {model_name}: {acc:.4f} ({acc*100:.1f}%)")

    if results:
        report_path = ModelEvaluator().generate_report(results)
        best_model = max(results, key=lambda name: results[name]["accuracy"])
        print(f"\n  최고 성능: {best_model} ({results[best_model]['accuracy']*100:.1f}%)")
        print(f"  평가 보고서: {report_path}")


if __name__ == "__main__":
    main()
