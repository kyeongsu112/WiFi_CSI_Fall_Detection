"""
알림 시스템
이상상황 감지 시 경고 메시지 출력 및 이력 저장
"""
import sys
import os
import json
import time
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALERT_COOLDOWN, LOG_DIR

logger = logging.getLogger(__name__)


class AlertManager:
    """이상상황 알림 관리자"""

    def __init__(self, cooldown=None, log_dir=None):
        self.cooldown = cooldown or ALERT_COOLDOWN
        self.log_dir = log_dir or LOG_DIR
        self.last_alert_time = 0
        self.alert_history = []

        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir,
            f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    def trigger_alert(self, detection_result):
        """
        이상상황 알림 발생
        
        Args:
            detection_result: 감지 결과 딕셔너리
        """
        current_time = time.time()

        # 쿨다운 확인
        if current_time - self.last_alert_time < self.cooldown:
            return False

        self.last_alert_time = current_time

        alert = {
            "alert_id": len(self.alert_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "type": detection_result.get("predicted_class", "unknown"),
            "confidence": detection_result.get("confidence", 0),
            "probabilities": detection_result.get("probabilities", {}),
            "message": self._generate_message(detection_result)
        }

        self.alert_history.append(alert)

        # 콘솔 경고
        self._print_alert(alert)

        # 파일 저장
        self._save_alert(alert)

        logger.warning(f"[WARN] 이상상황 감지: {alert['message']}")
        return True

    def _generate_message(self, result):
        """알림 메시지 생성"""
        cls = result.get("predicted_class", "unknown")
        conf = result.get("confidence", 0)

        messages = {
            "falling": f"[ALERT] 낙상 감지! (신뢰도: {conf:.1%})",
        }

        return messages.get(cls, f"[WARN] 이상 행동 감지: {cls} (신뢰도: {conf:.1%})")

    def _print_alert(self, alert):
        """콘솔에 경고 출력"""
        print("\n" + "!" * 60)
        print(f"  [ALERT] 경고 #{alert['alert_id']}")
        print(f"  시각: {alert['timestamp']}")
        print(f"  메시지: {alert['message']}")
        print("!" * 60 + "\n")

    def _save_alert(self, alert):
        """알림 이력 JSON 파일에 추가 저장"""
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.alert_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"알림 저장 실패: {e}")

    def get_recent_alerts(self, count=10):
        """최근 알림 조회"""
        return self.alert_history[-count:]

    def get_alert_count(self):
        """총 알림 수"""
        return len(self.alert_history)
