from inference.confirmation import ConfirmationConfig, ConfirmationEngine


def test_candidate_to_confirmed_to_cooldown():
    engine = ConfirmationEngine(ConfirmationConfig(candidate_threshold=0.7, inactivity_seconds=2.0, motion_threshold=0.2, cooldown_seconds=3.0))
    assert engine.step(0.8, 0.5, 1.0) == "candidate"
    assert engine.step(0.8, 0.1, 1.0) == "candidate"
    assert engine.step(0.8, 0.1, 1.0) == "confirmed"
    assert engine.step(0.2, 0.1, 1.0) == "cooldown"
