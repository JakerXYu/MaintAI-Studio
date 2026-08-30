"""Unit tests for the retraining recommendation rule engine."""

from maintai.monitoring.contracts import DriftReport
from maintai.monitoring.recommend import RecommendConfig, recommend


def _report(severity="LOW"):
    return DriftReport(overall_severity=severity, features=[])


def test_no_triggers_no_recommendation():
    rec = recommend(_report())
    assert rec.recommended is False
    assert rec.auto_deploy is False
    assert rec.triggers == []


def test_high_drift_trigger():
    rec = recommend(_report("HIGH"))
    assert rec.recommended is True
    assert "high_drift" in rec.triggers


def test_performance_drop_trigger():
    assert "performance_drop" in recommend(_report(), performance_drop=0.2).triggers
    assert "performance_drop" not in recommend(_report(), performance_drop=0.01).triggers


def test_anomaly_rate_jump_trigger():
    rec = recommend(_report(), anomaly_rate=0.2, baseline_anomaly_rate=0.05)
    assert "anomaly_rate_jump" in rec.triggers
    rec2 = recommend(_report(), anomaly_rate=0.1, baseline_anomaly_rate=0.05)
    assert "anomaly_rate_jump" not in rec2.triggers


def test_schedule_trigger():
    assert "schedule" in recommend(_report(), schedule_due=True).triggers
    assert "schedule" in recommend(_report(), days_since_last_retrain=40).triggers
    assert "schedule" not in recommend(_report(), days_since_last_retrain=5).triggers


def test_manual_trigger():
    assert "manual" in recommend(_report(), manual=True).triggers


def test_any_trigger_sets_recommended():
    rec = recommend(
        _report("HIGH"),
        performance_drop=0.3,
        anomaly_rate=0.5,
        baseline_anomaly_rate=0.0,
        schedule_due=True,
        manual=True,
    )
    assert rec.recommended is True
    assert len(rec.triggers) == 5


def test_auto_deploy_always_false():
    rec = recommend(_report("HIGH"), manual=True)
    assert rec.auto_deploy is False


def test_evidence_quantified_per_trigger():
    rec = recommend(_report("HIGH"), performance_drop=0.3)
    assert {e.trigger for e in rec.evidence} == set(rec.triggers)
    for evidence in rec.evidence:
        assert evidence.detail
        assert evidence.trigger in rec.triggers


def test_configurable_thresholds():
    config = RecommendConfig(performance_drop_threshold=0.5)
    assert recommend(_report(), performance_drop=0.3, config=config).recommended is False
    assert recommend(_report(), performance_drop=0.6, config=config).recommended is True


def test_proposed_action_present_when_recommended():
    rec = recommend(_report("HIGH"))
    assert rec.proposed_action
    assert "no automatic" in rec.proposed_action.lower()
