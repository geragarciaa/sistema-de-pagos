"""
Tests for the FastAPI microservice (app.py) that exposes /transaction.
Requires: httpx (for TestClient), fastapi, pydantic.
"""

from fastapi.testclient import TestClient

# Import the FastAPI instance from app.py
# If your app file lives elsewhere (e.g., src/app.py), change the import to:
#   from src.app import app as fastapi_app
from app import app as fastapi_app

import pandas as pd
import tempfile
import os
from decision_engine import run, DECISION_ACCEPTED, DECISION_IN_REVIEW, DECISION_REJECTED

client = TestClient(fastapi_app)


def test_health():
    """Basic healthcheck should return status ok."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_config_contains_score_mapping():
    """Config endpoint should expose current rule thresholds/weights."""
    r = client.get("/config")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, dict)
    assert "score_to_decision" in payload
    assert "amount_thresholds" in payload


def test_transaction_in_review_path():
    """Typical medium-risk digital transaction from NEW user at night -> IN_REVIEW."""
    body = {
        "transaction_id": 42,
        "amount_mxn": 5200.0,
        "customer_txn_30d": 1,
        "geo_state": "Nuevo León",
        "device_type": "mobile",
        "chargeback_count": 0,
        "hour": 23,
        "product_type": "digital",
        "latency_ms": 180,
        "user_reputation": "new",
        "device_fingerprint_risk": "low",
        "ip_risk": "medium",
        "email_risk": "new_domain",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 42
    assert data["decision"] in ("ACCEPTED", "IN_REVIEW", "REJECTED")
    # With the current defaults (reject_at=10, review_at=4), this should lean to IN_REVIEW
    # If you tuned env vars REJECT_AT/REVIEW_AT, this assertion may need adjustment.
    assert data["decision"] == "IN_REVIEW"


def test_transaction_hard_block_rejection():
    """Chargebacks>=2 with ip_risk=high should trigger hard block -> REJECTED."""
    body = {
        "transaction_id": 99,
        "amount_mxn": 300.0,
        "customer_txn_30d": 0,
        "geo_state": "Nuevo León",
        "device_type": "mobile",
        "chargeback_count": 2,
        "hour": 12,
        "product_type": "digital",
        "latency_ms": 100,
        "user_reputation": "new",
        "device_fingerprint_risk": "low",
        "ip_risk": "high",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 99
    assert data["decision"] == "REJECTED"


def test_transaction_accepted_path():
    """Typical low-risk physical transaction from trusted user during day -> ACCEPTED."""
    body = {
        "transaction_id": 7,
        "amount_mxn": 1500.0,
        "customer_txn_30d": 5,
        "geo_state": "Jalisco",
        "device_type": "desktop",
        "chargeback_count": 0,
        "hour": 14,
        "product_type": "physical",
        "latency_ms": 80,
        "user_reputation": "trusted",
        "device_fingerprint_risk": "low",
        "ip_risk": "low",
        "email_risk": "low",
        "bin_country": "MX",
        "ip_country": "MX"
    }
    r = client.post("/transaction", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["transaction_id"] == 7
    assert data["decision"] == "ACCEPTED"

def test_run_basic_functionality():
    """Test basic functionality of run() with sample CSV data."""
    # Create test data
    test_data = pd.DataFrame({
        'transaction_id': [1, 2, 3],
        'amount_mxn': [1000.0, 5000.0, 10000.0],
        'customer_txn_30d': [1, 5, 0],
        'chargeback_count': [0, 0, 1],
        'hour': [12, 23, 14],
        'product_type': ['digital', 'physical', 'subscription'],
        'latency_ms': [100, 200, 3000],
        'user_reputation': ['new', 'trusted', 'high_risk'],
        'device_fingerprint_risk': ['low', 'medium', 'high'],
        'ip_risk': ['low', 'medium', 'high'],
        'email_risk': ['low', 'medium', 'high'],
        'bin_country': ['MX', 'MX', 'US'],
        'ip_country': ['MX', 'US', 'MX']
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file:
        input_path = input_file.name
        test_data.to_csv(input_path, index=False)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_file:
        output_path = output_file.name
    
    try:
        # Run the function
        result_df = run(input_path, output_path)
        
        # Verify the result DataFrame has expected columns
        assert 'decision' in result_df.columns
        assert 'risk_score' in result_df.columns
        assert 'reasons' in result_df.columns
        
        # Verify decisions are valid
        valid_decisions = {DECISION_ACCEPTED, DECISION_IN_REVIEW, DECISION_REJECTED}
        assert all(decision in valid_decisions for decision in result_df['decision'])
        
        # Verify risk scores are integers
        assert all(isinstance(score, (int, float)) for score in result_df['risk_score'])
        
        # Verify output file was created
        assert os.path.exists(output_path)
        
        # Verify output file content matches result DataFrame
        output_df = pd.read_csv(output_path)
        # Handle NaN values in reasons column by converting to empty strings
        output_df["reasons"] = output_df["reasons"].fillna("")
        result_df["reasons"] = result_df["reasons"].fillna("")
        
        pd.testing.assert_frame_equal(result_df, output_df)
        
    finally:
        # Cleanup
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)