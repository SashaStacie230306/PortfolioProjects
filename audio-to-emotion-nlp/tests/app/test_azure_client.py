"""Unit tests for src/app/azure_client.py."""

import importlib
import os
import sys
from types import SimpleNamespace

import pytest
import requests

# Path to the module under test
MODULE_PATH = "app.azure_client"


def reload_azure_client(env_vars: dict):
    """Reload the azure_client module with specified environment variables.

    Args:
        env_vars: Dict of environment variables to set before import.

    Returns:
        The reloaded azure_client module.
    """
    # Clear module so import-time code runs again
    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]
    # Apply env vars
    for k, v in env_vars.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return importlib.import_module(MODULE_PATH)


@pytest.fixture(autouse=True)
def set_dummy_env(monkeypatch):
    """Provide dummy AZURE_ENDPOINT_URL and AZURE_API_KEY for all tests.

    Then reload the azure_client module to pick them up.
    """
    env = {
        "AZURE_ENDPOINT_URL": "https://example.com/endpoint",
        "AZURE_API_KEY": "fake-api-key",
    }
    client = reload_azure_client(env)
    return client


def test_azure_predict_success(set_dummy_env, monkeypatch):
    """azure_predict should return parsed JSON on 2xx with no 'error'."""
    azure_client = set_dummy_env

    # Fake response from requests.post
    fake_json = {"predicted_label": "happy", "confidence": 88.5}
    fake_resp = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: fake_json,
    )
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: fake_resp)

    result = azure_client.azure_predict("test text")
    assert result == fake_json
    # Ensure headers include correct bearer token
    # The module builds headers internally, so we check via captured call
    # For simplicity here we trust the returned result


def test_azure_predict_model_error(set_dummy_env, monkeypatch):
    """If JSON contains 'error', azure_predict should raise RuntimeError."""
    azure_client = set_dummy_env

    fake_json = {"error": "bad input"}
    fake_resp = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: fake_json,
    )
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: fake_resp)

    with pytest.raises(RuntimeError) as exc:
        azure_client.azure_predict("anything")
    assert "Model error: bad input" in str(exc.value)


def test_azure_predict_timeout(set_dummy_env, monkeypatch):
    """If requests.post raises Timeout, azure_predict wraps it in RuntimeError."""
    azure_client = set_dummy_env

    def raise_timeout(*args, **kwargs):
        raise requests.Timeout("timeout happened")

    monkeypatch.setattr(requests, "post", raise_timeout)

    with pytest.raises(RuntimeError) as exc:
        azure_client.azure_predict("foo")
    assert "Azure prediction timed out:" in str(exc.value)
    assert "timeout happened" in str(exc.value)


def test_azure_predict_http_error(set_dummy_env, monkeypatch):
    """Test that azure_predict handles HTTP errors correctly.

    If response.raise_for_status raises HTTPError, azure_predict wraps with status/body.
    """
    azure_client = set_dummy_env

    # Build a fake HTTPError with response
    fake_response = SimpleNamespace(status_code=418, text="I'm a teapot")
    http_err = requests.HTTPError(response=fake_response)

    def post_error(*args, **kwargs):
        return SimpleNamespace(
            raise_for_status=lambda: (_ for _ in ()).throw(http_err), json=lambda: {}
        )

    monkeypatch.setattr(requests, "post", post_error)

    with pytest.raises(RuntimeError) as exc:
        azure_client.azure_predict("x")
    msg = str(exc.value)
    assert "Azure returned HTTP 418: I'm a teapot" in msg


def test_azure_predict_request_exception(set_dummy_env, monkeypatch):
    """If requests.post raises generic RequestException, azure_predict wraps it."""
    azure_client = set_dummy_env

    def raise_req_exc(*args, **kwargs):
        raise requests.RequestException("network fail")

    monkeypatch.setattr(requests, "post", raise_req_exc)

    with pytest.raises(RuntimeError) as exc:
        azure_client.azure_predict("bar")
    assert "Azure prediction failed: network fail" in str(exc.value)
