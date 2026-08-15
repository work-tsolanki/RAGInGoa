import json
import sys

import httpx
import pytest

sys.path.insert(0, '.')

from src.stt_service import SttService


@pytest.fixture
def service():
    svc = SttService.__new__(SttService)  # skip __init__'s SARVAM_API_KEY check
    svc.configured = True
    svc._client = None
    return svc


def test_transcribe_raises_when_unconfigured():
    svc = SttService.__new__(SttService)
    svc.configured = False
    svc._client = None
    with pytest.raises(RuntimeError, match="SARVAM_API_KEY"):
        svc.transcribe(b"fake audio bytes", "clip.wav", "en")


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json_body = json_body
        self.status_code = status_code
        self.text = json.dumps(json_body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json_body


def test_transcribe_parses_response_and_maps_language_code(service, monkeypatch):
    captured = {}

    class _FakeClient:
        def post(self, url, headers, data, files):
            captured["url"] = url
            captured["headers"] = headers
            captured["data"] = data
            captured["files"] = files
            return _FakeResponse({"transcript": "  Hello world  ", "language_code": "hi-IN"})

    service._client = _FakeClient()

    result = service.transcribe(b"fake audio bytes", "clip.wav", "hi")

    assert result == {"transcript": "Hello world", "language_code": "hi-IN"}
    assert captured["data"]["language_code"] == "hi-IN"
    assert captured["files"]["file"][0] == "clip.wav"


def test_transcribe_falls_back_to_unknown_language_code(service, monkeypatch):
    captured = {}

    class _FakeClient:
        def post(self, url, headers, data, files):
            captured["data"] = data
            return _FakeResponse({"transcript": "namaste", "language_code": "unknown"})

    service._client = _FakeClient()

    result = service.transcribe(b"fake audio bytes", "clip.wav", "zz")  # not in _LANGUAGE_TO_SARVAM

    assert captured["data"]["language_code"] == "unknown"
    assert result["transcript"] == "namaste"


def test_transcribe_passes_through_content_type(service):
    captured = {}

    class _FakeClient:
        def post(self, url, headers, data, files):
            captured["files"] = files
            return _FakeResponse({"transcript": "hello", "language_code": "en-IN"})

    service._client = _FakeClient()

    service.transcribe(b"fake audio bytes", "recording.webm", "en", content_type="audio/webm;codecs=opus")

    assert captured["files"]["file"] == ("recording.webm", b"fake audio bytes", "audio/webm;codecs=opus")


def test_transcribe_defaults_content_type_when_not_given(service):
    captured = {}

    class _FakeClient:
        def post(self, url, headers, data, files):
            captured["files"] = files
            return _FakeResponse({"transcript": "hello", "language_code": "en-IN"})

    service._client = _FakeClient()

    service.transcribe(b"fake audio bytes", "clip.wav", "en")

    assert captured["files"]["file"][2] == "audio/wav"


def test_transcribe_raises_http_status_error_on_failure(service):
    class _FakeClient:
        def post(self, url, headers, data, files):
            return _FakeResponse({"error": "bad request"}, status_code=400)

    service._client = _FakeClient()

    with pytest.raises(httpx.HTTPStatusError):
        service.transcribe(b"fake audio bytes", "clip.wav", "en")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
