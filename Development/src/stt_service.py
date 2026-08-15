import httpx
from sarvamai import AsyncSarvamAI
from sarvamai.types.realtime_audio_input import RealtimeAudioInput
from sarvamai.types.realtime_end import RealtimeEnd

from config import DEBUG, SARVAM_API_KEY, SARVAM_STT_MODEL, SARVAM_BASE_URL

# Maps this system's language codes (see generation_service._LANGUAGE_NAMES)
# to Sarvam's BCP-47-style codes. Languages with no confirmed Sarvam Saarika
# *batch* support (ur, ne, as) fall back to "unknown", which triggers
# auto-detection there - the realtime model (saaras:v3-realtime) does
# support them directly, per its connect() language_code list.
_LANGUAGE_TO_SARVAM = {
    "en": "en-IN", "hi": "hi-IN", "gu": "gu-IN", "mr": "mr-IN",
    "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN", "ml": "ml-IN",
    "bn": "bn-IN", "pa": "pa-IN", "or": "od-IN",
    "ur": "ur-IN", "ne": "ne-IN", "as": "as-IN",
}
_SARVAM_TO_LANGUAGE = {v: k for k, v in _LANGUAGE_TO_SARVAM.items()}


def sarvam_code_to_language(sarvam_code: str, default: str = "en") -> str:
    """Reverse of _LANGUAGE_TO_SARVAM - maps Sarvam's detected BCP-47 code
    (e.g. "hi-IN", from a realtime transcript.final event's `language`
    field when language_code="auto" was used) back to this system's 2-letter
    code, for feeding into retrieval's language-matching and the LLM's
    answer-language instruction. Falls back to `default` if Sarvam detected
    a language this system otherwise doesn't have a name/config for."""
    return _SARVAM_TO_LANGUAGE.get(sarvam_code, default)


class SttService:
    """Speech-to-text via Sarvam AI's Saarika model.

    REST API, no official Python SDK - multipart file upload, see
    https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe.
    """

    def __init__(self):
        self.configured = bool(SARVAM_API_KEY) and SARVAM_API_KEY != "mock"
        self._client = httpx.Client(timeout=30.0) if self.configured else None

    def transcribe(self, audio_bytes: bytes, filename: str, language: str,
                    content_type: str = None) -> dict:
        """Returns {"transcript": str, "language_code": str}. Raises
        RuntimeError if unconfigured, or httpx.HTTPStatusError on a non-2xx
        response from Sarvam."""
        if not self.configured:
            raise RuntimeError("SARVAM_API_KEY not configured - cannot transcribe audio")

        language_code = _LANGUAGE_TO_SARVAM.get(language, "unknown")
        response = self._client.post(
            f"{SARVAM_BASE_URL}/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            data={"model": SARVAM_STT_MODEL, "language_code": language_code},
            files={"file": (filename or "audio.wav", audio_bytes, content_type or "audio/wav")},
        )
        if response.status_code >= 400:
            if DEBUG:
                print(f"[SttService] Sarvam STT {response.status_code}: {response.text[:500]}")
            response.raise_for_status()
        result = response.json()
        return {
            "transcript": (result.get("transcript") or "").strip(),
            "language_code": result.get("language_code") or language_code,
        }


class RealtimeSttSession:
    """One live saaras:v3-realtime WebSocket session (one per in-progress
    recording). Audio is streamed in as it's captured instead of uploaded as
    a single file after recording stops - transcription happens
    concurrently with speaking, so by the time the user stops there's
    little to no STT work left to do. This is what actually removes the
    ~300-500ms batch-API latency; see main_app.py's audio_stream_* handling
    for how chunks get here and how transcript.partial/transcript.final
    events get relayed back out.

    Verified against the real API (see conversation - not just SDK docs):
    encoding must be "linear16" (raw 16-bit signed PCM, mono), matching
    exactly what the dashboard's PCM capture pipeline sends."""

    def __init__(self, language: str):
        if not SARVAM_API_KEY or SARVAM_API_KEY == "mock":
            raise RuntimeError("SARVAM_API_KEY not configured - cannot transcribe audio")
        self._client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
        self._language_code = _LANGUAGE_TO_SARVAM.get(language, "auto")
        self._ctx = None
        self.socket = None

    async def __aenter__(self):
        self._ctx = self._client.speech_to_text_realtime_streaming.connect(
            language_code=self._language_code,
            model="saaras:v3-realtime",
            encoding="linear16",
            sample_rate="16000",
        )
        self.socket = await self._ctx.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._ctx is not None:
            await self._ctx.__aexit__(exc_type, exc_val, exc_tb)

    async def send_chunk(self, pcm_base64: str):
        await self.socket.send_realtime_audio_input(RealtimeAudioInput(audio=pcm_base64))

    async def end(self):
        await self.socket.send_realtime_end(RealtimeEnd())

    def __aiter__(self):
        return self.socket.__aiter__()
