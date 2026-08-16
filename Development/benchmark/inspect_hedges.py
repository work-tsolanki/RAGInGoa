"""
One-off: re-runs the 10 hedged queries (+ the 2 slow Gujarati queries) from
the percentile batch through the live WS pipeline, this time capturing the
actual answer text (percentile_batch_results.json only stored confidence/
timing, not the answer) so the hedges and the outlier can be inspected
directly instead of inferred from query text and numbers alone.
"""

import asyncio
import json
import sys
import time

sys.path.insert(0, '.')

import websockets

from config import RAG_API_KEY

WS_URL = "ws://127.0.0.1:8000/ws/query"
PACE_SECONDS = 10.0

TARGETS = [
    {"query": "What are voter ID requirements", "lang": "en"},
    {"query": "What is Goa famous for?", "lang": "en"},
    {"query": "What are the beaches like in Goa", "lang": "en"},
    {"query": "What documents are needed for a US passport card", "lang": "en"},
    {"query": "What is the history of Portuguese architecture in Goa", "lang": "en"},
    {"query": "What is the significance of Isthmia in Greek mythology", "lang": "en"},
    {"query": "गोवा किस लिए प्रसिद्ध है", "lang": "hi"},
    {"query": "મતદાર ID માટે શું જરૂરી છે", "lang": "gu"},
    {"query": "કાજુ કતલી શું છે", "lang": "gu"},
    {"query": "કંપની એટલે શું", "lang": "gu"},
    {"query": "ફેની શું છે", "lang": "gu"},
]


async def main():
    results = []
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"type": "auth", "api_key": RAG_API_KEY}))
        await ws.recv()

        for i, q in enumerate(TARGETS):
            t0 = time.time()
            await ws.send(json.dumps({"query_text": q["query"], "language": q["lang"]}))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                if msg.get("stage") == "final":
                    wall_ms = (time.time() - t0) * 1000
                    result = {
                        "query": q["query"], "lang": q["lang"],
                        "confidence": msg["confidence"], "hedged": msg.get("hedged", False),
                        "backend": msg.get("backend"),
                        "total_ms": msg["latency_breakdown"]["total"],
                        "wall_ms": round(wall_ms, 1),
                        "answer": msg["answer"],
                    }
                    results.append(result)
                    print(f"[{i+1}/{len(TARGETS)}] {q['lang']} hedged={result['hedged']!s:5s} "
                          f"conf={result['confidence']:.3f} total_ms={result['total_ms']:.0f} backend={result['backend']}", flush=True)
                    print(f"    Q: {q['query']}")
                    print(f"    A: {result['answer']}\n")
                    break
                elif msg.get("stage") == "error":
                    results.append({"query": q["query"], "lang": q["lang"], "error": msg.get("message")})
                    print(f"[{i+1}/{len(TARGETS)}] ERROR: {msg.get('message')}", flush=True)
                    break
            if i < len(TARGETS) - 1:
                await asyncio.sleep(PACE_SECONDS)

    with open("benchmark/hedge_inspection_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Written to benchmark/hedge_inspection_results.json")


if __name__ == "__main__":
    asyncio.run(main())
