"""Gemini Robotics-ER 2 client with a first-class replay cache.

Model: gemini-robotics-er-2-preview -- Google's embodied-reasoning VLM (vision,
function calling, thinking). Called through the Interactions API.

Note on determinism: the Interactions API exposes `seed`, NOT `temperature`. We set
seed=0 and thinking_level="low" and say so plainly in the README. Reproducibility of
the reported numbers comes from the replay cache, not from a temperature knob.

WHY THE CACHE KEY IS NOT THE PROMPT BYTES
-----------------------------------------
The key is (scene_id, condition, seed, step_index, call_kind) -- the coordinates of a
call inside the eval -- and deliberately NOT a hash of the request. Every prompt carries
a rendered frame, and those pixels come out of a floating-point physics engine: they are
reproducible on one machine and near-but-not-bit-identical on another. A prompt-bytes key
would therefore turn `make judge` into a wall of cache misses on a judge's laptop, which
is the one outcome the cache exists to prevent.

The prompt hash is still recorded, next to the response. When it does not match, the
cached response is still returned -- offline reproduction is the point -- but the call is
counted as REPLAY DRIFT and the count is reported. A replay whose prompts have shifted is
a weaker claim than one whose prompts match exactly, and the report says which one it is
rather than quietly presenting the second as the first.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
from dataclasses import asdict, dataclass, field

MODEL = os.environ.get("GEMINI_MODEL", "gemini-robotics-er-2-preview")

# Published Gemini Robotics-ER 2 rates. Thought tokens bill as OUTPUT (see _live).
PRICE_IN_PER_MTOK = 2.00
PRICE_OUT_PER_MTOK = 10.00


class CacheMiss(RuntimeError):
    """Raised when replay mode has no cached response for a call.

    Replay NEVER falls back to a live call. A silent fallback would make `make judge`
    quietly cost money, quietly need an API key, and quietly report numbers that were
    not the recorded ones -- three failures that all look like success from the outside.
    A miss is a loud error instead.
    """


@dataclass
class VLMCall:
    """One request, plus the eval coordinates that identify it in the cache."""

    scene_id: str
    condition: str
    seed: int
    step_index: int
    call_kind: str
    system: str
    text: str
    image_png: bytes | None = None
    tools: list = field(default_factory=list)

    def cache_key(self) -> str:
        """Machine-stable identity: where this call sits in the eval, not what it said."""
        return (f"{self.scene_id}_{self.condition}_s{self.seed}"
                f"_{self.step_index:03d}_{self.call_kind}")

    def prompt_hash(self) -> str:
        """Fingerprint of what was actually sent -- the drift detector, not the key."""
        image_sha = (hashlib.sha256(self.image_png).hexdigest()
                     if self.image_png is not None else None)
        payload = {"system": self.system, "text": self.text,
                   "tools": self.tools, "image_sha": image_sha}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class VLMResponse:
    text: str
    tool_calls: list
    input_tokens: int
    output_tokens: int
    model: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VLMResponse":
        return cls(text=data["text"], tool_calls=data["tool_calls"],
                   input_tokens=data["input_tokens"], output_tokens=data["output_tokens"],
                   model=data["model"])


class LLMClient:
    """Two modes, one interface: `live` calls the API, `replay` reads the cache.

    Both accumulate tokens, so a replayed run reports the same cost the recorded run
    actually incurred -- the headline cost number survives being reproduced offline.
    """

    def __init__(self, mode: str = "replay", cache_dir=None, model: str = MODEL):
        if mode not in ("replay", "live"):
            raise ValueError(f"mode must be 'replay' or 'live', got {mode!r}")
        self.mode = mode
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else pathlib.Path("cache")
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.drift_count = 0
        self._client = None

    # --- accounting ------------------------------------------------------
    def cost_usd(self) -> float:
        return (self.input_tokens / 1_000_000 * PRICE_IN_PER_MTOK
                + self.output_tokens / 1_000_000 * PRICE_OUT_PER_MTOK)

    # --- cache -----------------------------------------------------------
    def _cache_path(self, call: VLMCall) -> pathlib.Path:
        return self.cache_dir / f"{call.cache_key()}.json"

    def write_cache(self, call: VLMCall, response: VLMResponse) -> pathlib.Path:
        """Persist one response. Indented, sorted JSON so cache diffs are reviewable."""
        path = self._cache_path(call)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"prompt_hash": call.prompt_hash(), "response": response.to_dict()}
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        return path

    def _read_cache(self, call: VLMCall) -> VLMResponse:
        path = self._cache_path(call)
        if not path.exists():
            raise CacheMiss(
                f"no cached response for {call.cache_key()!r} at {path}. "
                "Replay mode never falls back to a live call. Either record this "
                "episode with `make judge-live` (needs SECRETS), or check that the "
                "cache directory is the one the recording was written to."
            )
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("prompt_hash") != call.prompt_hash():
            # Replay anyway -- offline reproduction is the point -- but count it. The
            # report shows the drift count so a shifted replay is never passed off as
            # a byte-exact one.
            self.drift_count += 1
        return VLMResponse.from_dict(record["response"])

    # --- calling ---------------------------------------------------------
    def complete(self, call: VLMCall) -> VLMResponse:
        response = self._read_cache(call) if self.mode == "replay" else self._live(call)
        self.calls += 1
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        return response

    def _lazy_client(self):
        if self._client is None:
            if not os.environ.get("GEMINI_API_KEY"):
                raise RuntimeError(
                    "GEMINI_API_KEY is not set, so live mode cannot run. Copy "
                    "SECRETS.example to SECRETS, put your key in it (get one at "
                    "https://aistudio.google.com/apikey), then "
                    "`set -a && . ./SECRETS && set +a`. Replay mode -- which is what "
                    "`make judge` runs -- needs no key at all."
                )
            from google import genai
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return self._client

    def _live(self, call: VLMCall) -> VLMResponse:
        parts = []
        if call.image_png is not None:
            parts.append({"type": "image",
                          "data": base64.b64encode(call.image_png).decode("ascii"),
                          "mime_type": "image/png"})
        parts.append({"type": "text", "text": call.text})

        result = self._lazy_client().interactions.create(
            model=self.model,
            system_instruction=call.system,
            input=parts,
            tools=call.tools,
            # `seed` is this API's only determinism lever; there is no temperature.
            generation_config={"seed": 0, "thinking_level": "low",
                               "max_output_tokens": 2048},
            store=False,
        )

        tool_calls = []
        for step in (getattr(result, "steps", None) or []):
            if getattr(step, "type", None) != "function_call":
                continue
            args = getattr(step, "arguments", None)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append({"name": getattr(step, "name", None),
                               "args": args or {},
                               "id": getattr(step, "id", None)})

        usage = getattr(result, "usage", None)
        inp = int(getattr(usage, "total_input_tokens", 0) or 0)
        # Thinking is billed at the output rate. A real episode came back with 2,344
        # output tokens of which a large share were thoughts; leaving them out would
        # understate the headline cost by more than a rounding error.
        out = (int(getattr(usage, "total_output_tokens", 0) or 0)
               + int(getattr(usage, "total_thought_tokens", 0) or 0))

        return VLMResponse(text=getattr(result, "output_text", "") or "",
                           tool_calls=tool_calls, input_tokens=inp,
                           output_tokens=out, model=self.model)
