from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime

import httpx

from toolery.core.models import TURN_SEPARATOR, Message, Scenario, ToolCall, TraceResult, TurnUsage
from toolery.core.text_utils import strip_reasoning_tags
from toolery.tools.mock_runtime import MockToolRuntime
from toolery.tools.registry import ToolRegistry

_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 30.0


# A dead or unreachable server should fail fast, not after the whole budget.
_CONNECT_TIMEOUT_S = 10.0


class MalformedToolCall(RuntimeError):
    """The server could not parse the tool call the model wrote (llama-server
    answers HTTP 500 "Failed to parse tool call arguments as JSON"). The
    model's fault, not the server's: at temperature 0 a retry writes the
    same broken call, so none is made."""


class ScenarioTimeBudgetExhausted(TimeoutError):
    """The scenario's time budget ran out before the next request could be
    sent or answered."""


def _request_timeout(deadline: float | None) -> httpx.Timeout | object:
    """Per-request timeout that follows the scenario's time budget.

    The client's fixed default used to cap every request at 120s, whatever
    --timeout-scale said: a reasoning model whose single turn took longer was
    cut by httpx before the runner's budget ever applied. Each request now
    gets whatever remains until the scenario deadline (read/write/pool), and a
    short connect timeout so a dead server fails fast.
    """
    if deadline is None:
        return httpx.USE_CLIENT_DEFAULT
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ScenarioTimeBudgetExhausted("scenario time budget exhausted")
    return httpx.Timeout(remaining, connect=min(_CONNECT_TIMEOUT_S, remaining))


def _retry_delay(attempt: int, resp: httpx.Response) -> float:
    """Seconds to wait before the next attempt. Honors a numeric ``Retry-After``
    header (RFC 7231 delta-seconds form) when present, otherwise falls back to
    capped exponential backoff (0.5, 1, 2, 4, ... ≤ 30s)."""
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), _BACKOFF_MAX_SECONDS)
        except ValueError:
            pass  # HTTP-date form unsupported; fall back to backoff
    return min(_BACKOFF_BASE_SECONDS * (2 ** attempt), _BACKOFF_MAX_SECONDS)


def _normalise_base_url(base_url: str) -> str:
    """OpenAI-compatible endpoints serve under `/vN/chat/completions`. Local
    vLLM/llama.cpp callers pass the bare host (`http://localhost:8888`) and
    expect the adapter to add `/v1`. We auto-detect to avoid the
    `/v1/v1/chat/completions` 404 if the caller already included `/v1`."""
    url = base_url.rstrip("/")
    if any(seg.startswith("v") and seg[1:].isdigit() for seg in url.split("/")):
        return url
    return url + "/v1"


_DECODER = json.JSONDecoder()


def _as_call(obj: object, allowed: set[str]) -> tuple[str, dict] | None:
    """(name, arguments) when `obj` is a tool call written as JSON:
    {"name", "arguments"|"parameters"}, possibly wrapped in "function" or
    "tool_call". Only names of tools the scenario exposes count."""
    if not isinstance(obj, dict):
        return None
    for key in ("tool_call", "function"):
        if isinstance(obj.get(key), dict) and "name" in obj[key]:
            obj = obj[key]
    name = obj.get("name")
    if not isinstance(name, str) or name not in allowed:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {"_raw": args}
    return name, args if isinstance(args, dict) else {"_raw": args}


def parse_text_tool_calls(text: str | None, allowed: set[str]) -> list[tuple[str, dict]]:
    """Tool calls a model wrote as JSON inside its reply, instead of in its
    native format. Neural-v1-24B (a Mistral Small 3.2 merge) answers
    ```python {"name": "read_file", "arguments": "{\"path\": ...}"} ``` and
    never emits [TOOL_CALLS], so llama-server returns plain text. Every
    JSON object or array in the text is tried, in order; nested objects of a
    call are not counted twice."""
    calls: list[tuple[str, dict]] = []
    i, text = 0, text or ""
    while i < len(text):
        if text[i] not in "{[":
            i += 1
            continue
        try:
            obj, end = _DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        found = [call for item in (obj if isinstance(obj, list) else [obj])
                 if (call := _as_call(item, allowed))]
        if found:
            calls += found
            i = end
        else:
            i += 1
    return calls


class OpenAIRawAdapter:
    name = "raw"
    version = "0.2"
    # The budget is enforced here but never shown to the model, so raising the
    # cutoff (Runner.budget_slack) leaves the model's trajectory unchanged up
    # to the strict limit — which is what makes strict_view() exact. Adapters
    # that put the budget in the prompt (hermes) must not opt in.
    supports_budget_slack = True

    def __init__(self, base_url: str, api_key: str = "", concurrency: int = 4,
                 max_retries: int = 4, text_tool_calls: bool = False,
                 max_tokens: int = 0) -> None:
        # Cap on the tokens generated per request when the scenario sets
        # none (0 = no cap). Mistral Small 3.2 at temperature 0 can loop until
        # the context is full: 64,600 tokens and ten minutes for one request
        # on hard-03-async-polling. A loop fails either way; the cap only
        # stops it sooner.
        self.max_tokens = max_tokens
        # Measurement only: also accept tool calls written as JSON in the
        # reply's text. Results carry the adapter name "raw-text" so they
        # never mix with a normal raw run.
        self.text_tool_calls = text_tool_calls
        if text_tool_calls:
            self.name = "raw-text"
        self.base_url = _normalise_base_url(base_url)
        self.api_key = api_key
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=120)

    async def aclose(self):
        await self._client.aclose()

    async def _post_with_retry(self, payload: dict, headers: dict,
                               deadline: float | None = None) -> tuple[httpx.Response, int]:
        """POST with retry on transient rate-limit (429) / server (5xx) errors.

        Cloud endpoints (MiniMax, OpenRouter, ...) throttle with 429 and may
        return transient 5xx from an overloaded gateway; without retry a single
        throttle gets recorded as a model failure and silently depresses the
        quality score. Other 4xx (bad request, auth) are real client errors and
        are NOT retried.

        Returns (response, latency_ms) where latency_ms times ONLY the
        successful attempt — retry backoff is excluded so the value is a clean
        denominator for effective-throughput math."""
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            resp = await self._client.post(url, json=payload, headers=headers,
                                           timeout=_request_timeout(deadline))
            latency_ms = int((time.monotonic() - t0) * 1000)
            if resp.status_code == 500 and "parse tool call" in resp.text:
                try:
                    detail = resp.json()["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    detail = resp.text[:300]
                raise MalformedToolCall(f"malformed tool call: {detail}")
            retryable = resp.status_code == 429 or 500 <= resp.status_code < 600
            if retryable and attempt < self.max_retries:
                await asyncio.sleep(_retry_delay(attempt, resp))
                continue
            resp.raise_for_status()
            return resp, latency_ms
        raise RuntimeError("unreachable")  # pragma: no cover

    async def run_scenario(self, scenario: Scenario, model: str, timeout: int) -> TraceResult:
        started = time.monotonic()
        # The runner's budget for the whole scenario. Requests time out against
        # this deadline themselves, a little before the runner's hard stop
        # (timeout + 5s), so a slow trial returns its partial trace — how far
        # the model got — instead of an empty one.
        deadline = started + timeout
        runtime = MockToolRuntime(scenario)
        reg = ToolRegistry.default()
        tools_schema = reg.openai_schemas(scenario.tools)

        messages: list[dict] = []
        if scenario.system_prompt:
            messages.append({"role": "system", "content": scenario.system_prompt})
        history_start = len(messages)
        messages.extend({"role": m.role, "content": m.content} for m in scenario.history)
        history_end = len(messages)
        messages.append({"role": "user", "content": scenario.prompt})

        sampling = scenario.sampling
        sampling_sent: dict = {"temperature": sampling.temperature if sampling else 0.0}
        if sampling:
            # min_p is a llama.cpp / KoboldCPP extension to the OpenAI body.
            for key in ("min_p", "seed", "max_tokens"):
                if getattr(sampling, key) is not None:
                    sampling_sent[key] = getattr(sampling, key)

        allowed_tools = {t["function"]["name"] for t in tools_schema}
        text_calls_parsed = 0
        if self.max_tokens and "max_tokens" not in sampling_sent:
            sampling_sent["max_tokens"] = self.max_tokens
        tool_calls_recorded: list[ToolCall] = []
        usage_records: list[TurnUsage] = []
        final_response: str | None = None
        pending_follow_ups = list(scenario.follow_ups)
        replies: list[str] = []
        error: str | None = None
        turn_idx = 0
        try:
            for _ in range(scenario.budget.max_turns + 1 + len(scenario.follow_ups)):
                payload = {
                    "model": model, "messages": messages,
                    **sampling_sent,
                }
                # vLLM (and some other servers) reject `tools: []` with HTTP 400.
                # Only include the key when the scenario actually exposes tools.
                if tools_schema:
                    payload["tools"] = tools_schema
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                resp, req_latency_ms = await self._post_with_retry(payload, headers, deadline)
                data = resp.json()
                usage = data.get("usage") or {}
                usage_records.append(TurnUsage(
                    turn_index=turn_idx,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    latency_ms=req_latency_ms,
                ))
                msg = data["choices"][0]["message"]
                # Reasoning-model fallback: some servers (vLLM with DeepSeek/QwQ etc.)
                # put the final assistant text in `reasoning` or `reasoning_content` when
                # `content` is null. Normalize so downstream sees one field.
                if not msg.get("content"):
                    fallback = msg.get("reasoning") or msg.get("reasoning_content")
                    if fallback:
                        msg["content"] = fallback
                # Strip <think>/<thinking>/<reasoning> blocks from content. Some
                # models (MiniMax-M2, DeepSeek-R1, QwQ, etc.) emit chain-of-thought
                # inline in content; without stripping, structured-output checks
                # (JSON/YAML schema, regex, response_contains) fail on otherwise
                # correct answers and the model carries its own scratchpad into
                # subsequent turn context.
                msg["content"] = strip_reasoning_tags(msg.get("content"))
                if self.text_tool_calls and tools_schema and not msg.get("tool_calls"):
                    parsed = parse_text_tool_calls(msg.get("content"), allowed_tools)
                    if parsed:
                        text_calls_parsed += len(parsed)
                        msg["tool_calls"] = [
                            {"id": f"t{turn_idx:02d}{k:06d}", "type": "function",
                             "function": {"name": name, "arguments": json.dumps(args)}}
                            for k, (name, args) in enumerate(parsed)]
                messages.append(msg)
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    if scenario.follow_ups:
                        replies.append(msg.get("content") or "")
                        if pending_follow_ups:
                            messages.append({"role": "user", "content": pending_follow_ups.pop(0)})
                            turn_idx += 1
                            continue
                        final_response = TURN_SEPARATOR.join(replies)
                    else:
                        final_response = msg.get("content")
                    break
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {"_raw": raw_args}
                    result, kind = runtime.respond(name, args)
                    tool_calls_recorded.append(ToolCall(
                        index=turn_idx, name=name, args=args,
                        result=result, result_kind=kind,
                    ))
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"],
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    })
                    if len(tool_calls_recorded) > scenario.budget.max_tool_calls:
                        break
                turn_idx += 1
                if len(tool_calls_recorded) > scenario.budget.max_tool_calls:
                    break
        except (httpx.TimeoutException, TimeoutError) as e:
            # Worded to classify as failure_kind "timeout" — same bucket as the
            # runner's hard stop, but with the partial trace kept.
            error = f"timeout: scenario time budget of {timeout}s exhausted ({type(e).__name__})"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        duration_ms = int((time.monotonic() - started) * 1000)
        metadata: dict = {"base_url": self.base_url, "model": model}
        if self.text_tool_calls:
            metadata["text_tool_calls"] = text_calls_parsed
        traced = [Message.model_validate(m) for m in messages]
        if scenario.sampling:
            metadata["sampling"] = sampling_sent
        if scenario.conversation:
            # The frozen conversation is identified by its hash, not copied
            # into every trace (a 64k-token history per trial adds up fast).
            conv = scenario.conversation
            metadata["conversation"] = conv.model_dump(exclude={"fixture"})
            marker = Message(role="system", content=(
                f"[frozen conversation {conv.name} ({conv.length}), "
                f"{history_end - history_start} messages omitted, sha256 {conv.sha256}]"))
            traced = traced[:history_start] + [marker] + traced[history_end:]
        return TraceResult(
            scenario_id=scenario.id,
            adapter=self.name,
            trial_index=0,
            messages=traced,
            tool_calls=tool_calls_recorded,
            final_response=final_response,
            started_at_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            duration_ms=duration_ms,
            error=error,
            adapter_metadata=metadata,
            usage=usage_records,
        )
