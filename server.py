"""Model Crawl - local web app for finding and downloading Hugging Face models/datasets.

Run:  python server.py   (then open http://127.0.0.1:8765)
"""
import fnmatch
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
HOST = os.environ.get("MODEL_CRAWL_HOST", "127.0.0.1")
PORT = int(os.environ.get("MODEL_CRAWL_PORT", "8765"))
# Extra host names the UI may be reached by, e.g. when HOST=0.0.0.0 on a home server.
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"} | {
    h.strip().lower() for h in os.environ.get("MODEL_CRAWL_ALLOWED_HOSTS", "").split(",") if h.strip()
}

# ---------------------------------------------------------------- config

SECRET_FIELDS = ["hf_token", "brave_api_key", "anthropic_api_key", "openai_api_key"]
DEFAULTS = {
    "hf_token": "",
    "brave_api_key": "",
    "llm_provider": "none",  # none | anthropic | openai_compatible
    "anthropic_api_key": "",
    "anthropic_model": "claude-opus-5",
    "openai_base_url": "http://localhost:11434/v1",  # Ollama / LM Studio / vLLM / OpenRouter ...
    "openai_api_key": "",
    "openai_model": "",
    "download_dir": str(APP_DIR / "downloads"),
    "max_concurrent": 1,  # queued downloads that may run at the same time
}
ENV_FALLBACKS = {
    "hf_token": "HF_TOKEN",
    "brave_api_key": "BRAVE_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
}
_config_lock = threading.Lock()


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    for key, env in ENV_FALLBACKS.items():
        if not cfg.get(key) and os.environ.get(env):
            cfg[key] = os.environ[env]
    return cfg


def save_config(cfg):
    with _config_lock:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def mask(value):
    return f"…{value[-4:]}" if value else ""


def public_config(cfg):
    out = {k: v for k, v in cfg.items() if k not in SECRET_FIELDS}
    for key in SECRET_FIELDS:
        out[key] = mask(cfg.get(key, ""))
        out[key + "_set"] = bool(cfg.get(key))
    out["llm_ready"] = llm_ready(cfg)
    return out


def llm_ready(cfg):
    if cfg["llm_provider"] == "anthropic":
        return bool(cfg["anthropic_api_key"] and cfg["anthropic_model"])
    if cfg["llm_provider"] == "openai_compatible":
        return bool(cfg["openai_base_url"] and cfg["openai_model"])
    return False


def hf_api():
    # None falls back to a token saved by `hf auth login`, if any.
    return HfApi(token=load_config()["hf_token"] or None)


# ---------------------------------------------------------------- app

@asynccontextmanager
async def lifespan(_app):
    load_jobs()
    threading.Thread(target=scheduler_loop, daemon=True).start()
    yield
    stop_workers()


app = FastAPI(title="Model Crawl", lifespan=lifespan)


@app.middleware("http")
async def local_only(request: Request, call_next):
    # Guard against DNS-rebinding: only answer requests addressed to this machine.
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]").lower()
    if host not in ALLOWED_HOSTS:
        return JSONResponse({"detail": "Forbidden host"}, status_code=403)
    return await call_next(request)


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/api/config")
def get_config():
    return public_config(load_config())


class ConfigUpdate(BaseModel):
    values: dict
    clear: list[str] = []


@app.post("/api/config")
def update_config(body: ConfigUpdate):
    cfg = load_config()
    for key, value in body.values.items():
        if key not in DEFAULTS:
            continue
        value = (value or "").strip() if isinstance(value, str) else value
        if key in SECRET_FIELDS and not value:
            continue  # blank secret field means "keep the saved one"
        if key == "max_concurrent":
            value = min(max(int(value or 1), 1), 8)
        cfg[key] = value
    for key in body.clear:
        if key in SECRET_FIELDS:
            cfg[key] = ""
    save_config(cfg)
    _wake.set()  # a higher concurrency limit may let queued jobs start
    return public_config(cfg)


# ---------------------------------------------------------------- repo helpers

HF_URL_RE = re.compile(r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/(datasets/)?([^/\s?#]+/[^/\s?#]+)", re.I)
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
RESERVED_OWNERS = {
    "docs", "blog", "spaces", "papers", "learn", "models", "datasets", "organizations", "collections",
    "posts", "tasks", "settings", "join", "login", "pricing", "api", "chat", "enterprise", "search", "new",
}


def parse_repo_ref(text, repo_type):
    """Accept 'owner/name', 'datasets/owner/name' or a huggingface.co URL."""
    text = text.strip()
    m = HF_URL_RE.match(text)
    if m:
        return ("dataset" if m.group(1) else "model"), m.group(2)
    if text.lower().startswith("datasets/"):
        return "dataset", text[len("datasets/"):]
    return repo_type, text


def looks_like_repo_id(text):
    return bool(HF_URL_RE.match(text.strip()) or REPO_ID_RE.match(text.strip().removeprefix("datasets/")))


def summarize(info, repo_type):
    card = getattr(info, "card_data", None) or {}
    card = card.to_dict() if hasattr(card, "to_dict") else dict(card)
    tags = getattr(info, "tags", None) or []
    base = next((t.split(":", 2)[2] for t in tags if t.startswith("base_model:quantized:")), None)
    last = getattr(info, "last_modified", None)
    return {
        "id": info.id,
        "repo_type": repo_type,
        "author": getattr(info, "author", None),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "gated": getattr(info, "gated", False) or False,
        "private": getattr(info, "private", False) or False,
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "library": getattr(info, "library_name", None),
        "license": card.get("license"),
        "last_modified": last.isoformat() if last else None,
        "quantized_from": base,
        "url": f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{info.id}",
    }


# Quantization names found in file names, e.g. Q4_K_M, IQ3_XXS, UD-Q4_K_XL, BF16, MXFP4.
QUANT_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:UD-)?(?:I?Q[1-8](?:_[A-Z0-9]{1,4})*|BF16|F16|FP16|F32|FP32|FP8|MXFP4|TQ[12]_0))(?![A-Za-z0-9])",
    re.I,
)
SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}", re.I)
ONNX_TAGS = {"fp16", "q4", "q4f16", "int8", "uint8", "quantized", "bnb4", "fp32", "q8"}
WEIGHT_EXTS = {
    ".safetensors": "safetensors", ".bin": "pytorch", ".pt": "pytorch", ".pth": "pytorch",
    ".gguf": "gguf", ".onnx": "onnx", ".h5": "tensorflow", ".msgpack": "flax", ".npz": "mlx",
}
META_EXTS = {".json", ".txt", ".md", ".model", ".tiktoken", ".py", ".jinja", ".yaml", ".yml", ".vocab"}


def ext_of(path):
    name = path.rsplit("/", 1)[-1].lower()
    return "." + name.rsplit(".", 1)[-1] if "." in name else ""


def is_metadata(f):
    name = f["path"].rsplit("/", 1)[-1].lower()
    small = (f["size"] or 0) < 50 * 1024 * 1024
    return small and (ext_of(f["path"]) in META_EXTS or name.startswith(("license", "tokenizer", "readme", "notice")))


def build_variants(files):
    """Group weight files into selectable download options (quantizations / formats)."""
    groups = {}

    def add(kind, key, label, f):
        g = groups.setdefault((kind, key), {"id": f"{kind}:{key}", "kind": kind, "label": label, "files": [], "size": 0})
        g["files"].append(f["path"])
        g["size"] += f["size"] or 0

    for f in files:
        path, low = f["path"], f["path"].lower()
        if low.endswith(".gguf"):
            if "mmproj" in low:
                add("extra", "mmproj", f"mmproj · {path.rsplit('/', 1)[-1]}", f)
                continue
            matches = list(QUANT_RE.finditer(path))
            if matches:
                key = matches[-1].group(1).upper()
            else:
                key = SPLIT_RE.sub("", path.rsplit("/", 1)[-1])[:-5]
            add("gguf", key, key, f)
        elif low.endswith((".onnx", ".onnx_data")):
            stem = low.rsplit("/", 1)[-1].split(".onnx")[0]
            tag = stem.rsplit("_", 1)[-1] if "_" in stem else ""
            key = tag if tag in ONNX_TAGS else "fp32"
            add("onnx", key, f"ONNX {key}", f)

    variants = sorted(groups.values(), key=lambda g: (g["kind"] == "extra", g["size"]))

    # When a repo ships the same weights in several formats, let the user pick one.
    formats = {}
    for f in files:
        fmt = WEIGHT_EXTS.get(ext_of(f["path"]))
        if fmt and fmt not in ("gguf", "onnx"):
            formats.setdefault(fmt, []).append(f)
    if len(formats) > 1 or (formats and variants):
        for fmt, fs in formats.items():
            variants.append({
                "id": f"format:{fmt}", "kind": "format", "label": f"{fmt} weights only",
                "files": [f["path"] for f in fs], "size": sum(f["size"] or 0 for f in fs),
            })
    return variants


def quantized_alternatives(base_id, exclude):
    try:
        found = hf_api().list_models(filter=f"base_model:quantized:{base_id}", sort="downloads", limit=30)
        out = []
        for m in found:
            if m.id == exclude:
                continue
            fmt = next((t for t in ("GGUF", "AWQ", "GPTQ", "MLX", "EXL2", "EXL3", "bnb", "FP8", "ONNX")
                        if t.lower() in m.id.lower() or t.lower() in (m.tags or [])), "other")
            out.append({"id": m.id, "downloads": m.downloads, "likes": m.likes, "format": fmt})
        return out[:20]
    except Exception:
        return []


# ---------------------------------------------------------------- resolve / verify

@app.get("/api/resolve")
def resolve(repo: str, repo_type: str = "model", revision: str | None = None):
    repo_type, repo_id = parse_repo_ref(repo, repo_type)
    api = hf_api()
    try:
        info = api.repo_info(repo_id, repo_type=repo_type, revision=revision or None, files_metadata=True)
    except GatedRepoError:
        raise HTTPException(403, f"{repo_id} is gated: accept its terms on huggingface.co with the account of your token.")
    except RevisionNotFoundError:
        raise HTTPException(404, f"Revision '{revision}' not found in {repo_id}.")
    except RepositoryNotFoundError:
        return {"exists": False, "repo_id": repo_id, "repo_type": repo_type,
                "suggestions": hf_search(repo_id.split("/")[-1], repo_type, limit=8)}
    except HfHubHTTPError as exc:
        raise HTTPException(502, f"Hugging Face error: {exc}")

    files = [{"path": s.rfilename, "size": s.size or 0} for s in (info.siblings or [])]
    summary = summarize(info, repo_type)

    branches = []
    try:
        branches = [b.name for b in api.list_repo_refs(repo_id, repo_type=repo_type).branches]
    except Exception:
        pass

    alternatives = []
    if repo_type == "model":
        alternatives = quantized_alternatives(summary["quantized_from"] or repo_id, exclude=repo_id)

    return {
        "exists": True,
        "repo_id": info.id,
        "repo_type": repo_type,
        "revision": revision or "main",
        "info": summary,
        "files": files,
        "total_size": sum(f["size"] for f in files),
        "variants": build_variants(files),
        "metadata_files": [f["path"] for f in files if is_metadata(f)],
        "branches": sorted(branches, key=lambda b: (b != "main", b)),
        "alternatives": alternatives,
    }


# ---------------------------------------------------------------- search

def hf_search(query, repo_type, limit=15):
    api = hf_api()
    lister = api.list_datasets if repo_type == "dataset" else api.list_models
    results = []
    # The Hub search is substring-based, so also try dash/space variants of a phrase,
    # then fall back to its individual words, longest first.
    words = sorted((w for w in re.split(r"\s+", query) if len(w) > 2), key=len, reverse=True)
    for q in dict.fromkeys([query, query.replace(" ", "-"), query.replace(" ", ""), *words]):
        try:
            results = list(lister(search=q, sort="downloads", limit=limit))
        except Exception:
            results = []
        if results:
            break
    return [{"id": r.id, "repo_type": repo_type, "downloads": r.downloads, "likes": r.likes,
             "source": ["hf"], "verified": True} for r in results]


def brave_search(query, repo_type, key):
    scope = "huggingface.co/datasets" if repo_type == "dataset" else "huggingface.co"
    r = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": f"site:{scope} {query}", "count": 20},
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
        timeout=15,
    )
    r.raise_for_status()
    seen, out = set(), []
    for item in (r.json().get("web") or {}).get("results", []):
        m = HF_URL_RE.match(item.get("url", ""))
        if not m:
            continue
        rtype = "dataset" if m.group(1) else "model"
        rid = m.group(2)
        if rid.split("/")[0].lower() in RESERVED_OWNERS or (rtype, rid) in seen:
            continue
        seen.add((rtype, rid))
        out.append({"id": rid, "repo_type": rtype, "title": item.get("title"), "source": ["brave"], "verified": None})
    return out


def verify_candidates(cands):
    """Confirm web-search hits actually exist on the Hub (in parallel)."""
    api = hf_api()

    def check(c):
        try:
            info = api.repo_info(c["id"], repo_type=c["repo_type"])
            c.update(verified=True, id=info.id, downloads=getattr(info, "downloads", None), likes=getattr(info, "likes", None))
        except GatedRepoError:
            c.update(verified=True, gated=True)
        except Exception:
            c["verified"] = False
        return c

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(check, cands))


def combined_search(query, repo_type):
    cfg = load_config()
    hf = hf_search(query, repo_type)
    brave, brave_error = [], None
    if cfg["brave_api_key"]:
        try:
            brave = verify_candidates(brave_search(query, repo_type, cfg["brave_api_key"]))
        except Exception as exc:
            brave_error = str(exc)
    merged = {(c["repo_type"], c["id"].lower()): c for c in hf}
    for c in brave:
        key = (c["repo_type"], c["id"].lower())
        if key in merged:
            merged[key]["source"].append("brave")
        elif c["verified"] is not False:
            merged[key] = c
    rejected = [c["id"] for c in brave if c["verified"] is False]
    ranked = sorted(merged.values(), key=lambda c: (c["repo_type"] != repo_type, -(c.get("downloads") or 0)))
    return {"results": ranked[:25], "brave_used": bool(cfg["brave_api_key"]), "brave_error": brave_error,
            "unverified_web_hits": rejected}


@app.get("/api/search")
def search(q: str, repo_type: str = "model"):
    if not q.strip():
        raise HTTPException(400, "Empty query")
    return combined_search(q.strip(), repo_type)


# ---------------------------------------------------------------- LLM request interpretation

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "repo_type": {"type": "string", "enum": ["model", "dataset"]},
        "repo_id": {"type": "string", "description": "Exact 'owner/name' if you are confident, else empty"},
        "search_query": {"type": "string", "description": "Short query to find the repo on the Hub, e.g. 'Qwen2.5-7B-Instruct GGUF'"},
        "quantization": {"type": "string", "description": "Requested quant like Q4_K_M, Q8_0, IQ4_XS, BF16, AWQ, GPTQ, MLX-4bit; empty if none"},
        "explanation": {"type": "string", "description": "One or two sentences on how you read the request"},
    },
    "required": ["repo_type", "repo_id", "search_query", "quantization", "explanation"],
    "additionalProperties": False,
}

INTENT_SYSTEM = """You turn a user's request into a Hugging Face Hub lookup.
Decide whether they want a model or a dataset, the most likely repo id, a short search query, and any quantization.
Map loose wording to concrete quant names: "4-bit"/"q4" for GGUF -> Q4_K_M, "8-bit" -> Q8_0, "half precision" -> F16/BF16.
If they want a quantized build (GGUF/AWQ/GPTQ/MLX) of a base model, the search query should target that format,
e.g. "Llama-3.1-8B-Instruct GGUF". Keep the specific quant (Q5_K_M etc.) out of search_query: one GGUF repo
usually holds every quant, and the quant goes in the quantization field. Only fill repo_id when you are confident it exists; otherwise leave it empty.
Respond with JSON only, matching this schema: """ + json.dumps(INTENT_SCHEMA)


def call_llm(cfg, user_text):
    if cfg["llm_provider"] == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=cfg["anthropic_api_key"])
        model = cfg["anthropic_model"]
        extra = {}
        if model in ("claude-opus-5", "claude-fable-5-1"):
            # Server-side fallback: re-run on another model if the request is declined.
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        resp = client.beta.messages.create(
            model=model,
            max_tokens=4000,
            system=INTENT_SYSTEM,
            messages=[{"role": "user", "content": user_text}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": INTENT_SCHEMA}},
            **extra,
        )
        if resp.stop_reason == "refusal":
            raise HTTPException(422, "The model declined this request.")
        text = next((b.text for b in resp.content if b.type == "text"), "")
    elif cfg["llm_provider"] == "openai_compatible":
        headers = {"Authorization": f"Bearer {cfg['openai_api_key']}"} if cfg["openai_api_key"] else {}
        r = httpx.post(
            cfg["openai_base_url"].rstrip("/") + "/chat/completions",
            headers=headers,
            json={"model": cfg["openai_model"], "temperature": 0,
                  "messages": [{"role": "system", "content": INTENT_SYSTEM}, {"role": "user", "content": user_text}]},
            timeout=120,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    else:
        raise HTTPException(400, "No LLM configured. Open Settings to add one.")

    m = re.search(r"\{.*\}", text, re.S)  # local models sometimes wrap JSON in prose / code fences
    if not m:
        raise HTTPException(502, f"LLM did not return JSON: {text[:300]}")
    intent = json.loads(m.group(0))
    return {k: intent.get(k) or ("model" if k == "repo_type" else "") for k in INTENT_SCHEMA["properties"]}


class InterpretBody(BaseModel):
    text: str


@app.post("/api/interpret")
def interpret(body: InterpretBody):
    cfg = load_config()
    try:
        intent = call_llm(cfg, body.text)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"LLM call failed: {type(exc).__name__}: {exc}")

    verified = None
    if intent["repo_id"] and looks_like_repo_id(intent["repo_id"]):
        try:
            rtype, rid = parse_repo_ref(intent["repo_id"], intent["repo_type"])
            info = hf_api().repo_info(rid, repo_type=rtype)
            verified = {"id": info.id, "repo_type": rtype}
        except GatedRepoError:
            verified = {"id": intent["repo_id"], "repo_type": intent["repo_type"]}
        except Exception:
            verified = None
    found = combined_search(intent["search_query"] or intent["repo_id"] or body.text, intent["repo_type"])
    return {"intent": intent, "verified": verified, **found}


# ---------------------------------------------------------------- downloads: queue, scheduler, workers
#
# Job status: scheduled -> queued -> running -> completed | failed | cancelled
# "Download now" starts a job immediately, whatever else is running. Queued jobs start in
# order while fewer than `max_concurrent` jobs run. Jobs are saved to jobs.json so the
# queue and schedules survive a restart; jobs that were running are queued again.

JOBS_PATH = APP_DIR / "jobs.json"
JOBS = {}
_jobs_lock = threading.RLock()
_wake = threading.Event()
_shutting_down = False
SAVED_FIELDS = (
    "id", "repo_id", "repo_type", "revision", "dest", "files", "sizes", "total", "done_files",
    "errors", "log", "status", "created", "started", "ended", "scheduled_at", "order",
)


def save_jobs():
    with _jobs_lock:
        data = [{k: j.get(k) for k in SAVED_FIELDS} for j in JOBS.values()]
        tmp = JOBS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, JOBS_PATH)


def load_jobs():
    if not JOBS_PATH.exists():
        return
    for job in json.loads(JOBS_PATH.read_text(encoding="utf-8")):
        job["partial"] = {}
        if job["status"] == "running":
            job["status"] = "queued"  # interrupted by a restart; files resume where they stopped
        JOBS[job["id"]] = job
    _wake.set()


def scheduler_loop():
    while True:
        _wake.wait(timeout=2)
        _wake.clear()
        if _shutting_down:
            return
        try:
            scheduler_tick()
        except Exception as exc:
            print(f"scheduler error: {exc}", file=sys.stderr)


def scheduler_tick():
    now = time.time()
    changed = False
    with _jobs_lock:
        for job in JOBS.values():
            if job["status"] == "scheduled" and job["scheduled_at"] <= now:
                job["status"] = "queued"
                job["order"] = job["scheduled_at"]
                changed = True
        limit = int(load_config().get("max_concurrent") or 1)
        running = sum(j["status"] == "running" for j in JOBS.values())
        queued = sorted((j for j in JOBS.values() if j["status"] == "queued"), key=lambda j: j["order"])
        for job in queued[: max(0, limit - running)]:
            spawn(job)
            changed = True
    if changed:
        save_jobs()


def stop_workers():
    """On shutdown, stop workers but keep them marked running so they are re-queued on the next start."""
    global _shutting_down
    _shutting_down = True
    _wake.set()
    with _jobs_lock:
        for job in JOBS.values():
            if job["status"] == "running" and job.get("_proc"):
                job["_proc"].kill()
        save_jobs()


class DownloadBody(BaseModel):
    repo_id: str
    repo_type: str = "model"
    revision: str | None = None
    files: list[str] = []
    patterns: list[str] = []
    dest: str | None = None
    when: str = "now"  # now | queue | schedule
    scheduled_at: float | None = None  # unix seconds, for when == "schedule"


def default_dest(repo_type, repo_id):
    return str(Path(load_config()["download_dir"]) / f"{repo_type}s" / repo_id.replace("/", "__"))


@app.get("/api/default-dest")
def get_default_dest(repo_id: str, repo_type: str = "model"):
    return {"dest": default_dest(repo_type, repo_id)}


@app.post("/api/download")
def start_download(body: DownloadBody):
    if body.when not in ("now", "queue", "schedule"):
        raise HTTPException(400, "when must be now, queue or schedule")
    if body.when == "schedule" and (not body.scheduled_at or body.scheduled_at < time.time() - 60):
        raise HTTPException(400, "Pick a start time in the future.")

    cfg = load_config()
    info = hf_api().repo_info(body.repo_id, repo_type=body.repo_type, revision=body.revision or None, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings or []}

    selected = [f for f in body.files if f in sizes]
    for pattern in body.patterns:
        selected += [f for f in sizes if fnmatch.fnmatch(f, pattern.strip())]
    if not body.files and not body.patterns:
        selected = list(sizes)
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise HTTPException(400, "Nothing selected: no files match.")

    dest = Path(body.dest or default_dest(body.repo_type, body.repo_id)).expanduser()
    if not dest.is_absolute():
        dest = Path(cfg["download_dir"]) / dest
    dest.mkdir(parents=True, exist_ok=True)

    now = time.time()
    job = {
        "id": uuid.uuid4().hex[:10],
        "repo_id": body.repo_id,
        "repo_type": body.repo_type,
        "revision": body.revision or "main",
        "dest": str(dest),
        "files": selected,
        "sizes": {f: sizes[f] for f in selected},
        "total": sum(sizes[f] for f in selected),
        "done_files": [],
        "errors": [],
        "log": [],
        "partial": {},
        "status": {"now": "running", "queue": "queued", "schedule": "scheduled"}[body.when],
        "created": now,
        "started": None,
        "ended": None,
        "scheduled_at": body.scheduled_at if body.when == "schedule" else None,
        "order": now,
    }
    with _jobs_lock:
        JOBS[job["id"]] = job
        if body.when == "now":
            spawn(job)
    save_jobs()
    _wake.set()
    return public_job(job)


def spawn(job):
    cfg = load_config()
    env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1")
    if cfg["hf_token"]:
        env["HF_TOKEN"] = cfg["hf_token"]
    todo = [f for f in job["files"] if f not in job["done_files"]]
    # Own process group/session, so Ctrl+C in the server terminal doesn't hit the workers
    # directly; stop_workers() ends them and they are re-queued on the next start.
    if sys.platform == "win32":
        platform_opts = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        platform_opts = {"start_new_session": True}
    proc = subprocess.Popen(
        [sys.executable, str(APP_DIR / "downloader.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, encoding="utf-8", errors="replace", **platform_opts,
    )
    proc.stdin.write(json.dumps({"repo_id": job["repo_id"], "repo_type": job["repo_type"],
                                 "revision": job["revision"], "local_dir": job["dest"], "files": todo}))
    proc.stdin.close()
    job.update(_proc=proc, partial={}, status="running", errors=[], started=time.time(), ended=None)
    threading.Thread(target=watch, args=(job, proc), daemon=True).start()


def watch(job, proc):
    for line in proc.stdout:
        line = line.strip()
        try:
            ev = json.loads(line)
        except ValueError:
            if line:
                job["log"] = (job["log"] + [line])[-20:]
            continue
        if ev.get("event") == "progress":
            job["partial"] = ev["files"]
        elif ev.get("event") == "file_done":
            job["done_files"].append(ev["path"])
        elif ev.get("event") == "file_error":
            job["errors"].append({"path": ev["path"], "error": ev["error"]})
    proc.wait()
    if _shutting_down:
        return
    with _jobs_lock:
        if job.get("_proc") is not proc:
            return  # superseded by a newer run of the same job
        if job["status"] == "running":
            job["status"] = "completed" if proc.returncode == 0 else "failed"
        job["ended"] = time.time()
        job["_proc"] = None
    save_jobs()
    _wake.set()  # a slot is free: start the next queued job


def downloaded_bytes(job):
    done = set(job["done_files"])
    finished = sum(job["sizes"].get(f, 0) for f in done)
    in_flight = sum(min(n, job["sizes"].get(f, 0)) for f, n in job["partial"].items() if f not in done)
    return min(finished + in_flight, job["total"])


def public_job(job):
    got = downloaded_bytes(job)
    started = job["started"]
    return {
        **{k: v for k, v in job.items() if not k.startswith("_") and k not in ("sizes", "done_files", "files", "partial")},
        "file_count": len(job["files"]),
        "files_done": len(set(job["done_files"])),
        "downloaded": got,
        "elapsed": ((job["ended"] or time.time()) - started) if started else 0,
    }


@app.get("/api/jobs")
def list_jobs():
    with _jobs_lock:
        jobs = sorted(JOBS.values(), key=lambda j: j["order"])
        queue = [j["id"] for j in jobs if j["status"] == "queued"]
        out = [public_job(j) for j in jobs]
    for j in out:
        j["queue_position"] = queue.index(j["id"]) + 1 if j["id"] in queue else None
    return {"jobs": out, "max_concurrent": int(load_config().get("max_concurrent") or 1), "now": time.time()}


def get_job(job_id):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    return job


def changed(job):
    save_jobs()
    _wake.set()
    return public_job(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with _jobs_lock:
        job = get_job(job_id)
        if job["status"] in ("running", "queued", "scheduled"):
            job["status"] = "cancelled"
            job["ended"] = time.time()
            if job.get("_proc"):
                job["_proc"].kill()
    return changed(job)


@app.post("/api/jobs/{job_id}/retry")
def queue_job(job_id: str):
    """Put a cancelled/failed/scheduled job at the end of the queue."""
    with _jobs_lock:
        job = get_job(job_id)
        if job["status"] in ("running", "queued"):
            raise HTTPException(409, f"Job is already {job['status']}")
        job.update(status="queued", order=time.time(), scheduled_at=None)
    return changed(job)


@app.post("/api/jobs/{job_id}/start")
def start_now(job_id: str):
    """Start immediately, ignoring the concurrency limit."""
    with _jobs_lock:
        job = get_job(job_id)
        if job["status"] == "running":
            raise HTTPException(409, "Job is already running")
        job["scheduled_at"] = None
        spawn(job)
    return changed(job)


class MoveBody(BaseModel):
    direction: str  # up | down | top


@app.post("/api/jobs/{job_id}/move")
def move_job(job_id: str, body: MoveBody):
    with _jobs_lock:
        job = get_job(job_id)
        queue = sorted((j for j in JOBS.values() if j["status"] == "queued"), key=lambda j: j["order"])
        if job not in queue:
            raise HTTPException(409, "Only queued jobs can be reordered")
        i = queue.index(job)
        if body.direction == "top" and i > 0:
            job["order"] = queue[0]["order"] - 1
        elif body.direction in ("up", "down"):
            k = i - 1 if body.direction == "up" else i + 1
            if 0 <= k < len(queue):
                job["order"], queue[k]["order"] = queue[k]["order"], job["order"]
    return changed(job)


class ScheduleBody(BaseModel):
    scheduled_at: float


@app.post("/api/jobs/{job_id}/schedule")
def schedule_job(job_id: str, body: ScheduleBody):
    if body.scheduled_at < time.time() - 60:
        raise HTTPException(400, "Pick a start time in the future.")
    with _jobs_lock:
        job = get_job(job_id)
        if job["status"] == "running":
            raise HTTPException(409, "Cancel the running job first")
        job.update(status="scheduled", scheduled_at=body.scheduled_at)
    return changed(job)


@app.delete("/api/jobs/{job_id}")
def forget_job(job_id: str):
    with _jobs_lock:
        job = get_job(job_id)
        if job["status"] == "running":
            raise HTTPException(409, "Cancel the job first")
        JOBS.pop(job_id, None)
    save_jobs()
    return {"ok": True}


@app.post("/api/jobs/clear-completed")
def clear_completed():
    with _jobs_lock:
        for job_id in [k for k, j in JOBS.items() if j["status"] == "completed"]:
            JOBS.pop(job_id)
    save_jobs()
    return {"ok": True}


@app.post("/api/jobs/{job_id}/open")
def open_folder(job_id: str):
    path = get_job(job_id)["dest"]
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        # e.g. a headless Linux box without a desktop
        raise HTTPException(501, f"No file manager available. The files are in {path}")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    print(f"Model Crawl running at http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
