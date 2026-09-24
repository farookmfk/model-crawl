# Model Crawl

A small self-hosted web app for finding, verifying and downloading **Hugging Face models and datasets**.
It can fetch a single quantization from a repo, queue and schedule downloads, and optionally use an
LLM to understand requests like *"qwen 2.5 coder 7b, 5-bit gguf"*.

It runs as a local Python server with a browser UI on **Windows, Linux and macOS**.

## Features

- **Verify before downloading**: paste `owner/name` or any huggingface.co URL. The app checks that the
  repo exists and shows whether it is gated or private, plus its license, size and download count. If the
  repo doesn't exist, you get similar names.
- **Search**: Hugging Face Hub search, plus optional [Brave Search](https://brave.com/search/api/)
  for fuzzy names. Web results are checked against the Hub, and ones that don't exist are dropped.
- **Pick a quantization**:
  - GGUF repos are grouped by quant (`Q4_K_M`, `IQ3_XXS`, `UD-Q4_K_XL`, `BF16`, ...), with multi-part
    files kept together.
  - ONNX repos are grouped by variant (`fp16`, `q4`, `int8`, ...).
  - Multi-format repos can be downloaded as a single format ("safetensors only").
  - Quants stored as branches (e.g. EXL2) are selectable as a revision.
  - Optional extras such as `mmproj` vision projectors can be added.
- **Find quantized versions**: for any model, lists other repos quantized from the same base model
  (GGUF, AWQ, GPTQ, MLX, ...).
- **Ask AI** (optional): an LLM turns free text into a repo, a search query and a quantization, then
  preselects the matching quant. Works with Anthropic Claude or any OpenAI-compatible endpoint (Ollama,
  LM Studio, vLLM, OpenRouter, ...).
- **Download manager**:
  - Downloads run in a background worker with live progress and speed, and can be cancelled or resumed.
  - **Download now**, **Add to queue** (reorderable, with a configurable number of parallel downloads) or
    **Schedule** for a later date and time.
  - The queue and schedules survive restarts.

## Requirements

- Python **3.10+**
- On Debian/Ubuntu: `sudo apt install python3-venv`

## Quick start

```sh
git clone https://github.com/farookmfk/model-crawl.git
cd model-crawl
```

**Linux / macOS**

```sh
sh start.sh
```

**Windows (PowerShell)**

```powershell
.\start.ps1
```

On first run the launcher creates a `.venv` and installs `requirements.txt`. It then starts the server and
opens http://127.0.0.1:8765.

To run it manually instead:

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
.venv/bin/python server.py                     # Windows: .venv\Scripts\python server.py
```

## Configuration

Open **Settings** in the UI. Everything is optional; with no settings you can still browse and download
public repos.

| Setting | Used for |
|---|---|
| Hugging Face token | gated/private repos and higher rate limits ([create one](https://huggingface.co/settings/tokens), *read* access is enough) |
| Brave Search API key | web search for repos whose exact name you don't know |
| LLM provider | **Anthropic** (API key and model, default `claude-opus-5`) or **OpenAI-compatible** (base URL, model, optional key) |
| Download folder | defaults to `./downloads/<models\|datasets>/<owner>__<name>` |
| Queued downloads at the same time | how many queued jobs run in parallel (default 1) |

Settings are stored in `config.json` next to the app. **That file holds your keys and is git-ignored.
Don't commit it.**

Environment variables are used when a setting is empty:

| Variable | Purpose |
|---|---|
| `HF_TOKEN`, `BRAVE_API_KEY`, `ANTHROPIC_API_KEY` | fallback credentials |
| `MODEL_CRAWL_PORT` | port (default `8765`) |
| `MODEL_CRAWL_NO_BROWSER=1` | don't open a browser on start (headless servers) |
| `MODEL_CRAWL_HOST` | bind address (default `127.0.0.1`) |
| `MODEL_CRAWL_ALLOWED_HOSTS` | extra host names allowed to reach the UI, comma separated |

### Security notes

- By default the server only listens on `127.0.0.1` and rejects requests addressed to any other host
  name, which protects against DNS rebinding.
- The app has **no login**. If you expose it with `MODEL_CRAWL_HOST=0.0.0.0`, anyone who can reach the
  port can use your tokens to download. Only do that on a trusted network.

## Queue and schedule

| Button | Behavior |
|---|---|
| **Download now** | starts immediately, even while other downloads run |
| **Add to queue** | waits its turn. Queued jobs start in order, *N* at a time. Reorder with ↑/↓ or use **Start now**. |
| **Schedule** | pick a date and time (browser's time zone). At that time the job joins the queue. |

Jobs are saved in `jobs.json`. After a restart, the queue and schedules are restored, and downloads that were
running are queued again. Files that already finished are skipped. A file that was cut off partway starts
again from the beginning.

Scheduled downloads only start while the server is running. To keep it running unattended, start it
automatically:

<details>
<summary>Linux: systemd user service</summary>

`~/.config/systemd/user/model-crawl.service`

```ini
[Unit]
Description=Model Crawl

[Service]
WorkingDirectory=/path/to/model-crawl
Environment=MODEL_CRAWL_NO_BROWSER=1
ExecStart=/bin/sh start.sh
Restart=on-failure

[Install]
WantedBy=default.target
```

```sh
systemctl --user enable --now model-crawl
loginctl enable-linger "$USER"   # keep running when logged out
```
</details>

<details>
<summary>Windows: Task Scheduler</summary>

*Create Task* → trigger **At log on** → action **Start a program**: `powershell`, with arguments

```
-ExecutionPolicy Bypass -WindowStyle Hidden -Command "$env:MODEL_CRAWL_NO_BROWSER=1; & 'C:\path\to\model-crawl\start.ps1'"
```
</details>

## How it works

```
browser UI (static/index.html)
      │  JSON API
      ▼
server.py (FastAPI) ── Hugging Face Hub API (verify, list files, search)
      │               ├─ Brave Search API (optional)
      │               └─ LLM: Anthropic / OpenAI-compatible (optional)
      │  one subprocess per running job
      ▼
downloader.py ── huggingface_hub.hf_hub_download → download folder
```

Each job runs in its own process, so cancelling it is instant and the server stays responsive. The worker
streams byte-level progress back to the server as JSON lines.

| File | Purpose |
|---|---|
| `server.py` | API, repo inspection, quantization grouping, search, LLM, queue and scheduler |
| `downloader.py` | download worker |
| `static/index.html` | single-page UI (no build step) |
| `start.sh`, `start.ps1` | launchers that create the venv on first run |

## License

[MIT](LICENSE)
