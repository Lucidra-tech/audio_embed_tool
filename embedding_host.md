# Embedding Host: Azure GPU Budget or Hugging Face Container Endpoint

This note is for hosting the audio embedding model currently used in [`core/embedding_engine.py`](core/embedding_engine.py).

The local engine uses:

- Model code: `MusicFM25Hz`
- Hugging Face repository: `minzwon/MusicFM`
- Weights: `pretrained_fma.pt`
- Stats file: `msd_stats.json`
- Input audio: 30 seconds, loaded by `librosa` at `sr=24000`
- Embedding logic: `model.get_latent(wav_tensor, layer_ix=7)`, mean pool over time, then L2-normalize
- Current local paths: `/content/data/pretrained_fma.pt` and `/content/data/msd_stats.json`

For a hosted endpoint, keep the remote contract identical: accept an audio file or URL, resample to 24 kHz, crop or cap to 30 seconds, run MusicFM layer 7, return the normalized vector.

## Recommendation

For this project, start with a single-GPU T4/L4 endpoint before buying larger GPU capacity.

MusicFM inference for 30-second previews is a batch/audio embedding workload, not a large language model workload. A 1x NVIDIA T4 class GPU should be enough for a first deployment unless you need high concurrency or very low latency. Use autoscaling or scheduled shutdown before moving to A10/A100-class instances.

Best first options:

| Option | Use when | Starting shape |
| --- | --- | --- |
| Hugging Face Inference Endpoint with custom container | You want the easiest managed deployment path | 1x NVIDIA T4 or L4 |
| Azure VM with Docker/FastAPI | You want full control and possibly lower cost with stop/start scheduling | `Standard_NC4as_T4_v3` or `Standard_NC8as_T4_v3` |
| Google Cloud Compute Engine with Docker/FastAPI | You want full VM control, cheap T4 GPU, and hard stop/start automation | `n1-standard-4` + 1x `nvidia-tesla-t4` |
| Google Cloud Run GPU | You want a managed container endpoint that can scale down to zero | 4 vCPU, 16 GiB RAM, 1x NVIDIA L4 |

## Budget Model

Use 730 hours as the rough month length for always-on compute.

```text
monthly_compute = hourly_gpu_rate * running_hours_per_month * replica_count
```

Examples:

| Scenario | Formula | Monthly compute estimate |
| --- | --- | --- |
| Hugging Face T4 always on | `$0.50/hr * 730 * 1` | `$365/month` |
| Hugging Face T4, 8 hours/day | `$0.50/hr * 240 * 1` | `$120/month` |
| Hugging Face T4, min 1 replica plus short scale bursts | `$0.50/hr * ((730 * 1) + extra_scaled_hours)` | `$365/month + burst cost` |
| Azure GPU VM always on | `current_Azure_rate * 730 * 1` | Use Azure Calculator |
| Azure GPU VM, stopped outside work hours | `current_Azure_rate * scheduled_running_hours` | Use Azure Calculator |
| Google Compute Engine N1 + T4 always on, us-central1 example | `($0.189999/hr n1-standard-4 + $0.35/hr T4) * 730` | About `$394/month`, before disk/network |
| Google Compute Engine N1 + T4, 3 hours/day | `($0.189999/hr + $0.35/hr) * 90` | About `$49/month`, before disk/network |
| Google Cloud Run GPU L4 always on | `(4 vCPU + 16 GiB RAM + L4 GPU) * 730 hours` | About `$767/month` no zonal redundancy in us-central1, before network |
| Google Cloud Run GPU L4, request-driven and scales to zero | `billed_instance_seconds only while instances exist` | Best for spiky endpoint use |

Hugging Face currently lists 1x NVIDIA T4 GPU endpoints at `$0.50/hr` and bills by the minute while the endpoint is initializing or running. For Azure, use the Azure Pricing Calculator or Retail Prices API as the source of truth because VM rates vary by region, OS, discounts, reservations, spot availability, and quota.

Google Cloud currently lists NVIDIA T4 GPU pricing in Iowa/us-central1 at `$0.35/hr` per GPU, and `n1-standard-4` Linux VM pricing in the same region at `$0.189999/hr`. Google Cloud Run GPU pricing in us-central1 lists NVIDIA L4 without zonal redundancy at `$0.0001867/sec`, CPU at `$0.000018/vCPU-sec`, and memory at `$0.000002/GiB-sec` under instance-based billing. Always confirm the target region in the Google Cloud Pricing Calculator because GPU zones and pricing vary.

Also budget for:

- Persistent disk or container registry storage
- Network egress if clients pull many embeddings or audio files across regions
- Azure Container Registry or Docker Hub private image storage, if used
- Logging/monitoring
- A small CPU service if you split API orchestration from GPU inference

## Azure GPU VM Deployment

Suggested first Azure size:

| VM size | GPU | vCPU/RAM | Fit |
| --- | --- | --- | --- |
| `Standard_NC4as_T4_v3` | 1x NVIDIA Tesla T4, 16 GB GPU memory | 4 vCPU, 28 GB RAM | Cheapest reasonable first GPU VM |
| `Standard_NC8as_T4_v3` | 1x NVIDIA Tesla T4, 16 GB GPU memory | 8 vCPU, 56 GB RAM | More CPU/RAM headroom for audio decode and concurrency |
| `Standard_NC16as_T4_v3` | 1x NVIDIA Tesla T4, 16 GB GPU memory | 16 vCPU, 110 GB RAM | Use only if CPU-side preprocessing is bottlenecked |

Deployment shape:

1. Create an Ubuntu GPU VM in the target region.
2. Install NVIDIA drivers using the Azure NVIDIA GPU driver extension.
3. Install Docker and NVIDIA Container Toolkit.
4. Build a FastAPI container that loads `MusicFM25Hz` once at startup.
5. Download or bake in `pretrained_fma.pt` and `msd_stats.json`.
6. Expose `/health` and `/embed`.
7. Put the VM behind HTTPS using Azure Application Gateway, Azure Container Apps with GPU where available, or a reverse proxy such as Nginx/Caddy.
8. Add a stop/start schedule if the endpoint is not needed 24/7.

Azure budget controls:

- Prefer a scheduled VM for internal/batch usage.
- Use spot only for restartable batch embedding, not for a production endpoint.
- Keep audio and client workloads in the same region to reduce latency and egress.
- Start with one replica and queue requests instead of scaling GPUs early.
- Track GPU utilization with Azure Monitor and `nvidia-smi`; if GPU utilization is low but requests are slow, the bottleneck may be audio download/decoding rather than MusicFM.

## Google Cloud Deployment

Google Cloud has two good deployment paths for this embedding service.

| Path | Best for | Notes |
| --- | --- | --- |
| Compute Engine VM + T4 | Lowest predictable cost for short manual/batch sessions | Attach 1x T4 to an N1 VM, run Docker/FastAPI, and auto-stop the VM |
| Cloud Run GPU + L4 | Managed container endpoint with scale-to-zero | No driver setup, 1x L4 per instance, minimum 4 vCPU and 16 GiB memory |

### Compute Engine T4 VM

Suggested first VM:

| Component | Choice | Why |
| --- | --- | --- |
| Machine | `n1-standard-4` | 4 vCPU and 15 GiB RAM is enough to start for audio decoding plus one T4 |
| GPU | `nvidia-tesla-t4` | 16 GB VRAM, good for single-host inference and cheaper than L4/A100 paths |
| Boot disk | 50-100 GiB `pd-balanced` | Enough for Ubuntu, Docker image layers, Python packages, and logs |
| Model storage | Same boot disk or a 20-50 GiB separate `pd-balanced` disk | Keeps `pretrained_fma.pt` and `msd_stats.json` persistent across stop/start |
| Container image | Artifact Registry regional Docker repository | Keeps the image close to the VM and avoids cross-region pull cost |

Approximate us-central1 compute examples:

| Usage pattern | Compute formula | Estimate |
| --- | --- | --- |
| Always on | `($0.189999 + $0.35) * 730` | About `$394/month` |
| 3 hours/day | `($0.189999 + $0.35) * 90` | About `$49/month` |
| 2 hours/day | `($0.189999 + $0.35) * 60` | About `$32/month` |

These examples include only the N1 VM and T4 GPU. Add disk, Artifact Registry, snapshots, public IP/load balancer, and network egress if applicable.

Useful storage planning:

| Storage | Current Google price reference | Practical estimate |
| --- | --- | --- |
| `pd-balanced` persistent disk | `$0.000136986/GiB-hour` in us-central1 | 100 GiB is about `$10/month` |
| `pd-standard` persistent disk | `$0.000054795/GiB-hour` above the free tier in us-central1 | Cheaper, but slower; fine for cold artifacts |
| Artifact Registry image storage | First 0.5 GB free, then `$0.10/GB-month` | A 10 GB CUDA image is about `$1/month` after the free tier |

For the MusicFM embedding endpoint, prefer `pd-balanced` for the boot/model disk and a regional Artifact Registry repo in the same region as the VM, for example `us-central1-docker.pkg.dev`.

Example creation command:

```bash
gcloud compute instances create musicfm-embedder \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced
```

After the VM starts:

1. Install NVIDIA drivers, Docker, and NVIDIA Container Toolkit, or use a Deep Learning VM image if you want less manual driver work.
2. Pull the container from Artifact Registry.
3. Mount or download the MusicFM artifacts into a persistent path such as `/opt/musicfm`.
4. Run the FastAPI container with GPU access.
5. Keep the API private behind an internal load balancer, IAP, or a token-authenticated reverse proxy unless this is intentionally public.

Example container run shape:

```bash
docker run --gpus all -p 8000:8000 \
  -e MUSICFM_MODEL_DIR=/opt/musicfm \
  -v /opt/musicfm:/opt/musicfm \
  us-central1-docker.pkg.dev/PROJECT_ID/embeddings/musicfm-embedder:latest
```

### Cloud Run GPU Container

Cloud Run GPU is the simpler managed container option. It currently supports one NVIDIA L4 GPU per instance, provides pre-installed GPU drivers, requires at least 4 CPU and 16 GiB memory, and can scale down to zero when no instances are needed.

Use it when:

- You want less VM administration.
- Traffic is bursty and scale-to-zero matters.
- L4 cost is acceptable compared with managing a cheaper T4 VM.

Avoid it when:

- You need the absolute cheapest always-on endpoint.
- You need T4 specifically.
- Your cold-start path downloads a large model every time. Bake the model into the image or mount/download it into a fast startup cache.

Example deployment shape:

```bash
gcloud run deploy musicfm-embedder \
  --image=us-central1-docker.pkg.dev/PROJECT_ID/embeddings/musicfm-embedder:latest \
  --region=us-central1 \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --cpu=4 \
  --memory=16Gi \
  --no-gpu-zonal-redundancy \
  --max-instances=1 \
  --min-instances=0 \
  --concurrency=1
```

For this project, `--min-instances=0` is important. It lets Cloud Run scale down instead of billing an idle GPU endpoint all day. Keep `--max-instances=1` until you know the model memory and concurrency behavior.

## Auto-Shutdown for 2-3 Hour Use

For a VM that you only use for short embedding sessions, use both a default auto-stop timer and a schedule. The timer protects you when you forget to stop the machine; the schedule handles predictable work hours.

### Best VM Pattern: Stop After 3 Hours

Add a startup script that schedules a shutdown each time the VM boots:

```bash
sudo shutdown -h +180 "Auto-stopping MusicFM embedding VM after 3 hours"
```

For a 2 hour session:

```bash
sudo shutdown -h +120 "Auto-stopping MusicFM embedding VM after 2 hours"
```

If you need more time during a session:

```bash
sudo shutdown -c
sudo shutdown -h +180 "Extended MusicFM embedding VM session"
```

To make the 3-hour shutdown automatic on every boot, put this in the VM startup script metadata:

```bash
#!/bin/bash
shutdown -h +180 "Auto-stopping MusicFM embedding VM after 3 hours"
```

Then add it with:

```bash
gcloud compute instances add-metadata musicfm-embedder \
  --zone=us-central1-a \
  --metadata-from-file=startup-script=shutdown-after-3h.sh
```

### Recurring Schedule Pattern

Google Compute Engine also supports instance schedules through resource policies. Use this when you know the VM should run only during a daily time window.

Example: start at 10:00 and stop at 13:00 on weekdays:

```bash
gcloud compute resource-policies create instance-schedule musicfm-weekday-3h \
  --region=us-central1 \
  --vm-start-schedule="0 10 * * 1-5" \
  --vm-stop-schedule="0 13 * * 1-5" \
  --timezone="Asia/Karachi"

gcloud compute instances add-resource-policies musicfm-embedder \
  --zone=us-central1-a \
  --resource-policies=musicfm-weekday-3h
```

Use the startup-script shutdown timer even with a recurring schedule. It is a cheap guardrail if someone manually starts the VM outside the planned window.

Note that Google says scheduled start/stop operations can run up to 15 minutes after the scheduled time. If you need a strict 3-hour maximum, rely on the guest startup-script shutdown timer as the hard cap.

### Cloud Run Pattern: Scale to Zero

For Cloud Run GPU, use:

```bash
--min-instances=0
--max-instances=1
```

This is not a 2-3 hour timer, but it is usually better for API-style usage: the container instance disappears when idle, so the GPU is not kept running for a fixed block of time. If you need batch embedding for exactly 2-3 hours, prefer a Compute Engine VM or a Cloud Run Job with an explicit timeout.

## Hugging Face Custom Container Deployment

Hugging Face Inference Endpoints are a good fit because this model needs custom Python dependencies and custom inference logic rather than a standard text embedding runtime.

Expected container behavior:

- Load model artifacts from `/repository` when deployed through Hugging Face.
- Or download `minzwon/MusicFM` artifacts at container startup if you do not package them into the model repo.
- Start a web server on the port expected by the endpoint platform.
- Return JSON embeddings.

Minimal API contract:

```http
GET /health
POST /embed
```

Example request:

```json
{
  "audio_url": "https://example.com/preview.mp3"
}
```

Example response:

```json
{
  "model": "minzwon/MusicFM",
  "layer_ix": 7,
  "sample_rate": 24000,
  "duration_seconds": 30.0,
  "embedding": [0.0123, -0.0441]
}
```

Container outline:

```dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt fastapi uvicorn python-multipart requests

COPY core /app/core
COPY embedding_server.py /app/embedding_server.py

ENV HF_HOME=/tmp/huggingface
EXPOSE 7860

CMD ["uvicorn", "embedding_server:app", "--host", "0.0.0.0", "--port", "7860"]
```

The server should avoid instantiating `EmbeddingEngine()` per request. Load it once at process startup:

```python
engine = EmbeddingEngine(
    model_path="/repository/pretrained_fma.pt",
    stats_path="/repository/msd_stats.json",
)
```

If the Hugging Face model repository does not contain `pretrained_fma.pt` and `msd_stats.json` as mounted artifacts, keep the existing `hf_hub_download(repo_id="minzwon/MusicFM", ...)` behavior but make the target directory configurable instead of hard-coding `/content/data`.

## Implementation Changes Needed Before Hosting

The current `embedding_engine.py` is Colab-shaped. Before container deployment, adjust it so hosting is cleaner:

- Do not create `/content/data` unconditionally; read a configurable model directory from `MUSICFM_MODEL_DIR`.
- Do not instantiate `engine = EmbeddingEngine()` at import time inside a web server if you need lazy startup or test imports.
- Add a method that accepts raw audio bytes or a temporary file produced by the API layer.
- Keep output as `float32` JSON or compressed NumPy depending on downstream usage.
- Add request limits: max audio duration, max file size, and timeout for URL downloads.

## Practical First Budget

For a prototype:

- Hugging Face: budget around `$120/month` if you run a 1x T4 endpoint for 8 hours/day, or around `$365/month` if it is always on.
- Azure: budget using the live Azure Calculator for `Standard_NC4as_T4_v3` in your target region, then multiply by the expected running hours. Add storage and public IP/load balancer costs.
- Google Compute Engine: in us-central1, budget around `$32-$49/month` for 2-3 hours/day on `n1-standard-4` + T4, plus disk and network. Always-on is about `$394/month` before disk/network.
- Google Cloud Run GPU: use it when scale-to-zero matters. If it is effectively always active, the 4 vCPU + 16 GiB + L4 shape is closer to the high hundreds per month than the short-session T4 VM budget.

For production:

- Run one 1x T4/L4 endpoint first.
- Measure embeddings/minute using your real Deezer preview workload.
- Scale only if GPU utilization or queue latency proves the need.
- Consider separating batch embedding from online embedding. Batch can use spot GPU capacity; online should use stable on-demand capacity.

## Conclusion

Cheapest practical option:

- Use Google Compute Engine with `n1-standard-4` + 1x `nvidia-tesla-t4`.
- Keep the VM stopped except during embedding sessions.
- Add the startup-script shutdown timer for 2-3 hour usage.
- Store the MusicFM weights and stats on a small `pd-balanced` persistent disk.
- This is the lowest-cost path in this document for short daily use, around `$32-$49/month` for 2-3 hours/day in the us-central1 example, before disk/network.

Easiest setup option:

- Use Hugging Face Inference Endpoints with a custom container if you want the least cloud infrastructure work.
- Package the FastAPI server and MusicFM dependencies into the container, deploy the endpoint, and let Hugging Face handle most serving operations.
- This is simpler than managing GPU drivers, VM schedules, firewalls, reverse proxies, and container runtime setup yourself.
- The tradeoff is cost: it is likely more expensive for always-on or lightly used workloads than a scheduled Google Compute Engine T4 VM.

Best overall starting choice:

- If you are experimenting, embedding tracks in batches, or only need 2-3 hour sessions: choose Google Compute Engine T4 with auto-shutdown.
- If you need a public/managed endpoint quickly and do not want to manage infrastructure: choose Hugging Face custom container.
- If you want managed scale-to-zero on Google Cloud and can accept L4 pricing/cold starts: choose Cloud Run GPU.
- Azure is still a good option if the rest of your stack already lives in Azure, but for this specific short-session MusicFM embedding workload it is not the first cheapest/easiest recommendation.

## Sources

- Azure NCasT4_v3 VM specs: https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ncast4v3-series
- Azure Retail Prices API: https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices
- Azure Pricing Calculator: https://azure.microsoft.com/pricing/calculator/
- Hugging Face Inference Endpoints pricing: https://huggingface.co/docs/inference-endpoints/pricing
- Hugging Face custom container deployment: https://huggingface.co/docs/inference-endpoints/engines/custom_container
- Google Cloud GPU machine types: https://cloud.google.com/compute/docs/gpus
- Google Cloud GPU pricing: https://cloud.google.com/compute/gpus-pricing
- Google Cloud VM instance pricing: https://cloud.google.com/compute/vm-instance-pricing
- Google Cloud disk and image pricing: https://cloud.google.com/compute/disks-image-pricing
- Google Cloud Artifact Registry pricing: https://cloud.google.com/artifact-registry/pricing
- Google Cloud Run GPU support: https://cloud.google.com/run/docs/configuring/services/gpu
- Google Cloud Run pricing: https://cloud.google.com/run/pricing
- Google Compute Engine instance schedules: https://cloud.google.com/compute/docs/instances/schedule-instance-start-stop
