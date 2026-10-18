# Eval PC setup (WSL2 + CUDA)

Read before setting up or syncing the Windows PC. From 2026-10-18 all training-adjacent work and evaluation (P7.1 comparison, P7.2 `rag-eval` + judge, `scribe-tune`, `ood-eval`, `baseline-eval`/`baseline-stress`, `urgency-eval`, `tune-tau`, `style-aug`, `make data` / `export-splits`) runs on the PC: Windows + WSL2 Ubuntu, RTX 3050 4 GB, CUDA. The Mac runs only the final app (`make dev`, `make demo`). The two-machine rule is in CLAUDE.md.

## 1. WSL2 and the NVIDIA driver

The GPU driver is installed **on Windows only** (a current Game Ready/Studio driver includes WSL support). Never install an NVIDIA Linux driver inside WSL; the CUDA runtime comes with the PyTorch wheel.

```powershell
# PowerShell
wsl --update
wsl --status                 # default version 2
wsl -l -v                    # Ubuntu, VERSION 2
```

```bash
# inside Ubuntu
nvidia-smi                   # lists the RTX 3050 with ~4096 MiB; this nvidia-smi lives in /usr/lib/wsl/lib
```

WSL2 gets half of Windows' RAM by default. To change that, set `memory=` (and `swap=`) under `[wsl2]` in `%UserProfile%\.wslconfig`, then run `wsl --shutdown`. The job guard measures the WSL VM's memory, not Windows' memory.

**Keep the repo on the Linux filesystem** (`~/GroundedTriage`), not under `/mnt/c`. Work under `/mnt/c` is many times slower, and the guard's `flock` is unreliable on that mount.

## 2. Toolchain

```bash
sudo apt update && sudo apt install -y build-essential git make curl
# Python 3.11 (requires-python is >=3.11,<3.12; Ubuntu 22.04 ships 3.10, 24.04 ships 3.12)
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.11 python3.11-venv python3.11-dev
# Go (go.mod says 1.26): install the linux-amd64 tarball from go.dev/dl into /usr/local/go
export PATH=$PATH:/usr/local/go/bin     # add to ~/.bashrc
```

## 3. Clone and install

```bash
git clone <the GitHub remote> ~/GroundedTriage && cd ~/GroundedTriage
cp /path/to/your/.env .env          # copy the Mac's .env by hand (not in git, not in the bundle)
make setup                          # venv ml-service/.venv (python3.11), dev+data extras, Go modules
make setup-baseline                 # torch + transformers (+ eval). setup-models installs the same packages
ml-service/.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The default Linux x86_64 torch wheel from PyPI is CUDA-enabled. If the check above prints `False` or `None` for CUDA, reinstall from the CUDA index: `ml-service/.venv/bin/pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124`.

Device selection (`app/device.py`) picks CUDA, then MPS, then CPU. `TORCH_DEVICE=cuda|mps|cpu` in `.env` overrides it; empty means auto. With `MODEL_FP16=true` (the default), the transformers run in fp16 on CUDA as on MPS. On a 4 GB card that is what keeps one model per role in VRAM. Rules for `.env` on the PC: keep the same values as on the Mac (same Supabase project, same Groq key, so the daily cap is shared). Leave `TORCH_DEVICE` empty, and keep `JUDGE_MAX_TOKENS=1000`.

Then check everything:

```bash
make lint && make test
python3 scripts/jobguard.py --name check --check-only -- true   # prints "start; memory {...}" with psi_avg10 / gpu_used_mb
make ood-eval                                                   # first guarded model job; downloads the HF weights below
```

The first run of each eval downloads its Hugging Face weights to `~/.cache/huggingface`, so they aren't copied: BART-MNLI ~1.6 GB, twitter-roberta-sentiment ~0.5 GB, bge-small-en-v1.5 ~0.13 GB.

## 4. Copy from the Mac (one-time)

These are gitignored or untracked files that the evals read. Sizes were measured on the Mac on 2026-10-18. Copy them into the same paths under the PC's repo root.

| Path | Size | Needed for |
|---|---|---|
| `datasets/processed/` | 32.8 MB | every classification eval, ood-eval, compare, tune-tau, urgency-eval (current export `e2552b04338b`) |
| `datasets/raw/wixqa/` | 51.6 MB | `rag-eval`, retrieval eval (WixQA `p72_*` questions + KB corpus) |
| `datasets/raw/emb_cache/` | 57.7 MB | bge-small embeddings (fast_tier_emb, retrieval); avoids re-embedding |
| `datasets/raw/llm_cache/` | 1.3 MB | Groq response caches: `rag_eval_drafts`, `judge`, `style_aug`, `synthetic_meetings`, `stress_meetings`, `ood_prelabel`. Losing them means paying the tokens again |
| `datasets/raw/rag_eval/` | 0.1 MB | `progress.jsonl`: rag-eval resumes from the 23 done rows; `judge-sample` reads it |
| `datasets/raw/tobi_bueck/` | 23.9 MB | `make data` (rebuild after style-aug, P3.8b) |
| `datasets/raw/spamassassin/` | 6.7 MB | `make data` (spam splits) |
| `datasets/raw/enron_spam/` | 5.2 MB | Enron OOD rows (`data.ood`) |
| `models/fast_tier/`, `models/fast_tier_aug/` | 2.2 + 2.3 MB | cascade fast tier + ablation |
| `models/fast_tier_emb/`, `models/fast_tier_emb_aug/` | < 0.1 MB | embedding fast-tier comparison |
| `models/spam_filter/` | 4.0 MB | spam filter in the cascade |
| `ml-service/eval/compare.py` | < 0.1 MB | P7.1 comparison harness (untracked until P7.1 commits it) |
| `results/rag/metrics.json`, `results/rag/report.md` | < 0.1 MB | PRELIMINARY day-1 P7.2 output (untracked; rewritten by the next run) |
| **Total** | **187.8 MB** (370 files; ~122 MB as .tgz) | |

```bash
# on the Mac, from the repo root (COPYFILE_DISABLE drops macOS ._ metadata files)
COPYFILE_DISABLE=1 tar czf ~/gt-pc-bundle.tgz \
  datasets/processed datasets/raw/wixqa datasets/raw/emb_cache datasets/raw/llm_cache \
  datasets/raw/rag_eval datasets/raw/tobi_bueck datasets/raw/spamassassin datasets/raw/enron_spam \
  models/fast_tier models/fast_tier_aug models/fast_tier_emb models/fast_tier_emb_aug models/spam_filter \
  ml-service/eval/compare.py results/rag/metrics.json results/rag/report.md
# move it to Windows (USB, shared folder, scp), then inside WSL:
tar xzf /mnt/c/Users/<you>/Downloads/gt-pc-bundle.tgz -C ~/GroundedTriage
```

Also copy `.env` by hand. Its contents are never written down here.

**Not copied:**
- `datasets/raw/twcs/` (493 MB): only used to rebuild the Twitter OOD candidates; the reviewed `datasets/ood/twitter.csv` is committed. Copy it only if `make ood-set` must run again.
- `datasets/export/`: the PC already has `groundedtriage-splits-e2552b04338b.tar.gz` (29 MB) for training. `c6539fff94a1` and `db976b0585b2` are superseded. `make export-splits` rebuilds the export.
- `datasets/raw/.jobguard.lock` and `jobguard.jsonl`: per machine.
- `token.json` and `credentials.json`: Google Calendar, for the Mac app only.
- `.venv`, `bin/` and the HF cache: rebuilt or downloaded on the PC.
- `models/deep_tier/` and `models/urgency/`: produced on the PC itself.

## 5. Moving work back to the Mac

The Mac app needs only what's committed plus `models/`. After training and evals, copy `models/deep_tier/`, `models/urgency/` and any re-trained fast tier or spam filter back to the Mac, or publish them with `make models-pack` (docs/deploy.md). Results live in `results/` and arrive through git.
