# Cloud burst — L15 QLoRA on RunPod (L40S)

RunPod so reishi's **5060 Ti stays idle**. Default target: **L40S 48GB**, 94GB RAM,
16 vCPU, **~$0.86/hr**.

One command on the pod runs smoke → QLoRA (2k steps, batch 4) → minimal SSM (10k steps).

**Estimated bill:** ~3–5 hrs → **~$3–5** (QLoRA-only ~1.5–2 hrs → **~$1.50–2**).

---

## When you come back — checklist

### Local (Windows, before starting pod)

- [ ] Export SFT (optional): `uv run python scripts/export_sft_from_lsm_home.py --out data/sft_v0.jsonl`
- [ ] Redact `data/sft_v0.jsonl` — remove anything you don't want on a remote GPU
- [ ] Or skip upload and use `--bootstrap` on the pod (synthetic recall_pref rows)

### RunPod console

- [ ] **Deploy pod:** GPU **L40S 48GB**, template **PyTorch 2.x / CUDA 12**, disk **≥50 GB**
- [ ] Copy **SSH command** (IP + port) from pod page
- [ ] *(Optional)* Upload dataset:

  ```powershell
  scp -P PORT data/sft_v0.jsonl root@POD_IP:/workspace/lsm/data/sft_v0.jsonl
  ```

### On pod (SSH or Web Terminal)

**Option A — GitHub (after push):**

```bash
cd /workspace
git clone https://github.com/kcpatt27/living-state-machine.git lsm
cd lsm
export HF_HOME=/workspace/hf_cache
bash scripts/runpod_l40s_burst.sh
```

**Option B — tarball (private, no GitHub):**

Local:

```powershell
.\scripts\pack_for_runpod.ps1
scp -P PORT lsm_runpod.tgz root@POD_IP:/workspace/
```

Pod:

```bash
mkdir -p /workspace/lsm
tar xzf /workspace/lsm_runpod.tgz -C /workspace/lsm
cd /workspace/lsm
export HF_HOME=/workspace/hf_cache
bash scripts/runpod_l40s_burst.sh
```

**Option C — step by step (after A or B):**

```bash
bash scripts/runpod_l15_setup.sh
uv run python scripts/runpod_l40s_burst.py --bootstrap
```

### Faster / cheaper variants

```bash
# QLoRA only (skip 10k-step SSM — saves ~2–3 hrs)
uv run python scripts/runpod_l40s_burst.py --bootstrap --skip-ssm

# Shorter SSM
uv run python scripts/runpod_l40s_burst.py --bootstrap --ssm-steps 5000

# Three QLoRA seeds (pick best adapter after local eval)
uv run python scripts/runpod_l40s_burst.py --bootstrap --qlora-seeds 0 1 2
```

### Download + stop pod

```powershell
scp -r -P PORT root@POD_IP:/workspace/lsm/runs/runpod_l40s_burst runs/
```

**Terminate the pod** in RunPod when download finishes.

---

## What the L40S burst runs

| Phase | Config | Output |
|-------|--------|--------|
| QLoRA smoke | 50 steps, 32 rows, seq 512 | `runs/runpod_l40s_burst/qlora_smoke/` |
| QLoRA train | 2000 steps, **batch 4**, grad_accum 4, seq 1024 | `runs/runpod_l40s_burst/qlora_qwen_7b/seed_0/` |
| Token prep | TinyStories → `data/tokens_small.bin` (once) | `data/tokens_small.bin` |
| SSM minimal | **10k steps**, batch 16, grad_accum 1 | `runs/runpod_l40s_burst/ssm/minimal/ckpts/` |

Summary: `runs/runpod_l40s_burst/summary.json` (elapsed time + estimated cost).

---

## L40S tuning reference

| Workload | 5060 Ti default | L40S burst |
|----------|-----------------|------------|
| QLoRA | batch 1, grad_accum 16 | **batch 4, grad_accum 4** |
| SSM minimal | batch 8, grad_accum 2, 2500 steps | **batch 16, grad_accum 1, 10k steps** |

Standalone QLoRA with L40S preset:

```bash
uv run python scripts/run_phase2_qlora.py --l40s --bootstrap --dataset data/sft_v0.jsonl
```

---

## Privacy

- Upload **only** redacted `sft_v0.jsonl` (or use bootstrap — no personal data)
- **Never** upload `$LSM_HOME`, sqlite, or LanceDB
- Terminate pod after download

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM | Rare on L40S; try `--skip-ssm` or `--ssm-steps 5000` |
| `CUDA required` | Not a GPU pod |
| CPU-only torch | Re-run `bash scripts/runpod_l15_setup.sh` |
| HF download slow | `export HF_HOME=/workspace/hf_cache` on a volume |
| Private repo | `scp` a tarball instead of `git clone` |

---

## After download (local)

**Completed 2026-06-10** via `runs/lsm_results.tgz` → extracted to
`runs/runpod_l40s_burst/`. See `runs/runpod_l40s_burst/DATA_PROVENANCE.md`.

1. Hybrid checkpoint default: `runs/runpod_l40s_burst/ssm/minimal/ckpts/step_010000.pt`
   (`LSM_HYBRID_CKPT_PATH` in `.env.example`; opt-in `LSM_STATE_UPDATER=hybrid`)
2. QLoRA adapters under `runs/runpod_l40s_burst/qlora_qwen_7b/seed_*/` (weights gitignored)
3. Daily chat: Ollama `qwen3.5:9b`; adapter eval when Transformers hook lands

See also: `docs/phase2_l15_gate.md`, `docs/phase2_training_perf.md`.
