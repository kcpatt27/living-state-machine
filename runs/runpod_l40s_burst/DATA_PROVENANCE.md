# RunPod L40S burst — data & artifact provenance

**Run date:** 2026-06-09 (pod) · **Downloaded:** 2026-06-10 (`runs/lsm_results.tgz`)
**Profile:** `scripts/runpod_l40s_burst.py --bootstrap`
**Summary:** `runs/runpod_l40s_burst/summary.json`
**GPU:** NVIDIA L40S · **Est. cost:** ~$8.04 @ $0.86/hr · **Wall time:** ~9.3 h

---

## 1. SFT dataset (QLoRA)

| Field | Value |
| --- | --- |
| Path on pod | `/workspace/lsm/data/sft_v0.jsonl` |
| Source | **Bootstrap** — `scripts/bootstrap_sft_recall_pref.py` |
| Generator | `build_bootstrap_rows(count=500, seed=0)` |
| Rows | 500 user→assistant pairs |
| Schema | `{"messages": [{"role","content"}, ...], "source": "bootstrap_recall_pref"}` |
| Personal data | **None** — synthetic preference/recall dialogues from `recall_pref_v1` ontology |
| Upload | Not required; `--bootstrap` wrote the file on-pod when missing |

### What the bootstrap contains

- Preference statements (`I prefer …`, `my {topic} is …`)
- Recall questions (`what is my {topic}?`)
- Distractor turns (~35% rate) from `recall_pref_v1` distractor pool

### Regenerate locally

```powershell
uv run python scripts/bootstrap_sft_recall_pref.py --count 500 --seed 0 --out data/sft_v0.jsonl
```

`data/` is gitignored; this file is the committed record. For real-user SFT later,
use `scripts/export_sft_from_lsm_home.py` and document a separate split in
`DATA_PROVENANCE` with redaction notes.

### Scientific disclosure

SFT rows are **in-domain** with the `recall_pref_v1` eval task (same preference
schema). The paper should describe this as memory-adjacent adaptation on a
public benchmark ontology, not general open-domain assistant tuning.

---

## 2. QLoRA adapter (L15)

| Field | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 via bitsandbytes) |
| Method | LoRA r=16, alpha=32, dropout=0.05 (`lsm/research/training/qlora.py`) |
| Train steps | 2000 · batch 4 · grad_accum 4 · seq_len 1024 |
| Seed | 0 |
| Final checkpoint | `qlora_qwen_7b/seed_0/step_002000/` |
| Metrics | `qlora_qwen_7b/seed_0/metrics.json` |
| Smoke | `qlora_smoke/` — 50 steps, 32 rows, seq 512 |

Weights (`*.safetensors`) are gitignored; reproduce via RunPod burst or local
`scripts/run_phase2_qlora.py --l40s --bootstrap`.

---

## 3. SSM minimal (L40S retrain)

| Field | Value |
| --- | --- |
| Model | `minimal` diagonal-SSM (~30.6M params) |
| Token data | `/workspace/lsm/data/tokens_small.bin` (prepared on-pod by burst script) |
| Corpus | TinyStories uint16 bin (see `runs/small/DATA_PROVENANCE.md` for local corpus) |
| Steps | 10,000 · batch 16 · grad_accum 1 · seq_len 512 · bf16 |
| Checkpoints kept | `step_009000.pt`, `step_009500.pt`, `step_010000.pt` (last 3) |
| **Production path** | `ssm/minimal/ckpts/step_010000.pt` |
| Final loss | 2.35 (`summary.json` → `phases.ssm_minimal.final_loss`) |

`*.pt` checkpoints are gitignored; shipped via `lsm_results.tgz` / RunPod download.

### Hybrid integration

Opt-in on Windows (CUDA):

```powershell
$env:LSM_STATE_UPDATER = "hybrid"
$env:LSM_HYBRID_CKPT_PATH = "runs/runpod_l40s_burst/ssm/minimal/ckpts/step_010000.pt"
uv run lsm chat
```

Default daily chat remains `facts_backed`; hybrid is for research / ablation.

---

## 4. Reproduce the burst

```bash
# On RunPod L40S after clone/tarball + runpod_l15_setup.sh
export HF_HOME=/workspace/hf_cache
nohup bash scripts/runpod_l40s_burst.sh > burst.log 2>&1 &
```

Download:

```powershell
scp -r -P PORT root@POD_IP:/workspace/lsm/runs/runpod_l40s_burst runs/
# or tarball as reishi used: runs/lsm_results.tgz
```

---

## 5. L15 exit — still open after this run

- [x] SFT dataset v0 (bootstrap 500 rows)
- [x] QLoRA train + checkpoints archived locally
- [x] SSM minimal 10k retrain on L40S
- [ ] `recall_pref_v1` regression with adapter loaded (Transformers + LoRA eval hook)
- [ ] Pre/post adapter numbers in Phase 2 report

See `docs/phase2_l15_gate.md`.
