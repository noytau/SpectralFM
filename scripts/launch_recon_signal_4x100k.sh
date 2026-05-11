#!/usr/bin/env bash
# Launch 4 signal-reconstruction experiments (FE×2 + transformer×2): random vs base_libri init.
# Each job uses exactly one visible GPU (CUDA_VISIBLE_DEVICES=one id).
#
# Defaults: 100k samples, 100k steps, lr=1e-4, cosine (warmup=10k), batch 512, grad accum 4,
# manifest single_channel_one, W&B project spectralfm-runai-recon-signal-100k.
#
# Usage (from repo root or any cwd):
#   bash scripts/launch_recon_signal_4x100k.sh
# Persist under tmux (recommended on SSH): survives disconnect; one session per job (4 GPUs)
#   or one session for all four in order (<4 GPUs):
#   USE_TMUX=1 bash scripts/launch_recon_signal_4x100k.sh
# Parallel (4 GPUs): default when nvidia-smi reports >= 4 GPUs.
# Sequential on GPU 0:   FORCE_SEQUENTIAL=1 bash scripts/launch_recon_signal_4x100k.sh
#
# Env overrides: REPO, WANDB_PROJECT, BASE_LIBRI_PT, MANIFEST_REL, N_SAMPLES, STEPS, LR, WARMUP,
#                BATCH_SIZE, GRAD_ACCUM, USE_TMUX

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="${REPO}/code:${REPO}/fairseq:${REPO}/fairseq/examples"

N_SAMPLES="${N_SAMPLES:-100000}"
STEPS="${STEPS:-100000}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-spectralfm-runai-recon-signal-100k}"
BASE_LIBRI_PT="${BASE_LIBRI_PT:-${REPO}/fairseq/base_libri_official.pt}"
MANIFEST="${MANIFEST:-${REPO}/fairseq/data/nova_data/single_channel_one/train.tsv}"
STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
ROOT_OUT="${REPO}/autoencoder_experiments"

if [[ ! -f "$BASE_LIBRI_PT" ]]; then
  echo "ERROR: base Libri checkpoint not found: $BASE_LIBRI_PT" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest not found: $MANIFEST" >&2
  exit 2
fi

echo "[*] Stopping existing train_reconstruction.py jobs (train mode)..."
pkill -f "train_reconstruction.py.*--mode.train" 2>/dev/null || pkill -f "train_reconstruction.py --mode train" 2>/dev/null || true
pkill -f "code/train_reconstruction.py" 2>/dev/null || true
sleep 3

NG="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
NG="${NG:-0}"
if [[ "${FORCE_SEQUENTIAL:-0}" == "1" ]]; then
  NG=1
fi
if [[ "$NG" -lt 1 ]]; then
  echo "ERROR: no NVIDIA GPUs visible (nvidia-smi). Need CUDA for --device cuda." >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1 && [[ "${USE_TMUX:-0}" == "1" ]]; then
  echo "ERROR: USE_TMUX=1 but tmux is not installed." >&2
  exit 1
fi

common_py=(
  python3 code/train_reconstruction.py --mode train
  --manifest "$MANIFEST"
  --n_samples "$N_SAMPLES" --steps "$STEPS"
  --lr "$LR" --warmup "$WARMUP"
  --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM"
  --wandb_project "$WANDB_PROJECT"
  --device cuda
)

write_meta() {
  local out="$1"
  shift
  mkdir -p "$out"
  {
    echo "utc_stamp=${STAMP}"
    echo "repo=${REPO}"
    echo "wandb_project=${WANDB_PROJECT}"
    echo "n_samples=${N_SAMPLES} steps=${STEPS} lr=${LR} warmup=${WARMUP}"
    echo "batch_size=${BATCH_SIZE} grad_accum_steps=${GRAD_ACCUM}"
    echo "manifest=${MANIFEST}"
    echo "base_libri_pt=${BASE_LIBRI_PT}"
    echo "command:" "$@"
    date -u "+%Y-%m-%dT%H:%M:%SZ"
  } >"${out}/RUN_INFO.txt"
}

run_bg() {
  local gpu="$1"
  local out="$2"
  shift 2
  write_meta "$out" "$@"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$@" >"${out}/train.log" 2>&1 &
  echo $! >"${out}/pid.txt"
  echo "[+] GPU ${gpu}  pid=$(cat "${out}/pid.txt")  -> ${out}"
}

# Detached tmux session running one training job (survives SSH disconnect).
run_tmux() {
  local session="$1" gpu="$2" out="$3"
  shift 3
  write_meta "$out" "$@"
  local inner="${out}/_tmux_train.sh"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    echo "set -o pipefail"
    echo "cd $(printf %q "$REPO")"
    echo "export PYTHONPATH=$(printf %q "$PYTHONPATH")"
    echo "export CUDA_VISIBLE_DEVICES=$(printf %q "$gpu")"
    printf 'CMD=('
    printf '%q ' "$@"
    echo ')'
    echo '"${CMD[@]}" 2>&1 | tee -a '"$(printf %q "$out")"'/train.log'
  } >"$inner"
  chmod +x "$inner"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -ds "$session" -c "$REPO" "$inner"
  echo "[+] tmux attach -t ${session}   (GPU ${gpu})  log: ${out}/train.log"
}

O1="${ROOT_OUT}/fe_signal_recon_100k_${STAMP}_random"
O2="${ROOT_OUT}/fe_signal_recon_100k_${STAMP}_pretrained_base_libri"
O3="${ROOT_OUT}/tr_signal_recon_100k_${STAMP}_random"
O4="${ROOT_OUT}/tr_signal_recon_100k_${STAMP}_pretrained_base_libri"

WB1="FE_signal_recon__init-random__n${N_SAMPLES}_s${STEPS}_lr${LR}_w${WARMUP}_cos_bs${BATCH_SIZE}_ga${GRAD_ACCUM}"
WB2="FE_signal_recon__init-base_libri__n${N_SAMPLES}_s${STEPS}_lr${LR}_w${WARMUP}_cos_bs${BATCH_SIZE}_ga${GRAD_ACCUM}"
WB3="TR_signal_recon__init-random__n${N_SAMPLES}_s${STEPS}_lr${LR}_w${WARMUP}_cos_bs${BATCH_SIZE}_ga${GRAD_ACCUM}"
WB4="TR_signal_recon__init-base_libri__n${N_SAMPLES}_s${STEPS}_lr${LR}_w${WARMUP}_cos_bs${BATCH_SIZE}_ga${GRAD_ACCUM}"

if [[ "${USE_TMUX:-0}" == "1" ]]; then
  echo "[*] USE_TMUX=1: jobs run in detached tmux sessions (attach to watch; logs in each out_dir/train.log)."
  S0="sfm_${STAMP}_fe_rand"
  S1="sfm_${STAMP}_fe_lib"
  S2="sfm_${STAMP}_tr_rand"
  S3="sfm_${STAMP}_tr_lib"
  if [[ "$NG" -ge 4 ]]; then
    run_tmux "$S0" 0 "$O1" "${common_py[@]}" --recon_path fe --ckpt none \
      --run_suffix random --wandb_run_name "$WB1" --out_dir "$O1"
    run_tmux "$S1" 1 "$O2" "${common_py[@]}" --recon_path fe --ckpt "$BASE_LIBRI_PT" \
      --run_suffix pretrained_base_libri --wandb_run_name "$WB2" --out_dir "$O2"
    run_tmux "$S2" 2 "$O3" "${common_py[@]}" --recon_path transformer --ckpt none \
      --run_suffix random --wandb_run_name "$WB3" --out_dir "$O3"
    run_tmux "$S3" 3 "$O4" "${common_py[@]}" --recon_path transformer --ckpt "$BASE_LIBRI_PT" \
      --run_suffix pretrained_base_libri --wandb_run_name "$WB4" --out_dir "$O4"
  else
    SEQ="${ROOT_OUT}/_sequential_${STAMP}.sh"
    write_meta "$O1" "${common_py[@]}" --recon_path fe --ckpt none \
      --run_suffix random --wandb_run_name "$WB1" --out_dir "$O1"
    write_meta "$O2" "${common_py[@]}" --recon_path fe --ckpt "$BASE_LIBRI_PT" \
      --run_suffix pretrained_base_libri --wandb_run_name "$WB2" --out_dir "$O2"
    write_meta "$O3" "${common_py[@]}" --recon_path transformer --ckpt none \
      --run_suffix random --wandb_run_name "$WB3" --out_dir "$O3"
    write_meta "$O4" "${common_py[@]}" --recon_path transformer --ckpt "$BASE_LIBRI_PT" \
      --run_suffix pretrained_base_libri --wandb_run_name "$WB4" --out_dir "$O4"
    SSQ="sfm_${STAMP}_seq4"
    {
      echo '#!/usr/bin/env bash'
      echo 'set -euo pipefail'
      echo "set -o pipefail"
      echo "cd $(printf %q "$REPO")"
      echo "export PYTHONPATH=$(printf %q "$PYTHONPATH")"
      echo "export CUDA_VISIBLE_DEVICES=0"
      echo "echo '=== 1/4 FE random ==='"
      printf 'CMD=('
      printf '%q ' "${common_py[@]}" --recon_path fe --ckpt none \
        --run_suffix random --wandb_run_name "$WB1" --out_dir "$O1"
      echo ')'
      echo '"${CMD[@]}" 2>&1 | tee -a '"$(printf %q "$O1")"'/train.log'
      echo "echo '=== 2/4 FE base_libri ==='"
      printf 'CMD=('
      printf '%q ' "${common_py[@]}" --recon_path fe --ckpt "$BASE_LIBRI_PT" \
        --run_suffix pretrained_base_libri --wandb_run_name "$WB2" --out_dir "$O2"
      echo ')'
      echo '"${CMD[@]}" 2>&1 | tee -a '"$(printf %q "$O2")"'/train.log'
      echo "echo '=== 3/4 TR random ==='"
      printf 'CMD=('
      printf '%q ' "${common_py[@]}" --recon_path transformer --ckpt none \
        --run_suffix random --wandb_run_name "$WB3" --out_dir "$O3"
      echo ')'
      echo '"${CMD[@]}" 2>&1 | tee -a '"$(printf %q "$O3")"'/train.log'
      echo "echo '=== 4/4 TR base_libri ==='"
      printf 'CMD=('
      printf '%q ' "${common_py[@]}" --recon_path transformer --ckpt "$BASE_LIBRI_PT" \
        --run_suffix pretrained_base_libri --wandb_run_name "$WB4" --out_dir "$O4"
      echo ')'
      echo '"${CMD[@]}" 2>&1 | tee -a '"$(printf %q "$O4")"'/train.log'
      echo "echo '=== all four finished ==='"
    } >"$SEQ"
    chmod +x "$SEQ"
    tmux kill-session -t "$SSQ" 2>/dev/null || true
    tmux new-session -ds "$SSQ" -c "$REPO" "$SEQ"
    echo "[+] Single-GPU queue in tmux: tmux attach -t ${SSQ}"
  fi
elif [[ "$NG" -ge 4 ]]; then
  echo "[*] Launching 4 jobs in parallel (GPUs 0–3), W&B project: ${WANDB_PROJECT}"
  run_bg 0 "$O1" "${common_py[@]}" --recon_path fe --ckpt none \
    --run_suffix random --wandb_run_name "$WB1" --out_dir "$O1"
  run_bg 1 "$O2" "${common_py[@]}" --recon_path fe --ckpt "$BASE_LIBRI_PT" \
    --run_suffix pretrained_base_libri --wandb_run_name "$WB2" --out_dir "$O2"
  run_bg 2 "$O3" "${common_py[@]}" --recon_path transformer --ckpt none \
    --run_suffix random --wandb_run_name "$WB3" --out_dir "$O3"
  run_bg 3 "$O4" "${common_py[@]}" --recon_path transformer --ckpt "$BASE_LIBRI_PT" \
    --run_suffix pretrained_base_libri --wandb_run_name "$WB4" --out_dir "$O4"
  wait
else
  echo "[!] Found ${NG} GPU(s) (<4); launching one job after another on GPU 0 (wall time ~4× one run)."
  run_bg 0 "$O1" "${common_py[@]}" --recon_path fe --ckpt none \
    --run_suffix random --wandb_run_name "$WB1" --out_dir "$O1"
  wait
  run_bg 0 "$O2" "${common_py[@]}" --recon_path fe --ckpt "$BASE_LIBRI_PT" \
    --run_suffix pretrained_base_libri --wandb_run_name "$WB2" --out_dir "$O2"
  wait
  run_bg 0 "$O3" "${common_py[@]}" --recon_path transformer --ckpt none \
    --run_suffix random --wandb_run_name "$WB3" --out_dir "$O3"
  wait
  run_bg 0 "$O4" "${common_py[@]}" --recon_path transformer --ckpt "$BASE_LIBRI_PT" \
    --run_suffix pretrained_base_libri --wandb_run_name "$WB4" --out_dir "$O4"
  wait
fi

LIST="${ROOT_OUT}/LAST_RECON_SIGNAL_4WAY_${STAMP}.txt"
{
  echo "$O1"
  echo "$O2"
  echo "$O3"
  echo "$O4"
} | tee "$LIST"
echo "[*] Output dirs listed in $LIST"
echo "[*] tail logs: tail -f $O1/train.log (etc.)"
