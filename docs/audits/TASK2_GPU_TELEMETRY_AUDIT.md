# XMCA-Net: Task 2.0.1 GPU Telemetry Forensic Audit

**Audit Target**: `src/generation/telemetry.py` & Execution Telemetry Lifecycle  
**Auditor**: Antigravity Static Code Analysis Suite

---

## 1. Origin Analysis of GPU Telemetry Metrics

All metrics reported by `src/generation/telemetry.py` were audited to confirm they derive from live hardware system calls and driver APIs rather than static approximations:

### 1.1 Driver-Level Telemetry (`query_nvidia_smi()`)
- **Code Location**: [`src/generation/telemetry.py#L12-L42`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/telemetry.py#L12-L42)
- **Implementation**:
  ```python
  nvidia_smi_path = shutil.which("nvidia-smi")
  if not nvidia_smi_path:
      return []

  cmd = [
      nvidia_smi_path,
      "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
      "--format=csv,noheader,nounits",
  ]
  result = subprocess.run(cmd, capture_output=True, text=True, check=True)
  ```
- **Forensic Verification**:
  - Does NOT invent GPU names or memory if `nvidia-smi` is missing (returns `[]` on macOS/CPU environments).
  - When present on Kaggle, parses genuine driver output fields: GPU index, name, memory used (MB), total memory (MB), and GPU utilization (%).

### 1.2 PyTorch CUDA Memory Accounting (`get_cuda_memory_snapshot()`)
- **Code Location**: [`src/generation/telemetry.py#L76-L94`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/telemetry.py#L76-L94)
- **Implementation**:
  Queries individual active CUDA devices:
  - `torch.cuda.memory_allocated(i)`: Current memory occupied by active tensors (GiB).
  - `torch.cuda.memory_reserved(i)`: Total memory managed by the PyTorch caching allocator (GiB).
  - `torch.cuda.max_memory_allocated(i)`: Peak watermark of allocated tensor memory.
  - `torch.cuda.max_memory_reserved(i)`: Peak watermark of reserved allocator memory.
  - `torch.cuda.reset_peak_memory_stats(i)`: Resets peak watermarks before each phase (lines 96-104).

### 1.3 Parameter Allocation across Devices (`get_parameter_distribution()`)
- **Code Location**: [`src/generation/telemetry.py#L45-L64`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/telemetry.py#L45-L64)
- **Implementation**:
  Iterates over `model.named_parameters()`, reading `str(param.device)`, summing `param.numel()`, and computing percentage of parameters resident on `cuda:0` vs `cuda:1`.
- **Significance**: Confirms true dual-GPU tensor residency instead of blindly assuming `device_map="auto"` placed layers across both accelerators.

### 1.4 Input Tensor Placement (`get_tensor_device_placement()`)
- **Code Location**: [`src/generation/telemetry.py#L67-L74`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/telemetry.py#L67-L74)
- **Implementation**:
  Inspects the `.device` attribute of input tensors (`input_ids`, `pixel_values`, `attention_mask`) confirming accelerator placement.

---

## 2. Telemetry Lifecycle Trace (`ModelMemoryTelemetry`)

- **Code Location**: [`src/generation/telemetry.py#L107-L156`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/telemetry.py#L107-L156)
- **Lifecycle Phases**:
  1. `capture_baseline()`: Captures driver memory and PyTorch allocations before model loading; resets peak memory stats.
  2. `capture_after_load(load_duration_sec)`: Records model loading time and resident VRAM.
  3. `capture_after_inference()`: Captures peak memory during image-conditioned generation passes.
  4. `capture_after_unload()`: Evaluates memory state after `model = None`, `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.ipc_collect()`.
- **Leakage Realism**: The code does NOT assert or force post-unload memory to `0.0 GiB`. It measures actual driver and CUDA allocator states, acknowledging that PyTorch contexts and driver handles may retain small overheads.

---

## 3. Current State of Artifact Reports

- File: [`artifacts/reports/task2_actual_kaggle_environment.json`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/artifacts/reports/task2_actual_kaggle_environment.json)
- Content check:
  ```json
  {
    "execution_mode": "dry_run",
    "system_platform": "macOS-26.7-arm64-arm-64bit",
    "is_kaggle": false,
    "dual_t4_verified": false,
    "accelerator_details": "PyTorch is not installed in the environment."
  }
  ```
- **Finding**: The report truthfully records that hardware execution has not taken place on local macOS. There are zero fabricated GPU numbers or false claims of Kaggle environment presence.

---

## 4. Audit Verdict

**VERDICT**: **PASS**  
All telemetry functions are wired to live PyTorch CUDA APIs and system `nvidia-smi` utilities. No constant or synthetic memory values exist. Real hardware execution remains **NOT VERIFIABLE** until executed on Kaggle dual-T4 GPUs.
