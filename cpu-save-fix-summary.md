# cpu_save mode fix for FusedMoE + KML

## Problem

`--cpu-save` mode crashed during model loading with:
```
AttributeError: 'FusedMoE' object has no attribute 'mlp'
```

The crash occurred because the `cpu_save` code path in `CPUMoEMethod.process_weights_after_loading()` tried to access `layer.mlp.experts`, but the actual layer type was `FusedMoE` (which uses `w13_weight`/`w2_weight` instead).

## Root cause

1. The original `cpu_save` implementation loaded bf16 weights directly from safetensor files via `SafeTensorLoader.load_experts()`, which supported falling back to original HuggingFace-format keys (`model.layers.X.mlp.experts.E.gate_proj.weight`).

2. A previous commit (3c60e05) simplified `load_experts()` to only support NUMA-sharded format keys, removing the fallback to original format.

3. A subsequent change added a new `cpu_save` branch that tried to extract weights from `layer.mlp.experts` directly, which only works for `DeepseekV2MoE` layers, not `FusedMoE`.

## Changes

### 1. `python/sglang/srt/layers/cpuinfer.py`

- `load_experts()`: Restored format auto-detection — tries NUMA → AMX → original HuggingFace format.
- Extracted NUMA-format loading into `_load_experts_numa_format()`.

### 2. `python/sglang/srt/layers/quantization/unquant.py`

- `cpu_save` branch in `CPUMoEMethod.process_weights_after_loading()`: Instead of accessing `layer.mlp.experts`, loads full bf16 weights from safetensor via `self.safetensor_loader.load_experts(f"model.layers.{self.layer_idx}")`.
- Fixed cleanup: `cpu_save` path deletes `gate_proj`/`up_proj`/`down_proj` (not `gate_weights`).

### 3. `python/sglang/srt/layers/quantization/w8a8_int8.py`

- Same changes as `unquant.py` for `NPU_W8A8CPUMoEMethod.process_weights_after_loading()`.

## Why loading from safetensor is the right approach

- `FusedMoE.w13_weight` is TP-sharded: with `--tensor-parallel-size 2`, each rank only has half the intermediate dimension.
- Loading from original safetensor files gives full (non-sharded) bf16 weights regardless of TP.
- The C++ `online quant from bf16` path expects full intermediate_size weights.

## Environment: ktransformers-dev 编译安装

### 硬件与依赖

- **硬件**: Kunpeng ARM (aarch64), Ascend NPU
- **HPCKit**: `/opt/HPCKit/25.1.0/` (提供 KML 库和 Bisheng/GCC 编译器)

### 编译步骤

```bash
# 1. 激活 conda 环境并加载 HPCKit（每次新 shell 需要）
conda activate kt
source /opt/HPCKit/latest/setvars.sh

# 2. 编译 ktransformers-dev
cd /home/shangyouren/ktransformers-dev
rm -rf build
export MAX_JOBS=$(nproc)
pip install -e . -v
```

### 关键注意事项

- **不要反复 source setvars.sh**: 同一个 shell 里 source 一次即可，重复 source 可能覆盖环境变量。
- `TORCH_DEVICE_BACKEND_AUTOLOAD=0`: 在 `env_vars.sh` 中设置，禁止 torch_npu 自动加载。
- `COMP_SV_LEN=32`: SVE 向量长度，编译时定义。
- 编译产物: `kt_kernel` 和 `kt_kernel_ext` 两个包，提供 `KTMoEWrapper`、`CPUInfer`、`MOEConfig`、`Int4_KERNEL_MOE`、`Int8_KERNEL_MOE` 等。

### sglang_awq 与 ktransformers 的关系

- `sglang_awq` 的 `kt_ep_wrapper.py` 通过 `from kt_kernel import KTMoEWrapper` 使用 ktransformers。
- `KTMoEWrapper` 是工厂类：`method="MOE_INT8"` 或 `"MOE_INT4"` 时创建 `GeneralMoEWrapper`，底层调用 KML 的 `GemmKernelInt8`/`GemmKernelInt4`。
- 权重加载有两种路径：
  - **`CPUMoEMethod`** (sglang 侧): 通过 `SafeTensorLoader` 加载 safetensor，创建 `Int4_KERNEL_MOE`。
  - **`GeneralMoEWrapper`** (kt-kernel 侧): 同样通过 `SafeTensorLoader` 加载，创建 `Int4_KERNEL_MOE`。
