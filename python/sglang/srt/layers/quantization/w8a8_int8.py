from __future__ import annotations

import importlib
import sys
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)
from safetensors import safe_open
import os,glob,logging
import numpy as np
import torch
from torch.nn.parameter import Parameter

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.cpuinfer import KExpertsCPUBuffer, SafeTensorLoader
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import (
    apply_module_patch,
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_npu,
    set_weight_attrs,
    use_intel_amx_backend,
)
try:
    import cpuinfer_ext
    from sglang.srt.distributed import get_tensor_model_parallel_rank
    CPUINFER_AVAILABLE = True
    if CPUINFER_AVAILABLE and cpuinfer_ext:
        # from cpuinfer_ext import QuantConfig
        from cpuinfer_ext.kvcache import ggml_type
        from cpuinfer_ext.moe import MOEConfig, KMLInt4_MOE, KMLInt8_MOE
except ImportError as e:
    print(f"[WARN]: CPUInfer is not available {e.msg}")
    CPUINFER_AVAILABLE = False
    cpuinfer_ext = None

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
if _is_cuda:
    from sgl_kernel import int8_scaled_mm
_is_npu = is_npu()

if _is_npu:
    import torch_npu

    try:
        from mindie_turbo import _ops as ops
        from mindie_turbo.quantize.quant_utils import quant_per_tensor
    except ImportError:
        useMindIETurbo = False
    else:
        useMindIETurbo = True

# func refers to RMSNorm.__init__
def npu_wrapper_rmsnorm_init(func):
    def init(self, hidden_size: int, **extra_args) -> None:
        func(self, hidden_size, **extra_args)
        self.ignore_anti = True
        # The Ascend w8a8_int8 quantization requires adding a bias in rmsnorm
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)

    return init


# func refers to RMSNorm.forward_oot
def npu_wrapper_rmsnorm_forward(func):
    def _rmsnorm_forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        if residual is not None:
            out, _, residual_out = torch_npu.npu_add_rms_norm(
                residual, x, self.weight.data, self.variance_epsilon
            )
            out = out + self.bias
            return out.to(x.dtype), residual_out

        out = torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]
        out = out + self.bias
        return out.to(x.dtype)

    return _rmsnorm_forward_oot


def npu_fused_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
):
    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch_npu.npu_moe_init_routing(
            hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )
    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    expert_tokens = expert_tokens.to(torch.int64)
    # gmm1: gate_up_proj
    hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        scale=[w13_scale.to(scale_dtype)],
        per_token_scale=[pertoken_scale],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    # act_fn: swiglu
    hidden_states = torch_npu.npu_swiglu(hidden_states)
    hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
    # gmm2: down_proj
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale.to(scale_dtype)],
        per_token_scale=[pertoken_scale],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    final_hidden_states = torch_npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


class W8A8Int8Config(QuantizationConfig):
    """Config class for W8A8 Int8 Quantization.

    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(self, quant_config: Dict[str, Any] = {}):
        super().__init__()
        self.quant_description = quant_config
        self.is_dynamic = quant_config.get("is_dynamic", False)
        ignore = cast(List[str], quant_config.get("ignore", []))
        self.ignore = ignore if ignore is not None else []
        packed_modules_mapping = quant_config.get("packed_modules_mapping", {})
        self.packed_modules_mapping = (
            packed_modules_mapping if packed_modules_mapping is not None else {}
        )

        if _is_npu:
            # Ascend w8a8_int8 quantization with bias, use wrappers to isolate the effects between models
            for name in self.quant_description.keys():
                if "norm.bias" in name:
                    apply_module_patch(
                        "sglang.srt.layers.layernorm.RMSNorm",
                        "__init__",
                        [npu_wrapper_rmsnorm_init],
                    )
                    apply_module_patch(
                        "sglang.srt.layers.layernorm.RMSNorm",
                        "forward_npu",
                        [npu_wrapper_rmsnorm_forward],
                    )

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return (
            [torch.float16, torch.bfloat16]
            if not _is_npu
            else [torch.int8, torch.float16, torch.bfloat16]
        )

    @classmethod
    def get_min_capability(cls) -> int:
        if _is_npu:
            raise NotImplementedError(
                'NPU hardware does not support "get_min_capability" feature.'
            )
        else:
            return 75

    @classmethod
    def get_name(self) -> str:
        return "w8a8_int8"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        filenames = []
        if _is_npu:
            filenames.append("quant_model_description.json")
        return filenames

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W8A8Int8Config:
        return cls(config)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if _is_npu:
            if isinstance(layer, LinearBase):
                key = "model"
                if "vision_model" in prefix:
                    key = "vision_model"
                elif "visual" in prefix:
                    key = "visual"
                packed_modules_mapping_subset = self.packed_modules_mapping.get(key, {})
                prefix_in_quant_config = prefix
                proj_name = prefix.split(".")[-1]
                if proj_name in packed_modules_mapping_subset:
                    prefix_in_quant_config = prefix.replace(
                        proj_name, packed_modules_mapping_subset[proj_name][0]
                    )
                self.is_dynamic = (
                    self.quant_description[prefix_in_quant_config + ".weight"]
                    == "W8A8_DYNAMIC"
                )
                if self.is_layer_skipped(prefix, packed_modules_mapping_subset):
                    return UnquantizedLinearMethod()
                return (
                    NPU_W8A8DynamicLinearMethod(self)
                    if self.is_dynamic
                    else NPU_W8A8LinearMethod(self)
                )
            elif isinstance(layer, FusedMoE):
                if os.environ.get('MOE_CPU_WEIGHT_PATH') is not None:
                    import re
                    match = re.search(r"(\d+)\.mlp", prefix)
                    assert match
                    layer_id = int(match.group(1))
                    return NPU_W8A8CPUEPMoEMethod(self, layer_id)
                else:
                    return NPU_W8A8MoEMethod(self)
            return None

        if should_ignore_layer(
            prefix, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
        ):
            return UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            return W8A8Int8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return W8A8Int8MoEMethod(self)
        return None

    def is_layer_skipped(
        self, prefix: str, fused_mapping: Mapping[str, List[str]] = MappingProxyType({})
    ):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = (
                    self.quant_description[shard_prefix + ".weight"] == "FLOAT"
                )

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision."
                    )
        else:
            is_skipped = self.quant_description[prefix + ".weight"] == "FLOAT"

        assert is_skipped is not None
        return is_skipped

    def get_scaled_act_names(self) -> List[str]:
        return []


class W8A8Int8LinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: W8A8Int8Config):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "W8A8Int8LinearMethod on CPU requires that CPU has AMX support"
            _amx_process_weight_after_loading(layer, ["weight"])
        else:
            layer.weight = Parameter(layer.weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        if use_intel_amx_backend(layer):
            return torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
                x,
                layer.weight,
                layer.weight_scale,
                bias,
                x.dtype,
                True,  # is_vnni
            )

        x_q, x_scale = per_token_quant_int8(x)

        return int8_scaled_mm(
            x_q, layer.weight, x_scale, layer.weight_scale, out_dtype=x.dtype, bias=bias
        )


class W8A8Int8MoEMethod(FusedMoEMethodBase):
    """MoE method for INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: W8A8Int8Config,num_gpu_experts : int = -1):
        self.quant_config = quant_config
        self.num_gpu_experts = num_gpu_experts


    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.num_gpu_experts!=-1: 
            num_experts=self.num_gpu_experts
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        tp_size = get_tensor_model_parallel_world_size()

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "W8A8Int8MoEMethod on CPU requires that CPU has AMX support"
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])
        else:
            layer.w13_weight = Parameter(layer.w13_weight, requires_grad=False)
            layer.w2_weight = Parameter(layer.w2_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace See [Note] inplace should be False in fused_experts.
                True,  # use_int8_w8a8
                False,  # use_fp8_w8a16
                layer.w13_weight_scale,  # w1_scale
                layer.w2_weight_scale,  # w2_scale
                None,  # block_size
                layer.w13_input_scale,  # a1_scale
                layer.w2_input_scale,  # a2_scale
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)

        quant_info = TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )
        return self.runner.run(dispatch_output, quant_info)


class NPU_W8A8LinearMethodImpl:
    """Linear method for NPU W8A8."""

    def __init__(self) -> None:
        # aclnn quant matmul requires to transpose matrix B, set to true by default.
        self.transpose_weight = True

    @staticmethod
    def get_weight(
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        params_dict["input_scale"] = torch.empty(1, dtype=params_dtype)
        params_dict["input_offset"] = torch.empty(1, dtype=params_dtype)
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        if params_dtype == torch.bfloat16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.float32)
        elif params_dtype == torch.float16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.int64)
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # To prevent import loops
        from sglang.srt.layers.linear import RowParallelLinear

        original_dtype = x.dtype
        if original_dtype != torch.int8:
            x = torch_npu.npu_quantize(
                x,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
                torch.qint8,
                -1,
                False,
            )
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in Attention TP>1 case)
        if isinstance(layer, RowParallelLinear) and layer.tp_rank > 0:
            quant_bias = None
        else:
            quant_bias = layer.quant_bias
        return torch_npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=original_dtype,
        )

    def process_weights_after_loading(self, layer):
        expanding_factor = layer.weight.data.shape[1]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
        layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)


class NPU_W8A8LinearMethodMTImpl:
    """Linear method for NPU W8A8."""

    def __init__(self) -> None:
        self.transpose_weight = True

    @staticmethod
    def get_weight(
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        params_dict["input_scale"] = torch.empty(1, dtype=params_dtype)
        params_dict["input_offset"] = torch.empty(1, dtype=torch.int8)
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        if params_dtype == torch.bfloat16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.float32)
        elif params_dtype == torch.float16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.int64)
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # To prevent import loops
        from sglang.srt.layers.linear import RowParallelLinear

        original_dtype = x.dtype
        if original_dtype != torch.int8:
            x = quant_per_tensor(x, layer.input_scale, layer.input_offset)

        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in Attention TP>1 case)
        if isinstance(layer, RowParallelLinear) and layer.tp_rank > 0:
            quant_bias = None
        else:
            quant_bias = layer.quant_bias

        return ops.quant_matmul(
            x=x, weight=layer.weight, deq_scale=layer.deq_scale, deq_bias=quant_bias
        )

    def process_weights_after_loading(self, layer):
        layer.aclnn_deq_scale = torch.nn.Parameter(
            torch_npu.npu_trans_quant_param(layer.deq_scale.npu()).to(device="npu"),
            requires_grad=False,
        )


class NPU_W8A8LinearMethod(LinearMethodBase):
    """Linear method for NPU quantization.

    This class search for specific quantization
    implementation supported on NPU hardware for linear methods.

    Args:
        quant_config: The NPU quantization config.
    """

    def __init__(self, quantization_config: W8A8Int8Config) -> None:
        self.quantization_config = quantization_config
        self.quant_method = (
            NPU_W8A8LinearMethodMTImpl()
            if useMindIETurbo
            else NPU_W8A8LinearMethodImpl()
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight_dict = self.quant_method.get_weight(
            input_size_per_partition, output_size_per_partition, params_dtype
        )
        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            layer.register_parameter(weight_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        pertensor_dict = self.quant_method.get_pertensor_param(params_dtype)
        for pertensor_name, pertensor_param in pertensor_dict.items():
            param = PerTensorScaleParameter(
                data=pertensor_param, weight_loader=weight_loader
            )
            # disable warning
            param.ignore_warning = True
            layer.register_parameter(pertensor_name, param)

        perchannel_dict = self.quant_method.get_perchannel_param(
            output_size_per_partition, params_dtype
        )
        for perchannel_name, perchannel_param in perchannel_dict.items():
            param = torch.nn.Parameter(perchannel_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(perchannel_name, param)
            set_weight_attrs(param, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.quant_method.apply(layer, x, bias)


class NPU_W8A8DynamicLinearMethodImpl:
    """Linear method for NPU W8A8_DYNAMIC."""

    def __init__(self):
        self.transpose_weight = True

    @staticmethod
    def get_weight(
        input_size: int, output_size: int, params_dtype: torch.dtype
    ) -> Dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        quant_out, dynamic_scale = torch_npu.npu_dynamic_quant(x)
        return torch_npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale,
            bias=bias,
            output_dtype=original_dtype,
        )

    def process_weights_after_loading(self, layer):
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)


class NPU_W8A8DynamicLinearMethod(LinearMethodBase):
    """Linear method for NPU quantization.

    This class search for specific quantization
    implementations supported on NPU hardware for linear methods.

    Args:
        quant_config: The NPU quantization config.
    """

    def __init__(self, quantization_config: W8A8Int8Config) -> None:
        self.quantization_config = quantization_config
        self.quant_method = NPU_W8A8DynamicLinearMethodImpl()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight_dict = self.quant_method.get_weight(
            input_size_per_partition, output_size_per_partition, params_dtype
        )
        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            layer.register_parameter(weight_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        pertensor_dict = self.quant_method.get_pertensor_param(params_dtype)
        for pertensor_name, pertensor_param in pertensor_dict.items():
            param = PerTensorScaleParameter(
                data=pertensor_param, weight_loader=weight_loader
            )
            # disable warning
            param.ignore_warning = True
            layer.register_parameter(pertensor_name, param)

        perchannel_dict = self.quant_method.get_perchannel_param(
            output_size_per_partition, params_dtype
        )
        for perchannel_name, perchannel_param in perchannel_dict.items():
            param = torch.nn.Parameter(perchannel_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(perchannel_name, param)
            set_weight_attrs(param, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.quant_method.apply(layer, x, bias)


class NPU_W8A8MoEMethod(FusedMoEMethodBase):
    """MoE method for NPU quantization.

    This class search for specific quantization
    implementations supported on NPU hardware for moe methods.

    Args:
        quant_config: The NPU quantization config.
    """

    def __init__(self, quantization_config: W8A8Int8Config, num_gpu_experts : int = -1) -> None:
        self.quantization_config = quantization_config
        self.quant_method = self
        self.num_gpu_experts = num_gpu_experts


    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if self.num_gpu_experts!=-1: 
            num_experts=self.num_gpu_experts
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts

        # weight
        if num_experts == 0:
            w13_weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            layer.register_parameter("w13_weight", w13_weight)
            set_weight_attrs(w13_weight, extra_weight_attrs)

            w2_weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            layer.register_parameter("w2_weight", w2_weight)
            set_weight_attrs(w2_weight, extra_weight_attrs)

            w13_weight_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)

            w2_weight_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

            w13_weight_offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            layer.register_parameter("w13_weight_offset", w13_weight_offset)
            set_weight_attrs(w13_weight_offset, extra_weight_attrs)

            w12_weight_offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            layer.register_parameter("w2_weight_offset", w12_weight_offset)
            set_weight_attrs(w12_weight_offset, extra_weight_attrs)
        else:
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
            )

            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size,
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight", w13_weight)
            set_weight_attrs(w13_weight, extra_weight_attrs)
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight", w2_weight)
            set_weight_attrs(w2_weight, extra_weight_attrs)
            # scale
            w13_weight_scale = torch.nn.Parameter(
                torch.empty(
                    num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            w2_weight_scale = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)
            # offset
            w13_weight_offset = torch.nn.Parameter(
                torch.empty(
                    num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_offset", w13_weight_offset)
            set_weight_attrs(w13_weight_offset, extra_weight_attrs)
            w2_weight_offset = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_offset", w2_weight_offset)
            set_weight_attrs(w2_weight_offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = Parameter(
            layer.w13_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )
        layer.w2_weight = Parameter(
            layer.w2_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data.squeeze(-1).contiguous(), requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data.squeeze(-1).contiguous(), requires_grad=False
        )
        layer.w13_weight_offset = Parameter(
            layer.w13_weight_offset.data.squeeze(-1).contiguous(), requires_grad=False
        )
        layer.w2_weight_offset = Parameter(
            layer.w2_weight_offset.data.squeeze(-1).contiguous(), requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)
        output = npu_fused_experts(
            hidden_states=x,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
        )
        return StandardCombineInput(hidden_states=output)

class NPU_W8A8CPUMoEMethod(FusedMoEMethodBase):
    """Pure CPU inference W8A8 MoE method"""
    
    CPU_INFER = None
    SafeTensor_Loader = None
    
    def __init__(self, quant_config: W8A8Int8Config,
                 layer_idx, num_gpu_experts, cpuinfer, subpool_count, 
                 cpu_save, cpu_original_weight_path, cpu_weight_path, chunked_prefill_size):
        self.tp_rank = get_tensor_model_parallel_rank()
        if self.tp_rank != 0:
            return
        if NPU_W8A8CPUMoEMethod.CPU_INFER is None:
            print(f"subpool count is {subpool_count}", flush=True)
            worker_config = cpuinfer_ext.WorkerPoolConfig()
            subpool_numa_map = list(range(subpool_count))
            subpool_thread_count = [
            cpuinfer // subpool_count + (1 if i < cpuinfer % subpool_count else 0)
            for i in range(subpool_count)
            ]

            worker_config.subpool_count = subpool_count
            worker_config.subpool_numa_map= subpool_numa_map
            worker_config.subpool_thread_count = subpool_thread_count
            NPU_W8A8CPUMoEMethod.CPU_INFER = cpuinfer_ext.CPUInfer(worker_config)
        self.cpu_infer = NPU_W8A8CPUMoEMethod.CPU_INFER
        # read safetensor weight
        self.load_merged_weight = False
        if cpu_save:
            if glob.glob(os.path.join(cpu_original_weight_path, "*.safetensors")):
                self.load_merged_weight = True
            else:
                raise RuntimeError(f"Cannot find safetensors in {cpu_original_weight_path} for cpu_save mode")
            if NPU_W8A8CPUMoEMethod.SafeTensor_Loader is None:
                NPU_W8A8CPUMoEMethod.SafeTensor_Loader = SafeTensorLoader(cpu_original_weight_path)
            self.safetensor_loader = NPU_W8A8CPUMoEMethod.SafeTensor_Loader
        else:
            if glob.glob(os.path.join(cpu_weight_path, "*.safetensors")):
                self.load_merged_weight = True
            if self.load_merged_weight:
                if NPU_W8A8CPUMoEMethod.SafeTensor_Loader is None:
                    NPU_W8A8CPUMoEMethod.SafeTensor_Loader = SafeTensorLoader(cpu_weight_path)
                self.safetensor_loader = NPU_W8A8CPUMoEMethod.SafeTensor_Loader
        self.layer_idx = layer_idx
        self.quant_config = quant_config
        self.num_gpu_experts = num_gpu_experts
        self.cpu_save = cpu_save
        self.cpu_weight_path = cpu_weight_path
        self.chunked_prefill_size = chunked_prefill_size
        
        if not CPUINFER_AVAILABLE:
            raise ImportError("CPUInfer is not available. Please install cpuinfer_ext.")
            
    
    def create_weights(self, layer: torch.nn.Module, num_experts: int, 
                      hidden_size: int, intermediate_size_per_partition: int, 
                      params_dtype: torch.dtype, **extra_weight_attrs):
        self.experts_num = num_experts
        self.num_experts_per_tok = extra_weight_attrs.pop("top_k")
        self.hidden_size = hidden_size
        self.moe_intermediate_size = extra_weight_attrs.pop("intermediate_size_full")
        
        if self.tp_rank != 0:
            return
            
        # No GPU weights needed for pure CPU inference
        logger.info(f"NPU_W8A8CPUMoEMethod creating weights for layer {self.layer_idx}")
    
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.tp_rank != 0:
            return

        torch.npu.synchronize()

        # Initialize W8A* MoE for CPU inference
        awq_moe_config = MOEConfig(
            self.experts_num,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size
        )
        awq_moe_config.layer_idx = self.layer_idx
        awq_moe_config.pool = self.cpu_infer.backend_
        awq_moe_config.max_len = self.chunked_prefill_size

        #TODO(Yue Chen) load from SafeTensor is not support

        if self.load_merged_weight:
            gate_proj_ptr = 0
            up_proj_ptr = 0
            down_proj_ptr = 0

            if self.load_merged_weight:
                base_key = f"model.layers.{self.layer_idx}"
                # base_key = f"blk.{self.layer_idx}"
                # Load pre-sliced NUMA experts
                w = self.safetensor_loader.load_experts(base_key)

                self.gate_proj = torch.cat(w["gate_weight"], dim=0).contiguous()
                self.up_proj   = torch.cat(w["up_weight"], dim=0).contiguous()
                self.down_proj = torch.cat(w["down_weight"], dim=0).contiguous()
                
                gate_proj_ptr = self.gate_proj.data_ptr()
                up_proj_ptr = self.up_proj.data_ptr()
                down_proj_ptr = self.down_proj.data_ptr()

            awq_moe_config.gate_proj = gate_proj_ptr
            awq_moe_config.up_proj = up_proj_ptr
            awq_moe_config.down_proj = down_proj_ptr

            if self.cpu_save:
                awq_moe_config.save = True
                awq_moe_config.load = False
            else:
                awq_moe_config.load = True
            awq_moe_config.path = self.cpu_weight_path
        else:
            awq_moe_config.load = True
            awq_moe_config.path = self.cpu_weight_path

        awq_moe_config.hidden_type = ggml_type.BF16
        awq_moe_config.output_type = ggml_type.FP32

        self.moe = KMLInt4_MOE(awq_moe_config)

        # from sglang.srt.eplb.expert_location_dispatch import get_global_expert_location_metadata
        # physical_to_logical_map_cpu = get_global_expert_location_metadata().physical_to_logical_map_cpu[self.layer_idx].contiguous()
        # print(physical_to_logical_map_cpu, flush=True)
        # self.cpu_infer.submit(
        #     self.moe.load_weights(physical_to_logical_map_cpu.data_ptr())
        # )

        self.cpu_infer.submit(
            self.moe.load_weights_task()
        )
        self.cpu_infer.sync()

        del awq_moe_config

        if self.load_merged_weight:
            del self.gate_proj
            del self.up_proj
            del self.down_proj

            del w

        logger.info(f"Loading W8A8 weights from {self.cpu_weight_path} for layer {self.layer_idx}")

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
            self, 
            layer: torch.nn.Module,
            dispatch_output: StandardDispatchOutput,
            **kwargs) -> torch.Tensor:

        raise RuntimeError("NPU_W8A8CPUMoEMethod shouldn't be used directly.")

class NPU_W8A8CPUEPMoEMethod(FusedMoEMethodBase):
    """Expert Parallelism W8A* MoE method (CPU + GPU hybrid)"""
    
    def __init__(self, quant_config: W8A8Int8Config, layer_idx: int):
        self.tp_rank = get_tensor_model_parallel_rank()

        if 'MOE_NUM_GPU_EXPERTS' not in os.environ or 'MOE_CPUINFER' not in os.environ or 'MOE_CPU_WEIGHT_PATH' not in os.environ:
            raise RuntimeError("the following arguments are required: --cpu-weight-path, --cpuinfer, --num-gpu-experts")
        
        self.num_gpu_experts = int(os.environ.get('MOE_NUM_GPU_EXPERTS'))
        self.enable_defer = os.environ.get("MOE_ENABLE_DEFER", "False").lower() == "true"
        cpuinfer = int(os.environ.get('MOE_CPUINFER'))
        cpu_save = os.environ.get('MOE_CPU_SAVE', 'False').lower() == 'true'
        cpu_original_weight_path = os.environ.get('MOE_CPU_ORIGINAL_WEIGHT_PATH', '')
        cpu_weight_path = os.environ.get('MOE_CPU_WEIGHT_PATH', '')
        subpool_count = int(os.environ.get('SUBPOOL_COUNT'))
        chunked_prefill_size = int(os.environ.get('MOE_CHUNKED_PREFILL_SIZE', 8192))
        
        # Create CPU and GPU methods
        self.cpu_method = NPU_W8A8CPUMoEMethod(
            quant_config, layer_idx, self.num_gpu_experts, 
            cpuinfer, subpool_count, cpu_save, cpu_original_weight_path, cpu_weight_path, chunked_prefill_size
        )
        self.marlin_method = NPU_W8A8MoEMethod(quant_config, self.num_gpu_experts)
        self.layer_idx = layer_idx
        
    def create_weights(self, layer: torch.nn.Module, num_experts: int, 
                      hidden_size: int, intermediate_size_per_partition: int, 
                      params_dtype: torch.dtype, **extra_weight_attrs):
        self.global_num_experts = num_experts
        # Create weights for both CPU and GPU methods
        self.cpu_method.create_weights(layer, num_experts, hidden_size, 
                                     intermediate_size_per_partition, params_dtype, 
                                     **extra_weight_attrs)
        self.marlin_method.create_weights(layer, num_experts, hidden_size, 
                                    intermediate_size_per_partition, params_dtype, 
                                    **extra_weight_attrs)
        
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.cpu_method.process_weights_after_loading(layer)
        if self.num_gpu_experts != 0:
            self.marlin_method.process_weights_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def _submit_to_cpu(self, moe_kexperts_param):
        bsz_tensor_cpu, expert_ids_cpu, weights_cpu, input_tensor_cpu, output_cpu = moe_kexperts_param
        self.cpu_method.cpu_infer.submit(
            self.cpu_method.moe.forward_task(
                bsz_tensor_cpu.data_ptr(), expert_ids_cpu.size(-1),
                expert_ids_cpu.data_ptr(), weights_cpu.data_ptr(),
                input_tensor_cpu.data_ptr(), output_cpu.data_ptr(),
                False
            )
        )

    def _sync_to_cpu(self, empty_param):
        self.cpu_method.cpu_infer.sync()

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
        **kwargs,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output

        if self.tp_rank == 0:
            input_tensor_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_tensor_cpu = \
                KExpertsCPUBuffer.get_buffer(x, self.cpu_method.num_experts_per_tok)
            topk_ids_long = topk_ids.to(torch.int64)

            if torch.npu.is_current_stream_capturing():
                input_tensor_cpu.copy_(x, non_blocking=True)
                expert_ids_cpu.copy_(topk_ids_long, non_blocking=True)
                weights_cpu.copy_(topk_weights, non_blocking=True)
            else:
                input_tensor_cpu.copy_(x, non_blocking=False)
                expert_ids_cpu.copy_(topk_ids_long, non_blocking=False)
                weights_cpu.copy_(topk_weights, non_blocking=False)

            self.moe_kexperts_param = (bsz_tensor_cpu, expert_ids_cpu, weights_cpu, input_tensor_cpu, output_cpu)
            if torch.npu.is_current_stream_capturing():
                torch_npu.npu._launch_host_func(
                    torch.npu.current_stream(),
                    self._submit_to_cpu,
                    self.moe_kexperts_param
                )
            else:
                self._submit_to_cpu(self.moe_kexperts_param)

        return StandardCombineInput(hidden_states=torch.zeros_like(x))

    def sync(self, x):
        input_tensor_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_tensor_cpu = KExpertsCPUBuffer.get_buffer(x, self.cpu_method.num_experts_per_tok)
        if torch.npu.is_current_stream_capturing():
            torch_npu.npu._launch_host_func(
                torch.npu.current_stream(),
                self._sync_to_cpu,
                ()
            )
            output_gpu = output_cpu.to(device=x.device, non_blocking=True)
        else:
            self._sync_to_cpu(())
            output_gpu = output_cpu.to(device=x.device, non_blocking=False)
        output = output_gpu.to(dtype=x.dtype)
        return output


    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
        **kwargs,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output

        if self.tp_rank == 0:
            input_tensor_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_tensor_cpu = \
                KExpertsCPUBuffer.get_buffer(x, self.cpu_method.num_experts_per_tok)
            topk_ids_long = topk_ids.to(torch.int64)

            if torch.npu.is_current_stream_capturing():
                input_tensor_cpu.copy_(x, non_blocking=True)
                expert_ids_cpu.copy_(topk_ids_long, non_blocking=True)
                weights_cpu.copy_(topk_weights, non_blocking=True)
            else:
                input_tensor_cpu.copy_(x, non_blocking=False)
                expert_ids_cpu.copy_(topk_ids_long, non_blocking=False)
                weights_cpu.copy_(topk_weights, non_blocking=False)

            self.moe_kexperts_param = (bsz_tensor_cpu, expert_ids_cpu, weights_cpu, input_tensor_cpu, output_cpu)
            if torch.npu.is_current_stream_capturing():
                torch_npu.npu._launch_host_func(
                    torch.npu.current_stream(),
                    self._submit_to_cpu,
                    self.moe_kexperts_param
                )
            else:
                self._submit_to_cpu(self.moe_kexperts_param)
        
        output = torch.zeros_like(x)

        # Wait for CPU and combine results
        if self.tp_rank == 0:
            if torch.npu.is_current_stream_capturing():
                torch_npu.npu._launch_host_func(
                    torch.npu.current_stream(),
                    self._sync_to_cpu,
                    ()
                )
                output_gpu = output_cpu.to(device=x.device, non_blocking=True)
            else:
                self._sync_to_cpu(())
                output_gpu = output_cpu.to(device=x.device, non_blocking=False)
            output = output + output_gpu.to(dtype=x.dtype)

        return StandardCombineInput(hidden_states=output)