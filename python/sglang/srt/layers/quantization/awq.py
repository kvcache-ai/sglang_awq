# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
import os
import re
import glob
import ctypes
import numpy as np
from safetensors import safe_open
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors_moe import KExpertsCPUBuffer
from sglang.srt.layers.linear import LinearBase, set_weight_attrs
from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.marlin_utils import (
    apply_awq_marlin_linear,
    awq_to_marlin_zero_points,
    check_marlin_supported,
    check_marlin_supports_layer,
    check_moe_marlin_supports_layer,
    marlin_make_empty_g_idx,
    marlin_make_workspace,
    marlin_moe_permute_scales,
    marlin_permute_scales,
    moe_awq_to_marlin_zero_points,
    verify_marlin_supported,
    verify_marlin_supports_shape,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import get_scalar_types, replace_parameter

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardDispatchOutput,
        CombineInput,
    )

from sglang.srt.utils import is_cuda, is_hip, is_npu

try:
    import cpuinfer_ext
    from sglang.srt.distributed import get_tensor_model_parallel_rank
    CPUINFER_AVAILABLE = True
    if CPUINFER_AVAILABLE and cpuinfer_ext:
        from cpuinfer_ext import QuantConfig
        from cpuinfer_ext.moe import MOEConfig, AMXInt4_1KGroup_MOE
except ImportError as e:
    print(f"[WARN]: CPUInfer is not available {e.msg}")
    CPUINFER_AVAILABLE = False
    cpuinfer_ext = None

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if _is_npu:
    import torch_npu

if _is_cuda:
    from sgl_kernel import (
        awq_dequantize,
        awq_marlin_moe_repack,
        awq_marlin_repack,
        fused_marlin_moe,
    )


elif _is_hip:
    from sglang.srt.layers.quantization.awq_triton import (
        awq_dequantize_triton as awq_dequantize,
    )

    warnings.warn(f"HIP does not support fused_marlin_moe currently.")
else:
    warnings.warn(f"Only CUDA and HIP support AWQ currently.")

logger = logging.getLogger(__name__)


ScalarType, scalar_types = get_scalar_types()


def is_layer_skipped_awq(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)

class SafeTensorLoader():
    tensor_file_map: dict
    tensor_type_map: dict
    file_handle_map: dict
    tensor_device_map: dict

    def __init__(self, file_path: str):
        self.__load_tensor_file_map(file_path)

    def __load_tensor_file_map(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Path not found: {file_path}")
        if os.path.isfile(file_path):
            folder_path = os.path.dirname(file_path)
        else:
            folder_path = file_path
        self.file_handle_map = {}
        self.tensor_file_map = {}
        self.tensor_type_map = {}
        self.tensor_device_map = {}

        found_safetensor = False
        for root, _, files in os.walk(folder_path):
            files = sorted(files)
            for file in files:
                if file.endswith(".safetensors"):
                    found_safetensor = True
                    file_path = os.path.join(root, file)
                    if file not in self.file_handle_map:
                        try:
                            handle = safe_open(file_path, framework="pt")
                            self.file_handle_map[file] = handle
                        except Exception as e:
                            print(f"Error opening Safetensor file {file_path}: {e}")
                            continue

                    f = self.file_handle_map.get(file)
                    if f is None:
                        continue
                    try:
                        for key in f.keys():
                            self.tensor_file_map[key] = file
                    except Exception as e:
                        print(f"Error reading Safetensor file {file_path}: {e}")

        if not found_safetensor:
            raise FileNotFoundError(f"No Safetensor files found in {folder_path}")

    def load_tensor(self, key: str, device: str="cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        return tensor.to(device)

    def close_all_handles(self):
        # for handle in self.file_handle_map.values():
        #     handle.close()
        self.file_handle_map.clear()

    def load_experts(self, base_key: str, device: str="cpu"):
        """Load experts with automatic format detection (NUMA vs raw AWQ)"""
        # Try NUMA format first (backward compatibility)
        amx_up_base = f"{base_key}.ffn_up_exps" 
        if self.has_tensor(f"{amx_up_base}.0.weight"):
            return self._load_experts_amx_format(base_key, device)
            
        raise ValueError(f"No experts found for key {base_key} in either format")
    
    def _load_experts_amx_format(self, base_key: str, device: str="cpu"):
        """Load experts in pre-generated AMX-Compatible format (original)"""
        up_base_key = f"{base_key}.ffn_up_exps"
        gate_base_key = f"{base_key}.ffn_gate_exps"
        down_base_key = f"{base_key}.ffn_down_exps"
        max_experts_count = -1
        
        while self.has_tensor(f"{up_base_key}.{max_experts_count+1}.weight"):
            max_experts_count += 1
        if max_experts_count == -1:
            raise ValueError(f"No experts found for key {base_key}")
            
        # Initialize empty lists to store tensors for each projection type
        up_weights = []
        gate_weights = []
        down_weights = []

        up_scales = []
        gate_scales = []
        down_scales = []

        up_zeros = []
        gate_zeros = []
        down_zeros = []
        
        for expert_id in range(max_experts_count+1):
            up_key = f"{up_base_key}.{expert_id}.weight"
            gate_key = f"{gate_base_key}.{expert_id}.weight"
            down_key = f"{down_base_key}.{expert_id}.weight"

            up_scale_key = f"{up_base_key}.{expert_id}.scale"
            gate_scale_key = f"{gate_base_key}.{expert_id}.scale"
            down_scale_key = f"{down_base_key}.{expert_id}.scale"

            up_zeros_key = f"{up_base_key}.{expert_id}.qzeros"
            gate_zeros_key = f"{gate_base_key}.{expert_id}.qzeros"
            down_zeros_key = f"{down_base_key}.{expert_id}.qzeros"
            
            up_tensor = self.load_tensor(up_key, device).numpy()
            gate_tensor = self.load_tensor(gate_key, device).numpy()
            down_tensor = self.load_tensor(down_key, device).numpy()

            up_scale_tensor = self.load_tensor(up_scale_key, device).numpy()
            gate_scale_tensor = self.load_tensor(gate_scale_key, device).numpy()
            down_scale_tensor = self.load_tensor(down_scale_key, device).numpy()

            up_zeros_tensor = self.load_tensor(up_zeros_key, device).numpy()
            gate_zeros_tensor = self.load_tensor(gate_zeros_key, device).numpy()
            down_zeros_tensor = self.load_tensor(down_zeros_key, device).numpy()

            up_weights.append(up_tensor)
            gate_weights.append(gate_tensor)
            down_weights.append(down_tensor)

            up_scales.append(up_scale_tensor)
            gate_scales.append(gate_scale_tensor)
            down_scales.append(down_scale_tensor)

            up_zeros.append(up_zeros_tensor)
            gate_zeros.append(gate_zeros_tensor)
            down_zeros.append(down_zeros_tensor)
                
        return {
            "up_weight": up_weights,
            "gate_weight": gate_weights,
            "down_weight": down_weights,
            "up_scale": up_scales,
            "gate_scale": gate_scales,
            "down_scale": down_scales,
            "up_zeros": up_zeros,
            "gate_zeros": gate_zeros,
            "down_zeros": down_zeros,
        }

    def has_tensor(self, name: str):
        return name in self.tensor_file_map
class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits."
            )
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (
            f"AWQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_name(self) -> str:
        return "awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16] if not _is_npu else [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        if _is_npu:
            raise NotImplementedError(
                'NPU hardware does not support "get_min_capability" feature.'
            )
        else:
            return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> AWQConfig:
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[LinearMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if _is_npu:
            if isinstance(layer, LinearBase):
                if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                    return UnquantizedLinearMethod()
                return AWQLinearAscendMethod(self)
            elif isinstance(layer, FusedMoE):
                if os.environ.get('MOE_AMX_WEIGHT_PATH') is not None:
                    match = re.search(r"(\d+)\.mlp", prefix)
                    assert match
                    layer_id = int(match.group(1))
                    return AWQAMXEPMoEMethod(self, layer_id)
                else:
                    return AWQMoEAscendMethod(self)
            return None

        if isinstance(layer, LinearBase):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return AWQLinearMethod(self)
        return None


class AWQMarlinConfig(QuantizationConfig):
    """Config class for AWQ Marlin"""

    # num_bits -> type
    TYPE_MAP = {
        4: scalar_types.uint4,
        8: scalar_types.uint8,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: Optional[list[str]],
        full_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.zero_point = zero_point
        self.lm_head_quantized = lm_head_quantized
        self.weight_bits = weight_bits
        self.modules_to_not_convert = modules_to_not_convert or []
        self.full_config = full_config

        if self.weight_bits not in self.TYPE_MAP:
            raise ValueError(
                f"Unsupported num_bits = {self.weight_bits}. "
                f"Supported num_bits = {self.TYPE_MAP.keys()}"
            )

        self.quant_type = self.TYPE_MAP[self.weight_bits]

        verify_marlin_supported(
            self.quant_type, group_size=self.group_size, has_zp=self.zero_point
        )

    def __repr__(self) -> str:
        return (
            f"AWQMarlinConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def get_name(cls) -> str:
        return "awq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AWQMarlinConfig:
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(
            weight_bits,
            group_size,
            zero_point,
            lm_head_quantized,
            modules_to_not_convert,
            config,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        can_convert = cls.is_awq_marlin_compatible(hf_quant_cfg)
        is_valid_user_quant = (
            user_quant is None or user_quant == "marlin" or user_quant == "awq_marlin"
        )

        if can_convert and is_valid_user_quant:
            msg = (
                "The model is convertible to {} during runtime."
                " Using {} kernel.".format(cls.get_name(), cls.get_name())
            )
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "awq":
            logger.info(
                "Detected that the model can run with awq_marlin"
                ", however you specified quantization=awq explicitly,"
                " so forcing awq. Use quantization=awq_marlin for"
                " faster inference"
            )
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            # Check if the layer is supported by AWQMarlin.
            if not check_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    "Layer '%s' is not supported by AWQMarlin. Falling back to unoptimized AWQ kernels.",  # noqa: E501
                    prefix,
                )
                return AWQConfig.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            return AWQMarlinLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config

            if not check_moe_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by AWQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                return MoeWNA16Config.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            return AWQMoEMethod(self)
        return None

    @classmethod
    def is_awq_marlin_compatible(cls, quant_config: dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        zero_point = quant_config.get("zero_point")

        if not _is_cuda:
            return False

        if quant_method != "awq":
            return False

        # If we cannot find the info needed in the config, cannot convert.
        if num_bits is None or group_size is None or zero_point is None:
            return False

        if num_bits not in cls.TYPE_MAP:
            return False

        return check_marlin_supported(
            quant_type=cls.TYPE_MAP[num_bits], group_size=group_size, has_zp=zero_point
        )


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

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
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = awq_dequantize(qweight, scales, qzeros)
        out = torch.matmul(reshaped_x, out)

        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)


class AWQMarlinLinearMethod(LinearMethodBase):
    """Linear method for AWQ Marlin.

    Args:
        quant_config: The AWQ Marlin quantization config.
    """

    def __init__(self, quant_config: AWQMarlinConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        verify_marlin_supports_shape(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            input_size=input_size,
            group_size=group_size,
        )

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        num_groups = input_size_per_partition // group_size

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.num_groups = num_groups

    # TODO: Update this docs
    # Checkpoints are serialized in AutoAWQ format, which is different from the
    # marlin format. This function is called after the weights are loaded.
    # Here, we handle the repacking
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.qweight.device
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        # Allocate marlin workspace
        layer.workspace = marlin_make_workspace(device)

        # Repack weights from AWQ format to marlin format.
        marlin_qweight = awq_marlin_repack(
            layer.qweight,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits,
        )
        replace_parameter(layer, "qweight", marlin_qweight)

        # Permute scales from AWQ format to marlin format.
        marlin_scales = marlin_permute_scales(
            layer.scales,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "scales", marlin_scales)

        # Permute zero-points from AWQ format to marlin format.
        marlin_zp = awq_to_marlin_zero_points(
            layer.qzeros,
            size_k=layer.num_groups,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits,
        )
        replace_parameter(layer, "qzeros", marlin_zp)

        # Not-used
        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_awq_marlin_linear(
            input=x,
            weight=layer.qweight,
            weight_scale=layer.scales,
            weight_zp=layer.qzeros,
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            quant_type=self.quant_config.quant_type,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            bias=bias,
        )


class AWQLinearAscendMethod(AWQLinearMethod):
    """Linear method for AWQ on Ascend.

    Args:
        quant_config: The AWQ quantization config.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)
        qweight_tmp = torch.zeros_like(layer.qweight.data)
        qzeros_tmp = layer.qzeros.data
        qzeros_list = []
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]

        for i in range(0, self.quant_config.pack_factor):
            shift_num = shifts[i] * 4
            qzeros_list.append((qzeros_tmp.reshape(-1, 1) >> shift_num) & 0xF)
            qweight_tmp.bitwise_or_(
                ((layer.qweight.data >> shift_num) * (2 ** (4 * i))) & (0xF << (4 * i))
            )

        qweight_tmp.bitwise_xor_(0x88888888)

        qzeros_tmp = torch.cat(qzeros_list, dim=-1).reshape(qzeros_tmp.shape[0], -1)
        qzeros_tmp = -(qzeros_tmp - 8)
        qzeros_tmp = qzeros_tmp.to(layer.scales.data.dtype)

        layer.qzeros = torch.nn.Parameter(qzeros_tmp, requires_grad=False)
        layer.qweight = torch.nn.Parameter(qweight_tmp, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        if bias is not None and bias.dtype == torch.bfloat16:
            bias = bias.float()

        out = torch_npu.npu_weight_quant_batchmatmul(
            reshaped_x,
            qweight,
            antiquant_scale=scales,
            antiquant_offset=qzeros,
            antiquant_group_size=self.quant_config.group_size,
            bias=bias,
        )

        return out.reshape(out_shape)


class AWQMoEMethod(FusedMoEMethodBase):

    def __init__(self, quant_config: AWQMarlinConfig,num_gpu_experts=-1):
        self.quant_config = quant_config
        if self.quant_config.weight_bits != 4:
            raise ValueError("AWQMoEMethod only supports 4bit now.")
        self.quant_type = scalar_types.uint4
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
        # Delay the import to avoid circular dependency
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        extra_weight_attrs.update(
            {
                "is_transposed": True,
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            }
        )

        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = intermediate_size_per_partition // self.quant_config.group_size

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                intermediate_size_per_partition * 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(
            torch.empty(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        # WEIGHT_ZERO_POINT
        # Allocate 2 zero points for w1 and w3 respectively.
        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

        device = layer.w13_qweight.device
        if not _is_npu:
            layer.workspace = marlin_make_workspace(device, 4)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.w13_qweight.shape[0]
        device = layer.w13_qweight.device

        layer.w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        layer.w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )

        marlin_w13_qweight = awq_marlin_moe_repack(
            layer.w13_qweight,
            layer.w13_g_idx_sort_indices,
            size_k=layer.w13_qweight.shape[1],
            size_n=layer.w13_qweight.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w13_qweight", marlin_w13_qweight)

        marlin_w2_qweight = awq_marlin_moe_repack(
            layer.w2_qweight,
            layer.w2_g_idx_sort_indices,
            size_k=layer.w2_qweight.shape[1],
            size_n=layer.w2_qweight.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w2_qweight", marlin_w2_qweight)

        # hidden_size->intermediate_size
        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_scales,
            size_k=layer.intermediate_size_per_partition,
            size_n=layer.w13_scales.shape[2],
            group_size=self.quant_config.group_size,
        )

        replace_parameter(layer, "w13_scales", marlin_w13_scales)

        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_scales,
            size_k=layer.intermediate_size_per_partition,
            size_n=layer.w2_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "w2_scales", marlin_w2_scales)

        marlin_w13_zp = moe_awq_to_marlin_zero_points(
            layer.w13_qzeros,
            size_k=layer.w13_qzeros.shape[1],
            size_n=layer.w13_qzeros.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w13_qzeros", marlin_w13_zp)

        marlin_w2_zp = moe_awq_to_marlin_zero_points(
            layer.w2_qzeros,
            size_k=layer.w2_qzeros.shape[1],
            size_n=layer.w2_qzeros.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w2_qzeros", marlin_w2_zp)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        # The input must currently be float16
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        orig_dtype = x.dtype
        x = x.half()

        topk_weights, topk_ids, router_logits = topk_output

        output = fused_marlin_moe(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            layer.w13_scales,
            layer.w2_scales,
            router_logits,
            topk_weights,
            topk_ids,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            w1_zeros=layer.w13_qzeros,
            w2_zeros=layer.w2_qzeros,
            num_bits=self.quant_config.weight_bits,
        ).to(orig_dtype)
        return StandardCombineInput(hidden_states=output)


def npu_fused_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_offset: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_offset: torch.Tensor,
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
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        antiquant_scale=[w13_scale],
        antiquant_offset=[w13_offset],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    # act_fn: swiglu
    hidden_states = torch_npu.npu_swiglu(hidden_states)
    # gmm2: down_proj
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        antiquant_scale=[w2_scale],
        antiquant_offset=[w2_offset],
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


class AWQMoEAscendMethod(AWQMoEMethod):
    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_qweight_tmp = torch.zeros_like(layer.w13_qweight.data)
        w2_qweight_tmp = torch.zeros_like(layer.w2_qweight.data)
        w13_qzeros_list = []
        w2_qzeros_list = []
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]
        for i in range(0, self.quant_config.pack_factor):
            shift_num = shifts[i] * 4
            w13_qzeros_list.append(
                (layer.w13_qzeros.data.reshape(-1, 1) >> shift_num) & 0xF
            )
            w2_qzeros_list.append(
                (layer.w2_qzeros.data.reshape(-1, 1) >> shift_num) & 0xF
            )
            w13_qweight_tmp.bitwise_or_(
                ((layer.w13_qweight.data >> shift_num) * (2 ** (4 * i)))
                & (0xF << (4 * i))
            )
            w2_qweight_tmp.bitwise_or_(
                ((layer.w2_qweight.data >> shift_num) * (2 ** (4 * i)))
                & (0xF << (4 * i))
            )

        w13_qweight_tmp.bitwise_xor_(0x88888888)
        w2_qweight_tmp.bitwise_xor_(0x88888888)

        w13_qzeros_tmp = torch.cat(w13_qzeros_list, dim=-1).reshape(
            layer.w13_qzeros.shape[0], layer.w13_qzeros.shape[1], -1
        )
        w13_qzeros_tmp = -(w13_qzeros_tmp - 8)
        w13_qzeros_tmp = w13_qzeros_tmp.to(layer.w13_scales.data.dtype)
        w2_qzeros_tmp = torch.cat(w2_qzeros_list, dim=-1).reshape(
            layer.w2_qzeros.shape[0], layer.w2_qzeros.shape[1], -1
        )
        w2_qzeros_tmp = -(w2_qzeros_tmp - 8)
        w2_qzeros_tmp = w2_qzeros_tmp.to(layer.w2_scales.data.dtype)

        layer.register_parameter(
            "w13_qzeros", torch.nn.Parameter(w13_qzeros_tmp, requires_grad=False)
        )
        layer.register_parameter(
            "w13_qweight", torch.nn.Parameter(w13_qweight_tmp, requires_grad=False)
        )
        layer.register_parameter(
            "w2_qzeros", torch.nn.Parameter(w2_qzeros_tmp, requires_grad=False)
        )
        layer.register_parameter(
            "w2_qweight", torch.nn.Parameter(w2_qweight_tmp, requires_grad=False)
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)
        output = npu_fused_experts(
            hidden_states=x,
            w13=layer.w13_qweight,
            w13_scale=layer.w13_scales,
            w13_offset=layer.w13_qzeros,
            w2=layer.w2_qweight,
            w2_scale=layer.w2_scales,
            w2_offset=layer.w2_qzeros,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
        )
        return StandardCombineInput(hidden_states=output)

class AWQAMXMoEMethod(FusedMoEMethodBase):
    """Pure CPU inference AWQ MoE method"""
    
    CPU_INFER = None
    SafeTensor_Loader = None
    
    def __init__(self, quant_config: AWQMarlinConfig,
                 layer_idx, num_gpu_experts, cpuinfer, amx_weight_path, chunked_prefill_size):
        self.tp_rank = get_tensor_model_parallel_rank()
        if self.tp_rank != 0:
            return
        if AWQAMXMoEMethod.CPU_INFER is None:
            AWQAMXMoEMethod.CPU_INFER = cpuinfer_ext.CPUInfer(cpuinfer)
        self.cpu_infer = AWQAMXMoEMethod.CPU_INFER
        # read safetensor weight
        self.load_merged_weight = False
        if glob.glob(os.path.join(amx_weight_path, "*.safetensors")):
            self.load_merged_weight = True
        if self.load_merged_weight:
            if AWQAMXMoEMethod.SafeTensor_Loader is None:
                AWQAMXMoEMethod.SafeTensor_Loader = SafeTensorLoader(amx_weight_path)
            self.safetensor_loader = AWQAMXMoEMethod.SafeTensor_Loader
        self.layer_idx = layer_idx
        self.quant_config = quant_config
        self.num_gpu_experts = num_gpu_experts
        self.amx_weight_path = amx_weight_path
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
        logger.info(f"AWQCPUInferMoEMethod creating weights for layer {self.layer_idx}")
    
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.tp_rank != 0:
            return

        torch.cuda.synchronize()
        gate_proj_ptr = 0
        up_proj_ptr = 0
        down_proj_ptr = 0

        gate_scale_ptr = 0
        up_scale_ptr = 0
        down_scale_ptr = 0

        gate_zeros_ptr = 0
        up_zeros_ptr = 0
        down_proj_ptr = 0

        if self.load_merged_weight:
            # base_key = f"model.layers.{self.layer_idx}"
            base_key = f"blk.{self.layer_idx}"
            # Load pre-sliced NUMA experts
            w = self.safetensor_loader.load_experts(base_key)

            self.gate_proj = np.concatenate(w["gate_weight"], axis=0)
            self.up_proj = np.concatenate(w["up_weight"], axis=0)
            self.down_proj = np.concatenate(w["down_weight"], axis=0)

            self.gate_scale = np.concatenate(w["gate_scale"], axis=0)
            self.up_scale = np.concatenate(w["up_scale"], axis=0)
            self.down_scale = np.concatenate(w["down_scale"], axis=0)

            self.gate_zeros = np.concatenate(w["gate_zeros"], axis=0)
            self.up_zeros = np.concatenate(w["up_zeros"], axis=0)
            self.down_zeros = np.concatenate(w["down_zeros"], axis=0)
            
            gate_proj_ptr = ctypes.addressof(
                ctypes.cast(self.gate_proj.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )
            up_proj_ptr = ctypes.addressof(
                ctypes.cast(self.up_proj.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )
            down_proj_ptr = ctypes.addressof(
                ctypes.cast(self.down_proj.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )
            gate_scale_ptr = ctypes.addressof(
                ctypes.cast(self.gate_scale.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )
            up_scale_ptr = ctypes.addressof(
                ctypes.cast(self.up_scale.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )
            down_scale_ptr = ctypes.addressof(
                ctypes.cast(self.down_scale.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )

            gate_zeros_ptr = ctypes.addressof(
                ctypes.cast(self.gate_zeros.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )
            up_zeros_ptr = ctypes.addressof(
                ctypes.cast(self.up_zeros.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )
            down_zeros_ptr = ctypes.addressof(
                ctypes.cast(self.down_zeros.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )

            
        # Initialize AWQ MoE for CPU inference
        awq_moe_config = MOEConfig(
            self.experts_num,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size,
            self.num_gpu_experts
        )
        awq_moe_config.layer_idx = self.layer_idx
        awq_moe_config.pool = self.cpu_infer.backend_
        awq_moe_config.max_len = self.chunked_prefill_size


        awq_moe_config.gate_proj = gate_proj_ptr
        awq_moe_config.up_proj = up_proj_ptr
        awq_moe_config.down_proj = down_proj_ptr

        awq_moe_config.gate_scale = gate_scale_ptr
        awq_moe_config.up_scale = up_scale_ptr
        awq_moe_config.down_scale = down_scale_ptr

        awq_moe_config.gate_zero = gate_zeros_ptr
        awq_moe_config.up_zero = up_zeros_ptr
        awq_moe_config.down_zero = down_zeros_ptr

        quant_config = QuantConfig()
        quant_config.quant_method = self.quant_config.get_name()
        quant_config.bits = self.quant_config.weight_bits
        quant_config.group_size = self.quant_config.group_size
        quant_config.zero_point = self.quant_config.zero_point

        awq_moe_config.quant_config = quant_config

        self.moe = AMXInt4_1KGroup_MOE(awq_moe_config)

        from sglang.srt.eplb.expert_location_dispatch import get_global_expert_location_metadata
        physical_to_logical_map_cpu = get_global_expert_location_metadata().physical_to_logical_map_cpu[self.layer_idx].contiguous()
        self.cpu_infer.submit(
            self.moe.load_weights(physical_to_logical_map_cpu.data_ptr())
        )
        self.cpu_infer.sync()

        if self.load_merged_weight:
            del self.gate_proj
            del self.up_proj
            del self.down_proj

            del self.gate_scale
            del self.up_scale
            del self.down_scale

            del self.gate_zeros
            del self.up_zeros
            del self.down_zeros

            del awq_moe_config
            del w
        
        logger.info(f"Loading AWQ weights from {self.amx_weight_path} for layer {self.layer_idx}")

    def submit(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_output: TopKOutput,
        *,
        activation: str = "silu",
        routed_scaling_factor: float,
        **kwargs,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.topk import select_experts

        assert activation == "silu", "Only SiLU activation is supported."
        if kwargs.get("expert_map") is not None:
            raise NotImplementedError(
                "Expert Parallelism is not supported for " "fused Marlin MoE method."
            )

        input_tensor_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_tensor_cpu, output_gpu = KExpertsCPUBuffer.get_buffer(x, self.num_experts_per_tok)
        
        topk_weights, topk_ids, router_logits = topk_output

        topk_ids = topk_ids.to(torch.long)

        input_tensor_cpu.copy_(x, non_blocking=True)
        expert_ids_cpu.copy_(topk_ids, non_blocking=True)
        weights_cpu.copy_(topk_weights, non_blocking=True)

        self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(x.device).cuda_stream, self.moe.forward(
            bsz_tensor_cpu.data_ptr(), expert_ids_cpu.size(-1), expert_ids_cpu.data_ptr(),
            weights_cpu.data_ptr(), input_tensor_cpu.data_ptr(), output_cpu.data_ptr()
        ))
        return None
        
    def sync(self, x):
        input_tensor_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_tensor_cpu, output_gpu = KExpertsCPUBuffer.get_buffer(x, self.num_experts_per_tok)
        self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(x.device).cuda_stream)
        output_gpu.copy_(output_cpu, non_blocking=True)
        return output_gpu
    
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, 
              topk_output: "TopKOutput", *, activation: str = "silu", 
              routed_scaling_factor: Optional[float] = None, **kwargs) -> torch.Tensor:
        
        if self.tp_rank != 0:
            return torch.zeros_like(x)
        
        input_tensor_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_tensor_cpu, output_gpu = \
            KExpertsCPUBuffer.get_buffer(x, self.num_experts_per_tok)
        
        topk_weights, topk_ids, router_logits = topk_output
        if routed_scaling_factor is not None:
            topk_weights = topk_weights * routed_scaling_factor
        topk_ids = topk_ids.to(torch.long)

        # Copy data to CPU
        input_tensor_cpu.copy_(x.to(torch.bfloat16), non_blocking=True)
        expert_ids_cpu.copy_(topk_ids, non_blocking=True)
        weights_cpu.copy_(topk_weights, non_blocking=True)

        print(bsz_tensor_cpu, flush=True)

        # Submit to CPUInfer
        self.cpu_infer.submit_with_cuda_stream(
            torch.cuda.current_stream(x.device).cuda_stream,
            self.moe.forward(
                bsz_tensor_cpu.data_ptr(), expert_ids_cpu.size(-1),
                expert_ids_cpu.data_ptr(), weights_cpu.data_ptr(),
                input_tensor_cpu.data_ptr(), output_cpu.data_ptr()
            )
        )
        
        # Wait for completion and copy back to GPU
        self.cpu_infer.sync_with_cuda_stream(
            torch.cuda.current_stream(x.device).cuda_stream
        )

        output_gpu.copy_(output_cpu, non_blocking=True)
        return output_gpu


class AWQAMXEPMoEMethod(FusedMoEMethodBase):
    """Expert Parallelism AWQ MoE method (CPU + GPU hybrid)"""
    
    def __init__(self, quant_config: AWQMarlinConfig, layer_idx: int):
        self.tp_rank = get_tensor_model_parallel_rank()

        if 'MOE_NUM_GPU_EXPERTS' not in os.environ or 'MOE_CPUINFER' not in os.environ or 'MOE_AMX_WEIGHT_PATH' not in os.environ:
            raise RuntimeError("the following arguments are required: --amx-weight-path, --cpuinfer, --num-gpu-experts")
        
        self.num_gpu_experts = int(os.environ.get('MOE_NUM_GPU_EXPERTS'))
        self.enable_defer = os.environ.get("MOE_ENABLE_DEFER", "False").lower() == "true"
        cpuinfer = int(os.environ.get('MOE_CPUINFER'))
        amx_weight_path = os.environ.get('MOE_AMX_WEIGHT_PATH', '')
        chunked_prefill_size = int(os.environ.get('MOE_CHUNKED_PREFILL_SIZE', 8192))
        
        # Create CPU and GPU methods
        self.cpu_method = AWQAMXMoEMethod(
            quant_config, layer_idx, self.num_gpu_experts, 
            cpuinfer, amx_weight_path, chunked_prefill_size
        )
        self.marlin_method = AWQMoEMethod(quant_config, self.num_gpu_experts)
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
        self.marlin_method.process_weights_after_loading(layer)

    def submit(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_output: TopKOutput,
        *,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        routed_scaling_factor: float,
        **kwargs,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert activation == "silu", "Only SiLU activation is supported."

        orig_dtype = x.dtype
        x = x.half()

        topk_weights, topk_ids, router_logits = topk_output

        if routed_scaling_factor is not None:
            topk_weights = topk_weights * routed_scaling_factor

        # Submit CPU inference (async)
        if self.tp_rank == 0:
            # print("Submit Task", flush=True)
            input_tensor_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_tensor_cpu, output_gpu = \
                KExpertsCPUBuffer.get_buffer(x, self.cpu_method.num_experts_per_tok)
            topk_ids_long = topk_ids.to(torch.long)

            input_tensor_cpu.copy_(x, non_blocking=True)
            expert_ids_cpu.copy_(topk_ids_long, non_blocking=True)
            weights_cpu.copy_(topk_weights, non_blocking=True)

            self.cpu_method.cpu_infer.submit_with_cuda_stream(
                torch.cuda.current_stream(x.device).cuda_stream,
                self.cpu_method.moe.forward(
                    bsz_tensor_cpu.data_ptr(), expert_ids_cpu.size(-1),
                    expert_ids_cpu.data_ptr(), weights_cpu.data_ptr(),
                    input_tensor_cpu.data_ptr(), output_cpu.data_ptr()
                )
            )
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)
        # GPU inference (sync)
        # output = fused_marlin_moe(
        #     x,
        #     layer.w13_qweight,
        #     layer.w2_qweight,
        #     layer.w13_scales,
        #     layer.w2_scales,
        #     router_logits,
        #     topk_weights,
        #     topk_ids,
        #     sort_indices1=layer.w13_g_idx_sort_indices,
        #     sort_indices2=layer.w2_g_idx_sort_indices,
        #     w1_zeros=layer.w13_qzeros,
        #     w2_zeros=layer.w2_qzeros,
        #     num_bits=self.marlin_method.quant_config.weight_bits,
        #     global_num_experts=self.global_num_experts,
        #     num_gpu_experts=self.num_gpu_experts
        # ).to(orig_dtype)
        output = npu_fused_experts(
            hidden_states=x,
            w13=layer.w13_qweight,
            w13_scale=layer.w13_scales,
            w13_offset=layer.w13_qzeros,
            w2=layer.w2_qweight,
            w2_scale=layer.w2_scales,
            w2_offset=layer.w2_qzeros,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
            global_num_experts=self.global_num_experts,
            num_gpu_experts=self.num_gpu_experts
        )
        return StandardCombineInput(hidden_states=output)

    def sync(self, x):
        orig_dtype = x.dtype
        input_tensor_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_tensor_cpu, output_gpu = KExpertsCPUBuffer.get_buffer(x, self.cpu_method.num_experts_per_tok)
        self.cpu_method.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(x.device).cuda_stream)
        output_gpu.copy_(output_cpu, non_blocking=True)
        return output_gpu.to(orig_dtype)

        
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_output: TopKOutput,
        *,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        
        assert activation == "silu", "Only SiLU activation is supported."
        orig_dtype = x.dtype
        x = x.half()

        topk_weights, topk_ids, router_logits = topk_output
        if routed_scaling_factor is not None:
            topk_weights = topk_weights * routed_scaling_factor
        

        if self.tp_rank == 0:
            input_tensor_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_tensor_cpu, output_gpu = \
                KExpertsCPUBuffer.get_buffer(x, self.cpu_method.num_experts_per_tok)
            topk_ids_long = topk_ids.to(torch.long)

            input_tensor_cpu.copy_(x, non_blocking=True)
            expert_ids_cpu.copy_(topk_ids_long, non_blocking=True)
            weights_cpu.copy_(topk_weights, non_blocking=True)

            self.cpu_method.cpu_infer.submit_with_cuda_stream(
                torch.cuda.current_stream(x.device).cuda_stream,
                self.cpu_method.moe.forward(
                    bsz_tensor_cpu.data_ptr(), expert_ids_cpu.size(-1),
                    expert_ids_cpu.data_ptr(), weights_cpu.data_ptr(),
                    input_tensor_cpu.data_ptr(), output_cpu.data_ptr()
                )
            )
        
        # GPU inference (sync)
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)
        output = npu_fused_experts(
            hidden_states=x,
            w13=layer.w13_qweight,
            w13_scale=layer.w13_scales,
            w13_offset=layer.w13_qzeros,
            w2=layer.w2_qweight,
            w2_scale=layer.w2_scales,
            w2_offset=layer.w2_qzeros,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
            global_num_experts=self.global_num_experts,
            num_gpu_experts=self.num_gpu_experts
        )
        # output = fused_marlin_moe(
        #     x,
        #     layer.w13_qweight,
        #     layer.w2_qweight,
        #     layer.w13_scales,
        #     layer.w2_scales,
        #     router_logits,
        #     topk_weights,
        #     topk_ids,
        #     sort_indices1=layer.w13_g_idx_sort_indices,
        #     sort_indices2=layer.w2_g_idx_sort_indices,
        #     w1_zeros=layer.w13_qzeros,
        #     w2_zeros=layer.w2_qzeros,
        #     num_bits=self.marlin_method.quant_config.weight_bits,
        #     global_num_experts=self.global_num_experts,
        #     num_gpu_experts=self.num_gpu_experts
        # ).to(orig_dtype)


        # Wait for CPU and combine results
        if self.tp_rank == 0:
            self.cpu_method.cpu_infer.sync_with_cuda_stream(
                torch.cuda.current_stream(x.device).cuda_stream
            )

            output_gpu.copy_(output_cpu, non_blocking=True)
            output += output_gpu.to(orig_dtype)
        
        return output