from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional
import logging
import torch
import glob
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
from sglang.srt.custom_op import CustomOp
from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_hip,
    is_npu,
    set_weight_attrs,
    use_intel_amx_backend,
)
from sglang.srt.layers.cpuinfer import KExpertsCPUBuffer, SafeTensorLoader
logger = logging.getLogger(__name__)
_is_npu = is_npu()

if _is_npu:
    import torch_npu
try:
    import kt_kernel_ext
    from sglang.srt.distributed import get_tensor_model_parallel_rank
    CPUINFER_AVAILABLE = True
    if CPUINFER_AVAILABLE and kt_kernel_ext:
        from kt_kernel_ext.moe import MOEConfig, Int4_KERNEL_MOE
except ImportError as e:
    print(f"[WARN]: CPUInfer is not available {e.msg}")
    CPUINFER_AVAILABLE = False
    kt_kernel = None

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_is_cpu = is_cpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

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
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

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
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["weight"])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_intel_amx_backend(layer):
            x_shapes = x.shape
            if len(x_shapes) == 3:
                x = x.view(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.weight_packed_linear(
                x,
                layer.weight,
                bias,
                True,  # is_vnni
            )
            if len(x_shapes) == 3:
                output = output.view(x_shapes[0], x_shapes[1], -1)
            return output

        return F.linear(x, layer.weight, bias)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(
        self,
        use_triton_kernels: bool = False,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False

        # Initialize CPUInfer
        self.tp_rank = get_tensor_model_parallel_rank()

        if 'MOE_CPUINFER' in os.environ:
            if 'MOE_NUM_GPU_EXPERTS' not in os.environ or 'MOE_CPU_WEIGHT_PATH' not in os.environ:
                raise RuntimeError("the following arguments are required: --cpu-weight-path, --num-gpu-experts")
            self.num_gpu_experts = int(os.environ.get('MOE_NUM_GPU_EXPERTS'))
            self.enable_defer = os.environ.get("MOE_ENABLE_DEFER", "False").lower() == "true"
            cpuinfer = int(os.environ.get('MOE_CPUINFER'))
            cpu_save = os.environ.get('MOE_CPU_SAVE', 'False').lower() == 'true'
            cpu_original_weight_path = os.environ.get('MOE_CPU_ORIGINAL_WEIGHT_PATH', '')
            cpu_weight_path = os.environ.get('MOE_CPU_WEIGHT_PATH', '')
            subpool_count = int(os.environ.get('SUBPOOL_COUNT'))
            chunked_prefill_size = int(os.environ.get('MOE_CHUNKED_PREFILL_SIZE', 8192))

            self.cpu_method = CPUMoEMethod(
                layer_idx, self.num_gpu_experts, cpuinfer, subpool_count,
                cpu_save, cpu_original_weight_path, cpu_weight_path, chunked_prefill_size
            )
        else:
            self.num_gpu_experts = -1
            self.cpu_method = None

        self.triton_kernel_moe_forward = None
        self.triton_kernel_moe_with_bias_forward = None
        if torch.cuda.is_available() and has_triton_kernels:
            from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
                triton_kernel_moe_forward as _tk_forward,
            )
            from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
                triton_kernel_moe_with_bias_forward as _tk_with_bias_forward,
            )

            self.triton_kernel_moe_forward = _tk_forward
            self.triton_kernel_moe_with_bias_forward = _tk_with_bias_forward

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        if self.cpu_method is not None:
            self.cpu_method.create_weights(
                layer, num_experts, hidden_size,
                intermediate_size_per_partition,
                params_dtype, **extra_weight_attrs
            )

        if self.num_gpu_experts !=-1:
            num_experts = self.num_gpu_experts
        self.num_gpu_experts = num_experts


        self.with_bias = with_bias

        if num_experts == 0:
            w13_weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            layer.register_parameter("w13_weight", w13_weight)
            set_weight_attrs(w13_weight, extra_weight_attrs)

            w2_weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            layer.register_parameter("w2_weight", w2_weight)
            set_weight_attrs(w2_weight, extra_weight_attrs)

            if self.with_bias:
                w13_weight_bias = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
                layer.register_parameter("w13_weight_bias", w13_weight_bias)
                set_weight_attrs(w13_weight_bias, extra_weight_attrs)

                w12_weight_bias = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
                layer.register_parameter("w2_weight_bias", w12_weight_bias)
                set_weight_attrs(w12_weight_bias, extra_weight_attrs)
        else:
            # Fused gate_up_proj (column parallel)
            w13_weight_n, w13_weight_k = 2 * intermediate_size_per_partition, hidden_size
            if self.use_triton_kernels:
                w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
            w13_weight = torch.nn.Parameter(
                torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight", w13_weight)
            set_weight_attrs(w13_weight, extra_weight_attrs)

            if self.with_bias:
                w13_weight_bias = torch.nn.Parameter(
                    torch.empty(
                        num_experts,
                        2 * intermediate_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                layer.register_parameter("w13_weight_bias", w13_weight_bias)
                set_weight_attrs(w13_weight_bias, extra_weight_attrs)

            # down_proj (row parallel)
            w2_weight_n, w2_weight_k = (
                hidden_size,
                intermediate_size_per_partition,
            )
            if self.use_triton_kernels:
                w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
            w2_weight = torch.nn.Parameter(
                torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight", w2_weight)
            set_weight_attrs(w2_weight, extra_weight_attrs)

            if self.with_bias:
                w2_weight_bias = torch.nn.Parameter(
                    torch.empty(num_experts, hidden_size, dtype=torch.float32),
                    requires_grad=False,
                )
                layer.register_parameter("w2_weight_bias", w2_weight_bias)
                set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.cpu_method is not None:
            self.cpu_method.process_weights_after_loading()

        if self.num_gpu_experts > 0 and _use_aiter:
            layer.w13_weight = torch.nn.Parameter(
                shuffle_weight(layer.w13_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                shuffle_weight(layer.w2_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()

        # Pack weight for get better performance on CPU
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])

        return

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        backend = (
            MoeRunnerBackend.TRITON_KERNELS
            if self.use_triton_kernels
            else MoeRunnerBackend.TRITON
        )
        self.runner = MoeRunner(backend, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return self.forward(
            layer=layer,
            dispatch_output=dispatch_output,
        )
    
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

    def sync(
        self, x
    ) -> torch.Tensor :
        _, _, _, output_cpu, _ = KExpertsCPUBuffer.get_buffer(x, self.cpu_method.num_experts_per_tok)
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

            self.moe_kexperts_param = (bsz_tensor_cpu, expert_ids_cpu, weights_cpu, input_tensor_cpu, output_cpu)

            if torch.npu.is_current_stream_capturing():
                input_tensor_cpu.copy_(x, non_blocking=True)
                expert_ids_cpu.copy_(topk_ids_long, non_blocking=True)
                weights_cpu.copy_(topk_weights, non_blocking=True)

                torch_npu.npu._launch_host_func(
                    torch.npu.current_stream(),
                    self._submit_to_cpu,
                    self.moe_kexperts_param
                )
            else:
                input_tensor_cpu.copy_(x, non_blocking=False)
                expert_ids_cpu.copy_(topk_ids_long, non_blocking=False)
                weights_cpu.copy_(topk_weights, non_blocking=False)

                self._submit_to_cpu(self.moe_kexperts_param)

        return StandardCombineInput(hidden_states=torch.zeros_like(x))

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        backend = self.runner.runner_backend
        if backend.is_triton_kernels():
            from sglang.srt.layers.moe.moe_runner.triton_kernels import (
                TritonKernelsQuantInfo,
            )

            quant_info = TritonKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_bias=getattr(layer, "w13_weight_bias", None),
                w2_bias=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)
        else:
            if _use_aiter:
                assert not moe_runner_config.no_combine, "unsupported"
                topk_weights, topk_ids, _ = topk_output
                if moe_runner_config.apply_router_weight_on_input:
                    assert (
                        topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
                    _, topk = topk_weights.shape
                    assert (
                        topk == 1
                    ), "Only support topk=1 when `apply_router_weight_on_input` is True"
                    x = x * topk_weights.to(x.dtype)
                    topk_weights = torch.ones_like(
                        topk_weights, dtype=torch.float32
                    )  # topk_weights must be FP32 (float32)
                output = fused_moe(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    activation=(
                        ActivationType.Silu
                        if moe_runner_config.activation == "silu"
                        else ActivationType.Gelu
                    ),
                )
                return StandardCombineInput(hidden_states=output)
            else:
                quant_info = TritonMoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    b13=getattr(layer, "w13_weight_bias", None),
                    b2=getattr(layer, "w2_weight_bias", None),
                )
                return self.runner.run(dispatch_output, quant_info)

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        assert (
            moe_runner_config.activation == "silu"
        ), f"activation = {moe_runner_config.activation} is not supported."

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace # See [Note] inplace should be False in fused_experts.
                False,  # use_int8_w8a8
                False,  # use_fp8_w8a16
                None,  # w1_scale
                None,  # w2_scale
                None,  # block_size
                None,  # a1_scale
                None,  # a2_scale
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)
        else:
            from sglang.srt.layers.moe.fused_moe_native import moe_forward_native

            output = moe_forward_native(
                layer,
                x,
                topk_output,
                moe_runner_config,
            )
            return StandardCombineInput(hidden_states=output)

    def forward_npu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        import torch_npu

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        if self.cpu_method is not None and self.tp_rank == 0:
            input_tensor_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_tensor_cpu = \
                KExpertsCPUBuffer.get_buffer(x, self.cpu_method.num_experts_per_tok)
            topk_ids_long = topk_ids.to(torch.int64)

            self.moe_kexperts_param = (bsz_tensor_cpu, expert_ids_cpu, weights_cpu, input_tensor_cpu, output_cpu)

            if torch.npu.is_current_stream_capturing():
                input_tensor_cpu.copy_(x, non_blocking=True)
                expert_ids_cpu.copy_(topk_ids_long, non_blocking=True)
                weights_cpu.copy_(topk_weights, non_blocking=True)

                torch_npu.npu._launch_host_func(
                    torch.npu.current_stream(),
                    self._submit_to_cpu,
                    self.moe_kexperts_param
                )
            else:
                input_tensor_cpu.copy_(x, non_blocking=False)
                expert_ids_cpu.copy_(topk_ids_long, non_blocking=False)
                weights_cpu.copy_(topk_weights, non_blocking=False)

                self._submit_to_cpu(self.moe_kexperts_param)

        if self.num_gpu_experts > 0:
            original_dtype = x.dtype
            num_tokens = x.shape[0]
            topk_weights = topk_weights.to(x.dtype)
            topk_ids = topk_ids.to(torch.int32)
            num_experts = layer.num_experts
            top_k = layer.top_k
            row_idx_len = num_tokens * top_k
            row_idx = (
                torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
                .view(top_k, -1)
                .permute(1, 0)
                .contiguous()
            )

            hidden_states, expanded_row_idx, expanded_expert_idx = (
                torch_npu.npu_moe_init_routing(
                    x, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
                )
            )

            expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
                expanded_expert_idx, num_experts
            )

            expert_tokens = expert_tokens.to(torch.int64)
            if layer.w13_weight.shape[-1] == layer.hidden_size:
                w13 = layer.w13_weight.transpose(1, 2)
                w2 = layer.w2_weight.transpose(1, 2)

            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[w13],
                split_item=2,
                group_list_type=0,
                group_type=0,
                group_list=expert_tokens,
                output_dtype=original_dtype,
            )[0]

            # act_fn:
            if self.moe_runner_config.activation == "silu":
                hidden_states = torch_npu.npu_swiglu(hidden_states)
            else:
                from sglang.srt.layers.activation import GeluAndMul

                hidden_states = GeluAndMul()(hidden_states)

            # gmm2: down_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[w2],
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
        else:
            final_hidden_states = torch.zeros_like(x)

        if self.cpu_method is not None and self.tp_rank == 0:
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
            output_cpuinfer = output_gpu.to(dtype=x.dtype)
            final_hidden_states = final_hidden_states + output_cpuinfer

        return StandardCombineInput(hidden_states=final_hidden_states)

    def forward_tpu(self, *args, **kwargs) -> CombineInput:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    forward_native = forward_cpu

class CPUMoEMethod():
    """Pure CPU inference MoE method"""
    
    CPU_INFER = None
    SafeTensor_Loader = None
    
    def __init__(self,
                 layer_idx, num_gpu_experts, cpuinfer, subpool_count, 
                 cpu_save, cpu_original_weight_path, cpu_weight_path, chunked_prefill_size):
        self.tp_rank = get_tensor_model_parallel_rank()
        if self.tp_rank != 0:
            return
        if CPUMoEMethod.CPU_INFER is None:
            print(f"subpool count is {subpool_count}", flush=True)
            worker_config = kt_kernel_ext.WorkerPoolConfig()
            subpool_numa_map = list(range(subpool_count))
            subpool_thread_count = [
            cpuinfer // subpool_count + (1 if i < cpuinfer % subpool_count else 0)
            for i in range(subpool_count)
            ]

            worker_config.subpool_count = subpool_count
            worker_config.subpool_numa_map= subpool_numa_map
            worker_config.subpool_thread_count = subpool_thread_count
            CPUMoEMethod.CPU_INFER = kt_kernel_ext.CPUInfer(worker_config)
        self.cpu_infer = CPUMoEMethod.CPU_INFER
        # read safetensor weight
        self.load_merged_weight = False
        if cpu_save:
            if glob.glob(os.path.join(cpu_original_weight_path, "*.safetensors")):
                self.load_merged_weight = True
            else:
                raise RuntimeError(f"Cannot find safetensors in {cpu_original_weight_path} for cpu_save mode")
            if CPUMoEMethod.SafeTensor_Loader is None:
                CPUMoEMethod.SafeTensor_Loader = SafeTensorLoader(cpu_original_weight_path)
            self.safetensor_loader = CPUMoEMethod.SafeTensor_Loader
        else:
            if glob.glob(os.path.join(cpu_weight_path, "*.safetensors")):
                self.load_merged_weight = True
            if self.load_merged_weight:
                if CPUMoEMethod.SafeTensor_Loader is None:
                    CPUMoEMethod.SafeTensor_Loader = SafeTensorLoader(cpu_weight_path)
                self.safetensor_loader = CPUMoEMethod.SafeTensor_Loader
        self.layer_idx = layer_idx
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
    
    def process_weights_after_loading(self) -> None:
        if self.tp_rank != 0:
            return

        if is_npu():
            torch.npu.synchronize()

        moe_config = MOEConfig(
            self.experts_num,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size
        )
        moe_config.layer_idx = self.layer_idx
        moe_config.pool = self.cpu_infer.backend_
        moe_config.max_len = self.chunked_prefill_size

        if self.load_merged_weight:
            gate_proj_ptr = 0
            up_proj_ptr = 0
            down_proj_ptr = 0

            if self.load_merged_weight:
                base_key = f"model.layers.{self.layer_idx}"
                # Load pre-sliced NUMA experts
                w = self.safetensor_loader.load_experts(base_key)

                self.gate_proj = torch.cat(w["gate_weight"], dim=0).contiguous()
                self.up_proj   = torch.cat(w["up_weight"], dim=0).contiguous()
                self.down_proj = torch.cat(w["down_weight"], dim=0).contiguous()
                
                gate_proj_ptr = self.gate_proj.data_ptr()
                up_proj_ptr = self.up_proj.data_ptr()
                down_proj_ptr = self.down_proj.data_ptr()

            moe_config.gate_proj = gate_proj_ptr
            moe_config.up_proj = up_proj_ptr
            moe_config.down_proj = down_proj_ptr

            if self.cpu_save:
                moe_config.save = True
                moe_config.load = False
            else:
                moe_config.load = True
            moe_config.path = self.cpu_weight_path
        else:
            moe_config.load = True
            moe_config.path = self.cpu_weight_path

        self.moe = Int4_KERNEL_MOE(moe_config)

        self.cpu_infer.submit(
            self.moe.load_weights_task()
        )
        self.cpu_infer.sync()

        del moe_config

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