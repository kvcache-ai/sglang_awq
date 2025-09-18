from typing import List, Dict
import os

import torch
from safetensors import safe_open

def override_config(
    num_gpu_experts, cpuinfer, subpool_count,
    cpu_save, cpu_original_weight_path, cpu_weight_path,
    cpu_method, chunked_prefill_size, enable_defer, cpu_embed):
    if num_gpu_experts is not None:
        os.environ['MOE_NUM_GPU_EXPERTS'] = str(num_gpu_experts)
    if cpuinfer is not None:
        os.environ['MOE_CPUINFER'] = str(cpuinfer)
    if subpool_count is not None:
        os.environ['SUBPOOL_COUNT'] = str(subpool_count)
    if cpu_original_weight_path is not None:
        os.environ['MOE_CPU_ORIGINAL_WEIGHT_PATH'] = str(cpu_original_weight_path)
    if cpu_weight_path is not None:
        os.environ['MOE_CPU_WEIGHT_PATH'] = str(cpu_weight_path)
    if cpu_method is not None:
        os.environ['MOE_CPU_METHOD'] = str(cpu_method)
    if cpu_embed is not None:
        os.environ['CPU_EMBED'] = str(cpu_embed)
    os.environ['MOE_CHUNKED_PREFILL_SIZE'] = str(chunked_prefill_size)
    os.environ['MOE_ENABLE_DEFER'] = str(enable_defer)
    os.environ['MOE_CPU_SAVE'] = str(cpu_save)

class KExpertsCPUBuffer():
    capture_bs: List = list()
    capture_buffers: Dict = dict()
    temp_bs: int = 0
    temp_buffer: tuple = tuple()
    @classmethod
    def get_buffer(cls, hidden_states: torch.Tensor, num_experts_per_tok):
        hidden_size = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_size)
        batch_size, hidden_size = hidden_states.shape
        
        if batch_size in KExpertsCPUBuffer.capture_buffers:
            return KExpertsCPUBuffer.capture_buffers[batch_size]
        if batch_size == KExpertsCPUBuffer.temp_bs:
            return KExpertsCPUBuffer.temp_buffer

        input_tensor_cpu = torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16)
        expert_ids_cpu = torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.int64, pin_memory=True)
        weights_cpu = torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True)
        output_cpu = torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=True, dtype=torch.float32)
        bsz_tensor_cpu = torch.tensor((batch_size), device="cpu", dtype=torch.int32, pin_memory=True)
        # output_gpu = torch.zeros_like(hidden_states)
        
        # cur_buffer = (input_tensor_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_tensor_cpu, output_gpu)
        cur_buffer = (input_tensor_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_tensor_cpu)
        if batch_size in KExpertsCPUBuffer.capture_bs:
            KExpertsCPUBuffer.capture_buffers[batch_size] = cur_buffer
        KExpertsCPUBuffer.temp_bs = batch_size
        KExpertsCPUBuffer.temp_buffer = cur_buffer
        return cur_buffer
    
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
        """Load experts with automatic format detection (NUMA vs raw W8A8)"""
        # Try NUMA format first (backward compatibility)
        amx_up_base = f"{base_key}.ffn_up_exps" 
        if self.has_tensor(f"{amx_up_base}.0.weight"):
            return self._load_experts_amx_format(base_key, device)

        original_up_base = f"{base_key}.mlp.experts.0.up_proj.weight"
        if self.has_tensor(f"{original_up_base}"):
            return self._load_experts_original_format(base_key, device)
            
        raise ValueError(f"No experts found for key {base_key} in either format")

    def _load_experts_original_format(self, base_key: str, device: str="cpu"):
        """Load experts in pre-generated AMX-Compatible format (original)"""
        expert_base = f"{base_key}.mlp.experts"
        max_experts_count = -1
        
        while self.has_tensor(f"{expert_base}.{max_experts_count+1}.up_proj.weight"):
            max_experts_count += 1
        if max_experts_count == -1:
            raise ValueError(f"No experts found for key {base_key}")
            
        # Initialize empty lists to store tensors for each projection type
        up_weights = []
        gate_weights = []
        down_weights = []
        
        for expert_id in range(max_experts_count+1):
            up_key = f"{expert_base}.{expert_id}.up_proj.weight"
            gate_key = f"{expert_base}.{expert_id}.gate_proj.weight"
            down_key = f"{expert_base}.{expert_id}.down_proj.weight"
            
            up_tensor = self.load_tensor(up_key, device)
            gate_tensor = self.load_tensor(gate_key, device)
            down_tensor = self.load_tensor(down_key, device)

            up_weights.append(up_tensor)
            gate_weights.append(gate_tensor)
            down_weights.append(down_tensor)

        return {
            "up_weight": up_weights,
            "gate_weight": gate_weights,
            "down_weight": down_weights,
        }
    
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
