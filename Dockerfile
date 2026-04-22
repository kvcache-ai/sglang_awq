ARG CANN_VERSION=8.3.rc2
ARG DEVICE_TYPE=910b
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=py3.11

FROM quay.io/ascend/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION

# -------------------------------------------------------------------
# Args
# -------------------------------------------------------------------
ARG PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
ARG PYTORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG PTA_URL="https://gitee.com/ascend/pytorch/releases/download/v7.1.0.1-pytorch2.6.0/torch_npu-2.6.0.post1-cp311-cp311-manylinux_2_28_aarch64.whl"
ARG TRITON_ASCEND_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/triton_ascend-3.2.0%2Bgitb0ea0850-cp311-cp311-linux_aarch64.whl"
ARG BISHENG_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/Ascend-BiSheng-toolkit_aarch64.run"
ARG MEMFABRIC_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl"
ARG SGLANG_URL="git@github.com:kvcache-ai/sglang_awq.git"
ARG SGLANG_TAG=mxcpu
ARG KTRANSFORMERS_URL="git@github.com:kvcache-ai/ktransformers-dev.git"
ARG KTRANSFORMERS_TAG=ascend-kunpeng
ARG SGLANG_KERNEL_NPU_URL="https://github.com/sgl-project/sgl-kernel-npu.git"
ARG SGLANG_KERNEL_NPU_TAG=main
ARG ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------------------------------------------
# System dependencies
# -------------------------------------------------------------------
RUN apt-get update -y && apt-get install -y \
    build-essential cmake vim wget curl net-tools \
    zlib1g-dev lld clang locales ccache \
    openssl libssl-dev pkg-config ca-certificates \
    protobuf-compiler libnuma-dev libhwloc-dev \
    python3.11-dev openssh-client libfmt-dev \
    && rm -rf /var/cache/apt/* /var/lib/apt/lists/* \
    && update-ca-certificates \
    && locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8
ENV PATH="/root/.cargo/bin:${PATH}"

# -------------------------------------------------------------------
# SSH setup for private repos
# -------------------------------------------------------------------
COPY ssh /root/.ssh
RUN chmod 600 /root/.ssh/id_rsa && \
    ssh-keyscan -p 443 ssh.github.com >> /root/.ssh/known_hosts

# -------------------------------------------------------------------
# Rust & build tools
# -------------------------------------------------------------------
RUN pip install setuptools setuptools-rust wheel build --no-cache-dir -i ${PIP_MIRROR}
RUN export RUSTUP_DIST_SERVER=https://mirrors.aliyun.com/rustup && \
    export RUSTUP_UPDATE_ROOT=https://mirrors.aliyun.com/rustup/rustup && \
    curl --proto '=https' --tlsv1.2 -sSf https://mirrors.aliyun.com/repo/rust/rustup-init.sh | sh -s -- -y \
    && rustc --version && cargo --version && protoc --version

# -------------------------------------------------------------------
# PyTorch + torch_npu (local wheel)
# -------------------------------------------------------------------
COPY torch_npu-2.6.0.post3-cp311-cp311-manylinux_2_28_aarch64.whl /tmp/
RUN pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION \
    --index-url https://download.pytorch.org/whl/cpu --no-cache-dir \
    && pip install /tmp/torch_npu-2.6.0.post3-cp311-cp311-manylinux_2_28_aarch64.whl --no-cache-dir \
    && rm /tmp/torch_npu-2.6.0.post3-cp311-cp311-manylinux_2_28_aarch64.whl

# -------------------------------------------------------------------
# triton-ascend + memfabric
# -------------------------------------------------------------------
RUN pip install ${TRITON_ASCEND_URL} --no-cache-dir \
    && pip install ${MEMFABRIC_URL} --no-cache-dir

# -------------------------------------------------------------------
# vLLM (empty target, for dependency satisfaction)
# -------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/vllm-project/vllm.git --branch v0.8.5 && \
    (cd vllm && VLLM_TARGET_DEVICE="empty" pip install -v . --no-cache-dir -i ${PIP_MIRROR}) && \
    rm -rf vllm

# -------------------------------------------------------------------
# Bisheng compiler
# -------------------------------------------------------------------
RUN wget ${BISHENG_URL} -O /tmp/bisheng.run && \
    chmod a+x /tmp/bisheng.run && \
    /tmp/bisheng.run --install && \
    rm /tmp/bisheng.run

# -------------------------------------------------------------------
# HPCKit (KML only, extracted from tar.gz)
# -------------------------------------------------------------------
COPY HPCKit_25.1.0_Linux-aarch64.tar.gz /workspace/
RUN cd /workspace && tar -zxf HPCKit_25.1.0_Linux-aarch64.tar.gz -C /opt && \
    mkdir -p /opt/HPCKit/25.1.0/compiler/gcc && \
    tar -zxf /opt/HPCKit_25.1.0_Linux-aarch64/package/gcc-12.3.1-2025.03-aarch64-linux.tar.gz \
        -C /opt/HPCKit/25.1.0/compiler/gcc --strip-components=1 && \
    cd /opt/HPCKit_25.1.0_Linux-aarch64/package && \
    tar -zxf KunpengHPCKit-kml.25.1.0.tar.gz && \
    cd KunpengHPCKit-kml.25.1.0 && cp -r gcclib lib && \
    mkdir -p /opt/HPCKit/25.1.0/kml && \
    cp -r /opt/HPCKit_25.1.0_Linux-aarch64/package/KunpengHPCKit-kml.25.1.0/* /opt/HPCKit/25.1.0/kml/ && \
    rm /workspace/HPCKit_25.1.0_Linux-aarch64.tar.gz && \
    rm -rf /opt/HPCKit_25.1.0_Linux-aarch64

# -------------------------------------------------------------------
# sgl-kernel-npu + deep-ep (pre-built wheels)
# -------------------------------------------------------------------
COPY sgl_kernel_npu-0.1.0-cp311-cp311-linux_aarch64.whl /tmp/
COPY deep_ep-1.0.0+aae2a1ad-cp311-cp311-linux_aarch64.whl /tmp/
RUN pip install wheel==0.45.1 -i ${PIP_MIRROR} && \
    pip install /tmp/sgl_kernel_npu-0.1.0-cp311-cp311-linux_aarch64.whl \
                /tmp/deep_ep-1.0.0+aae2a1ad-cp311-cp311-linux_aarch64.whl \
                --no-cache-dir -i ${PIP_MIRROR} && \
    rm /tmp/sgl_kernel_npu-0.1.0-cp311-cp311-linux_aarch64.whl \
       /tmp/deep_ep-1.0.0+aae2a1ad-cp311-cp311-linux_aarch64.whl

# -------------------------------------------------------------------
# ktransformers-dev → kt-kernel (local copy, build manually in container)
# -------------------------------------------------------------------
COPY ktransformers-dev /opt/ktransformers-dev

# -------------------------------------------------------------------
# sglang_awq
# -------------------------------------------------------------------
RUN git clone $SGLANG_URL --branch $SGLANG_TAG sglang && \
    (cd sglang/python && rm -rf pyproject.toml && mv pyproject_other.toml pyproject.toml && pip install -v ".[srt_npu]" --no-cache-dir -i ${PIP_MIRROR}) && \
    rm -rf sglang

# -------------------------------------------------------------------
# Runtime env
# -------------------------------------------------------------------
ENV TORCH_DEVICE_BACKEND_AUTOLOAD=0
ENV KT_KERNEL_PATH=/opt/ktransformers-dev/kt-kernel
# Driver paths must come first to avoid symbol conflicts with CANN toolkit stubs
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/opt/HPCKit/25.1.0/kml/lib:/opt/HPCKit/25.1.0/compiler/gcc/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${ASCEND_CANN_PATH}/latest/runtime/lib64/stub:${LD_LIBRARY_PATH}
# kt-kernel .so paths via ldconfig (works for non-interactive shells too)
RUN echo -e "${KT_KERNEL_PATH}/prefillint8gemm\n${KT_KERNEL_PATH}/prefillint4gemm\n${KT_KERNEL_PATH}/build" > /etc/ld.so.conf.d/kt-kernel.conf && ldconfig

# Ascend CANN environment (must source set_env.sh for full runtime)
# Source after Dockerfile ENV so driver paths stay first
RUN echo "source ${ASCEND_CANN_PATH}/latest/set_env.sh" >> /root/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH' >> /root/.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]
