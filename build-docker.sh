#!/bin/bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-sglang-kml:latest}"
BUILD_DIR="$(cd "$(dirname "$0")" && pwd)/.docker-build-context"

echo "==> Preparing build context in $BUILD_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/ssh"

# SSH keys for cloning private repos
echo "  -> SSH keys"
cp /home/shangyouren/.ssh/id_rsa "$BUILD_DIR/ssh/"
cp /home/shangyouren/.ssh/id_rsa.pub "$BUILD_DIR/ssh/"
cat > "$BUILD_DIR/ssh/config" << 'EOF'
Host github.com
  Hostname ssh.github.com
  Port 443
  User git
  IdentityFile /root/.ssh/id_rsa
EOF
chmod 600 "$BUILD_DIR/ssh/id_rsa"

# HPCKit tar.gz (download if not present)
HPCKIT_TAR="/home/shangyouren/HPCKit_25.1.0_Linux-aarch64.tar.gz"
if [ ! -f "$HPCKIT_TAR" ]; then
    echo "  -> Downloading HPCKit..."
    wget -q https://mirrors.huaweicloud.com/kunpeng/archive/HPC/HPCKit/HPCKit_25.1.0_Linux-aarch64.tar.gz -O "$HPCKIT_TAR"
fi
echo "  -> HPCKit"
cp -l "$HPCKIT_TAR" "$BUILD_DIR/"

# torch_npu local wheel
echo "  -> torch_npu wheel"
cp /home/shangyouren/torch_npu-2.6.0.post3-cp311-cp311-manylinux_2_28_aarch64.whl "$BUILD_DIR/"

# sgl_kernel_npu + deep_ep pre-built wheels
echo "  -> sgl_kernel_npu wheels"
cp /home/shangyouren/sglang_awq/sgl-kernel-npu/output/sgl_kernel_npu-0.1.0-cp311-cp311-linux_aarch64.whl "$BUILD_DIR/"
cp /home/shangyouren/sglang_awq/sgl-kernel-npu/output/deep_ep-1.0.0+aae2a1ad-cp311-cp311-linux_aarch64.whl "$BUILD_DIR/"

# ktransformers-dev (hardlinks, no data copy)
echo "  -> ktransformers-dev"
cp -al /home/shangyouren/ktransformers-dev "$BUILD_DIR/"

# Build
echo "==> Building Docker image: $IMAGE_NAME"
docker build -f "$BUILD_DIR/../Dockerfile" -t "$IMAGE_NAME" "$BUILD_DIR"

echo "==> Cleaning up context"
rm -rf "$BUILD_DIR"

echo "==> Done: $IMAGE_NAME"
echo "   Run: docker run --rm --device /dev/davinci0 --device /dev/davinci_manager $IMAGE_NAME"
