# AMD ROCm 环境下源码编译安装 Mooncake TransferEngine

## 环境要求

| 项目 | 版本 |
|------|------|
| GPU | AMD Instinct MI308 / MI300X 等 |
| ROCm | 7.0+ (需包含 hipcc, hipify-perl) |
| OS | Ubuntu 22.04+ |
| Python | 3.12 |
| CMake | 3.16+ |

## 为什么需要源码编译

PyPI 上的 `mooncake-transfer-engine` 仅提供 CUDA 编译的 wheel，`engine.so` 链接了 `libcudart.so.12`，无法在 ROCm 环境直接使用。

Mooncake 的 CMake 构建系统原生支持 `USE_HIP=ON`，编译后 `engine.so` 直接链接 `libamdhip64.so`，源码中 CUDA API 调用通过 `hipify-perl` 自动转换为 HIP API。

## 编译步骤

### 1. 克隆源码

```bash
git clone --depth 1 https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
```

### 2. 安装系统依赖

```bash
apt-get install -y \
    libibverbs1 \
    libjsoncpp-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libyaml-cpp-dev \
    libnuma-dev
```

### 3. 安装 Mooncake 自身依赖

```bash
bash dependencies.sh
```

该脚本会安装：
- yalantinglibs (C++20 协程/网络库)
- git submodules (pybind11 等)
- Go 1.23+ (用于 etcd wrapper)

### 4. CMake 配置

```bash
mkdir -p build && cd build
cmake .. \
    -DUSE_HIP=ON \
    -DUSE_TCP=ON \
    -DUSE_HTTP=ON \
    -DUSE_CUDA=OFF \
    -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DWITH_STORE=OFF \
    -DWITH_P2P_STORE=OFF \
    -DCMAKE_BUILD_TYPE=Release
```

关键参数说明：

| 参数 | 值 | 说明 |
|------|------|------|
| `USE_HIP` | ON | 启用 AMD HIP 支持，链接 `hip::host` |
| `USE_CUDA` | OFF | 禁用 CUDA |
| `USE_TCP` | ON | 启用 TCP 传输（无 RDMA 时必需） |
| `USE_HTTP` | ON | 启用 HTTP 元数据服务（P2PHANDSHAKE 模式） |

`USE_HIP=ON` 会触发：
- `find_package(HIP REQUIRED)` 从 `/opt/rocm/lib/cmake` 查找 HIP SDK
- 定义 `USE_HIP` 和 `__HIP_PLATFORM_AMD__` 编译宏
- 使用 `hipify-perl` 将源码中的 `cuda*` 调用自动转换为 `hip*`

### 5. 编译

```bash
make -j$(nproc)
```

### 6. 安装 Python 包

```bash
cd ..
pip install -e mooncake-wheel/

# 将编译产物拷贝到 wheel 包目录
cp build/mooncake-integration/engine.cpython-312-x86_64-linux-gnu.so \
    mooncake-wheel/mooncake/engine.so
cp build/mooncake-asio/libasio.so \
    mooncake-wheel/mooncake/
```

## 验证

### 检查动态链接

```bash
$ ldd mooncake-wheel/mooncake/engine.so | grep -E "hip|cuda"
    libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7
    # 应该只有 hip，没有 cuda
```

### 测试导入和初始化

```bash
export LD_LIBRARY_PATH=/path/to/Mooncake/mooncake-wheel/mooncake:/opt/rocm/lib:$LD_LIBRARY_PATH

python -c "
from mooncake.engine import TransferEngine
e = TransferEngine()
ret = e.initialize('127.0.0.1', 'P2PHANDSHAKE', 'tcp', '')
print(f'init ret={ret}, rpc_port={e.get_rpc_port()}')
"
```

预期输出：

```
init ret=0, rpc_port=xxxxx
```

### 运行通信单测

```bash
LD_LIBRARY_PATH=pd_distributed:$LD_LIBRARY_PATH \
HIP_VISIBLE_DEVICES=0 \
python pd_distributed/test_mooncake_comm.py
```

覆盖测试项：TransferEngine 初始化、CPU/GPU 内存注册与传输、Bootstrap Server、ZMQ 侧信道。
