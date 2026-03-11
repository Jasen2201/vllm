"""
Mooncake TransferEngine communication unit tests for ROCm (single machine).

Tests cover:
  1. TransferEngine TCP initialization
  2. CPU memory registration + transfer between two engines
  3. GPU memory registration + transfer between two engines
  4. Bootstrap server (FastAPI HTTP) register/query
  5. ZMQ side-channel messaging (MooncakeXferMetadata/Response)

Usage:
  LD_LIBRARY_PATH=pd_distributed:$LD_LIBRARY_PATH \
  HIP_VISIBLE_DEVICES=0,1 \
  python pd_distributed/test_mooncake_comm.py
"""

import asyncio
import ctypes
import os
import sys
import threading
import time

import numpy as np

# Ensure mooncake can find the cudart HIP shim
SHIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if SHIM_DIR not in os.environ.get("LD_LIBRARY_PATH", ""):
    print(f"NOTE: Make sure LD_LIBRARY_PATH includes {SHIM_DIR}")
    print(f"  export LD_LIBRARY_PATH={SHIM_DIR}:$LD_LIBRARY_PATH")

from mooncake.engine import TransferEngine

HOSTNAME = os.environ.get("MOONCAKE_TEST_HOST", "127.0.0.1")
PROTOCOL = "tcp"

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)


# ============================================================
# Test 1: TransferEngine basic TCP init
# ============================================================
def test_engine_init():
    print("\n" + "=" * 60)
    print("Test 1: TransferEngine TCP initialization")
    print("=" * 60)

    engine = TransferEngine()
    ret = engine.initialize(HOSTNAME, "P2PHANDSHAKE", PROTOCOL, "")
    report("initialize returns 0", ret == 0, f"ret={ret}")

    if ret == 0:
        port = engine.get_rpc_port()
        report("get_rpc_port > 0", port > 0, f"port={port}")
    else:
        report("get_rpc_port > 0", False, "skipped due to init failure")

    return ret == 0


# ============================================================
# Test 2: CPU memory register + P2P transfer
# ============================================================
def test_cpu_memory_transfer():
    print("\n" + "=" * 60)
    print("Test 2: CPU memory register + P2P transfer")
    print("=" * 60)

    BUF_SIZE = 4096

    # Create two engines simulating P and D
    engine_p = TransferEngine()
    ret_p = engine_p.initialize(HOSTNAME, "P2PHANDSHAKE", PROTOCOL, "")
    if ret_p != 0:
        report("engine_p init", False, f"ret={ret_p}")
        return False

    engine_d = TransferEngine()
    ret_d = engine_d.initialize(HOSTNAME, "P2PHANDSHAKE", PROTOCOL, "")
    if ret_d != 0:
        report("engine_d init", False, f"ret={ret_d}")
        return False

    port_p = engine_p.get_rpc_port()
    port_d = engine_d.get_rpc_port()
    report("both engines initialized", True, f"P port={port_p}, D port={port_d}")

    # Allocate CPU buffers
    src_buf = ctypes.create_string_buffer(BUF_SIZE)
    dst_buf = ctypes.create_string_buffer(BUF_SIZE)

    # Fill source with pattern
    pattern = b"\xAB" * BUF_SIZE
    ctypes.memmove(src_buf, pattern, BUF_SIZE)
    # Clear destination
    ctypes.memset(dst_buf, 0, BUF_SIZE)

    src_ptr = ctypes.addressof(src_buf)
    dst_ptr = ctypes.addressof(dst_buf)

    # Register memory with both engines
    ret = engine_p.batch_register_memory([src_ptr], [BUF_SIZE])
    report("P register src memory", ret == 0, f"ret={ret}")

    ret = engine_d.batch_register_memory([dst_ptr], [BUF_SIZE])
    report("D register dst memory", ret == 0, f"ret={ret}")

    # Transfer: P writes into D's memory
    remote_session = f"{HOSTNAME}:{port_d}"
    ret = engine_p.batch_transfer_sync_write(
        remote_session, [src_ptr], [dst_ptr], [BUF_SIZE]
    )
    report("batch_transfer_sync_write", ret == 0, f"ret={ret}")

    # Verify data
    dst_data = ctypes.string_at(dst_ptr, BUF_SIZE)
    data_match = dst_data == pattern
    report("data integrity check", data_match,
           f"first 16 bytes: {dst_data[:16].hex()}")

    return data_match


# ============================================================
# Test 3: GPU memory register + P2P transfer
# ============================================================
def test_gpu_memory_transfer():
    print("\n" + "=" * 60)
    print("Test 3: GPU memory register + P2P transfer")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        report("torch import", False, "torch not available")
        return False

    if not torch.cuda.is_available():
        report("GPU available", False, "no GPU")
        return False

    NUM_ELEMENTS = 1024
    DTYPE = torch.float32

    # Use GPU 0 for both P and D (single card test)
    device = torch.device("cuda:0")

    engine_p = TransferEngine()
    ret_p = engine_p.initialize(HOSTNAME, "P2PHANDSHAKE", PROTOCOL, "")
    engine_d = TransferEngine()
    ret_d = engine_d.initialize(HOSTNAME, "P2PHANDSHAKE", PROTOCOL, "")

    if ret_p != 0 or ret_d != 0:
        report("engines init", False, f"P={ret_p}, D={ret_d}")
        return False

    port_d = engine_d.get_rpc_port()

    # Create GPU tensors
    src_tensor = torch.arange(NUM_ELEMENTS, dtype=DTYPE, device=device)
    dst_tensor = torch.zeros(NUM_ELEMENTS, dtype=DTYPE, device=device)

    src_ptr = src_tensor.data_ptr()
    dst_ptr = dst_tensor.data_ptr()
    nbytes = src_tensor.nbytes

    report("GPU tensors created", True,
           f"src sum={src_tensor.sum().item()}, dst sum={dst_tensor.sum().item()}")

    # Register GPU memory
    ret = engine_p.batch_register_memory([src_ptr], [nbytes])
    report("P register GPU memory", ret == 0, f"ret={ret}")

    ret = engine_d.batch_register_memory([dst_ptr], [nbytes])
    report("D register GPU memory", ret == 0, f"ret={ret}")

    # Transfer
    remote_session = f"{HOSTNAME}:{port_d}"
    t0 = time.perf_counter()
    ret = engine_p.batch_transfer_sync_write(
        remote_session, [src_ptr], [dst_ptr], [nbytes]
    )
    elapsed = time.perf_counter() - t0
    report("GPU batch_transfer_sync_write", ret == 0,
           f"ret={ret}, time={elapsed*1000:.2f}ms")

    # Verify
    torch.cuda.synchronize()
    match = torch.equal(src_tensor, dst_tensor)
    report("GPU data integrity", match,
           f"dst sum={dst_tensor.sum().item()}, expected={src_tensor.sum().item()}")

    return match


# ============================================================
# Test 4: Bootstrap server register/query
# ============================================================
def test_bootstrap_server():
    print("\n" + "=" * 60)
    print("Test 4: MooncakeBootstrapServer register/query")
    print("=" * 60)

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        # We need a minimal VllmConfig mock for the bootstrap server
        # The server only uses it for type signature, not actual config
        import httpx
        import uvicorn
        from fastapi import FastAPI
    except ImportError as e:
        report("imports", False, str(e))
        return False

    # Manually create a simple bootstrap server (avoid VllmConfig dependency)
    app = FastAPI()
    workers = {}

    @app.post("/register")
    async def register(payload: dict):
        dp_rank = payload["dp_rank"]
        if dp_rank not in workers:
            workers[dp_rank] = {
                "engine_id": payload["engine_id"],
                "worker_addr": {},
            }
        entry = workers[dp_rank]
        tp_rank = payload["tp_rank"]
        if tp_rank not in entry["worker_addr"]:
            entry["worker_addr"][tp_rank] = {}
        entry["worker_addr"][tp_rank][payload["pp_rank"]] = payload["addr"]
        return {"status": "ok"}

    @app.get("/query")
    async def query():
        return workers

    PORT = 18998
    config = uvicorn.Config(app=app, host="0.0.0.0", port=PORT, log_level="error")
    server = uvicorn.Server(config=config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Wait for server ready
    for _ in range(30):
        try:
            resp = httpx.get(f"http://127.0.0.1:{PORT}/query", timeout=1)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.2)

    report("bootstrap server started", True, f"port={PORT}")

    # Register a worker
    payload = {
        "engine_id": "test-engine-001",
        "dp_rank": 0,
        "tp_rank": 0,
        "pp_rank": 0,
        "addr": "tcp://10.67.77.208:15000",
    }
    resp = httpx.post(f"http://127.0.0.1:{PORT}/register", json=payload, timeout=5)
    report("register worker", resp.status_code == 200, f"status={resp.status_code}")

    # Query
    resp = httpx.get(f"http://127.0.0.1:{PORT}/query", timeout=5)
    data = resp.json()
    report("query returns data", "0" in data, f"data={data}")

    has_addr = False
    if "0" in data:
        entry = data["0"]
        has_addr = (
            entry.get("engine_id") == "test-engine-001"
            and "0" in entry.get("worker_addr", {})
        )
    report("query data correct", has_addr, f"engine_id match and tp_rank 0 present")

    server.should_exit = True
    return has_addr


# ============================================================
# Test 5: ZMQ side-channel messaging
# ============================================================
def test_zmq_side_channel():
    print("\n" + "=" * 60)
    print("Test 5: ZMQ side-channel messaging (xfer metadata)")
    print("=" * 60)

    try:
        import msgspec
        import zmq
        import zmq.asyncio
    except ImportError as e:
        report("imports", False, str(e))
        return False

    # Re-define the message types locally to avoid heavy vllm imports
    from enum import IntEnum

    class XferResponseStatus(IntEnum):
        FINISH = 0
        CONTINUE = 1
        ERROR = 2

    class XferMetadata(msgspec.Struct, omit_defaults=True):
        remote_hostname: str
        remote_port: int
        remote_tp_size: int
        remote_tp_rank: int
        req_blocks: dict[str, tuple[str, list[int]]]
        kv_caches_base_addr: list[int]

    class XferResponse(msgspec.Struct, omit_defaults=True):
        status: XferResponseStatus
        ok_reqs: list[str] | None = None
        err_reqs: list[str] | None = None
        err_msg: str | None = None

    encoder = msgspec.msgpack.Encoder()
    meta_decoder = msgspec.msgpack.Decoder(XferMetadata)
    resp_decoder = msgspec.msgpack.Decoder(XferResponse)

    results = {"p_received": False, "d_received": False}

    async def run_test():
        ctx = zmq.asyncio.Context()

        # P side: ROUTER socket
        p_sock = ctx.socket(zmq.ROUTER)
        p_port = p_sock.bind_to_random_port(f"tcp://{HOSTNAME}")

        # D side: DEALER socket
        d_sock = ctx.socket(zmq.DEALER)
        d_sock.connect(f"tcp://{HOSTNAME}:{p_port}")

        # D sends XferMetadata to P
        meta = XferMetadata(
            remote_hostname=HOSTNAME,
            remote_port=12345,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={"req-001": ("xfer-001", [0, 1, 2, 3])},
            kv_caches_base_addr=[0x7f0000000000, 0x7f0000100000],
        )
        await d_sock.send(encoder.encode(meta))

        # P receives
        identity, data = await p_sock.recv_multipart()
        decoded_meta = meta_decoder.decode(data)
        results["p_received"] = (
            decoded_meta.remote_hostname == HOSTNAME
            and "req-001" in decoded_meta.req_blocks
            and decoded_meta.kv_caches_base_addr == [0x7f0000000000, 0x7f0000100000]
        )

        # P sends XferResponse back to D
        resp = XferResponse(
            status=XferResponseStatus.FINISH,
            ok_reqs=["req-001"],
        )
        await p_sock.send_multipart([identity, encoder.encode(resp)])

        # D receives response
        resp_data = await d_sock.recv()
        decoded_resp = resp_decoder.decode(resp_data)
        results["d_received"] = (
            decoded_resp.status == XferResponseStatus.FINISH
            and decoded_resp.ok_reqs == ["req-001"]
        )

        p_sock.close()
        d_sock.close()
        ctx.term()

    asyncio.run(run_test())

    report("P received XferMetadata", results["p_received"])
    report("D received XferResponse", results["d_received"])

    return results["p_received"] and results["d_received"]


# ============================================================
# Main
# ============================================================
def main():
    global passed, failed

    print("=" * 60)
    print("Mooncake Communication Unit Tests")
    print(f"Host: {HOSTNAME}, Protocol: {PROTOCOL}")
    print("=" * 60)

    test_engine_init()
    test_cpu_memory_transfer()
    test_gpu_memory_transfer()
    test_bootstrap_server()
    test_zmq_side_channel()

    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
