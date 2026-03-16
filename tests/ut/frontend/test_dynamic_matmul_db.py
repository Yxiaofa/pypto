"""Dynamic matmul tests with double buffer (ping-pong) optimization.

Double buffer allows overlapping compute and memory transfer:
- While computing on ping buffers, load data into pong buffers
- While computing on pong buffers, load data into ping buffers
This hides memory latency behind computation.
"""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


M = pl.DynVar('M')
K = pl.DynVar('K')
N = pl.DynVar('N')

@fe.kernel
def dynamic_matmul_db_kernel(
    a: pl.Tensor[[M, K], pl.FP16],
    b: pl.Tensor[[K, N], pl.FP16],
    c: pl.Tensor[[M, N], pl.FP32]
) -> pl.Tensor[[M, N], pl.FP32]:
    # ========== Ping buffer set ==========
    # Mat space: load buffers for ping
    tile_type_a_load_ping = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Mat,
        blayout=2,
        slayout=1,
    )
    tile_a_load_ping = plm.make_tile(tile_type_a_load_ping, addr=0x00000, size=32768)
    
    tile_type_b_load_ping = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Mat,
        blayout=2,
        slayout=1,
    )
    tile_b_load_ping = plm.make_tile(tile_type_b_load_ping, addr=0x08000, size=32768)
    
    # Left space: compute buffer for ping
    tile_type_a_ping = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Left,
        blayout=1,
        slayout=1,
    )
    tile_a_ping = plm.make_tile(tile_type_a_ping, addr=0x00000, size=32768)
    
    # Right space: compute buffer for ping
    tile_type_b_ping = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Right,
        blayout=1,
        slayout=2,
    )
    tile_b_ping = plm.make_tile(tile_type_b_ping, addr=0x00000, size=32768)
    
    # ========== Pong buffer set ==========
    # Mat space: load buffers for pong
    tile_type_a_load_pong = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Mat,
        blayout=2,
        slayout=1,
    )
    tile_a_load_pong = plm.make_tile(tile_type_a_load_pong, addr=0x10000, size=32768)
    
    tile_type_b_load_pong = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Mat,
        blayout=2,
        slayout=1,
    )
    tile_b_load_pong = plm.make_tile(tile_type_b_load_pong, addr=0x18000, size=32768)
    
    # Left space: compute buffer for pong
    tile_type_a_pong = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Left,
        blayout=1,
        slayout=1,
    )
    tile_a_pong = plm.make_tile(tile_type_a_pong, addr=0x08000, size=32768)
    
    # Right space: compute buffer for pong
    tile_type_b_pong = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Right,
        blayout=1,
        slayout=2,
    )
    tile_b_pong = plm.make_tile(tile_type_b_pong, addr=0x08000, size=32768)
    
    # ========== Accumulator (shared) ==========
    tile_type_c = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Acc,
        blayout=2,
        slayout=1,
        fractal=1024,
        valid_shape=[-1, -1],
    )
    tile_c = plm.make_tile(tile_type_c, addr=0x00000, size=65536)
    
    with pl.section_cube():
        M_dim = pl.tensor.dim(a, 0)
        K_dim = pl.tensor.dim(a, 1)
        N_dim = pl.tensor.dim(b, 1)
        
        for i in pl.range(0, M_dim, 128):
            for j in pl.range(0, N_dim, 128):
                for k in pl.range(0, K_dim, 128):
                    # Double buffering: alternate between ping and pong
                    # k_idx = k // 128, ping for even, pong for odd
                    k_idx = k // 128
                    
                    if k_idx % 2 == 0:
                        # === Ping iteration ===
                        # Event IDs: 0-2 for ping
                        plm.load(tile_a_load_ping, a, [i, k])
                        plm.load(tile_b_load_ping, b, [k, j])
                        
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                        
                        plm.move(tile_a_ping, tile_a_load_ping, target_memory=pl.MemorySpace.Left)
                        plm.move(tile_b_ping, tile_b_load_ping, target_memory=pl.MemorySpace.Right)
                        
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
                        
                        if k == 0:
                            plm.matmul(tile_c, tile_a_ping, tile_b_ping)
                        else:
                            plm.matmul_acc(tile_c, tile_c, tile_a_ping, tile_b_ping)
                        
                        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    else:
                        # === Pong iteration ===
                        # Event IDs: 3-5 for pong
                        plm.load(tile_a_load_pong, a, [i, k])
                        plm.load(tile_b_load_pong, b, [k, j])
                        
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=3)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=3)
                        
                        plm.move(tile_a_pong, tile_a_load_pong, target_memory=pl.MemorySpace.Left)
                        plm.move(tile_b_pong, tile_b_load_pong, target_memory=pl.MemorySpace.Right)
                        
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=4)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=4)
                        
                        plm.matmul_acc(tile_c, tile_c, tile_a_pong, tile_b_pong)
                        
                        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=5)
                        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=5)
                
                # Store result
                pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                plm.l0c_store(tile_c, [i, j], [128, 128], c)
                
                # Barrier to ensure l0c_store is complete before next iteration
                pl.system.bar_all()

    return c


@fe.jit()
def test_dynamic_matmul_db():
    compiled_lib = fe.compile(dynamic_matmul_db_kernel, arch="dav-c220-cube")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:1"
    torch.npu.set_device(device)

    shapes = [
        [128, 512, 128],
        [128, 512, 256],
        [256, 512, 256],
        [256, 512, 512],
        [512, 512, 512],
        [2048, 128, 2048],
    ]
    torch.manual_seed(0)

    for M_val, K_val, N_val in shapes:
        print(f"\nTesting shape: ({M_val}, {K_val}) x ({K_val}, {N_val}) = ({M_val}, {N_val})")
        
        a = torch.randn(M_val, K_val, dtype=torch.float16, device=device)
        b = torch.randn(K_val, N_val, dtype=torch.float16, device=device)
        c = torch.zeros(M_val, N_val, dtype=torch.float32, device=device)
        
        fe.launch(None, 1, compiled_lib, a, b, c)
        torch.npu.synchronize()
        
        print("***********npu output***********")
        print(c.shape, c.dtype)
        print(c)
        c_ref = torch.matmul(a.float(), b.float())
        print("***********golden output***********")
        print(c_ref.shape, c_ref.dtype)
        print(c_ref)
        
        torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)
        print("result equal!")


if __name__ == "__main__":
    test_dynamic_matmul_db()
    print("\nAll tests passed!")
