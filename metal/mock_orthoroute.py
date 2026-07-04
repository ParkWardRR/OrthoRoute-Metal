import sys
import time
import numpy as np
import scipy.sparse as sp

# Simulating the original OrthoRoute environment
# In a real environment, we would do:
# try:
#     import cupy as cp
# except ImportError:
#     cp = None

class Pathfinder:
    """
    Simulates the core Pathfinder from KiCad's OrthoRoute plugin.
    This demonstrates the Strategy Pattern to dynamically swap GPU backends.
    """
    def __init__(self, use_metal=False):
        self.use_metal = use_metal
        self.backend = None
        
        # Simulated Graph (PCB Routing Grid)
        self.n_nodes = 100_000
        nnz_per_node = 4
        
        print(f"Initializing Pathfinder grid ({self.n_nodes} nodes)...")
        # Generate grid graph
        row_ptr = np.arange(0, (self.n_nodes + 1) * nnz_per_node, nnz_per_node, dtype=np.int32)
        col_indices = np.zeros(self.n_nodes * nnz_per_node, dtype=np.int32)
        for i in range(self.n_nodes):
            for j in range(nnz_per_node):
                t = i + np.random.randint(-10, 10)
                col_indices[i * nnz_per_node + j] = max(0, min(self.n_nodes - 1, t))
                
        values = np.ones(self.n_nodes * nnz_per_node, dtype=np.float32)
        self.A = sp.csr_matrix((values, col_indices, row_ptr), shape=(self.n_nodes, self.n_nodes))
        
        # Deduplicate and extract
        self.row_ptr = self.A.indptr.astype(np.int32)
        self.col_indices = self.A.indices.astype(np.int32)
        self.values = self.A.data.astype(np.float32)
        
        # Distance array initialization
        self.distances = np.full(self.n_nodes, np.inf, dtype=np.float32)
        
        self._initialize_backend()

    def _initialize_backend(self):
        if self.use_metal:
            if sys.platform != "darwin":
                raise EnvironmentError("Metal backend is only supported on macOS")
                
            try:
                import orthoroute_mac
                print("[Backend] Loaded Apple Metal Backend (orthoroute_mac)")
                self.backend = orthoroute_mac.MetalDijkstra()
                
                # Zero-copy share NumPy arrays to Metal
                msg = self.backend.set_graph_csr(self.row_ptr, self.col_indices, self.values)
                print(f"[Backend] {msg}")
                
            except ImportError:
                raise ImportError("orthoroute_mac module not found. Did you run `maturin develop`?")
        else:
            print("[Backend] Loaded CUDA Backend (CuPy)")
            # In the real code, this would initialize CUDADijkstra
            pass
            
    def route_net(self, source, target):
        print(f"\n--- Routing Net (Source: {source}, Target: {target}) ---")
        
        # 1. Reset distances
        self.distances.fill(np.inf)
        self.distances[source] = 0.0
        
        start_t = time.time()
        
        if self.use_metal:
            # 2. Transfer initial distances (Zero-copy on UMA)
            self.backend.set_distances_csr(self.distances)
            self.backend.reset_predecessors()
            
            # 3. Setup SPFA (Shortest Path Faster Algorithm) Queues
            self.backend.setup_spfa()
            
            # 4. Dispatch the massive wavefront expansion!
            iters, converged = self.backend.execute_until_convergence(
                500, 1024, 512, 0.0
            )
            
            # 5. Retrieve final distances and predecessors
            final_dists = self.backend.get_distances()
            final_preds = self.backend.get_predecessors()
            
            exec_time = time.time() - start_t
            
            print(f"[Metal] Routing completed in {exec_time*1000:.2f} ms")
            print(f"[Metal] Converged: {converged}, Iters used: {iters}")
            print(f"[Metal] Distance to target: {final_dists[target]}")
            
            if final_preds[target] != -1:
                print(f"[Metal] Target is reachable!")
            else:
                print(f"[Metal] Target is unreachable.")
                
        else:
            # CUDA path fallback
            # dists = self.cuda_dijkstra.run(...)
            pass

if __name__ == "__main__":
    print("Testing OrthoRoute Core Integration...\n")
    
    try:
        router = Pathfinder(use_metal=True)
        router.route_net(source=0, target=50000)
    except Exception as e:
        print(f"Error during integration test: {e}")
