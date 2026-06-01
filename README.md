# lau-math-chapel — Chapel HPC Math Library

Chapel library for distributed mathematical computation: Laplacian eigendecomposition, heat kernels, agent fleet simulation, conservation law monitoring, and topological analysis.

## Chapel's Role in the Stack

```
┌─────────────────────────────────────────────┐
│  Go / Cloud Services (routing, orchestration)│
├─────────────────────────────────────────────┤
│  ★ Chapel / HPC Layer (this library)        │
│  Multi-locale distributed computation       │
│  Laplacian eigendecomposition across nodes  │
│  Parallel agent fleet simulation            │
│  Locality-aware graph processing (M+ verts) │
├─────────────────────────────────────────────┤
│  C / CUDA (single-machine, GPU kernels)     │
└─────────────────────────────────────────────┘
```

Chapel sits between single-machine performance (C/CUDA) and cloud orchestration (Go). It provides:
- **Multi-locale distributed computation** — scale across compute nodes transparently
- **Locality-aware data distribution** — Block/Cyclic distributions minimize data movement
- **Parallel-first design** — `forall` loops, atomic operations, reduction primitives
- **HPGAS model** — global address space with locality control

## Modules

| Module | Description |
|--------|-------------|
| `LauMatrix` | Distributed dense matrix (Block distribution), parallel multiply, QR, eigendecomposition |
| `LauLaplacian` | Sparse CSR graph Laplacian, spectral gap, Fiedler vector, Cheeger constant |
| `LauHeatKernel` | Heat kernel e^{-tL}, stochastic trace estimation, diffusion distances |
| `LauAgentFleet` | Distributed agent fleet simulation, CRDT merge, parallel interaction |
| `LauConservation` | Noether charge verification, conservation monitoring, drift detection |
| `LauTopology` | Simplicial complexes, Betti numbers (H⁰/H¹), Euler characteristic |

## Build

### Prerequisites

- [Chapel compiler](https://chapel-lang.org/) (`chpl` >= 1.31)

### Build

```bash
make
```

### Run Tests

```bash
make test
```

### Multi-Locale Execution

```bash
# Single node (development)
./test_main -nl 1

# Two locales
./test_main -nl 2

# N locales (production)
./test_main -nl <N>
```

For multi-node execution, configure your `CHPL_COMM` layer (GASNet, MPI, etc.):

```bash
# Example with GASNet over SSH
export CHPL_COMM=gasnet
export CHPL_COMM_SUBSTRATE=smp
./test_main -nl 4
```

## Project Structure

```
lau-math-chapel/
├── LauMatrix.chpl          # Distributed dense matrix
├── LauLaplacian.chpl       # Graph Laplacian & spectral methods
├── LauHeatKernel.chpl      # Heat kernel computation
├── LauAgentFleet.chpl      # Agent fleet simulation
├── LauConservation.chpl    # Conservation law verification
├── LauTopology.chpl        # Topological analysis
├── test_main.chpl          # Test suite (50+ tests)
├── Makefile                # Build system
└── README.md               # This file
```

## License

MIT
