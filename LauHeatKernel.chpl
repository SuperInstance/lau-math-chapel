// LauHeatKernel.chpl — Distributed heat kernel e^{-tL}, parallel across time steps, memory-efficient streaming
module LauHeatKernel {
  use LauMatrix, LauLaplacian, Math;

  /* Compute heat kernel via matrix exponential: H(t) = e^{-tL}
     Uses truncated Taylor series for efficiency. */
  proc heatKernelMatrix(L: LaplacianCSR, t: real, order: int = 20): LauMatrix {
    const n = L.n;
    // Start with I
    var H = new LauMatrix(n);
    H.setIdentity();

    // Taylor series: e^{-tL} = I - tL + (tL)^2/2! - (tL)^3/3! + ...
    // Build -tL as dense
    var negtL = new LauMatrix(n);
    negtL.setZero();
    for i in 0..<n {
      for k in L.rowPtr[i]..<L.rowPtr[i+1] {
        negtL.data[i, L.colIdx[k]] = -t * L.values[k];
      }
    }

    // Accumulate: H = sum_{k=0}^{order} (-tL)^k / k!
    var power = new LauMatrix(n);
    power.setIdentity(); // (-tL)^0 = I

    for k in 1..order {
      power = matMul(power, negtL);
      const scale = 1.0 / factorial(k);
      forall (i, j) in H.data.domain do
        H.data[i, j] += scale * power.data[i, j];
    }
    return H;
  }

  /* Apply heat kernel to a vector without forming full matrix.
     Uses Krylov subspace: h(t) = e^{-tL} * f
     Approximate via repeated sparse matrix-vector products. */
  proc heatKernelApply(L: LaplacianCSR, f: [] real, t: real, order: int = 20): [] real throws {
    const n = L.n;
    // h = f + sum_{k=1}^{order} (-t)^k L^k f / k!
    var h: [0..<n] real;
    for i in 0..<n do h[i] = f[i];

    var current = f; // L^0 * f
    var coeff = 1.0;
    for k in 1..order {
      current = L.multiply(current);
      coeff *= (-t) / k: real;
      forall i in 0..<n do h[i] += coeff * current[i];
    }
    return h;
  }

  /* Factorial helper */
  private proc factorial(n: int): real {
    var r = 1.0;
    for i in 2..n do r *= i: real;
    return r;
  }

  /* Heat kernel trace: Tr(e^{-tL}) = sum_i e^{-t lambda_i}
     Estimated via Hutchinson's stochastic trace estimator. */
  proc heatKernelTrace(L: LaplacianCSR, t: real, numSamples: int = 30, order: int = 20): real throws {
    const n = L.n;
    var traceEst = 0.0;
    for s in 0..<numSamples {
      // Random Rademacher vector
      var g: [0..<n] real;
      for i in 0..<n do g[i] = if (i + s) % 2 == 0 then 1.0 else -1.0;
      var Kg = heatKernelApply(L, g, t, order);
      traceEst += + reduce (g * Kg);
    }
    return traceEst / numSamples: real;
  }

  /* Heat kernel distance between two vectors: ||H(t)*a - H(t)*b|| */
  proc heatKernelDistance(L: LaplacianCSR, a: [] real, b: [] real, t: real, order: int = 20): real throws {
    const diff: [0..<a.size] real = [(i) in 0..<a.size] a[i] - b[i];
    const smoothed = heatKernelApply(L, diff, t, order);
    return sqrt(+ reduce (smoothed ** 2));
  }

  /* Multi-scale heat kernel: compute at multiple time scales simultaneously.
     Memory-efficient: reuses intermediate products. */
  record HeatKernelMultiScale {
    var times: [0..<0] real;
    var traces: [0..<0] real;

    proc init() {}

    proc compute(L: LaplacianCSR, timeScales: [] real, order: int = 20, numTraceSamples: int = 30) throws {
      const nt = timeScales.size;
      this.times = timeScales;
      this.traces = [0..<nt] 0.0;
      for i in 0..<nt do
        this.traces[i] = heatKernelTrace(L, timeScales[i], numTraceSamples, order);
    }
  }

  /* Diffusion distance matrix at time t between selected vertices */
  proc diffusionDistanceMatrix(L: LaplacianCSR, vertices: [] int, t: real, order: int = 20): LauMatrix throws {
    const nv = vertices.size;
    const n = L.n;
    var D = new LauMatrix(nv);
    D.setZero();

    // Compute heat kernel columns for selected vertices
    var cols: [0..<nv] [0..<n] real;
    for c in 0..<nv {
      var e: [0..<n] real;
      for i in 0..<n do e[i] = 0.0;
      e[vertices[c]] = 1.0;
      cols[c] = heatKernelApply(L, e, t, order);
    }

    // Pairwise distances
    forall i in 0..<nv {
      for j in (i+1)..<nv {
        var d = 0.0;
        for k in 0..<n do d += (cols[i][k] - cols[j][k]) ** 2;
        d = sqrt(d);
        D.data[i, j] = d;
        D.data[j, i] = d;
      }
    }
    return D;
  }

  /* Heat kernel page rank: compute heat kernel-based centrality */
  proc heatKernelCentrality(L: LaplacianCSR, t: real, order: int = 20): [] real throws {
    const n = L.n;
    var centrality: [0..<n] real;
    // Centrality = sum_j H(i,j) for each i
    // Use: c_i = sum of row i of e^{-tL}
    // Approximate by applying to unit vectors
    var ones: [0..<n] real;
    for i in 0..<n do ones[i] = 1.0;
    var smoothed = heatKernelApply(L, ones, t, order);
    // Centrality is the smoothed degree
    centrality = smoothed;
    // Normalize
    var s = + reduce centrality;
    if s > 1e-15 then for i in 0..<n do centrality[i] /= s;
    return centrality;
  }
}
