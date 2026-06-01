// LauMatrix.chpl — Distributed dense matrix with locale-aware Block distribution
module LauMatrix {
  use BlockDist, Math, Random;

  /* Distributed dense matrix stored in a block-distributed 2D array. */
  record LauMatrix {
    var nRows: int;
    var nCols: int;
    var data: [0..<nRows, 0..<nCols] real;

    proc init(nRows: int, nCols: int) {
      this.nRows = nRows;
      this.nCols = nCols;
      const dom = {0..<nRows, 0..<nCols};
      const blockedDom = dom dmapped Block(dom);
      this.data = blockedDom;
    }

    proc init(n: int) {
      this.init(n, n);
    }

    /* Identity matrix */
    proc ref setIdentity() {
      forall (i, j) in data.domain do
        data[i, j] = if i == j then 1.0 else 0.0;
    }

    /* Zero matrix */
    proc ref setZero() {
      forall (i, j) in data.domain do
        data[i, j] = 0.0;
    }

    /* Fill with random values in [lo, hi] */
    proc ref setRandom(lo: real = -1.0, hi: real = 1.0, seed: int = 42) {
      var rng = new randomStream(real, seed);
      forall (i, j) in data.domain with (var localRng = new randomStream(real, seed + i * nCols + j)) do
        data[i, j] = lo + localRng.getNext() * (hi - lo);
    }

    /* Symmetric: A = (A + A^T) / 2 */
    proc ref symmetrize() {
      forall i in 0..<nRows {
        for j in (i+1)..<nCols {
          const avg = (data[i, j] + data[j, i]) / 2.0;
          data[i, j] = avg;
          data[j, i] = avg;
        }
      }
    }

    /* Frobenius norm */
    proc frobeniusNorm(): real {
      var sumSq: atomic real;
      sumSq.write(0.0);
      forall (i, j) in data.domain with (ref sumSq) do
        sumSq.add(data[i, j] ** 2);
      return sqrt(sumSq.read());
    }

    /* Trace */
    proc trace(): real {
      var s = 0.0;
      for i in 0..<min(nRows, nCols) do s += data[i, i];
      return s;
    }

    /* Pretty-print a small slice */
    proc show(maxRows: int = 8, maxCols: int = 8) {
      const mr = min(nRows, maxRows);
      const mc = min(nCols, maxCols);
      for i in 0..<mr {
        for j in 0..<mc {
          writef("%.4f  ", data[i, j]);
        }
        if mc < nCols then write("...");
        writeln();
      }
      if mr < nRows then writeln("...");
    }
  }

  /* Distributed matrix multiply: C = A * B (parallel, locale-aware) */
  proc matMul(A: LauMatrix, B: LauMatrix): LauMatrix throws {
    if A.nCols != B.nRows then
      throw new Error("matMul dimension mismatch: " + A.nCols:string + " != " + B.nRows:string);
    var C = new LauMatrix(A.nRows, B.nCols);
    C.setZero();
    forall i in 0..<A.nRows {
      for k in 0..<A.nCols {
        const a_ik = A.data[i, k];
        forall j in 0..<B.nCols with (ref C) do
          C.data[i, j] += a_ik * B.data[k, j];
      }
    }
    return C;
  }

  /* Matrix-vector multiply: y = A * x */
  proc matVecMul(A: LauMatrix, x: [] real): [] real throws {
    if x.size != A.nCols then
      throw new Error("matVecMul dimension mismatch");
    var y: [0..<A.nRows] real;
    forall i in 0..<A.nRows {
      var s = 0.0;
      for j in 0..<A.nCols do s += A.data[i, j] * x[j];
      y[i] = s;
    }
    return y;
  }

  /* Transpose */
  proc transpose(A: LauMatrix): LauMatrix {
    var B = new LauMatrix(A.nCols, A.nRows);
    forall (i, j) in A.data.domain do
      B.data[j, i] = A.data[i, j];
    return B;
  }

  /* Add two matrices */
  proc matAdd(A: LauMatrix, B: LauMatrix): LauMatrix throws {
    if A.nRows != B.nRows || A.nCols != B.nCols then
      throw new Error("matAdd dimension mismatch");
    var C = new LauMatrix(A.nRows, A.nCols);
    forall (i, j) in C.data.domain do
      C.data[i, j] = A.data[i, j] + B.data[i, j];
    return C;
  }

  /* Scale: A * scalar */
  proc matScale(A: LauMatrix, s: real): LauMatrix {
    var B = new LauMatrix(A.nRows, A.nCols);
    forall (i, j) in B.data.domain do
      B.data[i, j] = A.data[i, j] * s;
    return B;
  }

  /* Power iteration for dominant eigenvalue/eigenvector */
  proc powerIteration(A: LauMatrix, maxIter: int = 200, tol: real = 1e-10): (real, [] real) {
    var v: [0..<A.nRows] real;
    // Initialize with ones, normalize
    var nrm = 0.0;
    for i in 0..<A.nRows { v[i] = 1.0; nrm += 1.0; }
    nrm = sqrt(nrm);
    for i in 0..<A.nRows do v[i] /= nrm;

    var lambda = 0.0;
    for iter in 0..<maxIter {
      // w = A * v
      var w: [0..<A.nRows] real;
      forall i in 0..<A.nRows {
        var s = 0.0;
        for j in 0..<A.nCols do s += A.data[i, j] * v[j];
        w[i] = s;
      }
      // eigenvalue estimate
      lambda = 0.0;
      for i in 0..<A.nRows do lambda += v[i] * w[i];
      // normalize
      var wNrm = 0.0;
      for i in 0..<A.nRows do wNrm += w[i] ** 2;
      wNrm = sqrt(wNrm);
      if wNrm < 1e-15 then break;
      for i in 0..<A.nRows do w[i] /= wNrm;
      // convergence check
      var diff = 0.0;
      for i in 0..<A.nRows do diff += (w[i] - v[i]) ** 2;
      diff = sqrt(diff);
      v = w;
      if diff < tol then break;
    }
    return (lambda, v);
  }

  /* Inverse power iteration for smallest eigenvalue */
  proc inversePowerIteration(A: LauMatrix, maxIter: int = 200, tol: real = 1e-10): (real, [] real) {
    // Shift to avoid singularity: A + sigma*I, solve (A+sigmaI)^{-1} v
    const n = A.nRows;
    const sigma = 1e-8;
    // Build shifted matrix as dense, do simple Gauss-Seidel-like solve
    // For simplicity, use direct iteration on A^{-1} approximated
    // We use: shift A -> A + sigma*I, then power iterate on inverse
    var B = new LauMatrix(n);
    forall (i, j) in B.data.domain do
      B.data[i, j] = A.data[i, j];
    forall i in 0..<n do B.data[i, i] += sigma;

    var v: [0..<n] real;
    for i in 0..<n do v[i] = 1.0 / sqrt(n: real);

    var lambda = 0.0;
    for iter in 0..<maxIter {
      // Solve B * w = v using simple Jacobi iteration
      var w: [0..<n] real;
      for i in 0..<n do w[i] = v[i];
      for _ in 0..<50 {
        var wNew: [0..<n] real;
        for i in 0..<n {
          var s = v[i];
          for j in 0..<n {
            if j != i then s -= B.data[i, j] * w[j];
          }
          wNew[i] = s / B.data[i, i];
        }
        w = wNew;
      }
      var wNrm = 0.0;
      for i in 0..<n do wNrm += w[i] ** 2;
      wNrm = sqrt(wNrm);
      if wNrm < 1e-15 then break;
      for i in 0..<n do w[i] /= wNrm;
      // eigenvalue estimate for A
      var Aw: [0..<n] real;
      for i in 0..<n {
        var s = 0.0;
        for j in 0..<n do s += A.data[i, j] * w[j];
        Aw[i] = s;
      }
      lambda = 0.0;
      for i in 0..<n do lambda += w[i] * Aw[i];
      var diff = 0.0;
      for i in 0..<n do diff += (w[i] - v[i]) ** 2;
      diff = sqrt(diff);
      v = w;
      if diff < tol then break;
    }
    return (lambda, v);
  }

  /* QR decomposition (Gram-Schmidt) for small matrices */
  proc qrDecompose(A: LauMatrix): (LauMatrix, LauMatrix) {
    const m = A.nRows, n = A.nCols;
    var Q = new LauMatrix(m, n);
    var R = new LauMatrix(n, n);
    Q.setZero();
    R.setZero();

    // Column storage
    var cols: [0..<n] [0..<m] real;
    for j in 0..<n {
      for i in 0..<m do cols[j][i] = A.data[i, j];
    }

    for j in 0..<n {
      var q = cols[j];
      for k in 0..<j {
        var dot = 0.0;
        for i in 0..<m do dot += q[i] * cols[k][i];
        R.data[k, j] = dot;
        for i in 0..<m do q[i] -= dot * cols[k][i];
      }
      var nrm = 0.0;
      for i in 0..<m do nrm += q[i] ** 2;
      nrm = sqrt(nrm);
      R.data[j, j] = nrm;
      if nrm > 1e-15 {
        for i in 0..<m do q[i] /= nrm;
      }
      cols[j] = q;
    }

    for j in 0..<n {
      for i in 0..<m do Q.data[i, j] = cols[j][i];
    }
    return (Q, R);
  }

  /* Full eigendecomposition via QR algorithm (for symmetric matrices) */
  proc eigenDecompose(A: LauMatrix, maxIter: int = 500, tol: real = 1e-10): ([] real, LauMatrix) {
    const n = A.nRows;
    var T = new LauMatrix(n);
    forall (i, j) in T.data.domain do T.data[i, j] = A.data[i, j];

    var V = new LauMatrix(n);
    V.setIdentity();

    for iter in 0..<maxIter {
      var (Q, R) = qrDecompose(T);
      // T = R * Q, V = V * Q
      var Tnew = new LauMatrix(n);
      Tnew.setZero();
      for i in 0..<n {
        for j in 0..<n {
          var s = 0.0;
          for k in 0..<n do s += R.data[i, k] * Q.data[k, j];
          Tnew.data[i, j] = s;
        }
      }
      var Vnew = new LauMatrix(n);
      Vnew.setZero();
      for i in 0..<n {
        for j in 0..<n {
          var s = 0.0;
          for k in 0..<n do s += V.data[i, k] * Q.data[k, j];
          Vnew.data[i, j] = s;
        }
      }
      // Check convergence: off-diagonal norm
      var offDiag = 0.0;
      for i in 0..<n {
        for j in 0..<n {
          if i != j then offDiag += Tnew.data[i, j] ** 2;
        }
      }
      T = Tnew;
      V = Vnew;
      if sqrt(offDiag) < tol then break;
    }

    var eigenvalues: [0..<n] real;
    for i in 0..<n do eigenvalues[i] = T.data[i, i];
    return (eigenvalues, V);
  }
}
