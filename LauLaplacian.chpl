// LauLaplacian.chpl — Distributed graph Laplacian (sparse CSR), parallel spectral gap, distributed power iteration
module LauLaplacian {
  use LauMatrix, Math;

  /* Sparse CSR matrix representation for graph Laplacian */
  record LaplacianCSR {
    var n: int;           // number of vertices
    var nnz: int;         // number of non-zeros
    var rowPtr: [0..<n+1] int;
    var colIdx: [0..<nnz] int;
    var values: [0..<nnz] real;

    proc init(n: int) {
      this.n = n;
      this.nnz = 0;
      this.rowPtr = [0..<n+1] 0;
      this.colIdx = [0..<0] 0;
      this.values = [0..<0] 0.0;
    }

    proc init(n: int, nnz: int, rowPtr: [] int, colIdx: [] int, values: [] real) {
      this.n = n;
      this.nnz = nnz;
      this.rowPtr = rowPtr;
      this.colIdx = colIdx;
      this.values = values;
    }

    /* Matrix-vector multiply: y = L * x */
    proc multiply(x: [] real): [] real throws {
      if x.size != n then throw new Error("LaplacianCSR multiply dimension mismatch");
      var y: [0..<n] real;
      forall i in 0..<n {
        var s = 0.0;
        for k in rowPtr[i]..<rowPtr[i+1] do
          s += values[k] * x[colIdx[k]];
        y[i] = s;
      }
      return y;
    }

    /* Frobenius norm (sparse) */
    proc frobeniusNorm(): real {
      var s = 0.0;
      for k in 0..<nnz do s += values[k] ** 2;
      return sqrt(s);
    }
  }

  /* Build Laplacian from edge list. Edges are (u, v, weight) triples. */
  proc buildLaplacian(n: int, edges: [] (int, int, real)): LaplacianCSR {
    // Count degrees
    var degree: [0..<n] real;
    for (u, v, w) in edges {
      degree[u] += w;
      degree[v] += w;
    }

    // Count non-zeros per row
    var nnzPerRow: [0..<n] int;
    for i in 0..<n do nnzPerRow[i] = 1; // diagonal
    for (u, v, w) in edges {
      nnzPerRow[u] += 1;
      nnzPerRow[v] += 1;
    }

    var totalNnz: int = + reduce nnzPerRow;
    var rowPtr: [0..<n+1] int;
    rowPtr[0] = 0;
    for i in 0..<n do rowPtr[i+1] = rowPtr[i] + nnzPerRow[i];

    var colIdx: [0..<totalNnz] int;
    var values: [0..<totalNnz] real;
    var cursor: [0..<n] int;
    for i in 0..<n do cursor[i] = rowPtr[i];

    // Fill diagonal (degree)
    for i in 0..<n {
      colIdx[cursor[i]] = i;
      values[cursor[i]] = degree[i];
      cursor[i] += 1;
    }

    // Fill off-diagonal (negative weights)
    for (u, v, w) in edges {
      colIdx[cursor[u]] = v;
      values[cursor[u]] = -w;
      cursor[u] += 1;
      colIdx[cursor[v]] = u;
      values[cursor[v]] = -w;
      cursor[v] += 1;
    }

    return new LaplacianCSR(n, totalNnz, rowPtr, colIdx, values);
  }

  /* Build normalized Laplacian: L_norm = I - D^{-1/2} A D^{-1/2} */
  proc buildNormalizedLaplacian(n: int, edges: [] (int, int, real)): LaplacianCSR {
    var degree: [0..<n] real;
    for (u, v, w) in edges {
      degree[u] += w;
      degree[v] += w;
    }
    var invSqrtDeg: [0..<n] real;
    for i in 0..<n do
      invSqrtDeg[i] = if degree[i] > 0.0 then 1.0 / sqrt(degree[i]) else 0.0;

    // Scale edges
    var scaledEdges: [0..<edges.size] (int, int, real);
    for i in 0..<edges.size {
      const (u, v, w) = edges[i];
      scaledEdges[i] = (u, v, w * invSqrtDeg[u] * invSqrtDeg[v]);
    }
    return buildLaplacian(n, scaledEdges);
  }

  /* Distributed power iteration for Fiedler vector (2nd smallest eigenvector) */
  proc fiedlerVector(L: LaplacianCSR, maxIter: int = 500, tol: real = 1e-10): (real, [] real) {
    const n = L.n;
    var v: [0..<n] real;
    for i in 0..<n do v[i] = 1.0 / sqrt(n: real);
    // Project out constant vector (eigenvalue 0)
    var mean = (+ reduce v) / n;
    for i in 0..<n do v[i] -= mean;
    var nrm = sqrt(+ reduce (v ** 2));
    for i in 0..<n do v[i] /= nrm;

    var lambda = 0.0;
    for iter in 0..<maxIter {
      // w = L * v
      var w = L.multiply(v);
      // Remove constant component
      var wMean = (+ reduce w) / n;
      for i in 0..<n do w[i] -= wMean;
      // Rayleigh quotient
      lambda = + reduce (v * w);
      // Normalize
      var wNrm = sqrt(+ reduce (w ** 2));
      if wNrm < 1e-15 then break;
      for i in 0..<n do w[i] /= wNrm;
      // Check convergence
      var diff = sqrt(+ reduce ((w - v) ** 2));
      v = w;
      if diff < tol then break;
    }
    return (lambda, v);
  }

  /* Spectral gap: difference between two smallest eigenvalues */
  proc spectralGap(L: LaplacianCSR, maxIter: int = 500, tol: real = 1e-10): real {
    // The Fiedler value IS the spectral gap for connected graphs (smallest eigenvalue is 0)
    const (lambda2, _) = fiedlerVector(L, maxIter, tol);
    return lambda2;
  }

  /* Number of connected components via BFS on Laplacian structure */
  proc connectedComponents(L: LaplacianCSR): int {
    const n = L.n;
    var visited: [0..<n] bool;
    var components = 0;

    for start in 0..<n {
      if visited[start] then continue;
      components += 1;
      // BFS
      var queue: [0..<n] int;
      var head = 0, tail = 0;
      queue[tail] = start; tail += 1;
      visited[start] = true;
      while head < tail {
        const u = queue[head]; head += 1;
        for k in L.rowPtr[u]..<L.rowPtr[u+1] {
          const v = L.colIdx[k];
          if !visited[v] && L.values[k] < 0.0 { // off-diagonal = edge
            visited[v] = true;
            queue[tail] = v; tail += 1;
          }
        }
      }
    }
    return components;
  }

  /* Algebraic connectivity: same as Fiedler value for connected graph */
  proc algebraicConnectivity(L: LaplacianCSR): real {
    return spectralGap(L);
  }

  /* Total volume of graph (sum of degrees) */
  proc graphVolume(L: LaplacianCSR): real {
    var vol = 0.0;
    for i in 0..<L.n {
      // Diagonal entry is the degree
      for k in L.rowPtr[i]..<L.rowPtr[i+1] {
        if L.colIdx[k] == i then vol += L.values[k];
      }
    }
    return vol;
  }

  /* Edge expansion (Cheeger constant estimate via Fiedler vector) */
  proc cheegerConstant(L: LaplacianCSR): real {
    const n = L.n;
    const (_, fied) = fiedlerVector(L);
    const vol = graphVolume(L);
    if vol < 1e-15 then return 0.0;

    // Sort vertices by Fiedler value
    var idx: [0..<n] int;
    for i in 0..<n do idx[i] = i;
    // Simple insertion sort (fine for testing)
    for i in 1..<n {
      var key = idx[i];
      var j = i - 1;
      while j >= 0 && fied[idx[j]] > fied[key] {
        idx[j+1] = idx[j];
        j -= 1;
      }
      idx[j+1] = key;
    }

    var bestCut = 1.0 / 0.0; // infinity
    var cutEdges = 0.0;
    var volS = 0.0;
    var inS: [0..<n] bool;

    for k in 0..<(n-1) {
      const v = idx[k];
      inS[v] = true;
      // Update volume of S
      for kk in L.rowPtr[v]..<L.rowPtr[v+1] {
        if L.colIdx[kk] == v then volS += L.values[kk];
      }
      // Update cut edges
      for kk in L.rowPtr[v]..<L.rowPtr[v+1] {
        if L.colIdx[kk] != v && inS[L.colIdx[kk]] then
          cutEdges += L.values[kk]; // subtracting negative = adding weight
        else if L.colIdx[kk] != v && !inS[L.colIdx[kk]] then
          cutEdges += abs(L.values[kk]);
      }
      // Cheeger: h(S) = cut(S) / min(vol(S), vol - vol(S))
      if volS > 0.0 && volS < vol then
        bestCut = min(bestCut, cutEdges / min(volS, vol - volS));
    }
    return bestCut;
  }
}
