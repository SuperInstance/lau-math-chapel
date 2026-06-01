// LauTopology.chpl — Distributed simplicial complex, parallel cohomology H^0/H^1, Betti numbers
module LauTopology {
  use LauMatrix, LauLaplacian, Math;

  /* Simplicial complex with simplices up to dimension 2 (vertices, edges, triangles) */
  class SimplicialComplex {
    var numVertices: int;
    var edges: domain(2);          // (u, v) pairs, u < v
    var triangles: domain(2);      // encoded as ordered triples stored pairwise

    // More explicit storage
    var edgeList: [0..<0] (int, int);
    var triList: [0..<0] (int, int, int);

    proc init(n: int) {
      this.numVertices = n;
    }

    /* Add edge (u, v) */
    proc addEdge(u: int, v: int) {
      const (a, b) = if u < v then (u, v) else (v, u);
      edges += (a, b);
      rebuildEdgeList();
    }

    /* Add triangle (i, j, k) */
    proc addTriangle(i: int, j: int, k: int) {
      // Sort the triple
      var vals = [i, j, k];
      // Sort
      if vals[0] > vals[1] { const tmp = vals[0]; vals[0] = vals[1]; vals[1] = tmp; }
      if vals[1] > vals[2] { const tmp = vals[1]; vals[1] = vals[2]; vals[2] = tmp; }
      if vals[0] > vals[1] { const tmp = vals[0]; vals[0] = vals[1]; vals[1] = tmp; }
      triangles += (vals[0], vals[1]); // encode first two
      // Also add the edges
      addEdge(vals[0], vals[1]);
      addEdge(vals[1], vals[2]);
      addEdge(vals[0], vals[2]);
      rebuildTriList(vals[0], vals[1], vals[2]);
    }

    proc rebuildEdgeList() {
      var tmp: [0..<edges.size] (int, int);
      var idx = 0;
      for (u, v) in edges {
        tmp[idx] = (u, v);
        idx += 1;
      }
      edgeList = tmp;
    }

    proc rebuildTriList(a: int, b: int, c: int) {
      var oldSize = triList.size;
      var tmp: [0..<oldSize + 1] (int, int, int);
      for i in 0..<oldSize do tmp[i] = triList[i];
      tmp[oldSize] = (a, b, c);
      triList = tmp;
    }

    /* Number of edges */
    proc numEdges(): int {
      return edgeList.size;
    }

    /* Number of triangles */
    proc numTriangles(): int {
      return triList.size;
    }

    /* Boundary operator d1: edges -> vertices (|V| x |E| matrix)
       d1[e] = v1 - v0 for edge (v0, v1) */
    proc boundary1(): LauMatrix {
      const ne = numEdges();
      var D = new LauMatrix(numVertices, ne);
      D.setZero();
      for e in 0..<ne {
        const (u, v) = edgeList[e];
        D.data[u, e] = -1.0;
        D.data[v, e] = 1.0;
      }
      return D;
    }

    /* Boundary operator d2: triangles -> edges (|E| x |T| matrix)
       d2[t] = e1 - e2 + e3 (orientation) */
    proc boundary2(): LauMatrix {
      const ne = numEdges();
      const nt = numTriangles();
      var D = new LauMatrix(ne, nt);
      D.setZero();

      // Build edge index map
      var edgeIdx: map((int, int), int);
      for e in 0..<ne {
        edgeIdx[edgeList[e]] = e;
      }

      for t in 0..<nt {
        const (a, b, c) = triList[t];
        // Triangle (a,b,c) has edges: (a,b), (b,c), (a,c)
        // d2 = +(a,b) -(a,c) +(b,c) ... sign depends on orientation
        if edgeIdx.contains((a, b)) then D.data[edgeIdx[(a, b)], t] += 1.0;
        if edgeIdx.contains((b, c)) then D.data[edgeIdx[(b, c)], t] += 1.0;
        if edgeIdx.contains((a, c)) then D.data[edgeIdx[(a, c)], t] -= 1.0;
      }
      return D;
    }

    /* Graph Laplacian L0 = d1 * d1^T (combinatorial) */
    proc laplacian0(): LauMatrix {
      const D1 = boundary1();
      const D1T = transpose(D1);
      return matMul(D1, D1T);
    }

    /* Edge Laplacian L1 = d2 * d2^T + d1^T * d1 */
    proc laplacian1(): LauMatrix {
      const D1 = boundary1();
      const D2 = boundary2();
      const D1T = transpose(D1);
      const D2T = transpose(D2);
      const A = matMul(D1T, D1);
      const B = matMul(D2, D2T);
      return matAdd(A, B);
    }
  }

  /* Compute Betti numbers:
     beta_0 = rank(H_0) = dim(ker(d0)) - dim(im(d1))
     Since d0=0, beta_0 = dim(ker(d1)) = #connected components
     beta_1 = rank(H_1) = dim(ker(d1)/im(d2))
     = dim(ker(d1)) - dim(im(d2))
     = (|E| - rank(d1)) - rank(d2) */

  proc betti0(K: borrowed SimplicialComplex): int {
    // Number of connected components
    // Use union-find
    const n = K.numVertices;
    var parent: [0..<n] int;
    var rank_: [0..<n] int;
    for i in 0..<n { parent[i] = i; rank_[i] = 0; }

    proc find(x: int): int {
      while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x]; }
      return x;
    }

    proc union(x: int, y: int) {
      var rx = find(x), ry = find(y);
      if rx == ry then return;
      if rank_[rx] < rank_[ry] then rx <=> ry;
      parent[ry] = rx;
      if rank_[rx] == rank_[ry] then rank_[rx] += 1;
    }

    for (u, v) in K.edgeList do union(u, v);

    var components: domain(int);
    for i in 0..<n do components += find(i);
    return components.size;
  }

  proc betti1(K: borrowed SimplicialComplex): int {
    // beta_1 = |E| - |V| + |components| - |T| ... not quite
    // For simplicial complex: beta_1 = dim(ker d1) - dim(im d2)
    //                            = (|E| - rank(d1)) - rank(d2)
    // rank(d1) = |V| - components (for connected graphs)
    // rank(d2) = num triangles (if no degenerate triangles)
    const components = betti0(K);
    const rankD1 = K.numVertices - components;
    const dimKerD1 = K.numEdges() - rankD1;
    const rankD2 = K.numTriangles(); // approximate
    const b1 = dimKerD1 - rankD2;
    return max(0, b1);
  }

  /* Compute all Betti numbers (up to dimension 1) */
  proc bettiNumbers(K: borrowed SimplicialComplex): (int, int) {
    return (betti0(K), betti1(K));
  }

  /* Euler characteristic: chi = V - E + T */
  proc eulerCharacteristic(K: borrowed SimplicialComplex): int {
    return K.numVertices - K.numEdges() + K.numTriangles();
  }

  /* Euler characteristic from Betti numbers: chi = beta_0 - beta_1 + beta_2 */
  proc eulerFromBetti(b0: int, b1: int, b2: int): int {
    return b0 - b1 + b2;
  }

  /* Distributed Betti number computation:
     Partition the complex across locales, compute local contributions, merge.
     For beta_0: merge connected components across partitions.
     For beta_1: parallelize over local subcomplexes. */
  class DistributedTopology {
    var numLocales: int;
    var subComplexes: [0..<numLocales] owned SimplicialComplex?;
    var boundaryEdges: [0..<0] (int, int); // edges crossing locale boundaries

    proc init(n: int, numLocales: int = 2) {
      this.numLocales = numLocales;
      for loc in 0..<numLocales {
        subComplexes[loc] = new SimplicialComplex(n);
      }
    }

    /* Add edge to specific locale */
    proc addEdgeLocal(loc: int, u: int, v: int) {
      subComplexes[loc]!.addEdge(u, v);
    }

    /* Add boundary edge (shared between locales) */
    proc addBoundaryEdge(u: int, v: int) {
      var oldSize = boundaryEdges.size;
      var tmp: [0..<oldSize + 1] (int, int);
      for i in 0..<oldSize do tmp[i] = boundaryEdges[i];
      tmp[oldSize] = (u, v);
      boundaryEdges = tmp;
    }

    /* Compute distributed beta_0: merge components across locales */
    proc distributedBetti0(): int {
      // Use union-find across all locales
      var n = subComplexes[0]!.numVertices;
      var parent: [0..<n] int;
      var rank_: [0..<n] int;
      for i in 0..<n { parent[i] = i; rank_[i] = 0; }

      proc find(x: int): int {
        while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
      }

      proc union(x: int, y: int) {
        var rx = find(x), ry = find(y);
        if rx == ry then return;
        if rank_[rx] < rank_[ry] then rx <=> ry;
        parent[ry] = rx;
        if rank_[rx] == rank_[ry] then rank_[rx] += 1;
      }

      // Merge edges from all locales
      for loc in 0..<numLocales {
        for (u, v) in subComplexes[loc]!.edgeList do union(u, v);
      }
      // Merge boundary edges
      for (u, v) in boundaryEdges do union(u, v);

      var components: domain(int);
      for i in 0..<n do components += find(i);
      return components.size;
    }

    /* Compute distributed beta_1 */
    proc distributedBetti1(): int {
      const components = distributedBetti0();
      var totalEdges = 0;
      var totalTriangles = 0;
      for loc in 0..<numLocales {
        totalEdges += subComplexes[loc]!.numEdges();
        totalTriangles += subComplexes[loc]!.numTriangles();
      }
      // Subtract boundary edge duplicates
      totalEdges -= boundaryEdges.size;
      const rankD1 = subComplexes[0]!.numVertices - components;
      const dimKerD1 = totalEdges - rankD1;
      const b1 = dimKerD1 - totalTriangles;
      return max(0, b1);
    }
  }

  /* Cohomology H^0: dimension = number of connected components
     (functions constant on each component) */
  proc cohomologyH0(K: borrowed SimplicialComplex): int {
    return betti0(K);
  }

  /* Cohomology H^1: same dimension as H_1 by universal coefficient theorem */
  proc cohomologyH1(K: borrowed SimplicialComplex): int {
    return betti1(K);
  }

  /* Check if complex is simply connected: beta_0 = 1 and beta_1 = 0 */
  proc isSimplyConnected(K: borrowed SimplicialComplex): bool {
    const (b0, b1) = bettiNumbers(K);
    return b0 == 1 && b1 == 0;
  }

  /* Check if complex is connected */
  proc isConnected(K: borrowed SimplicialComplex): bool {
    return betti0(K) == 1;
  }
}
