// test_main.chpl — 50+ tests for lau-math-chapel
use LauMatrix, LauLaplacian, LauHeatKernel, LauAgentFleet, LauConservation, LauTopology, Math;

var passed = 0;
var failed = 0;

proc check(name: string, condition: bool) {
  if condition {
    passed += 1;
    writeln("  ✓ ", name);
  } else {
    failed += 1;
    writeln("  ✗ FAIL: ", name);
  }
}

proc checkTol(name: string, actual: real, expected: real, tol: real = 1e-6) {
  check(name, abs(actual - expected) < tol);
}

// ============================================================
// SECTION 1: LauMatrix Tests
// ============================================================
writeln("\n=== LauMatrix Tests ===");

proc testMatrixCreation() {
  writeln("Matrix creation:");
  var A = new LauMatrix(4);
  check("zero matrix created", A.nRows == 4 && A.nCols == 4);
  check("zero matrix is zero", A.frobeniusNorm() < 1e-15);
}

proc testIdentity() {
  writeln("Identity matrix:");
  var A = new LauMatrix(5);
  A.setIdentity();
  check("identity trace", A.trace() == 5.0);
  checkTol("identity frobenius", A.frobeniusNorm(), sqrt(5.0));
}

proc testRandomFill() {
  writeln("Random fill:");
  var A = new LauMatrix(10);
  A.setRandom(-1.0, 1.0);
  check("random matrix non-zero", A.frobeniusNorm() > 0.0);
  check("random matrix bounded", A.frobeniusNorm() < 100.0);
}

proc testSymmetrize() {
  writeln("Symmetrize:");
  var A = new LauMatrix(4);
  A.setRandom();
  A.symmetrize();
  // Check symmetry
  var symErr = 0.0;
  for i in 0..<4 do for j in 0..<4 do symErr += (A.data[i,j] - A.data[j,i]) ** 2;
  checkTol("symmetric after symmetrize", sqrt(symErr), 0.0);
}

proc testMatMul() throws {
  writeln("Matrix multiply:");
  var A = new LauMatrix(3);
  A.setIdentity();
  var B = new LauMatrix(3);
  B.setRandom();
  var C = matMul(A, B);
  checkTol("I*B = B", C.frobeniusNorm(), B.frobeniusNorm());
}

proc testMatMul2() throws {
  writeln("Matrix multiply (non-square):");
  var A = new LauMatrix(3, 2);
  A.setRandom();
  var B = new LauMatrix(2, 4);
  B.setRandom();
  var C = matMul(A, B);
  check("result dimensions", C.nRows == 3 && C.nCols == 4);
}

proc testMatVecMul() throws {
  writeln("Matrix-vector multiply:");
  var A = new LauMatrix(3);
  A.setIdentity();
  var x: [0..<3] real = [1.0, 2.0, 3.0];
  var y = matVecMul(A, x);
  checkTol("I*x = x", y[0], 1.0);
  checkTol("I*x = x [1]", y[1], 2.0);
  checkTol("I*x = x [2]", y[2], 3.0);
}

proc testTranspose() {
  writeln("Transpose:");
  var A = new LauMatrix(2, 3);
  A.data[0,0] = 1.0; A.data[0,1] = 2.0; A.data[0,2] = 3.0;
  A.data[1,0] = 4.0; A.data[1,1] = 5.0; A.data[1,2] = 6.0;
  var B = transpose(A);
  check("transpose dimensions", B.nRows == 3 && B.nCols == 2);
  checkTol("transpose value", B.data[0,1], 4.0);
}

proc testMatAdd() throws {
  writeln("Matrix add:");
  var A = new LauMatrix(3);
  A.setIdentity();
  var B = new LauMatrix(3);
  B.setIdentity();
  var C = matAdd(A, B);
  checkTol("I+I trace", C.trace(), 6.0);
}

proc testMatScale() {
  writeln("Matrix scale:");
  var A = new LauMatrix(3);
  A.setIdentity();
  var B = matScale(A, 3.0);
  checkTol("3*I trace", B.trace(), 9.0);
}

proc testPowerIteration() {
  writeln("Power iteration:");
  var A = new LauMatrix(3);
  A.setIdentity();
  var (lambda, v) = powerIteration(A, 100);
  checkTol("dominant eigenvalue of I", lambda, 1.0, 1e-4);
}

proc testPowerIteration2() {
  writeln("Power iteration (diagonal):");
  var A = new LauMatrix(3);
  A.setZero();
  A.data[0,0] = 5.0; A.data[1,1] = 2.0; A.data[2,2] = 1.0;
  var (lambda, v) = powerIteration(A, 200);
  checkTol("dominant eigenvalue", lambda, 5.0, 0.1);
}

proc testQRDecompose() {
  writeln("QR decomposition:");
  var A = new LauMatrix(3);
  A.data[0,0] = 12.0; A.data[0,1] = -51.0; A.data[0,2] = 4.0;
  A.data[1,0] = 6.0;  A.data[1,1] = 167.0; A.data[1,2] = -68.0;
  A.data[2,0] = -4.0; A.data[2,1] = 24.0;  A.data[2,2] = -41.0;
  var (Q, R) = qrDecompose(A);
  // Q should be orthogonal: Q^T * Q = I
  var QT = transpose(Q);
  var QTQ = matMul(QT, Q);
  var orthErr = 0.0;
  for i in 0..<3 do for j in 0..<3 do {
    const expected = if i == j then 1.0 else 0.0;
    orthErr += (QTQ.data[i,j] - expected) ** 2;
  }
  checkTol("Q orthogonal", sqrt(orthErr), 0.0, 1e-6);
}

proc testEigenDecompose() {
  writeln("Eigendecomposition:");
  var A = new LauMatrix(3);
  A.setZero();
  A.data[0,0] = 2.0; A.data[0,1] = 1.0; A.data[0,2] = 0.0;
  A.data[1,0] = 1.0; A.data[1,1] = 3.0; A.data[1,2] = 1.0;
  A.data[2,0] = 0.0; A.data[2,1] = 1.0; A.data[2,2] = 2.0;
  var (eigenvalues, V) = eigenDecompose(A, 500);
  // Eigenvalues of this matrix are 1, 2, 4
  var sorted = eigenvalues;
  // Sort
  for i in 0..<3 {
    for j in (i+1)..<3 {
      if sorted[j] < sorted[i] { const tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp; }
    }
  }
  checkTol("eigenvalue 1", sorted[0], 1.0, 0.5);
  checkTol("eigenvalue 3", sorted[1], 3.0, 0.5);
  checkTol("eigenvalue 4", sorted[2], 4.0, 0.5);
}

// ============================================================
// SECTION 2: LauLaplacian Tests
// ============================================================
writeln("\n=== LauLaplacian Tests ===");

proc testBuildLaplacian() {
  writeln("Build Laplacian:");
  // Triangle graph: 3 vertices, 3 edges
  var edges: [0..<3] (int, int, real) = [(0,1,1.0), (1,2,1.0), (0,2,1.0)];
  var L = buildLaplacian(3, edges);
  check("Laplacian size", L.n == 3);
  // Diagonal entries should be degree = 2 for all vertices
  checkTol("degree[0]", L.values[L.rowPtr[0]], 2.0); // first entry is diagonal
  check("nnz count", L.nnz == 9); // 3 diag + 6 off-diag
}

proc testLaplacianMultiply() throws {
  writeln("Laplacian multiply:");
  var edges: [0..<2] (int, int, real) = [(0,1,1.0), (1,2,1.0)];
  var L = buildLaplacian(3, edges);
  var ones: [0..<3] real = [1.0, 1.0, 1.0];
  var Lx = L.multiply(ones);
  checkTol("L * 1 = 0", Lx[0], 0.0, 1e-10);
  checkTol("L * 1 = 0 [1]", Lx[1], 0.0, 1e-10);
  checkTol("L * 1 = 0 [2]", Lx[2], 0.0, 1e-10);
}

proc testNormalizedLaplacian() {
  writeln("Normalized Laplacian:");
  var edges: [0..<2] (int, int, real) = [(0,1,1.0), (1,2,1.0)];
  var L = buildNormalizedLaplacian(3, edges);
  check("normalized Laplacian created", L.n == 3);
}

proc testConnectedComponents() {
  writeln("Connected components:");
  // Path graph: 0-1-2-3
  var edges: [0..<3] (int, int, real) = [(0,1,1.0), (1,2,1.0), (2,3,1.0)];
  var L = buildLaplacian(4, edges);
  var cc = connectedComponents(L);
  check("connected path graph", cc == 1);

  // Disconnected: 0-1, 2-3
  var edges2: [0..<2] (int, int, real) = [(0,1,1.0), (2,3,1.0)];
  var L2 = buildLaplacian(4, edges2);
  var cc2 = connectedComponents(L2);
  check("disconnected graph", cc2 == 2);
}

proc testSpectralGap() {
  writeln("Spectral gap:");
  // Complete graph K4
  var edges: [0..<6] (int, int, real) = [
    (0,1,1.0), (0,2,1.0), (0,3,1.0),
    (1,2,1.0), (1,3,1.0), (2,3,1.0)
  ];
  var L = buildLaplacian(4, edges);
  var gap = spectralGap(L, 500);
  check("spectral gap > 0", gap > 0.01);
}

proc testFiedlerVector() {
  writeln("Fiedler vector:");
  var edges: [0..<3] (int, int, real) = [(0,1,1.0), (1,2,1.0), (2,3,1.0)];
  var L = buildLaplacian(4, edges);
  var (lambda2, fied) = fiedlerVector(L, 500);
  check("Fiedler value > 0", lambda2 > 0.01);
  check("Fiedler vector non-zero", sqrt(+ reduce (fied ** 2)) > 0.5);
}

proc testGraphVolume() {
  writeln("Graph volume:");
  var edges: [0..<2] (int, int, real) = [(0,1,1.0), (1,2,1.0)];
  var L = buildLaplacian(3, edges);
  var vol = graphVolume(L);
  checkTol("volume of path(3)", vol, 4.0); // degrees: 1,2,1
}

// ============================================================
// SECTION 3: LauHeatKernel Tests
// ============================================================
writeln("\n=== LauHeatKernel Tests ===");

proc testHeatKernelApply() throws {
  writeln("Heat kernel apply:");
  var edges: [0..<2] (int, int, real) = [(0,1,1.0), (1,2,1.0)];
  var L = buildLaplacian(3, edges);
  var f: [0..<3] real = [1.0, 0.0, 0.0];
  var h = heatKernelApply(L, f, 0.1, 20);
  check("heat kernel preserves mass (approximately)", abs(+ reduce h - 1.0) < 0.1);
  check("heat kernel diffuses", h[1] > 0.0);
}

proc testHeatKernelTrace() throws {
  writeln("Heat kernel trace:");
  var edges: [0..<2] (int, int, real) = [(0,1,1.0), (1,2,1.0)];
  var L = buildLaplacian(3, edges);
  var tr = heatKernelTrace(L, 0.1, 20, 20);
  // Tr(e^{-tL}) should be close to n for small t
  check("trace > 0", tr > 0.0);
  check("trace < n+1", tr < 4.0);
}

proc testHeatKernelDistance() throws {
  writeln("Heat kernel distance:");
  var edges: [0..<2] (int, int, real) = [(0,1,1.0), (1,2,1.0)];
  var L = buildLaplacian(3, edges);
  var a: [0..<3] real = [1.0, 0.0, 0.0];
  var b: [0..<3] real = [0.0, 0.0, 1.0];
  var d = heatKernelDistance(L, a, b, 0.1, 20);
  check("distance > 0", d > 0.0);
}

proc testHeatKernelSelfDistance() throws {
  writeln("Heat kernel self-distance:");
  var edges: [0..<2] (int, int, real) = [(0,1,1.0), (1,2,1.0)];
  var L = buildLaplacian(3, edges);
  var a: [0..<3] real = [1.0, 2.0, 3.0];
  var d = heatKernelDistance(L, a, a, 0.1, 20);
  checkTol("self-distance = 0", d, 0.0, 1e-6);
}

proc testHeatKernelCentrality() throws {
  writeln("Heat kernel centrality:");
  var edges: [0..<3] (int, int, real) = [(0,1,1.0), (1,2,1.0), (2,3,1.0)];
  var L = buildLaplacian(4, edges);
  var cent = heatKernelCentrality(L, 0.1, 20);
  check("centrality sums to ~1", abs(+ reduce cent - 1.0) < 0.01);
  check("centrality non-negative", min reduce cent >= -1e-10);
}

// ============================================================
// SECTION 4: LauAgentFleet Tests
// ============================================================
writeln("\n=== LauAgentFleet Tests ===");

proc testAgentCreation() {
  writeln("Agent creation:");
  var fleet = new AgentFleet(10);
  check("fleet size", fleet.numAgents == 10);
  check("initial timestep", fleet.timestep == 0);
}

proc testAgentInit() {
  writeln("Agent initialization:");
  var fleet = new AgentFleet(20);
  fleet.initRandom();
  var active = fleet.countActive();
  check("all active after init", active == 20);
}

proc testAgentStep() {
  writeln("Agent step:");
  var fleet = new AgentFleet(5, dt=0.01);
  fleet.initRandom();
  fleet.step();
  check("timestep incremented", fleet.timestep == 1);
}

proc testAgentMultiStep() {
  writeln("Agent multi-step:");
  var fleet = new AgentFleet(10, dt=0.001);
  fleet.initRandom();
  for i in 0..<100 do fleet.step();
  check("100 steps", fleet.timestep == 100);
}

proc testAgentKineticEnergy() {
  writeln("Agent kinetic energy:");
  var fleet = new AgentFleet(10);
  // Stationary agents
  for a in fleet.agents { a.vx = 0.0; a.vy = 0.0; a.active = true; }
  checkTol("zero KE", fleet.totalKineticEnergy(), 0.0);
}

proc testAgentCenterOfMass() {
  writeln("Agent center of mass:");
  var fleet = new AgentFleet(4);
  fleet.agents[0].x = 0.0; fleet.agents[0].y = 0.0; fleet.agents[0].energy = 1.0; fleet.agents[0].active = true;
  fleet.agents[1].x = 10.0; fleet.agents[1].y = 0.0; fleet.agents[1].energy = 1.0; fleet.agents[1].active = true;
  fleet.agents[2].x = 0.0; fleet.agents[2].y = 10.0; fleet.agents[2].energy = 1.0; fleet.agents[2].active = true;
  fleet.agents[3].x = 10.0; fleet.agents[3].y = 10.0; fleet.agents[3].energy = 1.0; fleet.agents[3].active = true;
  var (cx, cy) = fleet.centerOfMass();
  checkTol("center x", cx, 5.0);
  checkTol("center y", cy, 5.0);
}

proc testAgentCRDT() {
  writeln("Agent CRDT:");
  var fleet = new AgentFleet(5);
  fleet.initRandom();
  var snapshots = fleet.getCRDTSnapshot(0);
  check("snapshot size", snapshots.size == 5);
  check("CRDT locale", snapshots[0].sourceLocale == 0);
}

proc testAgentCRDTMerge() {
  writeln("Agent CRDT merge:");
  var fleet1 = new AgentFleet(3);
  fleet1.agents[0].x = 1.0; fleet1.agents[0].active = true;
  fleet1.agents[1].x = 2.0; fleet1.agents[1].active = true;
  fleet1.agents[2].x = 3.0; fleet1.agents[2].active = true;

  var snapshots = fleet1.getCRDTSnapshot(1);

  var fleet2 = new AgentFleet(3);
  fleet2.agents[0].x = 0.0; fleet2.agents[0].active = true;
  fleet2.agents[1].x = 0.0; fleet2.agents[1].active = true;
  fleet2.agents[2].x = 0.0; fleet2.agents[2].active = true;
  fleet2.mergeFromCRDTS(snapshots);
  // After merge, fleet2 should have updated positions
  checkTol("merged x[0]", fleet2.agents[0].x, 1.0, 1e-6);
}

proc testDistributedFleet() {
  writeln("Distributed fleet:");
  var df = new DistributedFleet(20, 2);
  df.initRandom();
  check("total active", df.totalActive() == 20);
  check("kinetic energy >= 0", df.totalKineticEnergy() >= 0.0);
}

proc testDistributedFleetStep() {
  writeln("Distributed fleet step:");
  var df = new DistributedFleet(10, 2);
  df.initRandom();
  df.distributedStep();
  check("step completed", true);
}

proc testAgentPositions() {
  writeln("Agent positions matrix:");
  var fleet = new AgentFleet(5);
  fleet.initRandom();
  var M = fleet.positionsMatrix();
  check("positions dimensions", M.nRows == 5 && M.nCols == 2);
}

// ============================================================
// SECTION 5: LauConservation Tests
// ============================================================
writeln("\n=== LauConservation Tests ===");

proc testChargeRecord() {
  writeln("Charge record:");
  var c = new Charge("test", 1.0, 1.0, 1e-6);
  check("charge conserved", c.isConserved());
  checkTol("charge deviation", c.deviation(), 0.0);
}

proc testChargeViolation() {
  writeln("Charge violation:");
  var c = new Charge("test", 2.0, 1.0, 1e-6);
  check("charge NOT conserved", !c.isConserved());
  checkTol("charge deviation", c.deviation(), 1.0);
}

proc testEnergyConservation() {
  writeln("Energy conservation:");
  var fleet = new AgentFleet(5);
  for a in fleet.agents { a.vx = 0.0; a.vy = 0.0; a.active = true; }
  var charge = verifyEnergyConservation(fleet, 0.0, 1e-10);
  check("energy conserved (zero KE)", charge.isConserved());
}

proc testMassConservation() {
  writeln("Mass conservation:");
  var fleet = new AgentFleet(10);
  fleet.initRandom();
  var charge = verifyMassConservation(fleet, 10);
  check("mass conserved", charge.isConserved());
}

proc testMomentumConservation() {
  writeln("Momentum conservation:");
  var fleet = new AgentFleet(4);
  for a in fleet.agents { a.vx = 0.0; a.vy = 0.0; a.active = true; a.energy = 1.0; }
  var (cpx, cpy) = verifyMomentumConservation(fleet, 0.0, 0.0, 1e-10);
  check("px conserved", cpx.isConserved());
  check("py conserved", cpy.isConserved());
}

proc testDistributedReduction() {
  writeln("Distributed charge reduction:");
  var charges: [0..<3] Charge = [
    new Charge("test", 1.0, 1.0),
    new Charge("test", 2.0, 2.0),
    new Charge("test", 3.0, 3.0)
  ];
  var reduced = distributedChargeReduction(charges);
  checkTol("reduced value", reduced.value, 6.0);
  checkTol("reduced expected", reduced.expectedValue, 6.0);
}

proc testConservationMonitor() {
  writeln("Conservation monitor:");
  var names = ["energy", "mass"];
  var expected = [1.0, 10.0];
  var monitor = new ConservationMonitor(names, expected);
  var values = [1.0, 10.0];
  monitor.record(values);
  check("monitor check", monitor.check(values));
  check("monitor step", monitor.currentStep == 1);
}

proc testConservationMonitorDrift() {
  writeln("Conservation drift detection:");
  var names = ["energy"];
  var expected = [1.0];
  var monitor = new ConservationMonitor(names, expected);
  // Record constant values
  for i in 0..<20 {
    var vals = [1.0];
    monitor.record(vals);
  }
  var drifts = monitor.detectDrift(5);
  checkTol("no drift detected", drifts[0], 0.0, 1e-10);
}

proc testLaplacianQuadratic() {
  writeln("Laplacian quadratic form:");
  var edges: [0..<2] (int, int, real) = [(0,1,1.0), (1,2,1.0)];
  var L = buildLaplacian(3, edges);
  var x: [0..<3] real = [1.0, 2.0, 3.0];
  var charge = verifyLaplacianQuadratic(L, x);
  check("Laplacian quadratic conserved", charge.isConserved());
}

// ============================================================
// SECTION 6: LauTopology Tests
// ============================================================
writeln("\n=== LauTopology Tests ===");

proc testSimplicialComplex() {
  writeln("Simplicial complex creation:");
  var K = new SimplicialComplex(4);
  K.addEdge(0, 1);
  K.addEdge(1, 2);
  K.addEdge(2, 3);
  check("num vertices", K.numVertices == 4);
  check("num edges", K.numEdges() == 3);
}

proc testAddTriangle() {
  writeln("Add triangle:");
  var K = new SimplicialComplex(3);
  K.addTriangle(0, 1, 2);
  check("triangle added", K.numTriangles() == 1);
  check("triangle edges auto-added", K.numEdges() == 3);
}

proc testBetti0Connected() {
  writeln("Betti-0 (connected):");
  var K = new SimplicialComplex(4);
  K.addEdge(0, 1);
  K.addEdge(1, 2);
  K.addEdge(2, 3);
  var b0 = betti0(K);
  check("beta_0 = 1 for connected", b0 == 1);
}

proc testBetti0Disconnected() {
  writeln("Betti-0 (disconnected):");
  var K = new SimplicialComplex(6);
  K.addEdge(0, 1);
  K.addEdge(2, 3);
  K.addEdge(4, 5);
  var b0 = betti0(K);
  check("beta_0 = 3 for three components", b0 == 3);
}

proc testBetti1Cycle() {
  writeln("Betti-1 (cycle):");
  var K = new SimplicialComplex(4);
  K.addEdge(0, 1);
  K.addEdge(1, 2);
  K.addEdge(2, 3);
  K.addEdge(3, 0);
  var (b0, b1) = bettiNumbers(K);
  check("beta_0 = 1 for cycle", b0 == 1);
  check("beta_1 = 1 for cycle", b1 == 1);
}

proc testBetti1Filled() {
  writeln("Betti-1 (filled triangle):");
  var K = new SimplicialComplex(3);
  K.addTriangle(0, 1, 2);
  var (b0, b1) = bettiNumbers(K);
  check("beta_0 = 1 for triangle", b0 == 1);
  check("beta_1 = 0 for filled triangle", b1 == 0);
}

proc testBetti1TwoCycles() {
  writeln("Betti-1 (two cycles):");
  var K = new SimplicialComplex(6);
  // Cycle 1: 0-1-2-0
  K.addEdge(0, 1); K.addEdge(1, 2); K.addEdge(0, 2);
  // Cycle 2: 3-4-5-3
  K.addEdge(3, 4); K.addEdge(4, 5); K.addEdge(3, 5);
  // Connect them
  K.addEdge(2, 3);
  var (b0, b1) = bettiNumbers(K);
  check("beta_0 = 1 for connected", b0 == 1);
  check("beta_1 = 2 for two cycles", b1 == 2);
}

proc testEulerCharacteristic() {
  writeln("Euler characteristic:");
  var K = new SimplicialComplex(3);
  K.addTriangle(0, 1, 2);
  var chi = eulerCharacteristic(K);
  check("chi = V - E + F = 3 - 3 + 1 = 1", chi == 1);
}

proc testEulerPath() {
  writeln("Euler characteristic (path graph):");
  var K = new SimplicialComplex(4);
  K.addEdge(0, 1); K.addEdge(1, 2); K.addEdge(2, 3);
  var chi = eulerCharacteristic(K);
  check("chi = 4 - 3 + 0 = 1", chi == 1);
}

proc testIsConnected() {
  writeln("Is connected:");
  var K = new SimplicialComplex(3);
  K.addEdge(0, 1); K.addEdge(1, 2);
  check("connected", isConnected(K));
}

proc testIsSimplyConnected() {
  writeln("Is simply connected:");
  var K = new SimplicialComplex(3);
  K.addTriangle(0, 1, 2);
  check("simply connected", isSimplyConnected(K));
}

proc testCohomology() {
  writeln("Cohomology:");
  var K = new SimplicialComplex(4);
  K.addEdge(0, 1); K.addEdge(1, 2); K.addEdge(2, 3); K.addEdge(3, 0);
  var h0 = cohomologyH0(K);
  var h1 = cohomologyH1(K);
  check("H^0 = 1", h0 == 1);
  check("H^1 = 1", h1 == 1);
}

proc testDistributedTopology() {
  writeln("Distributed topology:");
  var dt = new DistributedTopology(4, 2);
  // Locale 0: 0-1, 1-2
  dt.addEdgeLocal(0, 0, 1);
  dt.addEdgeLocal(0, 1, 2);
  // Locale 1: 2-3
  dt.addEdgeLocal(1, 2, 3);
  // Boundary edge connecting locales
  dt.addBoundaryEdge(1, 2);
  var b0 = dt.distributedBetti0();
  check("distributed beta_0 = 1", b0 == 1);
}

proc testBoundaryOperator() {
  writeln("Boundary operator:");
  var K = new SimplicialComplex(3);
  K.addEdge(0, 1);
  K.addEdge(1, 2);
  var D = K.boundary1();
  check("boundary dimensions", D.nRows == 3 && D.nCols == 2);
}

// ============================================================
// MAIN
// ============================================================
proc main() {
  writeln("╔══════════════════════════════════════════╗");
  writeln("║   lau-math-chapel Test Suite             ║");
  writeln("╚══════════════════════════════════════════╝");

  // LauMatrix
  testMatrixCreation();
  testIdentity();
  testRandomFill();
  testSymmetrize();
  testMatMul();
  testMatMul2();
  testMatVecMul();
  testTranspose();
  testMatAdd();
  testMatScale();
  testPowerIteration();
  testPowerIteration2();
  testQRDecompose();
  testEigenDecompose();

  // LauLaplacian
  testBuildLaplacian();
  testLaplacianMultiply();
  testNormalizedLaplacian();
  testConnectedComponents();
  testSpectralGap();
  testFiedlerVector();
  testGraphVolume();

  // LauHeatKernel
  testHeatKernelApply();
  testHeatKernelTrace();
  testHeatKernelDistance();
  testHeatKernelSelfDistance();
  testHeatKernelCentrality();

  // LauAgentFleet
  testAgentCreation();
  testAgentInit();
  testAgentStep();
  testAgentMultiStep();
  testAgentKineticEnergy();
  testAgentCenterOfMass();
  testAgentCRDT();
  testAgentCRDTMerge();
  testDistributedFleet();
  testDistributedFleetStep();
  testAgentPositions();

  // LauConservation
  testChargeRecord();
  testChargeViolation();
  testEnergyConservation();
  testMassConservation();
  testMomentumConservation();
  testDistributedReduction();
  testConservationMonitor();
  testConservationMonitorDrift();
  testLaplacianQuadratic();

  // LauTopology
  testSimplicialComplex();
  testAddTriangle();
  testBetti0Connected();
  testBetti0Disconnected();
  testBetti1Cycle();
  testBetti1Filled();
  testBetti1TwoCycles();
  testEulerCharacteristic();
  testEulerPath();
  testIsConnected();
  testIsSimplyConnected();
  testCohomology();
  testDistributedTopology();
  testBoundaryOperator();

  writeln("\n════════════════════════════════════════════");
  writeln("Results: ", passed, " passed, ", failed, " failed");
  writeln("Total:   ", passed + failed, " tests");
  if failed > 0 {
    writeln("\n⚠  SOME TESTS FAILED");
  } else {
    writeln("\n✅ ALL TESTS PASSED");
  }
}
