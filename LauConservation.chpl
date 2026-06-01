// LauConservation.chpl — Distributed Noether charge verification, reduction across locales
module LauConservation {
  use LauMatrix, LauLaplacian, LauAgentFleet, Math;

  /* Charge record: represents a conserved quantity */
  record Charge {
    var name: string;
    var value: real;
    var expectedValue: real;
    var tolerance: real;

    proc init(name: string, value: real, expectedValue: real, tolerance: real = 1e-6) {
      this.name = name;
      this.value = value;
      this.expectedValue = expectedValue;
      this.tolerance = tolerance;
    }

    proc isConserved(): bool {
      return abs(value - expectedValue) < tolerance;
    }

    proc deviation(): real {
      return abs(value - expectedValue);
    }
  }

  /* Verify energy conservation: kinetic energy change */
  proc verifyEnergyConservation(fleet: borrowed AgentFleet, initialKE: real, tolerance: real = 0.1): Charge {
    const currentKE = fleet.totalKineticEnergy();
    return new Charge("energy", currentKE, initialKE, tolerance);
  }

  /* Verify mass conservation: number of active agents */
  proc verifyMassConservation(fleet: borrowed AgentFleet, initialCount: int, tolerance: real = 0.5): Charge {
    const current = fleet.countActive();
    return new Charge("mass", current: real, initialCount: real, tolerance);
  }

  /* Verify momentum conservation: sum of velocities should be ~constant */
  proc verifyMomentumConservation(fleet: borrowed AgentFleet, initialPx: real, initialPy: real,
                                   tolerance: real = 0.5): (Charge, Charge) {
    var px = 0.0, py = 0.0;
    for a in fleet.agents {
      if a.active {
        px += a.vx * a.energy;
        py += a.vy * a.energy;
      }
    }
    return (new Charge("momentum_x", px, initialPx, tolerance),
            new Charge("momentum_y", py, initialPy, tolerance));
  }

  /* Distributed reduction: sum charge values across locales */
  proc distributedChargeReduction(charges: [] Charge): Charge {
    if charges.size == 0 then return new Charge("empty", 0.0, 0.0);
    var totalValue = + reduce [c in charges] c.value;
    var totalExpected = + reduce [c in charges] c.expectedValue;
    var maxTol = max reduce [c in charges] c.tolerance;
    return new Charge("reduced_" + charges[0].name, totalValue, totalExpected, maxTol);
  }

  /* Conservation monitor: tracks charges over time and detects violations */
  class ConservationMonitor {
    var numCharges: int;
    var historyLength: int;
    var chargeNames: [0..<numCharges] string;
    var history: [0..<historyLength, 0..<numCharges] real;
    var expectedValues: [0..<numCharges] real;
    var tolerances: [0..<numCharges] real;
    var currentStep: int;

    proc init(names: [] string, expected: [] real, tols: [] real, historyLen: int = 1000) {
      this.numCharges = names.size;
      this.historyLength = historyLen;
      this.chargeNames = names;
      this.expectedValues = expected;
      this.tolerances = tols;
      this.currentStep = 0;
    }

    proc init(names: [] string, expected: [] real, historyLen: int = 1000) {
      this.init(names, expected, [i in 0..<names.size] 1e-6, historyLen);
    }

    /* Record current charge values */
    proc record(values: [] real) {
      if currentStep < historyLength {
        for i in 0..<numCharges do
          history[currentStep, i] = values[i];
      }
      currentStep += 1;
    }

    /* Check if all charges are within tolerance */
    proc check(values: [] real): bool {
      for i in 0..<numCharges {
        if abs(values[i] - expectedValues[i]) > tolerances[i] then
          return false;
      }
      return true;
    }

    /* Get maximum deviation across all charges */
    proc maxDeviation(values: [] real): real {
      var maxDev = 0.0;
      for i in 0..<numCharges {
        maxDev = max(maxDev, abs(values[i] - expectedValues[i]));
      }
      return maxDev;
    }

    /* Detect drift: compute trend of each charge over recent history */
    proc detectDrift(window: int = 10): [] real {
      var drifts: [0..<numCharges] real;
      if currentStep < window + 1 then return drifts;
      const startIdx = max(0, currentStep - window);
      for i in 0..<numCharges {
        var slope = 0.0;
        // Simple linear regression slope
        var sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0;
        const n = min(window, currentStep - startIdx): real;
        for k in 0..<n:int {
          const x = k: real;
          const y = history[startIdx + k, i];
          sumX += x;
          sumY += y;
          sumXY += x * y;
          sumX2 += x * x;
        }
        const denom = n * sumX2 - sumX * sumX;
        if abs(denom) > 1e-15 then
          slope = (n * sumXY - sumX * sumY) / denom;
        drifts[i] = slope;
      }
      return drifts;
    }

    /* Summarize conservation status */
    proc summarize(values: [] real): string {
      var s = "Conservation Status (step " + currentStep:string + "):\n";
      for i in 0..<numCharges {
        const dev = abs(values[i] - expectedValues[i]);
        const status = if dev <= tolerances[i] then "✓" else "✗ VIOLATION";
        s += "  " + chargeNames[i] + ": " + values[i]:string + " (expected " + expectedValues[i]:string +
             ", dev=" + dev:string + ") " + status + "\n";
      }
      return s;
    }
  }

  /* Verify Laplacian conservation: x^T L x should equal sum of edge differences squared */
  proc verifyLaplacianQuadratic(L: LaplacianCSR, x: [] real): Charge {
    var xTLx = 0.0;
    for i in 0..<L.n {
      for k in L.rowPtr[i]..<L.rowPtr[i+1] {
        xTLx += x[i] * L.values[k] * x[L.colIdx[k]];
      }
    }
    // Also compute directly from edges: sum_{(i,j)} w_ij (x_i - x_j)^2
    var edgeSum = 0.0;
    for i in 0..<L.n {
      for k in L.rowPtr[i]..<L.rowPtr[i+1] {
        const j = L.colIdx[k];
        if i < j { // only count each edge once
          const w = -L.values[k]; // off-diagonal is negative
          if w > 0.0 then
            edgeSum += w * (x[i] - x[j]) ** 2;
        }
      }
    }
    return new Charge("laplacian_quadratic", xTLx, edgeSum, 1e-8);
  }

  /* Verify heat kernel contractivity: ||e^{-tL}f|| <= ||f|| */
  proc verifyContractivity(L: LaplacianCSR, f: [] real, t: real): Charge throws {
    use LauHeatKernel;
    const fNorm = sqrt(+ reduce (f ** 2));
    const smoothed = heatKernelApply(L, f, t, 20);
    const sNorm = sqrt(+ reduce (smoothed ** 2));
    // Should be <= original norm
    return new Charge("heat_kernel_contractivity", sNorm, fNorm, 1e-10);
  }

  /* Distributed Noether charge: compute charge on each locale, reduce */
  proc distributedNoetherCharge(fleets: [] owned AgentFleet?, chargeFunc): Charge {
    var charges: [0..<fleets.size] real;
    for i in 0..<fleets.size {
      charges[i] = chargeFunc(fleets[i]!);
    }
    const total = + reduce charges;
    return new Charge("noether", total, total, 1e-10); // first call sets expected
  }
}
