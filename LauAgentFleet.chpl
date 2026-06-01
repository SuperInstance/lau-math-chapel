// LauAgentFleet.chpl — Distributed agent fleet: locale-partitioned agents, parallel update, CRDT merge
module LauAgentFleet {
  use LauMatrix, Math;

  /* Agent state: position, velocity, energy, type */
  record Agent {
    var id: int;
    var x: real;
    var y: real;
    var vx: real;
    var vy: real;
    var energy: real;
    var agentType: int;
    var active: bool;

    proc init(id: int, x: real = 0.0, y: real = 0.0, vx: real = 0.0, vy: real = 0.0,
              energy: real = 1.0, agentType: int = 0) {
      this.id = id;
      this.x = x;
      this.y = y;
      this.vx = vx;
      this.vy = vy;
      this.energy = energy;
      this.agentType = agentType;
      this.active = true;
    }
  }

  /* CRDT-mergeable agent state for distributed aggregation.
     Uses last-writer-wins with timestamp. */
  record AgentCRDT {
    var agentId: int;
    var x: real;
    var y: real;
    var vx: real;
    var vy: real;
    var energy: real;
    var timestamp: int;
    var sourceLocale: int;

    proc merge(other: AgentCRDT): AgentCRDT {
      if other.timestamp > this.timestamp then return other;
      return this;
    }
  }

  /* Convert Agent to CRDT */
  proc agentToCRDT(a: Agent, timestamp: int, localeId: int): AgentCRDT {
    return new AgentCRDT(a.id, a.x, a.y, a.vx, a.vy, a.energy, timestamp, localeId);
  }

  /* Convert CRDT back to Agent */
  proc crdtToAgent(c: AgentCRDT): Agent {
    var a = new Agent(c.agentId, c.x, c.y, c.vx, c.vy, c.energy);
    return a;
  }

  /* Distributed agent fleet */
  class AgentFleet {
    var numAgents: int;
    var agents: [0..<numAgents] Agent;
    var timestep: int;
    var dt: real;

    proc init(numAgents: int, dt: real = 0.01) {
      this.numAgents = numAgents;
      this.agents = [i in 0..<numAgents] new Agent(i);
      this.timestep = 0;
      this.dt = dt;
    }

    /* Initialize agents randomly */
    proc initRandom(xRange: real = 100.0, yRange: real = 100.0, maxVel: real = 1.0, numTypes: int = 3) {
      for i in 0..<numAgents {
        // Deterministic pseudo-random initialization
        agents[i].x = ((i * 12345 + 67890) % 100000): real / 100000.0 * xRange;
        agents[i].y = ((i * 54321 + 98765) % 100000): real / 100000.0 * yRange;
        agents[i].vx = (((i * 11111 + 22222) % 1000): real / 1000.0 - 0.5) * 2.0 * maxVel;
        agents[i].vy = (((i * 33333 + 44444) % 1000): real / 1000.0 - 0.5) * 2.0 * maxVel;
        agents[i].agentType = i % numTypes;
        agents[i].energy = 1.0;
        agents[i].active = true;
      }
    }

    /* Simple update: Euler integration with boundary reflection */
    proc step() {
      forall a in agents {
        if !a.active then continue;
        a.x += a.vx * dt;
        a.y += a.vy * dt;
        // Boundary reflection
        if a.x < 0.0 { a.x = -a.x; a.vx = -a.vx; }
        if a.x > 100.0 { a.x = 200.0 - a.x; a.vx = -a.vx; }
        if a.y < 0.0 { a.y = -a.y; a.vy = -a.vy; }
        if a.y > 100.0 { a.y = 200.0 - a.y; a.vy = -a.vy; }
        // Energy decay
        a.energy *= 0.999;
        if a.energy < 0.01 then a.active = false;
      }
      timestep += 1;
    }

    /* Apply force field: agents attracted to center */
    proc attractToCenter(strength: real = 0.1) {
      forall a in agents {
        if !a.active then continue;
        a.vx -= strength * (a.x - 50.0) * dt;
        a.vy -= strength * (a.y - 50.0) * dt;
      }
    }

    /* Agent-agent interaction: simple repulsion within radius */
    proc interact(repulsionRadius: real = 5.0, repulsionStrength: real = 0.5) {
      // For efficiency in testing, use O(n^2) but parallelize outer loop
      forall i in 0..<numAgents {
        if !agents[i].active then continue;
        for j in 0..<numAgents {
          if i == j || !agents[j].active then continue;
          const dx = agents[i].x - agents[j].x;
          const dy = agents[i].y - agents[j].y;
          const distSq = dx * dx + dy * dy;
          const rSq = repulsionRadius * repulsionRadius;
          if distSq < rSq && distSq > 0.01 {
            const dist = sqrt(distSq);
            const force = repulsionStrength / distSq;
            agents[i].vx += force * dx / dist * dt;
            agents[i].vy += force * dy / dist * dt;
          }
        }
      }
    }

    /* Get CRDT snapshots for all active agents */
    proc getCRDTSnapshot(localeId: int): [] AgentCRDT {
      var activeCount = 0;
      for a in agents do if a.active then activeCount += 1;
      var snapshots: [0..<activeCount] AgentCRDT;
      var idx = 0;
      for a in agents {
        if a.active {
          snapshots[idx] = agentToCRDT(a, timestep, localeId);
          idx += 1;
        }
      }
      return snapshots;
    }

    /* Merge CRDT updates from remote locales */
    proc mergeFromCRDTS(remoteSnapshots: [] AgentCRDT) {
      for c in remoteSnapshots {
        if c.agentId >= 0 && c.agentId < numAgents {
          // Last-writer-wins: only update if remote is newer
          if c.timestamp >= this.timestep {
            agents[c.agentId].x = c.x;
            agents[c.agentId].y = c.y;
            agents[c.agentId].vx = c.vx;
            agents[c.agentId].vy = c.vy;
            agents[c.agentId].energy = c.energy;
          }
        }
      }
    }

    /* Count active agents */
    proc countActive(): int {
      return + reduce [a in agents] a.active:int;
    }

    /* Total kinetic energy */
    proc totalKineticEnergy(): real {
      return + reduce [a in agents] (if a.active then 0.5 * (a.vx**2 + a.vy**2) else 0.0);
    }

    /* Center of mass */
    proc centerOfMass(): (real, real) {
      var totalMass = 0.0;
      var cx = 0.0, cy = 0.0;
      for a in agents {
        if a.active {
          totalMass += a.energy;
          cx += a.x * a.energy;
          cy += a.y * a.energy;
        }
      }
      if totalMass > 1e-15 {
        cx /= totalMass;
        cy /= totalMass;
      }
      return (cx, cy);
    }

    /* Get agent positions as dense matrix (n x 2) */
    proc positionsMatrix(): LauMatrix {
      var M = new LauMatrix(numAgents, 2);
      forall i in 0..<numAgents {
        M.data[i, 0] = agents[i].x;
        M.data[i, 1] = agents[i].y;
      }
      return M;
    }
  }

  /* Distributed fleet: each locale manages a subset of agents.
     This simulates the distribution pattern. */
  class DistributedFleet {
    var totalAgents: int;
    var numLocales: int;
    var localFleets: [0..<numLocales] owned AgentFleet?;

    proc init(totalAgents: int, numLocales: int = 2, dt: real = 0.01) {
      this.totalAgents = totalAgents;
      this.numLocales = numLocales;
      // Partition agents across locales
      const agentsPerLocale = totalAgents / numLocales;
      const remainder = totalAgents % numLocales;
      for loc in 0..<numLocales {
        const count = agentsPerLocale + if loc < remainder then 1 else 0;
        localFleets[loc] = new AgentFleet(count, dt);
      }
    }

    proc initRandom(xRange: real = 100.0, yRange: real = 100.0, maxVel: real = 1.0) {
      for loc in 0..<numLocales {
        localFleets[loc]!.initRandom(xRange, yRange, maxVel);
        // Fix IDs to be globally unique
        var offset = 0;
        for prev in 0..<loc {
          offset += localFleets[prev]!.numAgents;
        }
        for i in 0..<localFleets[loc]!.numAgents {
          localFleets[loc]!.agents[i].id = offset + i;
        }
      }
    }

    /* Parallel step across all locales */
    proc distributedStep() {
      forall loc in 0..<numLocales {
        localFleets[loc]!.step();
      }
    }

    /* CRDT merge across locales */
    proc mergeAll() {
      // Collect all snapshots
      var allSnapshots: [0..<numLocales] [] AgentCRDT;
      for loc in 0..<numLocales {
        allSnapshots[loc] = localFleets[loc]!.getCRDTSnapshot(loc);
      }
      // Merge into all locales
      for loc in 0..<numLocales {
        for remote in 0..<numLocales {
          if remote != loc {
            localFleets[loc]!.mergeFromCRDTS(allSnapshots[remote]);
          }
        }
      }
    }

    /* Total active agents across all locales */
    proc totalActive(): int {
      var total = 0;
      for loc in 0..<numLocales do total += localFleets[loc]!.countActive();
      return total;
    }

    /* Total kinetic energy across all locales */
    proc totalKineticEnergy(): real {
      var total = 0.0;
      for loc in 0..<numLocales do total += localFleets[loc]!.totalKineticEnergy();
      return total;
    }
  }
}
