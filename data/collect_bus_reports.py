"""
Phase 2: Run the SUMO simulation and, at every REPORT_INTERVAL seconds,
have each BUS (not regular cars) broadcast (edge_id, mean_speed, occupancy,
timestamp). This mimics buses acting as mobile traffic sensors.

Ground truth for training comes from ALL vehicles on each edge (what SUMO
actually observed), while the model only ever sees the bus-reported subset
at inference time -- this is the realistic constraint: buses are a sparse
sample of full traffic state.

Output: data/bus_reports.csv   (sparse, what buses actually shared)
        data/edge_ground_truth.csv (dense, all-vehicle truth per edge per interval)
"""
import os
import sys
import csv

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci  # noqa: E402

REPORT_INTERVAL = 30  # seconds between broadcasts
NET_DIR = os.path.join(os.path.dirname(__file__), "..", "network")


def main():
    sumo_cfg = os.path.join(NET_DIR, "sim.sumocfg")
    traci.start(["sumo", "-c", sumo_cfg, "--no-step-log", "true",
                 "--no-warnings", "true"])

    bus_reports = []
    ground_truth = []

    edge_ids = [e for e in traci.edge.getIDList() if not e.startswith(":")]

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % REPORT_INTERVAL == 0:
            t = traci.simulation.getTime()

            # Dense ground truth: every edge's actual mean speed/occupancy
            for e in edge_ids:
                mean_speed = traci.edge.getLastStepMeanSpeed(e)
                occupancy = traci.edge.getLastStepOccupancy(e)
                veh_count = traci.edge.getLastStepVehicleNumber(e)
                ground_truth.append([t, e, mean_speed, occupancy, veh_count])

            # Sparse bus reports: only what buses currently on the road see
            for veh_id in traci.vehicle.getIDList():
                if traci.vehicle.getTypeID(veh_id) != "bus":
                    continue
                edge = traci.vehicle.getRoadID(veh_id)
                if edge.startswith(":"):
                    continue  # skip internal junction edges
                speed = traci.vehicle.getSpeed(veh_id)
                bus_reports.append([t, veh_id, edge, speed])

        step += 1

    traci.close()

    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "bus_reports.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "bus_id", "edge_id", "speed"])
        w.writerows(bus_reports)

    with open(os.path.join(os.path.dirname(__file__), "edge_ground_truth.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "edge_id", "mean_speed", "occupancy", "veh_count"])
        w.writerows(ground_truth)

    print(f"Collected {len(bus_reports)} bus reports across {len(edge_ids)} edges.")
    print(f"Ground truth rows: {len(ground_truth)}")


if __name__ == "__main__":
    main()
