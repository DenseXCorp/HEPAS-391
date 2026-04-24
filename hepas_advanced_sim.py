#!/usr/bin/env python3
"""
HEPAS-391: Hybrid Elevator Priority Allocation System
Stochastic Simulation Engine v2.0
© 2026 DenseX Corporation — Released under AGPL-3.0
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

CONFIG = {
    "floors": list(range(-3, 13)), "interchanges": [3, 5],
    "zones": {
        "A": {"floors": list(range(-3, 4))},
        "B": {"floors": list(range(0, 6))},
        "C": {"floors": list(range(5, 13))},
    },
    "elevators": {
        1: {"capacity": 12, "speed": 2.5, "zone": "A", "status": "active"},
        2: {"capacity": 12, "speed": 2.5, "zone": "A", "status": "active"},
        3: {"capacity": 16, "speed": 3.0, "zone": "B", "status": "active"},
        4: {"capacity": 16, "speed": 3.0, "zone": "B", "status": "active"},
        5: {"capacity": 10, "speed": 4.0, "zone": "C", "status": "active"},
        6: {"capacity": 10, "speed": 4.0, "zone": "C", "status": "repair"},
    },
}

NOISE_CONFIG = {
    "passenger_arrival_noise": 0.15, "wait_time_jitter": 0.20,
    "sensor_error": 0.05, "human_behavior_variance": 0.25,
    "outlier_probability": 0.02,
}

WEIGHTS = {
    "alpha1": 2.0, "alpha2": 0.05, "omega_emg_fire": 500,
    "omega_emg_med": 200, "omega_vip": 80, "beta_fire": 300,
    "gamma_scarcity": 50, "delta_interchange": 15, "eta_wait_penalty": 1.0,
}

class HEPAS_AdvancedSimulator:
    def __init__(self):
        self.cfg, self.noise, self.w = CONFIG, NOISE_CONFIG, WEIGHTS

    def _get_zone(self, floor):
        for z, data in self.cfg["zones"].items():
            if floor in data["floors"]: return z
        return "A"

    def _active_in_zone(self, zone):
        return sum(1 for e in self.cfg["elevators"].values() if e["zone"] == zone and e["status"] == "active")

    def generate_state(self, time_step, fire_loc=None):
        rows = []
        for f in self.cfg["floors"]:
            zone = self._get_zone(f)
            e_avail = self._active_in_zone(zone)
            
            base_lam = 12 if f == 0 else 8 if f in self.cfg["interchanges"] else 3
            n_wait = int(np.random.poisson(base_lam) * np.random.normal(1.0, self.noise["human_behavior_variance"]))
            if np.random.random() < self.noise["outlier_probability"]: 
                n_wait += np.random.randint(3, 10)
            
            waits = [max(0.0, (min(time_step * 0.5, 120) + np.random.exponential(15)) * (1 + np.random.uniform(-self.noise["wait_time_jitter"], self.noise["wait_time_jitter"]))) for _ in range(n_wait)]
            max_w, avg_w = (max(waits), np.mean(waits)) if waits else (0.0, 0.0)
            
            vip = 1 if np.random.random() < 0.01 else 0
            emg = 2 if fire_loc == f else 2 if np.random.random() < 0.001 else 1 if np.random.random() < 0.005 else 0
            
            det = (self.w["alpha1"]*n_wait + self.w["alpha2"]*max_w + 
                   (self.w["omega_emg_fire"] if emg==2 else self.w["omega_emg_med"] if emg==1 else 0) + 
                   self.w["omega_vip"]*vip + 
                   (self.w["beta_fire"]/(abs(f-fire_loc)+0.1) if fire_loc is not None else 0) + 
                   self.w["gamma_scarcity"]/(e_avail+0.1) + 
                   self.w["delta_interchange"]*(1 if f in self.cfg["interchanges"] else 0) + 
                   (self.w["eta_wait_penalty"]*np.exp((avg_w-60)/30) if avg_w>60 else 0))
            
            noisy_score = det + np.random.normal(0, 2.0) + np.random.normal(0, det*0.05) + np.random.exponential(1.5)*(1 if np.random.random()>0.5 else -1)

            n_wait_m = max(0, n_wait + np.random.normal(0, n_wait * self.noise["sensor_error"]))
            max_w_m  = max(0, max_w + np.random.normal(0, max_w * self.noise["sensor_error"]))
            avg_w_m  = max(0, avg_w + np.random.normal(0, avg_w * self.noise["sensor_error"]))

            rows.append({
                "floor_id": f, "zone_id": zone,
                "num_waiting": round(n_wait_m, 1), "max_wait_sec": round(max_w_m, 1), "avg_wait_sec": round(avg_w_m, 1),
                "vip_present": vip, "emergency_level": emg, "fire_distance": abs(f-fire_loc) if fire_loc is not None else -1,
                "elevators_available": e_avail, "is_interchange": 1 if f in self.cfg["interchanges"] else 0,
                "capacity_pressure": round(n_wait / (12 * e_avail + 1), 3),
                "target_priority_deterministic": round(det, 2),
                "target_priority_noisy": round(float(np.clip(noisy_score, 0, 100)), 2),
                "time_step": time_step,
                "floor_type": "lobby" if f == 0 else "interchange" if f in self.cfg["interchanges"] else "standard"
            })
        return pd.DataFrame(rows)

    def generate_dataset(self, n_snapshots=100000):
        frames, fire_loc = [], None
        for i in range(n_snapshots):
            if np.random.random() < 0.02: fire_loc = np.random.choice(self.cfg["floors"])
            elif fire_loc is not None and np.random.random() < 0.1: fire_loc = None
            frames.append(self.generate_state(i, fire_loc))
            if (i+1) % 10000 == 0: print(f"    → {i+1:,} snapshots")
        return pd.get_dummies(pd.concat(frames, ignore_index=True), columns=["zone_id"])

if __name__ == "__main__":
    print("🚧 Generating HEPAS‑391 dataset...")
    sim = HEPAS_AdvancedSimulator()
    df = sim.generate_dataset(100000)
    df.to_csv("hepas_dataset_v2_noisy.csv", index=False)
    print(f"✅ Dataset saved: {df.shape[0]:,} rows")
