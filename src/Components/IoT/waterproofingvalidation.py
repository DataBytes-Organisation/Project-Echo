#!/usr/bin/env python3
"""
Environmental Test Orchestrator
- Executes temperature/humidity/vibration/water-spray profiles
- Subscribes to DUT MQTT telemetry and logs to CSV + JSONL
- Computes dew point, detects ingress/condensation, thermal deltas
- Produces pass/fail summary mapped to your deliverables

Dependencies:
  pip install paho-mqtt pydantic ruamel.yaml numpy pandas
"""

import os, csv, json, time, math, socket
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import threading

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from ruamel.yaml import YAML
import paho.mqtt.client as mqtt

# -----------------------------
# Config & Models
# -----------------------------

class VibrationStep(BaseModel):
    duration_s: int
    rms_g: float = Field(..., description="Target RMS acceleration (g)")
    note: Optional[str] = None

class ThermalStep(BaseModel):
    duration_s: int
    chamber_temp_c: float
    chamber_rh_pct: Optional[float] = None
    note: Optional[str] = None

class SprayStep(BaseModel):
    duration_s: int
    flow_rate_lpm: float
    note: Optional[str] = None

class TestLimits(BaseModel):
    max_board_temp_c: float = 85.0
    max_internal_rh_pct: float = 90.0
    max_dewpoint_delta_c: float = 2.0  # internal_t - dewpoint > 2C indicates buffer
    max_moisture_kohm: float = 50.0    # <50kΩ on moisture probe = ingress
    max_vibe_rms_g: float = 8.0

class TestPlan(BaseModel):
    name: str
    chamber: Dict[str, Any] = {}        # SCPI or GPIO relay config (placeholder)
    shaker: Dict[str, Any] = {}
    spray: Dict[str, Any] = {}
    mqtt: Dict[str, Any]
    limits: TestLimits
    thermal_profile: List[ThermalStep] = []
    vibration_profile: List[VibrationStep] = []
    spray_profile: List[SprayStep] = []
    soak_between_blocks_s: int = 60
    simulate: bool = False

# -----------------------------
# Utilities
# -----------------------------

def dew_point_celsius(t_c: float, rh_pct: float) -> float:
    # Magnus formula (valid for typical ranges)
    a, b = 17.62, 243.12
    gamma = (a * t_c) / (b + t_c) + math.log(max(rh_pct, 1e-6) / 100.0)
    return (b * gamma) / (a - gamma)

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

# -----------------------------
# Hardware Abstraction (stubs)
# Replace these with your chamber/shaker/spray drivers
# -----------------------------

class ClimateChamber:
    def __init__(self, cfg: Dict[str, Any], simulate: bool=False):
        self.cfg, self.simulate = cfg, simulate
        self._set = {"temp": 25.0, "rh": 50.0}

    def set(self, temp_c: float, rh_pct: Optional[float]=None):
        self._set["temp"] = temp_c
        if rh_pct is not None:
            self._set["rh"] = rh_pct
        if self.simulate:
            print(f"[SIM] Chamber set: T={temp_c}C RH={self._set['rh']}%")
        # TODO: SCPI or MODBUS call here

    def get_setpoints(self):
        return dict(self._set)

class Shaker:
    def __init__(self, cfg: Dict[str, Any], simulate: bool=False):
        self.cfg, self.simulate = cfg, simulate
        self._rms = 0.0

    def set_rms_g(self, rms_g: float):
        self._rms = rms_g
        if self.simulate:
            print(f"[SIM] Shaker RMS set to {rms_g} g")
        # TODO: talk to controller (e.g., TCP SCPI)

    def stop(self):
        self.set_rms_g(0.0)

class SprayRig:
    def __init__(self, cfg: Dict[str, Any], simulate: bool=False):
        self.cfg, self.simulate = cfg, simulate
        self._flow = 0.0

    def set_flow_lpm(self, flow: float):
        self._flow = flow
        if self.simulate:
            print(f"[SIM] Spray flow {flow} LPM")
        # TODO: GPIO relay + flow controller

    def stop(self):
        self.set_flow_lpm(0.0)

# -----------------------------
# MQTT Telemetry
# -----------------------------

class TelemetryStore:
    def __init__(self):
        self.latest = {}
        self.lock = threading.Lock()

    def update(self, payload: Dict[str, Any]):
        with self.lock:
            self.latest = payload | {"timestamp": utc_now_iso()}

    def snapshot(self):
        with self.lock:
            return dict(self.latest)

def make_mqtt_client(broker_host, broker_port, topic, store: TelemetryStore):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    def on_connect(c, userdata, flags, rc, properties=None):
        print(f"[MQTT] Connected rc={rc}; subscribing {topic}")
        c.subscribe(topic, qos=1)

    def on_message(c, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            store.update(data)
        except Exception as e:
            print("[MQTT] Bad payload:", e)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker_host, broker_port, keepalive=60)
    return client

# -----------------------------
# Orchestrator
# -----------------------------

class Orchestrator:
    def __init__(self, plan: TestPlan, out_dir="results"):
        self.plan = plan
        self.chamber = ClimateChamber(plan.chamber, plan.simulate)
        self.shaker = Shaker(plan.shaker, plan.simulate)
        self.spray  = SprayRig(plan.spray, plan.simulate)
        self.store  = TelemetryStore()
        os.makedirs(out_dir, exist_ok=True)
        self.out_csv = os.path.join(out_dir, f"{plan.name}_telemetry.csv")
        self.out_jsonl = os.path.join(out_dir, f"{plan.name}_events.jsonl")
        self.out_report = os.path.join(out_dir, f"{plan.name}_report.json")
        self._csv_init()

        self.mqtt = make_mqtt_client(
            plan.mqtt["host"], plan.mqtt.get("port", 1883),
            plan.mqtt["topic"], self.store
        )

        self.fail_reasons = []

    def _csv_init(self):
        hdr = ["utc_ts","phase","set_temp_c","set_rh_pct",
               "dut_temp_c","dut_rh_pct","dut_board_temp_c",
               "dew_point_c","dew_margin_c","moisture_kohm",
               "accel_rms_g"]
        with open(self.out_csv, "w", newline="") as f:
            csv.writer(f).writerow(hdr)

    def _log_row(self, phase, set_temp, set_rh, snap):
        t = snap.get("t_c")
        rh = snap.get("rh_pct")
        board_t = snap.get("board_temp_c")
        dew = dew_point_celsius(t or 0.0, rh or 0.0) if t is not None and rh is not None else None
        dew_margin = (t - dew) if (dew is not None and t is not None) else None
        row = [
            utc_now_iso(), phase, set_temp, set_rh,
            t, rh, board_t, dew, dew_margin,
            snap.get("moisture_kohm"), snap.get("accel_rms_g")
        ]
        with open(self.out_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

        event = dict(
            ts=utc_now_iso(), phase=phase, set_temp_c=set_temp, set_rh_pct=set_rh,
            telemetry=snap
        )
        with open(self.out_jsonl, "a") as f:
            f.write(json.dumps(event) + "\n")

        # Checks
        lim = self.plan.limits
        if board_t is not None and board_t > lim.max_board_temp_c:
            self.fail_reasons.append(f"Board temp {board_t:.1f}C exceeds {lim.max_board_temp_c}C")
        if rh is not None and rh > lim.max_internal_rh_pct:
            self.fail_reasons.append(f"Internal RH {rh:.1f}% exceeds {lim.max_internal_rh_pct}%")
        if dew_margin is not None and dew_margin < lim.max_dewpoint_delta_c:
            self.fail_reasons.append(f"Dew margin {dew_margin:.2f}C < {lim.max_dewpoint_delta_c}C (condensation risk)")
        mk = snap.get("moisture_kohm")
        if mk is not None and mk < lim.max_moisture_kohm:
            self.fail_reasons.append(f"Moisture probe {mk:.1f}kΩ < {lim.max_moisture_kohm}kΩ (ingress)")
        ar = snap.get("accel_rms_g")
        if ar is not None and ar > lim.max_vibe_rms_g * 1.5:  # sanity bound on DUT accel
            self.fail_reasons.append(f"Accel RMS {ar:.2f}g suspiciously high")

    def _hold(self, seconds, phase, set_temp, set_rh):
        t0 = time.time()
        while time.time() - t0 < seconds:
            snap = self.store.snapshot()
            self._log_row(phase, set_temp, set_rh, snap)
            time.sleep(1.0)

    def run(self):
        # Background MQTT loop
        threading.Thread(target=self.mqtt.loop_forever, daemon=True).start()
        print(f"[RUN] {self.plan.name} started at {utc_now_iso()}")

        # Thermal block
        for i, step in enumerate(self.plan.thermal_profile, 1):
            self.chamber.set(step.chamber_temp_c, step.chamber_rh_pct)
            phase = f"THERMAL_{i}"
            print(f"[PHASE] {phase}: T={step.chamber_temp_c} RH={step.chamber_rh_pct} dur={step.duration_s}s")
            self._hold(step.duration_s, phase, step.chamber_temp_c, step.chamber_rh_pct)
            time.sleep(self.plan.soak_between_blocks_s)

        # Vibration block
        for i, step in enumerate(self.plan.vibration_profile, 1):
            self.shaker.set_rms_g(step.rms_g)
            phase = f"VIBE_{i}"
            print(f"[PHASE] {phase}: RMS={step.rms_g}g dur={step.duration_s}s")
            self._hold(step.duration_s, phase, *self._current_setpoints())
            self.shaker.stop()
            time.sleep(self.plan.soak_between_blocks_s)

        # Spray block
        for i, step in enumerate(self.plan.spray_profile, 1):
            self.spray.set_flow_lpm(step.flow_rate_lpm)
            phase = f"SPRAY_{i}"
            print(f"[PHASE] {phase}: Flow={step.flow_rate_lpm}LPM dur={step.duration_s}s")
            self._hold(step.duration_s, phase, *self._current_setpoints())
            self.spray.stop()
            time.sleep(self.plan.soak_between_blocks_s)

        # Finalize report
        self.shaker.stop(); self.spray.stop()
        report = self._summarize()
        with open(self.out_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[DONE] Report -> {self.out_report}")
        return report

    def _current_setpoints(self):
        sp = self.chamber.get_setpoints()
        return sp.get("temp", None), sp.get("rh", None)

    def _summarize(self):
        df = pd.read_csv(self.out_csv)
        stats = {
            "rows": len(df),
            "temp_c_min": float(df["dut_temp_c"].min(skipna=True)),
            "temp_c_max": float(df["dut_temp_c"].max(skipna=True)),
            "rh_pct_max": float(df["dut_rh_pct"].max(skipna=True)),
            "board_temp_c_max": float(df["dut_board_temp_c"].max(skipna=True)),
            "moisture_kohm_min": float(df["moisture_kohm"].min(skipna=True)),
            "accel_rms_g_max": float(df["accel_rms_g"].max(skipna=True)),
            "dew_margin_c_min": float(df["dew_margin_c"].min(skipna=True)),
        }
        passed = len(self.fail_reasons) == 0
        return {
            "test_name": self.plan.name,
            "utc_finished": utc_now_iso(),
            "limits": self.plan.limits.model_dump(),
            "stats": stats,
            "passed": passed,
            "fail_reasons": sorted(set(self.fail_reasons)),
            "artifacts": {
                "csv": self.out_csv,
                "events_jsonl": self.out_jsonl
            },
            "recommendations": self._recommendations(stats)
        }

    def _recommendations(self, stats):
        recs = []
        if stats["dew_margin_c_min"] < self.plan.limits.max_dewpoint_delta_c:
            recs.append("Improve vapor barrier / desiccant; review vent membrane specs.")
        if stats["moisture_kohm_min"] < self.plan.limits.max_moisture_kohm:
            recs.append("Upgrade gasket (Shore A + compression set), add double O-ring or potting in feed-throughs.")
        if stats["board_temp_c_max"] > self.plan.limits.max_board_temp_c:
            recs.append("Add thermal pads to housing, improve heat spreader or finning; consider lighter PCB solder mask color.")
        return recs

# -----------------------------
# CLI Entry
# -----------------------------

def load_plan(path: str) -> TestPlan:
    yaml = YAML(typ="safe")
    with open(path, "r") as f:
        cfg = yaml.load(f)
    return TestPlan(**cfg)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="YAML plan file")
    ap.add_argument("--out", default="results", help="Output directory")
    args = ap.parse_args()

    plan = load_plan(args.plan)
    orch = Orchestrator(plan, out_dir=args.out)
    rep = orch.run()
    print(json.dumps(rep, indent=2))
