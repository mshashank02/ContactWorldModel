#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def set_object_pose(model: mujoco.MjModel, data: mujoco.MjData, pos, quat) -> None:
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
    if jnt_id < 0:
        raise ValueError("Could not find joint 'object:joint'")
    if int(model.jnt_type[jnt_id]) != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("'object:joint' is not a free joint")

    qpos_adr = int(model.jnt_qposadr[jnt_id])
    dof_adr = int(model.jnt_dofadr[jnt_id])
    data.qpos[qpos_adr:qpos_adr + 3] = np.asarray(pos, dtype=float)
    data.qpos[qpos_adr + 3:qpos_adr + 7] = np.asarray(quat, dtype=float)
    data.qvel[dof_adr:dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)


def object_position(model: mujoco.MjModel, data: mujoco.MjData) -> list[float]:
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
    qpos_adr = int(model.jnt_qposadr[jnt_id])
    return [float(x) for x in data.qpos[qpos_adr:qpos_adr + 3]]


def has_robot_object_contact(model: mujoco.MjModel, data: mujoco.MjData):
    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        if g1 < 0 or g2 < 0:
            continue
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
        if (n1.startswith("robot0:") and n2 == "object") or (n2.startswith("robot0:") and n1 == "object"):
            return True, (n1, n2)
    return False, None


def read_fixed_friction(xml_path: Path) -> str:
    root = ET.parse(xml_path).getroot()
    contact = root.find(".//body[@name='object']//flexcomp[@name='soft']/contact")
    if contact is None:
        return ""
    return contact.attrib.get("friction", "")


def write_variant_xml(
    base_xml_path: Path,
    solver: str,
    timestep: float,
    young: float,
    damping: float,
    run_tag: str,
) -> Path:
    tree = ET.parse(base_xml_path)
    root = tree.getroot()

    option = root.find("option")
    if option is None:
        raise ValueError("Could not find top-level <option> in XML")
    option.set("solver", solver)
    option.set("timestep", f"{timestep:.12g}")

    elasticity = root.find(".//body[@name='object']//flexcomp[@name='soft']/elasticity")
    if elasticity is None:
        raise ValueError("Could not find object flex elasticity block in XML")
    elasticity.set("young", f"{young:.12g}")
    elasticity.set("damping", f"{damping:.12g}")

    out_path = base_xml_path.parent / f"{base_xml_path.stem}.{run_tag}.xml"
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def worker_mode(args: argparse.Namespace) -> int:
    base_xml = Path(args.xml_path).expanduser().resolve()
    if not base_xml.is_file():
        raise FileNotFoundError(base_xml)

    run_tag = f"sweep_tmp_{os.getpid()}_{args.run_id:05d}"
    variant_xml = write_variant_xml(
        base_xml_path=base_xml,
        solver=args.solver,
        timestep=args.timestep,
        young=args.young,
        damping=args.damping,
        run_tag=run_tag,
    )

    result: dict[str, Any] = {
        "run_id": args.run_id,
        "solver": args.solver,
        "young": args.young,
        "damping": args.damping,
        "timestep": args.timestep,
        "drop_pos_x": args.drop_pos[0],
        "drop_pos_y": args.drop_pos[1],
        "drop_pos_z": args.drop_pos[2],
        "drop_quat_w": args.drop_quat[0],
        "drop_quat_x": args.drop_quat[1],
        "drop_quat_y": args.drop_quat[2],
        "drop_quat_z": args.drop_quat[3],
        "sim_time": args.sim_time,
        "return_code": 0,
        "exception_type": "",
        "exception_message": "",
        "time_reset_detected": False,
        "nonfinite_detected": False,
        "first_contact_step": -1,
        "first_contact_time": "",
        "first_contact_bodies": "",
        "max_ncon": 0,
        "final_time": "",
        "final_obj_x": "",
        "final_obj_y": "",
        "final_obj_z": "",
        "steps_planned": int(math.ceil(args.sim_time / args.timestep)),
        "steps_completed": 0,
        "variant_xml": str(variant_xml),
    }

    print(
        f"RUN_START run_id={args.run_id} solver={args.solver} young={args.young:.6e} "
        f"damping={args.damping:.6e} timestep={args.timestep:.6e} "
        f"drop_pos={args.drop_pos} sim_time={args.sim_time}"
    )

    try:
        model = mujoco.MjModel.from_xml_path(str(variant_xml))
        data = mujoco.MjData(model)

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        set_object_pose(model, data, args.drop_pos, args.drop_quat)

        total_steps = result["steps_planned"]
        prev_time = float(data.time)

        for step_idx in range(total_steps):
            mujoco.mj_step(model, data)
            result["steps_completed"] = step_idx + 1
            current_time = float(data.time)

            if current_time + 1e-12 < prev_time:
                result["time_reset_detected"] = True
                print(
                    f"TIME_RESET step={step_idx} prev_time={prev_time:.6f} "
                    f"current_time={current_time:.6f}"
                )
            prev_time = current_time

            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                result["nonfinite_detected"] = True
                raise FloatingPointError("Non-finite qpos/qvel detected")

            if data.ncon > result["max_ncon"]:
                result["max_ncon"] = int(data.ncon)

            in_contact, bodies = has_robot_object_contact(model, data)
            if in_contact and result["first_contact_step"] == -1:
                result["first_contact_step"] = step_idx
                result["first_contact_time"] = f"{current_time:.6f}"
                result["first_contact_bodies"] = str(bodies)
                print(
                    f"FIRST_CONTACT step={step_idx} time={current_time:.6f} "
                    f"bodies={bodies} ncon={data.ncon}"
                )

        obj_pos = object_position(model, data)
        result["final_time"] = f"{float(data.time):.6f}"
        result["final_obj_x"] = f"{obj_pos[0]:.6f}"
        result["final_obj_y"] = f"{obj_pos[1]:.6f}"
        result["final_obj_z"] = f"{obj_pos[2]:.6f}"

    except Exception as exc:
        result["return_code"] = 2
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        print(f"WORKER_EXCEPTION {type(exc).__name__}: {exc}")

    finally:
        try:
            variant_xml.unlink(missing_ok=True)
        except Exception:
            pass

    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    return int(result["return_code"])


def parse_result_from_output(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])
    raise ValueError("Could not find RESULT_JSON in worker output")


def scan_log_for_warnings(output: str) -> dict[str, Any]:
    lower = output.lower()
    unstable = "simulation is unstable" in lower or "nan, inf or huge value in qacc" in lower
    too_many_contacts = "too many contacts" in lower

    dof_match = re.search(r"QACC at DOF\s+(\d+)", output)
    unstable_dof = int(dof_match.group(1)) if dof_match else ""

    unstable_time = ""
    unstable_line_match = re.search(
        r"QACC at DOF\s+\d+.*?Time\s*=\s*([0-9.]+)",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if unstable_line_match:
        unstable_time = unstable_line_match.group(1)

    return {
        "unstable_warning": unstable,
        "unstable_dof": unstable_dof,
        "unstable_time": unstable_time,
        "too_many_contacts_warning": too_many_contacts,
    }


def build_values(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [float(start)]
    return [float(x) for x in np.geomspace(start, stop, count)]


def run_sweep(args: argparse.Namespace) -> int:
    base_xml = Path(args.xml_path).expanduser().resolve()
    if not base_xml.is_file():
        raise FileNotFoundError(base_xml)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path("generated") / f"sweep_unfrozen_hand_{timestamp}"
    )
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    young_values = build_values(args.young_min, args.young_max, args.young_count)
    damping_values = build_values(args.damping_min, args.damping_max, args.damping_count)
    timestep_values = build_values(args.timestep_min, args.timestep_max, args.timestep_count)
    solvers = list(args.solvers)

    fixed_friction = read_fixed_friction(base_xml)
    manifest = {
        "base_xml": str(base_xml),
        "fixed_friction": fixed_friction,
        "drop_pos": args.drop_pos,
        "drop_quat": args.drop_quat,
        "sim_time": args.sim_time,
        "solvers": solvers,
        "young_values": young_values,
        "damping_values": damping_values,
        "timestep_values": timestep_values,
        "total_runs": len(solvers) * len(young_values) * len(damping_values) * len(timestep_values),
        "notes": "Hand unfrozen. XML friction kept fixed from the input XML.",
    }
    (out_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    run_id = 0
    total_runs = manifest["total_runs"]

    for solver in solvers:
        for young in young_values:
            for damping in damping_values:
                for timestep in timestep_values:
                    run_id += 1
                    run_name = (
                        f"run_{run_id:05d}_solver-{solver}"
                        f"_young-{young:.3e}_damping-{damping:.3e}_dt-{timestep:.3e}"
                    )
                    log_path = logs_dir / f"{run_name}.log"

                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--run-id", str(run_id),
                        "--xml-path", str(base_xml),
                        "--solver", solver,
                        "--young", f"{young:.12g}",
                        "--damping", f"{damping:.12g}",
                        "--timestep", f"{timestep:.12g}",
                        "--sim-time", f"{args.sim_time:.12g}",
                        "--drop-pos", *(str(x) for x in args.drop_pos),
                        "--drop-quat", *(str(x) for x in args.drop_quat),
                    ]

                    print(
                        f"[{run_id:03d}/{total_runs}] solver={solver} young={young:.3e} "
                        f"damping={damping:.3e} dt={timestep:.3e}"
                    )

                    proc = subprocess.run(
                        cmd,
                        cwd=str(Path.cwd()),
                        capture_output=True,
                        text=True,
                    )
                    combined_output = proc.stdout
                    if proc.stderr:
                        combined_output += "\n[STDERR]\n" + proc.stderr

                    log_header = (
                        f"run_id={run_id}\n"
                        f"solver={solver}\n"
                        f"young={young:.12g}\n"
                        f"damping={damping:.12g}\n"
                        f"timestep={timestep:.12g}\n"
                        f"friction={fixed_friction}\n"
                        f"drop_pos={args.drop_pos}\n"
                        f"drop_quat={args.drop_quat}\n"
                        f"sim_time={args.sim_time}\n"
                        f"returncode={proc.returncode}\n\n"
                    )
                    log_path.write_text(log_header + combined_output, encoding="utf-8")

                    try:
                        row = parse_result_from_output(combined_output)
                    except Exception as exc:
                        row = {
                            "run_id": run_id,
                            "solver": solver,
                            "young": young,
                            "damping": damping,
                            "timestep": timestep,
                            "drop_pos_x": args.drop_pos[0],
                            "drop_pos_y": args.drop_pos[1],
                            "drop_pos_z": args.drop_pos[2],
                            "drop_quat_w": args.drop_quat[0],
                            "drop_quat_x": args.drop_quat[1],
                            "drop_quat_y": args.drop_quat[2],
                            "drop_quat_z": args.drop_quat[3],
                            "sim_time": args.sim_time,
                            "return_code": proc.returncode,
                            "exception_type": "ResultParseError",
                            "exception_message": str(exc),
                            "time_reset_detected": "",
                            "nonfinite_detected": "",
                            "first_contact_step": "",
                            "first_contact_time": "",
                            "first_contact_bodies": "",
                            "max_ncon": "",
                            "final_time": "",
                            "final_obj_x": "",
                            "final_obj_y": "",
                            "final_obj_z": "",
                            "steps_planned": "",
                            "steps_completed": "",
                            "variant_xml": "",
                        }

                    row["subprocess_returncode"] = proc.returncode
                    row["friction"] = fixed_friction
                    row["log_file"] = str(log_path.relative_to(out_dir))
                    row.update(scan_log_for_warnings(combined_output))
                    rows.append(row)

    csv_path = out_dir / "sweep_summary.csv"
    fieldnames = [
        "run_id",
        "solver",
        "young",
        "damping",
        "timestep",
        "friction",
        "drop_pos_x",
        "drop_pos_y",
        "drop_pos_z",
        "drop_quat_w",
        "drop_quat_x",
        "drop_quat_y",
        "drop_quat_z",
        "sim_time",
        "return_code",
        "subprocess_returncode",
        "unstable_warning",
        "unstable_dof",
        "unstable_time",
        "too_many_contacts_warning",
        "time_reset_detected",
        "nonfinite_detected",
        "first_contact_step",
        "first_contact_time",
        "first_contact_bodies",
        "max_ncon",
        "final_time",
        "final_obj_x",
        "final_obj_y",
        "final_obj_z",
        "steps_planned",
        "steps_completed",
        "exception_type",
        "exception_message",
        "log_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    print(f"\nSweep complete.")
    print(f"Output folder: {out_dir}")
    print(f"Summary CSV:   {csv_path}")
    print(f"Logs folder:   {logs_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep unfrozen-hand MuJoCo drop tests over young, damping, timestep, and solver."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument(
        "--xml-path",
        default="generated/block_16_0.4_0.3/manipulate_block_touch_sensors_16_0.4_0.3.xml",
        help="Path to the unfrozen-hand top-level XML",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Folder where logs and CSV will be written. Defaults to generated/sweep_unfrozen_hand_<timestamp>",
    )
    parser.add_argument("--sim-time", type=float, default=0.6, help="Simulation time per run in seconds")
    parser.add_argument(
        "--drop-pos",
        type=float,
        nargs=3,
        default=[1.0, 0.87, 0.2],
        help="Object drop position XYZ. Default reduces height to z=0.2.",
    )
    parser.add_argument(
        "--drop-quat",
        type=float,
        nargs=4,
        default=[1.0, 0.0, 0.0, 0.0],
        help="Object drop quaternion wxyz",
    )

    parser.add_argument("--young-min", type=float, default=1e6, help="Minimum Young's modulus")
    parser.add_argument("--young-max", type=float, default=2e11, help="Maximum Young's modulus")
    parser.add_argument("--young-count", type=int, default=6, help="Number of log-spaced Young values")

    parser.add_argument("--damping-min", type=float, default=1e-3, help="Minimum damping")
    parser.add_argument("--damping-max", type=float, default=1e-1, help="Maximum damping")
    parser.add_argument("--damping-count", type=int, default=5, help="Number of log-spaced damping values")

    parser.add_argument("--timestep-min", type=float, default=1e-4, help="Minimum timestep")
    parser.add_argument("--timestep-max", type=float, default=1e-2, help="Maximum timestep")
    parser.add_argument("--timestep-count", type=int, default=5, help="Number of log-spaced timestep values")

    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["CG", "Newton"],
        help="MuJoCo solvers to test",
    )

    parser.add_argument("--run-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--solver", type=str, default="CG", help=argparse.SUPPRESS)
    parser.add_argument("--young", type=float, default=1e6, help=argparse.SUPPRESS)
    parser.add_argument("--damping", type=float, default=1e-3, help=argparse.SUPPRESS)
    parser.add_argument("--timestep", type=float, default=1e-4, help=argparse.SUPPRESS)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker_mode(args)
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
