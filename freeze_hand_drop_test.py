#!/usr/bin/env python3
import argparse
import os
import sys
import time

import numpy as np
import mujoco


def joint_qpos_count(jnt_type: int) -> int:
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    if jnt_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        return 1
    raise ValueError(f"Unknown joint type: {jnt_type}")


def joint_dof_count(jnt_type: int) -> int:
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    if jnt_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        return 1
    raise ValueError(f"Unknown joint type: {jnt_type}")


def collect_hand_joint_indices(model: mujoco.MjModel):
    qpos_idx = []
    qvel_idx = []
    hand_joint_names = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not name.startswith("robot0:"):
            continue
        jtype = int(model.jnt_type[j])
        qpos_adr = int(model.jnt_qposadr[j])
        qvel_adr = int(model.jnt_dofadr[j])
        qpos_idx.extend(range(qpos_adr, qpos_adr + joint_qpos_count(jtype)))
        qvel_idx.extend(range(qvel_adr, qvel_adr + joint_dof_count(jtype)))
        hand_joint_names.append(name)
    return np.array(qpos_idx, dtype=np.int32), np.array(qvel_idx, dtype=np.int32), hand_joint_names


def compute_hold_ctrl(model: mujoco.MjModel, data: mujoco.MjData):
    hold = np.array(data.ctrl, copy=True)
    ctrlrange = model.actuator_ctrlrange
    for act_id in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id) or ""
        if not aname.startswith("robot0:A_"):
            continue
        jname = aname.replace(":A_", ":")
        try:
            qpos = data.joint(jname).qpos
        except Exception:
            continue
        qpos_arr = np.asarray(qpos)
        if qpos_arr.size != 1:
            continue
        hold[act_id] = float(qpos_arr.reshape(-1)[0])
    if model.nu > 0:
        hold = np.clip(hold, ctrlrange[:, 0], ctrlrange[:, 1])
    return hold


def set_object_pose(model: mujoco.MjModel, data: mujoco.MjData, pos, quat):
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


def object_position(model: mujoco.MjModel, data: mujoco.MjData):
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
    qpos_adr = int(model.jnt_qposadr[jnt_id])
    return np.array(data.qpos[qpos_adr:qpos_adr + 3], copy=True)


def has_robot_object_contact(model: mujoco.MjModel, data: mujoco.MjData):
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        if g1 < 0 or g2 < 0:
            continue
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
        if (n1.startswith("robot0:") and n2 == "object") or (n2.startswith("robot0:") and n1 == "object"):
            return True, (n1, n2)
    return False, None


def parse_args():
    p = argparse.ArgumentParser(description="Freeze ShadowHand joints, drop object, then unfreeze.")
    p.add_argument(
        "--xml-path",
        default="generated/block_16_0.4_0.3/manipulate_block_touch_sensors_16_0.4_0.3.xml",
        help="Path to MuJoCo XML",
    )
    p.add_argument("--freeze-steps", type=int, default=400, help="Physics steps to keep hand frozen")
    p.add_argument("--unfreeze-steps", type=int, default=2000, help="Physics steps after release")
    p.add_argument("--drop-pos", type=float, nargs=3, default=[1.0, 0.87, 0.20], help="Object XYZ to start from")
    p.add_argument("--drop-quat", type=float, nargs=4, default=[1.0, 0.0, 0.0, 0.0], help="Object quaternion wxyz")
    p.add_argument("--print-every", type=int, default=50, help="Print status every N steps")
    p.add_argument("--viewer", action="store_true", help="Open MuJoCo passive viewer")
    p.add_argument("--realtime", action="store_true", help="Sleep to roughly match simulation time (viewer mode)")
    return p.parse_args()


def main():
    args = parse_args()
    xml_path = os.path.abspath(args.xml_path)
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    hand_qpos_idx, hand_qvel_idx, hand_joint_names = collect_hand_joint_indices(model)
    if hand_qpos_idx.size == 0:
        print("No hand joints found (expected names starting with 'robot0:').")
        return 1

    set_object_pose(model, data, args.drop_pos, args.drop_quat)
    locked_qpos = data.qpos[hand_qpos_idx].copy()
    hold_ctrl = compute_hold_ctrl(model, data)

    total_steps = args.freeze_steps + args.unfreeze_steps
    first_contact_step = None
    release_step = args.freeze_steps
    phase = "frozen"

    print(f"XML: {xml_path}")
    print(f"dt: {model.opt.timestep}")
    print(f"hand joints: {len(hand_joint_names)}")
    print(f"freeze_steps={args.freeze_steps}, unfreeze_steps={args.unfreeze_steps}, total={total_steps}")
    print(f"drop_pos={args.drop_pos}, drop_quat={args.drop_quat}")

    viewer_ctx = None
    if args.viewer:
        try:
            from mujoco import viewer as mj_viewer
            viewer_ctx = mj_viewer.launch_passive(model, data)
        except Exception as e:
            print(f"Viewer unavailable, continuing headless: {e}")
            viewer_ctx = None

    def one_step(step_idx: int):
        nonlocal phase, first_contact_step

        if model.nu > 0:
            data.ctrl[:] = hold_ctrl

        mujoco.mj_step(model, data)

        if step_idx < args.freeze_steps:
            data.qpos[hand_qpos_idx] = locked_qpos
            data.qvel[hand_qvel_idx] = 0.0
            mujoco.mj_forward(model, data)
        elif phase == "frozen":
            phase = "unfrozen"
            print(f"[step {step_idx}] hand released (t={data.time:.6f}s)")

        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise FloatingPointError("Non-finite qpos/qvel detected")

        in_contact, bodies = has_robot_object_contact(model, data)
        if in_contact and first_contact_step is None:
            first_contact_step = step_idx
            print(f"[step {step_idx}] first robot-object contact at t={data.time:.6f}s bodies={bodies}")

        if args.print_every > 0 and (step_idx % args.print_every == 0 or step_idx == total_steps - 1):
            opos = object_position(model, data)
            print(
                f"[step {step_idx:5d}] phase={phase:8s} t={data.time:8.4f}s "
                f"obj=({opos[0]: .4f}, {opos[1]: .4f}, {opos[2]: .4f}) "
                f"ncon={data.ncon}"
            )

        if viewer_ctx is not None:
            try:
                viewer_ctx.sync()
                if args.realtime:
                    time.sleep(model.opt.timestep)
            except Exception:
                pass

    try:
        if viewer_ctx is not None:
            while viewer_ctx.is_running() and total_steps > 0:
                for step_idx in range(total_steps):
                    if not viewer_ctx.is_running():
                        break
                    one_step(step_idx)
                break
        else:
            for step_idx in range(total_steps):
                one_step(step_idx)

    except Exception as e:
        crash_phase = "frozen" if step_idx < release_step else "unfrozen"
        print(f"\nSimulation error at step {step_idx} ({crash_phase} phase): {type(e).__name__}: {e}")
        return 2
    finally:
        if viewer_ctx is not None:
            try:
                viewer_ctx.close()
            except Exception:
                pass

    print("\nCompleted without Python-level MuJoCo exception.")
    if first_contact_step is None:
        print("No robot-object contact detected during the run.")
    else:
        when = "before release" if first_contact_step < release_step else "after release"
        print(f"First robot-object contact step: {first_contact_step} ({when})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
