import mujoco

DOF = 27
model = mujoco.MjModel.from_xml_path("/home/mshashank02/ShadowHand-TQC/generated/block_16_0.4_0.3/manipulate_block_touch_sensors_16_0.4_0.3.xml")

def jnt_dof_count(jnt_type: int) -> int:
    # mujoco.mjtJoint enum values
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    if jnt_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        return 1
    raise ValueError(f"Unknown joint type: {jnt_type}")

jnt_id = None
for j in range(model.njnt):
    adr = int(model.jnt_dofadr[j])
    nd  = jnt_dof_count(int(model.jnt_type[j]))
    if adr <= DOF < adr + nd:
        jnt_id = j
        break

if jnt_id is None:
    print(f"DOF {DOF} not found in any joint. model.nv={model.nv}")
else:
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or f"joint_{jnt_id}"
    bid   = int(model.jnt_bodyid[jnt_id])
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
    jtype = int(model.jnt_type[jnt_id])

    print("DOF:", DOF)
    print("joint id:", jnt_id, "name:", jname)
    print("body  id:", bid,   "name:", bname)
    print("joint type:", mujoco.mjtJoint(jtype).name)
    print("jnt_dofadr:", int(model.jnt_dofadr[jnt_id]), "dof_count:", jnt_dof_count(jtype))
