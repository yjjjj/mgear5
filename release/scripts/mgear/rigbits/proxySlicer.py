"""
Rigbits proxy mesh slicer using maya.cmds.

This script creates proxy geometry for a skinned object by splitting its faces
into groups based on skinCluster influence weights.
"""

import datetime
import maya.cmds as cmds
import mgear

from mgear.core import applyop, node


def matchWorldTransform_cmds(source, target):
    """Match target's world transform to source using cmds.xform.

    Args:
        source (str): Source transform name.
        target (str): Target transform name.
    """
    m = cmds.xform(source, q=True, ws=True, m=True)
    cmds.xform(target, ws=True, m=m)


def slice(parent=False, oSel=None, *args):
    """Create a proxy geometry from a skinned object using maya.cmds.

    Args:
        parent (bool): If True, parent the proxy under the skin influence.
        oSel (str): Name of the object to process. If not provided, the first
            selected object is used.
        *args: Additional arguments (ignored).

    Raises:
        RuntimeError: If no object is selected or if necessary nodes are missing.
    """
    start_time = datetime.datetime.now()
    print(oSel)
    if not oSel:
        sels = cmds.ls(sl=True, long=True)
        if not sels:
            cmds.error("No object selected.")
        oSel = sels[0]
        print("----\n{}".format(oSel))

    # Get skinCluster influences.
    infs = cmds.skinCluster(oSel, q=True, inf=True)
    n_faces = cmds.polyEvaluate(oSel, face=True)
    # Get face components as strings.
    o_faces = cmds.ls(oSel + ".f[*]", flatten=True)
    face_groups = [[] for _ in infs]

    # Get the shape node and skinCluster.
    shapes = cmds.listRelatives(oSel, s=True, fullPath=True)
    if not shapes:
        cmds.error("No shape node found for '{}'.".format(oSel))
    s_cluster = cmds.listConnections(shapes[0], type="skinCluster")
    if not s_cluster:
        cmds.error("No skinCluster found on '{}'.".format(shapes[0]))
    sC_name = s_cluster[0]

    # Process each face.
    for i_face in range(n_faces):
        face = o_faces[i_face]
        info = cmds.polyInfo(face, faceToVertex=True)
        if info:
            # polyInfo returns a string like: "Face 0: 0 1 2 3"
            vtxs = list(map(int, info[0].split()[2:]))
        else:
            vtxs = []
        o_sum = None
        for i_vtx in vtxs:
            vtx = "{}.vtx[{}]".format(oSel, i_vtx)
            vals = cmds.skinPercent(sC_name, vtx, q=True, v=True)
            if o_sum is None:
                o_sum = vals
            else:
                o_sum = [a + b for a, b in zip(o_sum, vals)]
        if o_sum:
            max_idx = o_sum.index(max(o_sum))
            print("adding face: {} to group in: {}"
                  .format(i_face, infs[max_idx]))
            face_groups[max_idx].append(i_face)

    original = oSel

    # Create a parent group if needed.
    if not parent:
        if cmds.objExists("ProxyGeo"):
            parent_grp = "ProxyGeo"
        else:
            parent_grp = cmds.createNode("transform", name="ProxyGeo")
    if cmds.objExists("rig_proxyGeo_grp"):
        proxy_set = "rig_proxyGeo_grp"
    else:
        proxy_set = cmds.sets(empty=True, name="rig_proxyGeo_grp")

    # Process each face group.
    for idx, bone_list in enumerate(face_groups):
        if bone_list:
            proxy_name = infs[idx] + "_Proxy"
            dup = cmds.duplicate(original, rr=True, name=proxy_name)
            new_obj = dup[0]
            # Unlock transform attributes.
            for attr in ("translateX", "translateY", "translateZ",
                         "rotateX", "rotateY", "rotateZ",
                         "scaleX", "scaleY", "scaleZ"):
                cmds.setAttr(new_obj + "." + attr, lock=False)
            new_faces = cmds.ls(new_obj + ".f[*]", flatten=True)
            faces_to_del = list(new_faces)
            for face_idx in sorted(bone_list, reverse=True):
                if face_idx < len(faces_to_del):
                    faces_to_del.pop(face_idx)
            cmds.delete(faces_to_del)
            if parent:
                cmds.parent(new_obj, infs[idx])
            else:
                # Reparent proxy under the designated group (removes it from the
                # scene root if necessary)
                cmds.parent(new_obj, parent_grp)
                # Duplicate new_obj to transfer its shape cleanly.
                dummy = cmds.duplicate(new_obj, rr=True)[0]
                # Remove existing shapes from new_obj.
                children = cmds.listRelatives(new_obj, s=True, fullPath=True)
                if children:
                    cmds.delete(children)
                # Match world transform.
                matchWorldTransform_cmds(infs[idx], new_obj)
                dummy_shapes = cmds.listRelatives(dummy, s=True, fullPath=True)
                if dummy_shapes:
                    cmds.parent(dummy_shapes[0], new_obj, shape=True)
                cmds.delete(dummy)
                new_shape = cmds.listRelatives(new_obj, s=True, fullPath=True)
                if new_shape:
                    cmds.rename(new_shape[0], new_obj + "_offset")
                mulmat_node = applyop.gear_mulmatrix_op(
                    infs[idx] + ".worldMatrix",
                    new_obj + ".parentInverseMatrix")
                out_plug = mulmat_node + ".output"
                dm_node = node.createDecomposeMatrixNode(out_plug)
                cmds.connectAttr(dm_node + ".outputTranslate",
                                 new_obj + ".translate")
                cmds.connectAttr(dm_node + ".outputRotate",
                                 new_obj + ".rotate")
                cmds.connectAttr(dm_node + ".outputScale",
                                 new_obj + ".scale")
            print("Creating proxy for: {}".format(infs[idx]))
            cmds.sets(new_obj, addElement=proxy_set)

    end_time = datetime.datetime.now()
    final_time = end_time - start_time
    mgear.log("=============== Slicing for: {} finish ======= [ {} ] ==="
              .format(oSel, str(final_time)))
