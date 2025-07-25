from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import yaml
import omni.kit.commands
from isaacsim.asset.importer.urdf import _urdf

print(dir(_urdf.ImportConfig()))

config_path = "source/isaaclab_assets/data/g1_rubber_hand/config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

import_config = _urdf.ImportConfig()
import_config.make_instanceable       = cfg.get("make_instanceable", False)
import_config.import_inertia_tensor   = cfg.get("import_inertia_tensor", False)
import_config.fix_base                = cfg.get("fix_base", False)
import_config.merge_fixed_joints      = cfg.get("merge_fixed_joints", False)
import_config.self_collision          = cfg.get("self_collision", False)
import_config.default_drive_type      = cfg.get("default_drive_type", "none")
import_config.override_joint_dynamics = cfg.get("override_joint_dynamics", False)
import_config.default_drive_stiffness = cfg.get("default_drive_stiffness", 0.0)
import_config.default_drive_damping   = cfg.get("default_drive_damping", 0.0)
import_config.link_density            = cfg.get("link_density", 0.0)
import_config.convex_decomp           = cfg.get("convex_decompose_mesh", False)

urdf_path = "source/isaaclab_assets/data/g1_UDRF/g1_29dof_rev_1_0_rubber_hand.urdf"
usd_out   = "source/isaaclab_assets/data/g1_rubber_hand/g1_rubber_hand.usd"
print(f"Converting {urdf_path} → {usd_out}")

result, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=import_config,
    dest_path=usd_out
)

print(f"URDF → USD 변환 완료: {usd_out}")
simulation_app.close()
