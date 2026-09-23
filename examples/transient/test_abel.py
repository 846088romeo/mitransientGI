from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import mitsuba as mi

mi.set_variant("llvm_ad_rgb")

import mitransient as mitr

scene_path = Path(__file__).parent / "cornell-box" / "cbox_diffuse.xml"
scene = mi.load_file(str(scene_path))
data_steady, data_transient = mi.render(scene, spp=128)

print("Steady:", data_steady.shape)
print("Transient:", data_transient.shape)

transient = mitr.vis.tonemap_transient(data_transient)

mitr.vis.save_video("transient.mp4", transient, axis_video=2)
print("Saved transient video to transient.mp4")
# mitr.vis.show_video(transient, axis_video=2, normalize=True)