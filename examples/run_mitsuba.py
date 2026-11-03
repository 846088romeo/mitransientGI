import mitsuba as mi

mi.set_variant('llvm_ad_rgb')
scene = mi.load_file('scenes/letter_T_big.xml')
image = mi.render(scene, spp=512)
mi.util.write_bitmap('output_volpath.png', image)