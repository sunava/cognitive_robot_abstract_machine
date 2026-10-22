"""
Conversion of a composed USD stage (an Isaac Sim twin) into a self-contained CRAMERA
scene bundle: per-link GLB meshes with physically based materials, a URDF built from the
UsdPhysics joints, and the ``scene.json`` sidecars the viewer reads.

Only :mod:`cramera.onboard.usd_scene.stage_reader` imports ``pxr``; every other module
works on the neutral description in :mod:`cramera.onboard.usd_scene.scene_description`.
"""
