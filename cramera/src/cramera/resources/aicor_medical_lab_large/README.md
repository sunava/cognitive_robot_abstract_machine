# AICOR Medical Laboratory — expanded world

This seven-by-six-metre room preserves the original sample-preparation bench, its five instruments, calibrated object poses and existing robot task frames.

The rear workbench contains fifteen separate new instrument/accessory resources. The clear rear aisle is 1.75 metres wide. All new links are namespaced; the independent model packages remain in `devices/`.

`environment.urdf` is the composed CRAM environment. `scene.json` and `semantics.json` provide `medicalLab` station cameras, instrument poses, interaction frames and joint capabilities. `room/model.blend` edits the new architecture; `laboratory_large.blend`, when rendered, contains the assembled scene.

The pH meter's nested support bearing uses one explicit base/arm self-contact exclusion. Its joint represents the bearing motion; its mating visual surfaces are not intended as a detailed bearing contact simulation. External contacts remain enabled.

Instrument models provide geometry and articulated motion. Biological, chemical, thermal and optical outcomes require a separate controller or simulation. See each instrument package for its detailed scope and licence.
