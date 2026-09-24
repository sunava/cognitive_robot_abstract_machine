# AICOR centrifuge

Load `model.urdf` independently or copy this complete folder into another project.
All visual files and artwork are included under `assets/`.

The root `laboratory_centrifuge_base` is at the centre of the instrument's support
surface. Coordinates are metres, with Z up and the front toward negative Y.
The lid joint retains its bounded opening angle and negative-X hinge axis.
The rotor retains its continuous vertical axis and four open sample sleeves.
Joint names and limits are contained in the URDF; the original laboratory
placement is recorded in `metadata.json`.

This model supplies geometry, contact shapes and articulation. The laboratory's
motor commands, sample clamps, manual knowledge and safety interlocks require its
simulation controller and are not implemented by the URDF alone.
