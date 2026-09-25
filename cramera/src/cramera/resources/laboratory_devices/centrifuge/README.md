# AICOR centrifuge

**Principal author:** Vanessa Hassouna · **Institution:** AICOR Institute.
Original laboratory assets require prior written permission for reuse; see
[LICENSE.txt](LICENSE.txt) and [NOTICE.md](NOTICE.md). Keep these notices with
authorized copies. Existing permissions and third-party rights are preserved.

For authorized reuse, load `model.urdf` independently or copy the complete folder
into another project.
All visual files and artwork are included under `assets/`.

The root `laboratory_centrifuge_base` is at the centre of the instrument's support
surface. Coordinates are metres, with Z up and the front toward negative Y.
The lid joint retains its bounded opening angle and negative-X hinge axis.
The rotor retains its continuous vertical axis and four open sample sleeves.
Joint names and limits are contained in the URDF; the original laboratory
placement is recorded in `metadata.json`.

The fixed child link `laboratory_centrifuge_lid_handle` identifies the centre of
the grip bar on `laboratory_centrifuge_lid`. Its axes match the lid's axes and it
moves with the hinge. Its visual and collision geometry remains on the lid.

This model supplies geometry, contact shapes and articulation. The laboratory's
motor commands, sample clamps, manual knowledge and safety interlocks require its
simulation controller and are not implemented by the URDF alone.
