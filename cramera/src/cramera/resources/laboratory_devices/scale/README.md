# AICOR scale

**Principal author:** Vanessa Hassouna · **Institution:** AICOR Institute.
Original laboratory assets require prior written permission for reuse; see
[LICENSE.txt](LICENSE.txt) and [NOTICE.md](NOTICE.md). Keep these notices with
authorized copies. Existing permissions and third-party rights are preserved.

For authorized reuse, load `model.urdf` independently or copy the complete folder
into another project.
All visual files and artwork are included under `assets/`.

The root `laboratory_balance` is at the centre of the weighing pan, rather than at
the table surface. The housing bottom lies 0.066 m below this root. Coordinates
are metres, with Z up and the front toward negative Y. The URDF retains separate
weighing-pan, housing and open sample-support contact primitives. The original
laboratory placement is recorded in `metadata.json`.

The visual contains the housing, pan, display and open sample-support collar.
The collar's visual segments match the contact geometry, so it remains visible
without the browser. Weight readings, tare, settling detection and the live display require the
laboratory simulation controller.
