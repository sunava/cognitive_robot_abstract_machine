# Laboratory asset rights and attribution

The project's specified principal author is **Vanessa Hassouna**, with
**AICOR Institute** as the institution. Original laboratory asset reuse requires
prior written permission, subject to applicable law, existing permissions and
third-party rights. The operative scope is in each bundle's `LICENSE.txt` and
`NOTICE.md`; the local identifier is `LicenseRef-AICOR-Laboratory-Assets`.

This policy covers the original laboratory model data. CRAMERA's software stays
under its existing GPL-3.0-only terms; the repository and other packages retain
their existing terms. The separately obtained PR2 description, existing logo
artwork, fonts and other third-party material are excluded from the asset claim.
The package metadata identifies its combination of software and asset terms.

## Where the notices travel

- The assembled `resources/aicor_medical_lab/` bundle contains `LICENSE.txt`,
  `NOTICE.md` and `rights.json`.
- The device catalog and every independent `resources/laboratory_devices/<device>/`
  folder contain the same three documents.
- Generated scene and device JSON carry `assetRights` metadata.
- Generated URDF comments identify the author and licence documents.
- GLB metadata contains the copyright notice and asset terms without changing
  geometry or embedded images. Existing distinct copyright text is preserved.
- The laboratory controls display the credits and expandable rights information.

The canonical documents live in `cramera/tools/laboratory/asset_rights/`.
The scene-description, Blender and device generators propagate them on export.
`LaboratoryDeviceResource.copy_to()` copies the complete device directory,
including its notices. Authorized users should retain the full folder when
moving an instrument between projects.

## Provenance and limits

The models were developed procedurally for this project with AI-assisted
implementation and human design direction. A copyright label does not itself
establish protection or decide which person or institution holds particular
rights. In Germany, protection depends on human creative contributions; the
assessment for AI-assisted work depends on the individual case. See the
[DPMA explanation](https://www.dpma.de/service/schutzrechte_kurz_erklaert/geistigeseigentumdigital/index.html).

The asset README previously referred to GPL-3.0-only. On 2026-09-25 the owner
reported only private personal use so far and requested permission-only asset
terms. This update does not revoke any permission that was validly granted.
The [GNU licensing FAQ](https://www.gnu.org/licenses/gpl-faq.en.html#ReleaseUnderGPLAndNF)
distinguishes a rights holder's choice of licensing terms from recipients'
existing permissions. Git history and these notices record the policy change;
they are not a registration of legal rights.
