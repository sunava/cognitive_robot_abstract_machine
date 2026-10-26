# cramera-bridge

Publishes a running CRAM world to the cramera viewer, records runs into scene bundles,
onboards demos, turns bundles back into CRAM worlds, answers questions in the entity
query language about live worlds and recorded scenes (`cramera_bridge.questions`, with
`cramera-record-answers`) and performs the plans composed in the plan builder
(`cramera_bridge.plans`). It is the only package that imports the CRAM
workspace, and needs it installed. See the repository README and
`docs/architecture.md`.
