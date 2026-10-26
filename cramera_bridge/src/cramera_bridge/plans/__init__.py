"""
Plans composed in the cramera plan builder, as the CRAM workspace performs them: the
catalog of robots, objects, surfaces and native actions the builder offers, and the
robot task demo a generated demo runs.

The plan builder imports nothing of the CRAM workspace; it reads the catalog through
:class:`~cramera_bridge.plans.workspace.PlanWorkspace` as protocol documents, and a
generated demo hands its :class:`~cramera_protocol.plan_builder.DemoDocument` to
:class:`~cramera_bridge.plans.command.PlanDemoCommand`.
"""
