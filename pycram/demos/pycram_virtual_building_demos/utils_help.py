from urdf_parser_py.urdf import URDF

kitchen = URDF.from_parameter_server("robot_description")
print(kitchen)
f = open("../../resources/robocanes.urdf", "w")
f.write(kitchen.to_xml_string())
