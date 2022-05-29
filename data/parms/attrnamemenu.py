# -*- coding: utf-8 -*-

node = kwargs["node"]
multiparm_index = kwargs["script_multiparm_index"]

geo = node.inputGeometry(0)
attr_type = node.parm("attr_{}_type".format(multiparm_index)).evalAsString()

if attr_type == "detail":
    attr_names = tuple(x.name() for x in geo.globalAttribs())
elif attr_type == "point":
    attr_names = tuple(x.name() for x in geo.pointAttribs())
elif attr_type == "primitive":
    attr_names = tuple(x.name() for x in geo.primAttribs())
elif attr_type == "vertex":
    attr_names = tuple(x.name() for x in geo.vertexAttribs())
else:
    raise RuntimeError("Unrecognized attribute type `{}`.".format(attr_type))

return tuple(y for x in attr_names for y in (x, x))
