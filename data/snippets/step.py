# -*- coding: utf-8 -*-

import warp

# Define the input and output geometries. As per the Solver SOP node that is
# responsible for evaluating this snippet, the input geometry corresponds to
# the geometry from the previous frame, as to enable building simulations.
node = hou.pwd()
input_geo = node.inputGeometry(1)
output_geo = node.geometry()

# Retrieve the HDA node and its Python module.
hda_node = hou.node(hou.chsop("hdanode"))
hda_module = hda_node.hdaModule()

# Retrieve the context attached to this node instance, such as the parameter
# values, and the kernel function to run. This context is initialized upon
# cooking the node for the first time, or just after hitting the reset button.
context = hda_module.get_context(hda_node, input_geo)

# Prepare the time information to pass to the kernel function. Note that
# querying the current frame (or time) in Houdini is important to ensure that
# the node is time-dependent and is hence being recooked at each time step.
frame_rate = hou.fps()
time = (hou.frame() - context.start_frame) / frame_rate
time_step = 1.0 / (frame_rate * context.substeps)

# Query the input geometry by fetching the values for the requested attributes.
# Also pass the output geometry to make sure that the array sizes conform to
# the expected sizes of the corresponding attributes at the destination.
arrays = hda_module.get_attribute_value_arrays(
    hda_node,
    input_geo,
    output_geo,
    context.attr_descs,
    context.kernel_device,
)

# Define how many threads to use when evaluating the kernel, which basically
# corresponds to the number of entities that we want to iterate over.
kernel_dim = hda_module.get_kernel_dim(hda_node, output_geo)

# Prepare the arguments to pass to the kernel function. Their order matches how
# the attributes are defined in the HDA's parameters. The time and time step
# are always appended to the list of arguments.
kernel_inputs = list(arrays) + [time, time_step]

# Run the kernel function.
warp.launch(
    context.kernel_fn,
    dim=kernel_dim,
    inputs=kernel_inputs,
    device=context.kernel_device,
)

# Update the geometry with the resulting attribute values.
hda_module.set_attribute_value_arrays(
    hda_node,
    output_geo,
    context.attr_descs,
    arrays,
)
