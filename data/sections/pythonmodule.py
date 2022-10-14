# -*- coding: utf-8 -*-

"""Module accessible to all the node instances of the Warp HDA.

Since the same module object is reused across all the node instances, special
care must be taken when caching data associated with a specific node instance.


Notes
-----
Do not attempt to retrieve the geometry object of a node instance from this
module or else it will break Houdini's evaluation. Always request a geometry
instance to be passed as an argument if needed.
"""

import sys

if sys.version_info.major < 3:
    raise RuntimeError(
        "This node requires to be run within a Python 3 environment."
    )

import enum
import importlib.util
import os
import tempfile

from typing import (
    Any,
    Callable,
    NamedTuple,
    Sequence,
    Tuple,
)

import hou
import numpy
import warp


# We need to initialize Warp's runtime globally before any of the node
# instances can start using it.
warp.init()


class Access(enum.IntFlag):
    """Access flags."""

    Read = 1
    Write = 2


class AttributeDesc(NamedTuple):
    """Geometry attribute description."""

    name: str
    type: hou.attribType
    data_type: hou.attribData
    size: int
    access: Access

    def get_instance(self, geo: hou.Geometry) -> hou.Attrib:
        """Retrieve a geometry attribute from its description."""
        if self.type == hou.attribType.Global:
            return geo.findGlobalAttrib(self.name)

        if self.type == hou.attribType.Point:
            return geo.findPointAttrib(self.name)

        if self.type == hou.attribType.Prim:
            return geo.findPrimAttrib(self.name)

        if self.type == hou.attribType.Vertex:
            return geo.findVertexAttrib(self.name)

        # This should be unreachable.
        assert False

    def get_count(self, geo: hou.Geometry) -> hou.Attrib:
        """Retrieve the number of values for that attribute."""
        if self.type == hou.attribType.Global:
            return 1

        if self.type == hou.attribType.Point:
            return geo.intrinsicValue("pointcount")

        if self.type == hou.attribType.Prim:
            return geo.intrinsicValue("primitivecount")

        if self.type == hou.attribType.Vertex:
            return geo.intrinsicValue("vertexcount")

        # This should be unreachable.
        assert False

    def get_value_array(self, geo: hou.Geometry) -> numpy.array:
        """Retrieve the attribute values from the given geometry."""
        if self.type == hou.attribType.Global:
            if self.size == 1:
                values = (geo.attribValue(self.name),)
            else:
                values = geo.attribValue(self.name)

            if self.data_type == hou.attribData.Int:
                dtype = numpy.int32
            elif self.data_type == hou.attribData.Float:
                dtype = numpy.float32
            else:
                raise RuntimeError(
                    "Unsupported data type for attribute `{}`."
                    .format(self.name)
                )

            array = numpy.array(values, dtype=dtype)
        elif self.type == hou.attribType.Point:
            if self.data_type == hou.attribData.Int:
                values = geo.pointIntAttribValuesAsString(
                    self.name,
                    int_type=hou.numericData.Int32,
                )
                array = numpy.frombuffer(values, dtype=numpy.int32)
            elif self.data_type == hou.attribData.Float:
                values = geo.pointFloatAttribValuesAsString(
                    self.name,
                    float_type=hou.numericData.Float32,
                )
                array = numpy.frombuffer(values, dtype=numpy.float32)
            else:
                raise RuntimeError(
                    "Unsupported data type for attribute `{}`."
                    .format(self.name)
                )
        elif self.type == hou.attribType.Prim:
            if self.data_type == hou.attribData.Int:
                values = geo.primIntAttribValuesAsString(
                    self.name,
                    int_type=hou.numericData.Int32,
                )
                array = numpy.frombuffer(values, dtype=numpy.int32)
            elif self.data_type == hou.attribData.Float:
                values = geo.primFloatAttribValuesAsString(
                    self.name,
                    float_type=hou.numericData.Float32,
                )
                array = numpy.frombuffer(values, dtype=numpy.float32)
            else:
                raise RuntimeError(
                    "Unsupported data type for attribute `{}`."
                    .format(self.name)
                )
        elif self.type == hou.attribType.Vertex:
            if self.data_type == hou.attribData.Int:
                values = geo.vertexIntAttribValuesAsString(
                    self.name,
                    int_type=hou.numericData.Int32,
                )
                array = numpy.frombuffer(values, dtype=numpy.int32)
            elif self.data_type == hou.attribData.Float:
                values = geo.vertexFloatAttribValuesAsString(
                    self.name,
                    float_type=hou.numericData.Float32,
                )
                array = numpy.frombuffer(values, dtype=numpy.float32)
            else:
                raise RuntimeError(
                    "Unsupported data type for attribute `{}`."
                    .format(self.name)
                )
        else:
            # This should be unreachable.
            assert False

        return array.reshape(-1, self.size)

    def set_value_array(self, geo: hou.Geometry, array: numpy.array) -> None:
        """Set the attribute values onto the given geometry."""
        if self.type == hou.attribType.Global:
            geo.setGlobalAttribValue(self.name, array.tolist())
        elif self.type == hou.attribType.Point:
            if self.data_type == hou.attribData.Int:
                geo.setPointIntAttribValuesFromString(
                    self.name,
                    array,
                    int_type=hou.numericData.Int32,
                )
            elif self.data_type == hou.attribData.Float:
                geo.setPointFloatAttribValuesFromString(
                    self.name,
                    array,
                    float_type=hou.numericData.Float32,
                )
            else:
                raise RuntimeError(
                    "Unsupported data type for attribute `{}`."
                    .format(self.name)
                )
        elif self.type == hou.attribType.Prim:
            if self.data_type == hou.attribData.Int:
                geo.setPrimIntAttribValuesFromString(
                    self.name,
                    array,
                    int_type=hou.numericData.Int32,
                )
            elif self.data_type == hou.attribData.Float:
                geo.setPrimFloatAttribValuesFromString(
                    self.name,
                    array,
                    float_type=hou.numericData.Float32,
                )
            else:
                raise RuntimeError(
                    "Unsupported data type for attribute `{}`."
                    .format(self.name)
                )
        elif self.type == hou.attribType.Vertex:
            if self.data_type == hou.attribData.Int:
                geo.setVertexIntAttribValuesFromString(
                    self.name,
                    array,
                    int_type=hou.numericData.Int32,
                )
            elif self.data_type == hou.attribData.Float:
                geo.setVertexFloatAttribValuesFromString(
                    self.name,
                    array,
                    float_type=hou.numericData.Float32,
                )
            else:
                raise RuntimeError(
                    "Unsupported data type for attribute `{}`."
                    .format(self.name)
                )
        else:
            # This should be unreachable.
            assert False

    def get_warp_dtype(self) -> Any:
        """Retrieve the corresponding Warp dtype."""
        dtype = _ATTR_TO_WARP_DTYPE.get((self.data_type, self.size), None)
        if dtype is None:
            raise RuntimeError(
                "Unsupported data type for attribute `{}`."
                .format(self.name)
            )

        return dtype


class Context(NamedTuple):
    """Context data stored for each node instance of this HDA.

    It's initialized upon cooking a node instance for the first time, or just
    after hitting the reset button for that node.
    """

    start_frame: int
    substeps: int
    attr_descs: Tuple[AttributeDesc, ...]
    kernel_fn: Callable
    kernel_device: str


#   Helpers
# ------------------------------------------------------------------------------

_ARRAY_CACHE = {}
_CONTEXT_CACHE = {}


_ATTR_TO_WARP_DTYPE = {
    (hou.attribData.Int, 1): warp.int32,
    (hou.attribData.Float, 1): warp.float32,
    (hou.attribData.Float, 2): warp.vec2,
    (hou.attribData.Float, 3): warp.vec3,
    (hou.attribData.Float, 4): warp.vec4,
    (hou.attribData.Float, 9): warp.mat33,
    (hou.attribData.Float, 16): warp.mat44,
}

_WARP_DTYPE_TO_REPR = {
    warp.int32: "int32",
    warp.float32: "float32",
    warp.vec2: "vec2",
    warp.vec3: "vec3",
    warp.vec4: "vec4",
    warp.mat33: "mat33",
    warp.mat44: "mat44",
}

assert all(x in _WARP_DTYPE_TO_REPR for x in _ATTR_TO_WARP_DTYPE.values())


_CODE_TEMPLATE = """# -*- coding: utf-8 -*-

import warp as wp

@wp.kernel
def run(
{}
):
    tid = wp.tid()

"""


def _load_kernel_module(file_path: str, node_id: int) -> Any:
    """Load the Python module containing the Warp kernel."""
    module_name = "nvidia-warp-{}".format(node_id)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get_kernel_fn(hda_node: hou.Node) -> Callable:
    """Retrieve the Warp kernel function to run."""
    source = hda_node.parm("kernel_source").evalAsString()
    node_id = hda_node.sessionId()
    if source == "embedded":
        code = hda_node.evalParm("kernel_source_embedded")

        # Create a temporary file.
        file, file_path = tempfile.mkstemp(suffix=".py", text=True)
        os.close(file)

        try:
            # Save the embedded code in the temporary file.
            with open(file_path, "w") as f:
                f.write(code)

            # Import the temporary file as a Python module.
            module = _load_kernel_module(file_path, node_id)
        finally:
            # The resulting Python module is stored in memory so it's safe to
            # clean-up the temporary file now.
            os.remove(file_path)
    elif source == "file":
        file_path = hda_node.evalParm("kernel_source_file")

        # Import the given file as a Python module.
        module = _load_kernel_module(file_path, node_id)
    else:
        raise RuntimeError("Unrecognized kernel source `{}`.".format(source))

    if not hasattr(module, "run"):
        raise RuntimeError(
            "The kernel code is expected to define a function named `run`."
        )

    return module.run


def _get_attribute_descs(
    hda_node: hou.Node,
    geo: hou.Geometry,
) -> Tuple[AttributeDesc, ...]:
    """Retrieve the attribute descriptions from the HDA's parameters."""
    attr_descs = []
    count = hda_node.evalParm("attr_count")
    for i in range(1, count + 1):
        name = hda_node.evalParm("attr_{}_name".format(i))
        type = hda_node.parm("attr_{}_type".format(i)).evalAsString()
        access = hda_node.parm("attr_{}_access".format(i)).evalAsString()

        if type == "detail":
            attr = geo.findGlobalAttrib(name)
        elif type == "point":
            attr = geo.findPointAttrib(name)
        elif type == "primitive":
            attr = geo.findPrimAttrib(name)
        elif type == "vertex":
            attr = geo.findVertexAttrib(name)
        else:
            raise RuntimeError(
                "Unrecognized attribute type `{}`.".format(type)
            )

        if attr is None:
            raise RuntimeError(
                "The {} attribute `{}` could not be found.".format(type, name)
            )

        if access == "read":
            access = Access.Read
        elif access == "write":
            access = Access.Write
        elif access == "readwrite":
            access = Access.Read | Access.Write
        else:
            raise RuntimeError(
                "Unrecognized attribute access `{}`.".format(access)
            )

        attr_descs.append(
            AttributeDesc(
                name=name,
                type=attr.type(),
                data_type=attr.dataType(),
                size=attr.size(),
                access=access,
            )
        )

    return tuple(attr_descs)


#   UI Events
# ------------------------------------------------------------------------------


def handle_reset_pressed(hda_node: hou.Node) -> None:
    """Reset the state of the simulation."""
    clear_caches(hda_node)

    # Reset the simulation state of the SOP solver.
    hda_node.parm("./solver/resimulate").pressButton()


def handle_kernel_source_generate_pressed(
    hda_node: hou.Node,
    geo: hou.Geometry,
) -> None:
    """Generate the embedded kernel code from the attributes."""
    # Declare the kernel's function parameters based on the attributes defined
    # as parameters.
    params = []
    attr_descs = _get_attribute_descs(hda_node, geo)
    for attr_desc in attr_descs:
        warp_dtype = attr_desc.get_warp_dtype()
        warp_dtype_repr = _WARP_DTYPE_TO_REPR[warp_dtype]
        params.append(
            "    {}_array: wp.array(dtype=wp.{}),"
            .format(attr_desc.name, warp_dtype_repr)
        )

    # Append the time-related parameters.
    params.append("    time: wp.float32,")
    params.append("    time_step: wp.float32,")

    # Format the kernel's code.
    code = _CODE_TEMPLATE.format("\n".join(params))

    # Overwrite the parameter value with the generated code.
    hda_node.setParms(
        {
            "kernel_source_embedded": code,
        }
    )


#   Public API
# ------------------------------------------------------------------------------


def get_kernel_dim(hda_node: hou.Node, geo: hou.Geometry) -> int:
    """Retrieve the numer of time to run the kernel at each time step."""
    run_over = hda_node.parm("runover").evalAsString()
    if run_over == "detail":
        return 1

    if run_over == "points":
        return geo.intrinsicValue("pointcount")

    if run_over == "primitives":
        return geo.intrinsicValue("primitivecount")

    if run_over == "vertices":
        return geo.intrinsicValue("vertexcount")

    if run_over == "numbers":
        return hda_node.evalParm("runover_numbercount")

    raise RuntimeError("Unrecognized run over value `{}`.".format(run_over))


def get_context(hda_node: hou.Node, geo: hou.Geometry) -> Context:
    """Retrieve the context associated with the given node instance.

    The data is retrieved from the cache however, if no cached data is found,
    a new instance is initialized.
    """
    node_id = hda_node.sessionId()
    context = _CONTEXT_CACHE.get(node_id, None)

    if context is not None:
        return context

    kernel_fn = _get_kernel_fn(hda_node)
    kernel_device = hda_node.parm("device").evalAsString()
    attr_descs = _get_attribute_descs(hda_node, geo)

    context = Context(
        start_frame=hda_node.evalParm("startframe"),
        substeps=hda_node.evalParm("substeps"),
        attr_descs=attr_descs,
        kernel_fn=kernel_fn,
        kernel_device=kernel_device,
    )
    _CONTEXT_CACHE[node_id] = context
    return context


def get_attribute_value_arrays(
    hda_node: hou.Node,
    src_geo: hou.Geometry,
    dst_geo: hou.Geometry,
    attr_descs: Sequence[AttributeDesc],
    kernel_device: str,
) -> Tuple[warp.array, ...]:
    """Retrieve the array values for the given attributes from a geometry."""
    node_id = hda_node.sessionId()
    array_cache = _ARRAY_CACHE.get(node_id, {})

    arrays = []
    for attr_desc in attr_descs:
        # Retrieve the cached array, if any.
        array = array_cache.get(attr_desc.name, None)

        # Retrieve the number of components currently stored in the cached
        # array as well as the desired target count.
        current_count = None if array is None else len(array)
        target_count = attr_desc.get_count(dst_geo)

        # Retrieve the corresponding Warp dtype.
        warp_dtype = attr_desc.get_warp_dtype()

        # Prepare the value array for the attribute.
        # When the attribute is readable, we always want to query its values
        # from the geometry to pick up any possible update. And when it's
        # writable, we need to make sure that we own a copy instead of
        # having it referencing Houdini's internal buffer.

        if attr_desc.access == Access.Write and target_count == current_count:
            # Reuse the cached array as-is. We don't need to bother about
            # updating its values since they shouldn't be read by the solver.
            arrays.append(array)
            continue

        if attr_desc.access == Access.Read:
            # Since this attribute is read-only, we can directly take
            # the values from the output geometry because:
            # - the resulting array is guaranteed to have the right size.
            # - we don't need to build upon the values from the previous step,
            # like a simulation would, since its values are not supposed to
            # ever be updated by the solver.
            array = attr_desc.get_value_array(dst_geo)
            copy = False
        elif attr_desc.access == Access.Write:
            # When the attribute is set to write-only, we only need to query
            # the latest values from the geometry if we don't have any cached
            # array or that its size needs to change, which is the case here
            # since we already handled the special case where the size doesn't
            # need to change.
            # We could very well create an uninitialized array but retrieving
            # it from Houdini is more convenient since it already comes with
            # the right buffer size and it's plenty fast (no copy is made yet).
            array = attr_desc.get_value_array(dst_geo)
            copy = True
        elif current_count is None or target_count > current_count:
            # With a readable attribute that has its number of components
            # increased, we still query the input geometry for its values at
            # the previous step but we also force the desired shape to make
            # room for the new components.
            shape = (target_count, attr_desc.size)
            array = attr_desc.get_value_array(src_geo)
            array.reshape(shape)
            copy = True
        else:
            # A readable (and writable) attribute requires us to retrieve
            # the latest values from the geometry while making sure that
            # we have ownership over the resulting buffer.
            array = attr_desc.get_value_array(src_geo)
            copy = True

        # Cast the array to Warp.
        array = warp.array(
            array,
            dtype=warp_dtype,
            device=kernel_device,
            copy=copy,
        )

        arrays.append(array)

    return tuple(arrays)


def set_attribute_value_arrays(
    hda_node: hou.Node,
    geo: hou.Geometry,
    attr_descs: Sequence[AttributeDesc],
    arrays: Sequence[warp.array],
) -> None:
    """Set the array values for the given attributes onto a geometry."""
    array_cache = {}
    for attr_desc, array in zip(attr_descs, arrays):
        if Access.Write in attr_desc.access:
            # Write the array values onto the geometry attribute.
            attr_desc.set_value_array(geo, array.numpy())

        attr = attr_desc.get_instance(geo)
        array_cache[attr_desc.name] = array

    node_id = hda_node.sessionId()
    _ARRAY_CACHE[node_id] = array_cache


def clear_caches(hda_node: hou.Node) -> None:
    """Clear all the cached data for the given node."""
    node_id = hda_node.sessionId()
    _ARRAY_CACHE.pop(node_id, None)
    _CONTEXT_CACHE.pop(node_id, None)
