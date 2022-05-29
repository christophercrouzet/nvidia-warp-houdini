# -*- coding: utf-8 -*-

#
#   .--.--.--.---.-.----.-----.
#   |  |  |  |  _  |   _|  _  |
#   |________|___._|__| |   __|
#                       |__|
#

"""Builder for NVIDIA's Warp HDA."""

import os
import re
import shutil
import tempfile

from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import defaulttools
import hou


_NAMESPACE = "nvidia"
_NAMESPACE_LABEL = "NVIDIA"

_NAME = "warp"
_LABEL = "Warp ({})".format(_NAMESPACE_LABEL)

_VERSION = "0.0.1"


_ROOT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
    ),
)
_DATA_DIR = os.path.join(_ROOT_PATH, "data")

_INPUT_LABEL_EXPR = re.compile(r"^label(\d+)$")


#   Helpers
# ------------------------------------------------------------------------------


def _read_data_file(file) -> str:
    """Read the contents of a data file."""
    file_path = os.path.join(_DATA_DIR, file)
    with open(file_path, "r") as f:
        return f.read()


def _create_node(
    parent: hou.Node,
    type_name: str,
    name: Optional[str] = None,
    inputs: Optional[
        Sequence[Tuple[Union[hou.Node, hou.SubnetIndirectInput], int]]
    ] = None,
    spare_parms: Optional[Sequence[hou.ParmTemplate]] = None,
) -> hou.Node:
    """Create a new node."""
    node = parent.createNode(type_name, node_name=name)

    # Wire any inputs.
    if inputs is not None:
        for i, input in enumerate(inputs):
            if input is None:
                continue

            input_node, input_index = input
            node.setInput(i, input_node, input_index)

    # Add any spare parameters.
    if spare_parms is not None:
        parm_template_group = hou.ParmTemplateGroup(
            node.parmTemplateGroup().parmTemplates() + spare_parms
        )
        node.setParmTemplateGroup(parm_template_group)

    return node


def _create_output_node(
    parent: hou.Node,
    input: Tuple[Union[hou.Node, hou.SubnetIndirectInput], int],
) -> hou.Node:
    """Create an output node and set its flags."""
    node = _create_node(parent, "output", name="OUT", inputs=(input,))
    node.setGenericFlag(hou.nodeFlag.Display, True)
    node.setGenericFlag(hou.nodeFlag.Render, True)
    parent.setOutputForViewFlag(0)
    return node


def _create_hda(
    file_path: str,
    parent: hou.Node,
    name: str,
    scope: Optional[str] = None,
    namespace: Optional[str] = None,
    version: Optional[str] = None,
    label: Optional[str] = None,
    input_count: Optional[int] = None,
    output_count: Optional[int] = None,
) -> hou.Node:
    """Create a new HDA."""
    input_count = 1 if input_count is None else input_count
    output_count = 1 if output_count is None else output_count
    label = name if label is None else label

    # Create the initial subnetwork.
    type_name = hou.hda.fullNodeTypeNameFromComponents(
        scope,
        namespace,
        name,
        version,
    )
    subnet = _create_node(parent, "subnet")

    # Clean-up any existing contents.
    for child in subnet.children():
        child.destroy()

    # Rename the input labels.
    for parm in subnet.parms():
        match = _INPUT_LABEL_EXPR.match(parm.name())
        if match is None:
            continue

        digit = match.group(1)
        parm.set("Input {}".format(digit))

    # Convert the subnetwork into an HDA.
    return subnet.createDigitalAsset(
        name=type_name,
        hda_file_name=file_path,
        description=label,
        min_num_inputs=input_count,
        max_num_inputs=input_count,
        compress_contents=True,
        change_node_type=True,
        create_backup=False,
    )


def _set_hda_tab_menu_paths(
    definition: hou.HDADefinition,
    paths: Sequence[str],
) -> None:
    """Set the tab menu paths for the HDA."""
    node_type = definition.nodeType()
    node_category = definition.nodeTypeCategory()

    file, file_path = tempfile.mkstemp(suffix=".xml")
    os.close(file)
    try:
        tool_name = hou.shelves.defaultToolName(
            node_category.name(),
            node_type.name(),
        )
        tool = defaulttools.createDefaultHDATool(
            file_path,
            node_type,
            tool_name,
            locations=paths,
        )
        defaulttools.setHDAToolVariables(tool, definition)

        with open(file_path, "rb") as f:
            definition.addSection("Tools.shelf", f.read())
    finally:
        os.remove(file_path)


#   Parameters
# ------------------------------------------------------------------------------

def _build_parms() -> Tuple[hou.ParmTemplate, ...]:
    """Build the parameter interface for the HDA."""
    return (
        hou.ButtonParmTemplate(
            "reset",
            "Reset",
            script_callback="hou.phm().handle_reset_pressed(kwargs[\"node\"])",
            script_callback_language=hou.scriptLanguage.Python,
        ),
        hou.IntParmTemplate(
            "startframe",
            "Start Frame",
            1,
            default_value=(1,),
            min=0,
            max=100,
        ),
        hou.IntParmTemplate(
            "substeps",
            "Substeps",
            1,
            default_value=(1,),
            min=1,
            max=10,
            min_is_strict=True,
        ),
        hou.MenuParmTemplate(
            "device",
            "Device",
            (
                "cpu",
                "cuda",
            ),
            (
                "CPU",
                "CUDA",
            ),
            default_value=0,
        ),
        hou.MenuParmTemplate(
            "runover",
            "Run Over",
            (
                "detail",
                "points",
                "primitives",
                "vertices",
                "numbers",
            ),
            (
                "Detail (only once)",
                "Points",
                "Primitives",
                "Vertices",
                "Numbers",
            ),
            default_value=1,
        ),
        hou.IntParmTemplate(
            "runover_numbers",
            "Numbers",
            1,
            default_value=(1,),
            min=0,
            max=10000,
            min_is_strict=True,
            conditionals={
                hou.parmCondType.HideWhen:
                    "{ runover != numbers }",
            },
        ),
        hou.FolderParmTemplate(
            "attr_folder",
            "Attributes",
            parm_templates=(
                hou.FolderParmTemplate(
                    "attr_count",
                    "Count",
                    parm_templates=(
                        hou.MenuParmTemplate(
                            "attr_#_type",
                            "Class",
                            (
                                "detail",
                                "point",
                                "primitive",
                                "vertex",
                            ),
                            (
                                "Detail",
                                "Point",
                                "Primitive",
                                "Vertex",
                            ),
                            default_value=1,
                        ),
                        hou.MenuParmTemplate(
                            "attr_#_access",
                            "Access",
                            (
                                "read",
                                "write",
                                "readwrite",
                            ),
                            (
                                "Read Only",
                                "Write Only",
                                "Read and Write",
                            ),
                            default_value=2,
                        ),
                        hou.StringParmTemplate(
                            "attr_#_name",
                            "Name",
                            1,
                            item_generator_script=_read_data_file(
                                "parms/attrnamemenu.py",
                            ),
                            menu_type=hou.menuType.StringReplace,
                        ),
                        hou.SeparatorParmTemplate(
                            "attr_#_sep",
                            tags={
                                "sidefx::look": "blank",
                            },
                        ),
                    ),
                    folder_type=hou.folderType.MultiparmBlock,
                ),
            ),
        ),
        hou.FolderParmTemplate(
            "kernel_folder",
            "Kernel",
            parm_templates=(
                hou.MenuParmTemplate(
                    "kernel_source",
                    "Source",
                    (
                        "embedded",
                        "file",
                    ),
                    (
                        "Embedded",
                        "File",
                    ),
                    default_value=0,
                ),
                hou.ButtonParmTemplate(
                    "kernel_source_embedded_generate",
                    "Generate From Attributes",
                    script_callback=(
                        "hou.phm().handle_kernel_source_generate_pressed("
                        "kwargs[\"node\"], "
                        "kwargs[\"node\"].inputGeometry(0)"
                        ")"
                    ),
                    script_callback_language=hou.scriptLanguage.Python,
                    conditionals={
                        hou.parmCondType.HideWhen:
                            "{ kernel_source != embedded }",
                    },
                ),
                hou.StringParmTemplate(
                    "kernel_source_embedded",
                    "Code",
                    1,
                    tags={
                        "editor": "1",
                        "editorlang": "python",
                        "editorlines": "8-40",
                    },
                    conditionals={
                        hou.parmCondType.HideWhen:
                            "{ kernel_source != embedded }",
                    },
                ),
                hou.StringParmTemplate(
                    "kernel_source_file",
                    "File",
                    1,
                    string_type=hou.stringParmType.FileReference,
                    conditionals={
                        hou.parmCondType.HideWhen: "{ kernel_source != file }",
                    },
                ),
            ),
        ),
    )


#   Networks
# ------------------------------------------------------------------------------


def _build_solver_network(parent: hou.Node) -> None:
    """Build the network for the Solver SOP."""
    step = _create_node(
        parent,
        "python",
        name="step",
        inputs=(
            (parent.node("Input_1"), 0),
            (parent.node("Prev_Frame"), 0),
        ),
        spare_parms=(
            hou.StringParmTemplate(
                "hdanode",
                "HDA Node",
                1,
                default_value=("../../../..",),
                string_type=hou.stringParmType.NodeReference,
            ),
        ),
    )
    step.setParms(
        {
            "python": _read_data_file("snippets/step.py"),
        }
    )

    output = _create_output_node(parent, (step, 0))

    parent.layoutChildren()


def _build_hda_network(parent: hou.Node) -> None:
    """Build the network for the HDA."""
    solver = _create_node(
        parent,
        "solver",
        name="solver",
        inputs=(
            (parent.indirectInputs()[0], 0),
        ),
    )
    solver.setParmExpressions(
        {
            "startframe": "ch(\"../startframe\")",
            "substep": "ch(\"../substeps\")",
        },
    )
    _build_solver_network(solver.node("d/s"))

    output = _create_output_node(parent, (solver, 0))

    parent.layoutChildren()


#   Public API
# ------------------------------------------------------------------------------


def build(dest_path: str) -> hou.Node:
    """Create the Warp SOP HDA and write it into the given destination path."""
    # We create a temporary directory where to save the HDA because:
    # - we want to write to the final destination path only at the last minute,
    #   when we know that the build operation succeeded.
    # - updates to the definition instance causes Houdini to create backup
    #   files, which are unneeded here and should be cleaned up.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "{}.hda".format(_NAME))

        # Create the new HDA within the current scene.
        root = hou.node("/obj")
        parent = root.createNode("geo")
        hda_node = _create_hda(
            temp_path,
            parent,
            _NAME,
            namespace=_NAMESPACE,
            version=_VERSION,
            label=_LABEL,
            input_count=1,
            output_count=1,
        )
        definition = hda_node.type().definition()

        # Set its version.
        definition.setVersion(_VERSION)

        # Set its tab menu path.
        _set_hda_tab_menu_paths(definition, (_NAMESPACE_LABEL,))

        # Build its parameters.
        parms = _build_parms()
        parm_template_group = hou.ParmTemplateGroup(parms)
        definition.setParmTemplateGroup(parm_template_group)

        # Define its Python module.
        definition.addSection(
            "PythonModule",
            _read_data_file("sections/pythonmodule.py"),
        )
        definition.setExtraFileOption("PythonModule/IsPython", True)

        # Define its OnDeleted event.
        definition.addSection(
            "OnDeleted",
            _read_data_file("sections/ondeleted.py"),
        )
        definition.setExtraFileOption("OnDeleted/IsPython", True)

        # Define its help document.
        definition.addSection(
            "Help",
            _read_data_file("sections/help.txt"),
        )

        # Build its network contents.
        _build_hda_network(hda_node)

        # Save the changes back into the definition file.
        definition.updateFromNode(hda_node)

        # Copy the resulting file onto its final destination.
        shutil.copyfile(temp_path, dest_path)
