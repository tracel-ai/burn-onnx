#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
# ]
# ///

"""Fetch ONNX operator specs and write per-operator markdown files to ops/.

Each file contains the latest spec plus a version history showing all opset
versions with their changes.

Usage: ./onnx-spec/fetch-specs.py
"""

import os
from pathlib import Path

import onnx
from onnx import defs

# Only the default ONNX domain
DOMAIN = ""

# Attribute type name mapping
ATTR_TYPE_NAMES = {
    0: "UNDEFINED",
    1: "FLOAT",
    2: "INT",
    3: "STRING",
    4: "TENSOR",
    5: "GRAPH",
    6: "FLOATS",
    7: "INTS",
    8: "STRINGS",
    9: "TENSORS",
    10: "GRAPHS",
    11: "SPARSE_TENSOR",
    12: "SPARSE_TENSORS",
    13: "TYPE_PROTO",
    14: "TYPE_PROTOS",
}

# Input/output option mapping
FORMAL_PARAM_OPTION = {
    0: "Single",
    1: "Optional",
    2: "Variadic",
}


def get_all_schemas_by_name():
    """Get all versions of each operator schema, grouped by name."""
    all_schemas = defs.get_all_schemas_with_history()
    by_name = {}
    for schema in all_schemas:
        if schema.domain != DOMAIN:
            continue
        name = schema.name
        if name not in by_name:
            by_name[name] = []
        by_name[name].append(schema)
    # Sort each operator's versions by opset
    for name in by_name:
        by_name[name].sort(key=lambda s: s.since_version)
    return by_name


def format_attribute(attr):
    """Format a single attribute as markdown."""
    type_name = ATTR_TYPE_NAMES.get(attr.type, f"UNKNOWN({attr.type})")
    required = "required" if attr.required else "optional"
    line = f"- **{attr.name}** ({type_name}, {required})"
    if attr.description:
        desc = attr.description.strip().split("\n")[0]  # first line only
        line += f": {desc}"
    return line


def format_io(param, direction):
    """Format a single input or output as markdown."""
    option = FORMAL_PARAM_OPTION.get(param.option, "Single")
    type_str = param.type_str if param.type_str else ""
    line = f"- **{param.name}**"
    parts = []
    if type_str:
        parts.append(type_str)
    if option != "Single":
        parts.append(option.lower())
    if parts:
        line += f" ({', '.join(parts)})"
    if param.description:
        desc = param.description.strip().split("\n")[0]
        line += f": {desc}"
    return line


def format_type_constraints(schema):
    """Format type constraints as markdown."""
    lines = []
    for tc in schema.type_constraints:
        types = ", ".join(sorted(tc.allowed_type_strs))
        lines.append(f"- **{tc.type_param_str}**: {types}")
        if tc.description:
            desc = tc.description.strip().split("\n")[0]
            lines.append(f"  {desc}")
    return lines


def schema_to_markdown(latest, all_versions):
    """Convert an ONNX schema to a markdown string with version history."""
    lines = []
    lines.append(f"# {latest.name}")
    lines.append("")

    # Version history summary
    first = all_versions[0]
    lines.append(f"First introduced in opset **{first.since_version}**")
    if len(all_versions) > 1:
        versions = ", ".join(str(s.since_version) for s in all_versions)
        lines.append(f"")
        lines.append(f"All versions: {versions}")
    lines.append("")

    if latest.doc:
        lines.append("## Description")
        lines.append("")
        lines.append(latest.doc.strip())
        lines.append("")

    # Attributes
    attrs = list(latest.attributes.values())
    if attrs:
        lines.append("## Attributes")
        lines.append("")
        for attr in sorted(attrs, key=lambda a: a.name):
            lines.append(format_attribute(attr))
        lines.append("")

    # Inputs
    if latest.inputs:
        min_inputs = latest.min_input
        max_inputs = latest.max_input
        lines.append(f"## Inputs ({min_inputs} - {max_inputs})")
        lines.append("")
        for inp in latest.inputs:
            lines.append(format_io(inp, "input"))
        lines.append("")

    # Outputs
    if latest.outputs:
        min_outputs = latest.min_output
        max_outputs = latest.max_output
        lines.append(f"## Outputs ({min_outputs} - {max_outputs})")
        lines.append("")
        for out in latest.outputs:
            lines.append(format_io(out, "output"))
        lines.append("")

    # Type constraints
    tc_lines = format_type_constraints(latest)
    if tc_lines:
        lines.append("## Type Constraints")
        lines.append("")
        lines.extend(tc_lines)
        lines.append("")

    # Version history details
    if len(all_versions) > 1:
        lines.append("## Version History")
        lines.append("")
        for schema in reversed(all_versions):
            tc = format_type_constraints(schema)
            type_summary = ""
            if tc:
                # Extract just the type list from the first constraint
                for constraint in schema.type_constraints:
                    types = ", ".join(sorted(constraint.allowed_type_strs))
                    type_summary = f" Types: {types}"
                    break
            lines.append(
                f"- **Opset {schema.since_version}**:{type_summary}"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    script_dir = Path(__file__).resolve().parent
    ops_dir = script_dir / "ops"
    ops_dir.mkdir(exist_ok=True)

    # Clean existing files
    for f in ops_dir.glob("*.md"):
        f.unlink()

    schemas_by_name = get_all_schemas_by_name()
    print(f"Writing {len(schemas_by_name)} operator specs to {ops_dir}/")

    for name in sorted(schemas_by_name):
        versions = schemas_by_name[name]
        latest = versions[-1]
        md = schema_to_markdown(latest, versions)
        filepath = ops_dir / f"{name}.md"
        filepath.write_text(md)

    print("Done.")


if __name__ == "__main__":
    main()
