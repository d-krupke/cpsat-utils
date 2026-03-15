"""
Import/export utilities for CP-SAT models.

Allows saving and loading CpModel instances to/from files in either
binary protobuf or human-readable text format. Useful for comparing
solver performance across machines without sharing code.

Usage:
    from cpsat_utils.io import export_model, import_model

    export_model(model, "my_model.pb")     # binary
    export_model(model, "my_model.pbtxt")  # text
    loaded = import_model("my_model.pb")

When to modify:
    - If ortools changes its protobuf API
    - To add support for additional serialization formats
"""

from pathlib import Path

from ortools.sat.python import cp_model

_TEXT_EXTENSIONS = {".txt", ".pbtxt"}
_BINARY_EXTENSIONS = {".pb", ".bin", ".dat"}


def _is_binary(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    if suffix in _TEXT_EXTENSIONS:
        return False
    if suffix in _BINARY_EXTENSIONS:
        return True
    msg = (
        f"Cannot determine format from extension '{suffix}'. "
        f"Use one of {sorted(_TEXT_EXTENSIONS)} for text "
        f"or {sorted(_BINARY_EXTENSIONS)} for binary."
    )
    raise ValueError(msg)


def export_model(model: cp_model.CpModel, filename: str) -> None:
    """
    Export a CpModel to a file.

    The format (binary protobuf or human-readable text) is determined
    by the file extension:
      - Text: .txt, .pbtxt
      - Binary: .pb, .bin, .dat

    Args:
        model: The CpModel to export.
        filename: Destination file path.

    Raises:
        ValueError: If the file extension is not recognized.
    """
    binary = _is_binary(filename)
    path = Path(filename)

    # Prefer export_to_file (available in modern ortools).
    # It detects format by extension: .txt -> text, everything else -> binary.
    if hasattr(model, "export_to_file"):
        if binary:
            # export_to_file writes binary for non-.txt extensions
            if path.suffix == ".txt":
                # .txt would be interpreted as text; use a temp name
                tmp = path.with_suffix(".pb")
                model.export_to_file(str(tmp))
                tmp.rename(path)
            else:
                model.export_to_file(str(path))
        else:
            # export_to_file writes text for .txt extension
            if path.suffix == ".txt":
                model.export_to_file(str(path))
            else:
                tmp = path.with_suffix(".txt")
                model.export_to_file(str(tmp))
                tmp.rename(path)
        return

    # Fallback for older ortools: Proto() returns a real protobuf message.
    from google.protobuf import text_format

    proto = model.Proto()
    if binary:
        path.write_bytes(proto.SerializeToString())
    else:
        path.write_text(text_format.MessageToString(proto))


def import_model(filename: str) -> cp_model.CpModel:
    """
    Import a CpModel from a file.

    The format is determined by the file extension (see ``export_model``).

    Args:
        filename: Source file path.

    Returns:
        A CpModel loaded from the file.

    Raises:
        ValueError: If the file extension is not recognized.
        FileNotFoundError: If the file does not exist.
    """
    binary = _is_binary(filename)
    path = Path(filename)
    model = cp_model.CpModel()
    proto = model.Proto()

    # Modern ortools: proto wrapper with parse_text_format
    if hasattr(proto, "parse_text_format"):
        if binary:
            # parse_text_format only accepts text, so convert via
            # the pb2 class which supports ParseFromString.
            from google.protobuf import text_format
            from ortools.sat import cp_model_pb2

            pb2_proto = cp_model_pb2.CpModelProto()
            pb2_proto.ParseFromString(path.read_bytes())
            proto.parse_text_format(text_format.MessageToString(pb2_proto))
        else:
            proto.parse_text_format(path.read_text())
        return model

    # Fallback for older ortools: Proto() is a real protobuf message.
    from google.protobuf import text_format
    from ortools.sat import cp_model_pb2

    pb2_proto = cp_model_pb2.CpModelProto()
    if binary:
        pb2_proto.ParseFromString(path.read_bytes())
    else:
        text_format.Parse(path.read_text(), pb2_proto)
    proto.CopyFrom(pb2_proto)
    return model
