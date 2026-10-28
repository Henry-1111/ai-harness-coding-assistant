"""Tree-sitter structure extraction for Go and Java source files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import tree_sitter_go
import tree_sitter_java
from tree_sitter import Language, Node, Parser

from .models import RepoSymbol

_LANGUAGES = {
    ".go": Language(tree_sitter_go.language()),
    ".java": Language(tree_sitter_java.language()),
}
_JAVA_TYPE_KINDS: Dict[str, Literal["class", "interface", "record", "enum"]] = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "record_declaration": "record",
    "enum_declaration": "enum",
    "annotation_type_declaration": "interface",
}


def _text(node: Optional[Node]) -> Optional[str]:
    return node.text.decode("utf-8", errors="replace") if node is not None and node.text else None


def _descendant(node: Optional[Node], node_type: str) -> Optional[Node]:
    if node is None:
        return None
    pending = [node]
    while pending:
        current = pending.pop()
        if current.type == node_type:
            return current
        pending.extend(reversed(current.named_children))
    return None


def _go_owner(node: Node) -> Optional[str]:
    if node.type == "method_declaration":
        receiver = node.child_by_field_name("receiver")
        return _text(_descendant(receiver, "type_identifier"))
    parent = node.parent
    while parent is not None:
        if parent.type == "type_spec":
            return _text(parent.child_by_field_name("name"))
        parent = parent.parent
    return None


def _java_owner(node: Node) -> Optional[str]:
    parent = node.parent
    while parent is not None:
        if parent.type in {
            "class_declaration", "interface_declaration", "record_declaration",
            "enum_declaration", "annotation_type_declaration",
        }:
            return _text(parent.child_by_field_name("name"))
        parent = parent.parent
    return None


def _go_import(node: Node) -> Optional[str]:
    literal = node.child_by_field_name("path")
    if literal is None or literal.type not in {
        "interpreted_string_literal", "raw_string_literal"
    }:
        return None
    value = _text(literal)
    return value[1:-1] if value and len(value) >= 2 else None


def _java_import(node: Node) -> Optional[str]:
    qualified = next(
        (
            child for child in node.named_children
            if child.type in {"scoped_identifier", "identifier"}
        ),
        None,
    )
    value = _text(qualified)
    if not value:
        return None
    is_static = any(child.type == "static" for child in node.children)
    is_wildcard = any(child.type == "asterisk" for child in node.named_children)
    if is_static and not is_wildcard:
        return value.rsplit(".", 1)[0] if "." in value else None
    if is_wildcard and not is_static:
        return f"{value}.*"
    return value


def go_java_structure(path: Path) -> Tuple[List[RepoSymbol], List[str]]:
    """Extract declared types/functions/methods and static imports from Go or Java."""
    try:
        source = path.read_bytes()
    except OSError:
        return [], []
    suffix = path.suffix.lower()
    root = Parser(_LANGUAGES[suffix]).parse(source).root_node
    symbols: List[RepoSymbol] = []
    imports: Set[str] = set()
    pending = [root]

    while pending:
        node = pending.pop()
        pending.extend(reversed(node.named_children))
        line = node.start_point.row + 1

        if suffix == ".go":
            if node.type == "import_spec":
                specifier = _go_import(node)
                if specifier:
                    imports.add(specifier)
            elif node.type in {"type_spec", "type_alias"}:
                name = _text(node.child_by_field_name("name"))
                type_node = node.child_by_field_name("type")
                if name:
                    kind = "type"
                    if type_node is not None and type_node.type == "struct_type":
                        kind = "struct"
                    elif type_node is not None and type_node.type == "interface_type":
                        kind = "interface"
                    symbols.append(RepoSymbol(name=name, kind=kind, line=line))
            elif node.type == "function_declaration":
                name = _text(node.child_by_field_name("name"))
                if name:
                    symbols.append(RepoSymbol(name=name, kind="function", line=line))
            elif node.type in {"method_declaration", "method_elem"}:
                name_node = node.child_by_field_name("name") or _descendant(
                    node, "field_identifier"
                )
                name = _text(name_node)
                if name:
                    owner = _go_owner(node)
                    symbols.append(
                        RepoSymbol(name=f"{owner}.{name}" if owner else name,
                                   kind="method", line=line)
                    )
        else:
            if node.type == "import_declaration":
                specifier = _java_import(node)
                if specifier:
                    imports.add(specifier)
            elif node.type in _JAVA_TYPE_KINDS:
                name = _text(node.child_by_field_name("name"))
                if name:
                    kind = _JAVA_TYPE_KINDS[node.type]
                    symbols.append(RepoSymbol(name=name, kind=kind, line=line))
            elif node.type in {"method_declaration", "constructor_declaration"}:
                name = _text(node.child_by_field_name("name"))
                if name:
                    owner = _java_owner(node)
                    symbols.append(
                        RepoSymbol(name=f"{owner}.{name}" if owner else name,
                                   kind="method", line=line)
                    )

    return sorted(symbols, key=lambda item: (item.line, item.name)), sorted(imports)
