"""Symbol and import extraction for JavaScript, JSX, TypeScript and TSX."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, Tuple

import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

from .models import RepoSymbol

_LANGUAGES = {
    ".js": Language(tree_sitter_javascript.language()),
    ".jsx": Language(tree_sitter_javascript.language()),
    ".mjs": Language(tree_sitter_javascript.language()),
    ".cjs": Language(tree_sitter_javascript.language()),
    ".ts": Language(tree_sitter_typescript.language_typescript()),
    ".tsx": Language(tree_sitter_typescript.language_tsx()),
    ".mts": Language(tree_sitter_typescript.language_typescript()),
    ".cts": Language(tree_sitter_typescript.language_typescript()),
}


def _name(node: Optional[Node]) -> Optional[str]:
    if node is None or node.type not in {
        "identifier",
        "property_identifier",
        "private_property_identifier",
        "type_identifier",
    }:
        return None
    return node.text.decode("utf-8", errors="replace") if node.text else None


def _string_value(node: Optional[Node]) -> Optional[str]:
    if node is None or node.type != "string" or node.text is None:
        return None
    value = node.text.decode("utf-8", errors="replace")
    return value[1:-1] if len(value) >= 2 else None


def _enclosing_type(node: Node) -> Optional[str]:
    parent = node.parent
    while parent is not None:
        if parent.type in {"class_declaration", "class", "interface_declaration"}:
            return _name(parent.child_by_field_name("name"))
        parent = parent.parent
    return None


def _is_async(node: Node) -> bool:
    return any(child.type == "async" for child in node.children)


def javascript_structure(path: Path) -> Tuple[List[RepoSymbol], List[str]]:
    """Return declaration-level symbols and static module specifiers from a source file.

    Tree-sitter preserves valid declarations around syntax errors. Dynamic imports and
    computed names are deliberately omitted because their targets are not statically known.
    """
    try:
        source = path.read_bytes()
    except OSError:
        return [], []
    language = _LANGUAGES[path.suffix.lower()]
    root = Parser(language).parse(source).root_node
    symbols: List[RepoSymbol] = []
    imports: Set[str] = set()
    pending = [root]

    while pending:
        node = pending.pop()
        pending.extend(reversed(node.named_children))
        line = node.start_point.row + 1

        if node.type in {"import_statement", "export_statement"}:
            specifier = _string_value(node.child_by_field_name("source"))
            if specifier:
                imports.add(specifier)
        elif node.type == "call_expression":
            function = node.child_by_field_name("function")
            arguments = node.child_by_field_name("arguments")
            if function is not None and function.text == b"require" and arguments is not None:
                first = next(iter(arguments.named_children), None)
                specifier = _string_value(first)
                if specifier:
                    imports.add(specifier)

        name = _name(node.child_by_field_name("name"))
        if name and node.type in {"class_declaration", "interface_declaration"}:
            kind = "interface" if node.type == "interface_declaration" else "class"
            symbols.append(RepoSymbol(name=name, kind=kind, line=line))
        elif name and node.type in {"function_declaration", "generator_function_declaration"}:
            kind = "async_function" if _is_async(node) else "function"
            symbols.append(RepoSymbol(name=name, kind=kind, line=line))
        elif name and node.type == "type_alias_declaration":
            symbols.append(RepoSymbol(name=name, kind="type", line=line))
        elif name and node.type in {"method_definition", "method_signature"}:
            owner = _enclosing_type(node)
            symbols.append(
                RepoSymbol(name=f"{owner}.{name}" if owner else name, kind="method", line=line)
            )
        elif node.type == "variable_declarator":
            declared_name = _name(node.child_by_field_name("name"))
            value = node.child_by_field_name("value")
            if declared_name and value is not None:
                if value.type in {"arrow_function", "function_expression", "generator_function"}:
                    kind = "async_function" if _is_async(value) else "function"
                    symbols.append(RepoSymbol(name=declared_name, kind=kind, line=line))
                elif value.type == "class":
                    symbols.append(RepoSymbol(name=declared_name, kind="class", line=line))

    return sorted(symbols, key=lambda item: (item.line, item.name)), sorted(imports)
