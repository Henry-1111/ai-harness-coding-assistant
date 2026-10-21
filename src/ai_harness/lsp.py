from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence
from urllib.parse import unquote, urlparse

from .models import SearchHit


class LSPError(RuntimeError):
    pass


@dataclass(frozen=True)
class DefinitionLocation:
    uri: str
    start_line: int
    start_character: int
    end_line: int
    end_character: int


def encode_message(payload: Dict[str, Any]) -> bytes:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def read_message(stream: BinaryIO) -> Dict[str, Any]:
    headers: Dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            raise EOFError("LSP server closed the stream")
        if line == b"\r\n":
            break
        name, value = line.decode("ascii").split(":", 1)
        headers[name.lower()] = value.strip()
    if "content-length" not in headers:
        raise LSPError("LSP response is missing Content-Length")
    body = stream.read(int(headers["content-length"]))
    result = json.loads(body.decode("utf-8"))
    if not isinstance(result, dict):
        raise LSPError("LSP response must be an object")
    return result


class LSPClient:
    """Small synchronous JSON-RPC client for an explicitly configured language server."""

    def __init__(self, command: Sequence[str], repository: Path) -> None:
        if not command:
            raise ValueError("language server command is required")
        self.command = list(command)
        self.repository = repository.resolve()
        self.process: Optional[subprocess.Popen[bytes]] = None
        self._next_id = 1

    def __enter__(self) -> "LSPClient":
        self.process = subprocess.Popen(
            self.command,
            cwd=self.repository,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.request(
            "initialize",
            {
                "processId": None,
                "rootUri": self.repository.as_uri(),
                "capabilities": {},
            },
        )
        self.notify("initialized", {})
        return self

    def __exit__(self, *_: object) -> None:
        if self.process and self.process.poll() is None:
            try:
                self.request("shutdown", None)
                self.notify("exit", None)
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
        self.process = None

    def definition(self, relative_path: str, line: int, character: int) -> List[DefinitionLocation]:
        path = (self.repository / relative_path).resolve()
        try:
            path.relative_to(self.repository)
        except ValueError as exc:
            raise LSPError("document path escapes repository") from exc
        result = self.request(
            "textDocument/definition",
            {
                "textDocument": {"uri": path.as_uri()},
                "position": {"line": line, "character": character},
            },
        )
        locations = result if isinstance(result, list) else [result] if result else []
        return [self._location(item) for item in locations]

    def open_document(self, relative_path: str, language_id: str, text: str) -> None:
        path = (self.repository / relative_path).resolve()
        try:
            path.relative_to(self.repository)
        except ValueError as exc:
            raise LSPError("document path escapes repository") from exc
        self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": path.as_uri(),
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                }
            },
        )

    def request(self, method: str, params: Any) -> Any:
        request_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            message = self._receive()
            if "method" in message and "id" in message:
                self._send({"jsonrpc": "2.0", "id": message["id"], "result": None})
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise LSPError(str(message["error"]))
            return message.get("result")

    def notify(self, method: str, params: Any) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, payload: Dict[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            raise LSPError("LSP client is not running")
        self.process.stdin.write(encode_message(payload))
        self.process.stdin.flush()

    def _receive(self) -> Dict[str, Any]:
        if not self.process or not self.process.stdout:
            raise LSPError("LSP client is not running")
        return read_message(self.process.stdout)

    @staticmethod
    def _location(item: Dict[str, Any]) -> DefinitionLocation:
        range_value = item["range"]
        return DefinitionLocation(
            uri=item["uri"],
            start_line=range_value["start"]["line"],
            start_character=range_value["start"]["character"],
            end_line=range_value["end"]["line"],
            end_character=range_value["end"]["character"],
        )


class LSPDefinitionEnricher:
    """Resolve definitions for query symbols found in the highest-ranked source hits."""

    WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

    def __init__(self, command: Sequence[str], repository: Path, max_files: int = 3) -> None:
        self.command = list(command)
        self.repository = repository.resolve()
        self.max_files = max_files

    def related_paths(self, query: str, hits: Sequence[SearchHit]) -> List[str]:
        terms = {item.lower() for item in self.WORD.findall(query)}
        if not terms:
            return []
        related: List[str] = []
        with LSPClient(self.command, self.repository) as client:
            for hit in hits[: self.max_files]:
                path = (self.repository / hit.path).resolve()
                if not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                client.open_document(hit.path, self._language_id(path), text)
                lines = text.splitlines()
                for line_index, line in enumerate(lines):
                    match = next(
                        (
                            match
                            for match in self.WORD.finditer(line)
                            if match.group(0).lower() in terms
                        ),
                        None,
                    )
                    if match is None:
                        continue
                    for location in client.definition(hit.path, line_index, match.start()):
                        relative = self._relative_path(location.uri)
                        if relative and relative not in related:
                            related.append(relative)
                    break
        return related

    def _relative_path(self, uri: str) -> Optional[str]:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            return None
        path = Path(unquote(parsed.path)).resolve()
        try:
            return path.relative_to(self.repository).as_posix()
        except ValueError:
            return None

    @staticmethod
    def _language_id(path: Path) -> str:
        return {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
        }.get(path.suffix.lower(), "plaintext")
