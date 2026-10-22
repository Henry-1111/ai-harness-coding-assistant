import shutil
import sys
from pathlib import Path

import pytest

from ai_harness.lsp import LSPDefinitionEnricher
from ai_harness.models import SearchHit

PYRIGHT_COMMAND = shutil.which("pyright-langserver")
if PYRIGHT_COMMAND is None:
    sibling = Path(sys.executable).with_name("pyright-langserver")
    PYRIGHT_COMMAND = str(sibling) if sibling.is_file() else None


@pytest.mark.skipif(
    PYRIGHT_COMMAND is None,
    reason="install the optional lsp dependency to run this integration test",
)
def test_real_pyright_definition_enriches_related_file(tmp_path: Path):
    (tmp_path / "policy.py").write_text(
        "def allow_request(user: str) -> bool:\n    return user == 'admin'\n",
        encoding="utf-8",
    )
    (tmp_path / "handler.py").write_text(
        "from policy import allow_request\n\n"
        "def authorize(user: str) -> bool:\n"
        "    return allow_request(user)\n",
        encoding="utf-8",
    )
    enricher = LSPDefinitionEnricher(
        [PYRIGHT_COMMAND, "--stdio"],
        tmp_path,
    )

    paths = enricher.related_paths(
        "allow_request",
        [SearchHit(path="handler.py", line=1, text="allow_request")],
    )

    assert paths == ["policy.py"]
