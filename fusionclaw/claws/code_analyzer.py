"""Code analyzer claw — analyzes Python files and directories.

Provides structural analysis of Python code: classes, functions,
imports, complexity metrics. No external dependencies needed.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

from ..claw import BaseClaw
from ..fuser import count_tokens
from ..models import Fact, StateObject


class CodeAnalyzerClaw(BaseClaw):
    """Analyzes Python source files for structure, complexity, and patterns.

    Pass a file path or directory as input. Returns structural analysis
    as a StateObject for fusion with other claws.
    """

    claw_id = "code_analyzer"
    description = "Analyzes Python code for structure, classes, functions, and complexity"

    def __init__(self, max_files: int = 20, max_file_size: int = 50_000):
        self.max_files = max_files
        self.max_file_size = max_file_size

    async def run(self, input: str) -> StateObject:
        path = Path(input.strip())

        if not path.exists():
            return StateObject(
                claw_id=self.claw_id,
                summary=f"Path not found: {input}",
                key_facts=[Fact(key="error", value="path_not_found")],
                raw_context="",
                token_count=0,
            )

        if path.is_file():
            files = [path] if path.suffix == ".py" else []
        else:
            files = sorted(path.rglob("*.py"))[: self.max_files]

        if not files:
            return StateObject(
                claw_id=self.claw_id,
                summary=f"No Python files found in: {input}",
                key_facts=[],
                raw_context="",
                token_count=0,
            )

        all_classes: list[str] = []
        all_functions: list[str] = []
        all_imports: list[str] = []
        total_lines = 0
        file_summaries: list[str] = []
        errors: list[str] = []

        for f in files:
            try:
                source = f.read_text(errors="ignore")
                if len(source) > self.max_file_size:
                    source = source[: self.max_file_size]

                lines = source.count("\n") + 1
                total_lines += lines

                tree = ast.parse(source, filename=str(f))
                analysis = _analyze_ast(tree)

                all_classes.extend(analysis["classes"])
                all_functions.extend(analysis["functions"])
                all_imports.extend(analysis["imports"])

                rel_path = str(f)
                file_summaries.append(
                    f"### {rel_path} ({lines} lines)\n"
                    f"  Classes: {', '.join(analysis['classes']) or 'none'}\n"
                    f"  Functions: {', '.join(analysis['functions']) or 'none'}\n"
                    f"  Imports: {', '.join(analysis['imports'][:10]) or 'none'}"
                )
            except Exception as e:
                errors.append(f"{f}: {e}")

        facts = [
            Fact(key="total_files", value=str(len(files))),
            Fact(key="total_lines", value=str(total_lines)),
            Fact(key="total_classes", value=str(len(all_classes))),
            Fact(key="total_functions", value=str(len(all_functions))),
        ]
        if all_classes:
            facts.append(Fact(key="classes", value=", ".join(all_classes[:15])))
        if all_functions:
            facts.append(Fact(key="functions", value=", ".join(all_functions[:15])))

        unique_imports = sorted(set(all_imports))[:20]
        if unique_imports:
            facts.append(Fact(key="imports", value=", ".join(unique_imports)))

        raw_context = "\n\n".join(file_summaries)
        if errors:
            raw_context += "\n\n### Errors\n" + "\n".join(errors)

        summary = (
            f"Analyzed {len(files)} Python files ({total_lines} lines): "
            f"{len(all_classes)} classes, {len(all_functions)} functions, "
            f"{len(unique_imports)} unique imports"
        )

        return StateObject(
            claw_id=self.claw_id,
            summary=summary,
            key_facts=facts,
            raw_context=raw_context,
            token_count=count_tokens(raw_context),
        )


def _analyze_ast(tree: ast.Module) -> dict[str, list[str]]:
    """Extract classes, functions, and imports from an AST."""
    classes: list[str] = []
    functions: list[str] = []
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not isinstance(getattr(node, "_parent", None), ast.ClassDef):
                functions.append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return {"classes": classes, "functions": functions, "imports": imports}
