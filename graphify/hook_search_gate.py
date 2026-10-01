"""Project-membership gate for the `hook-guard search` arm (#3882).

The read arm skips out-of-project targets (#1840 (a)); the search arm nudged on
every Bash/Grep search while a graph existed, wherever it pointed. This module
decides whether a search provably touches ONLY paths outside the project root,
so the nudge can stay quiet for it.

Strict allow-list. The worst failure here is silencing a search over project
code, and a shell line has endless ways to change what a command touches
(variables, globs, brace expansion, `$'..'`, conditional `cd`, a previous
command creating the target, zsh-only expansions...). Modelling those proved
open-ended, so the gate only ever goes quiet for ONE simple shape, written with
literal words:

    [cd ABS_DIR &&] SEARCH [2>/dev/null | 2>&1] [| FILTER ...]

- SEARCH is grep/egrep/fgrep/zgrep or rg with KNOWN options, or find with plain
  tests; every other tool (ack, ag, fd, git grep, ...) keeps nudging;
- FILTER only reads the pipe: head, tail, wc, sort, uniq, cut with options only;
- every word is literal: unquoted `$ * ? [ ] { } ( ) < > ; & ! # \\` and
  backquotes, a newline, or `$`/backslash/backquote inside double quotes make
  the line undecidable (only `~/` at the start of a word is expanded, and a `$`
  right before a closing double quote is a literal regex anchor);
- every search target must exist; it is "in" when it is inside the project or
  an ANCESTOR of it (a recursive search walks through), compared by file
  identity, so macOS firmlinks and case-insensitive spellings count;
- `cd` takes an absolute, existing directory without `..`, followed by `&&`.

Anything else keeps the nudge. POSIX shells only: on Windows the gate stays off.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


class _Undecidable(Exception):
    """The command cannot be proven to stay outside the project."""


# --------------------------------------------------------------------------- #
# Per-tool option tables. A flag missing here is UNKNOWN and makes the search
# undecidable; that is deliberate. Flags that follow symlinks during traversal
# (grep -R, rg -L/--follow) and modes without a positional pattern (rg --files)
# are left out on purpose.
# --------------------------------------------------------------------------- #
_GREP = {
    "short_flags": frozenset("rinlLcvwxosqhHEFGPaIzZbUTuy"),
    "short_valued": frozenset("efmABCdD"),
    "long_flags": frozenset({
        "--recursive", "--ignore-case", "--no-ignore-case", "--line-number",
        "--files-with-matches", "--files-without-match", "--count", "--invert-match",
        "--word-regexp", "--line-regexp", "--only-matching", "--no-messages", "--quiet",
        "--silent", "--no-filename", "--with-filename", "--extended-regexp",
        "--fixed-strings", "--basic-regexp", "--perl-regexp", "--text", "--null",
        "--null-data", "--byte-offset", "--binary", "--initial-tab",
        "--color", "--colour",  # grep: value only as --color=WHEN
        "--line-buffered",
    }),
    "long_valued": frozenset({
        "--regexp", "--file", "--max-count", "--context", "--after-context",
        "--before-context", "--include", "--exclude", "--exclude-dir", "--exclude-from",
        "--label", "--binary-files", "--devices", "--directories",
    }),
    "pattern_opts": frozenset({"-e", "-f", "--regexp", "--file"}),
    "pattern_file_opts": frozenset({"-f", "--file", "--exclude-from"}),
}
_RG = {
    "short_flags": frozenset("isSnNlcvwxoqHIFauUPzb0p"),
    "short_valued": frozenset("efgtTmABCdEjMr"),
    "long_flags": frozenset({
        "--ignore-case", "--case-sensitive", "--smart-case", "--line-number",
        "--no-line-number", "--files-with-matches", "--files-without-match", "--count",
        "--count-matches", "--invert-match", "--word-regexp", "--line-regexp",
        "--only-matching", "--quiet", "--with-filename", "--no-filename",
        "--fixed-strings", "--text", "--unrestricted", "--multiline", "--pcre2",
        "--search-zip", "--null", "--byte-offset", "--pretty", "--hidden",
        "--no-hidden", "--no-ignore", "--no-ignore-vcs", "--no-ignore-parent",
        "--no-heading", "--heading", "--vimgrep", "--json", "--stats", "--trim",
        "--column", "--no-column", "--no-messages", "--one-file-system",
    }),
    "long_valued": frozenset({
        "--regexp", "--file", "--glob", "--iglob", "--type", "--type-not",
        "--max-count", "--context", "--after-context", "--before-context",
        "--max-depth", "--encoding", "--threads", "--max-columns", "--replace",
        "--sort", "--sortr", "--max-filesize", "--colors",
        "--color",  # rg: `--color WHEN` consumes the next token
    }),
    "pattern_opts": frozenset({"-e", "-f", "--regexp", "--file"}),
    "pattern_file_opts": frozenset({"-f", "--file"}),
}
_TABLES = {"grep": _GREP, "egrep": _GREP, "fgrep": _GREP, "zgrep": _GREP,
           "rg": _RG, "ripgrep": _RG}

# find: leading options, then roots, then an expression made only of these.
_FIND_LEADING_OK = frozenset({"-P"})
_FIND_UNARY = frozenset({
    "-print", "-print0", "-empty", "-true", "-false", "-prune", "-not", "!", "-o",
    "-a", "-or", "-and", "-readable", "-writable", "-executable", "-depth", "-xdev",
    "-mount", "-nouser", "-nogroup", "-daystart", "-ls",
})
_FIND_VALUED = frozenset({
    "-name", "-iname", "-path", "-ipath", "-wholename", "-iwholename", "-regex",
    "-iregex", "-regextype", "-type", "-maxdepth", "-mindepth", "-size", "-mtime",
    "-mmin", "-atime", "-amin", "-ctime", "-cmin", "-newer", "-perm", "-user",
    "-group", "-links", "-inum",
})

# Pipe-only filters after the search: every argument must be an option or a
# number (no file operand, no output file; `sort -o` is refused).
_FILTERS = frozenset({"head", "tail", "wc", "sort", "uniq", "cut"})
_FILTER_ARG_RE = re.compile(r"^(-[A-Za-z0-9:,=.\-]*|[0-9]+)$")

# Unquoted characters that start an expansion, a redirection, a control
# operator or a comment. `|` and `&&` are tokenised separately; `~` only at the
# start of a word, followed by `/` or the end.
_UNQUOTED_SPECIAL = set("$*?[]{}()<>;&!#\\`")
# The only redirections accepted, removed before tokenising.
_STDERR_RE = re.compile(r"(?<=\s)2>(?:/dev/null|&1)(?=\s|$|\|)")


def _tokens(cmd: str) -> list:
    """Words and the operators `|` / `&&` of a line made only of literal words."""
    if "\n" in cmd or "\r" in cmd:
        raise _Undecidable("multi-line command")
    cmd = _STDERR_RE.sub(" ", cmd)
    toks: list = []
    word: list = []
    in_word = False
    i, n = 0, len(cmd)
    while i < n:
        c = cmd[i]
        if c.isspace():
            if in_word:
                toks.append("".join(word))
                word, in_word = [], False
        elif c == "'":
            j = cmd.find("'", i + 1)
            if j < 0:
                raise _Undecidable("unterminated quote")
            word.append(cmd[i + 1:j])
            in_word, i = True, j
        elif c == '"':
            j = i + 1
            while j < n and cmd[j] != '"':
                if cmd[j] in "\\`" or (cmd[j] == "$" and cmd[j + 1:j + 2] != '"'):
                    raise _Undecidable("expansion inside double quotes")
                j += 1
            if j >= n:
                raise _Undecidable("unterminated quote")
            word.append(cmd[i + 1:j])
            in_word, i = True, j
        elif c == "|":
            if cmd[i + 1:i + 2] == "|":
                raise _Undecidable("||")
            if in_word:
                toks.append("".join(word))
                word, in_word = [], False
            toks.append("|")
        elif c == "&" and cmd[i + 1:i + 2] == "&":
            if in_word:
                toks.append("".join(word))
                word, in_word = [], False
            toks.append("&&")
            i += 1
        elif c == "~" and not in_word and cmd[i + 1:i + 2] in ("/", "") + tuple(" \t"):
            home = os.environ.get("HOME")
            if not home:
                raise _Undecidable("~ without HOME")
            word.append(home)
            in_word = True
        elif c in _UNQUOTED_SPECIAL or c == "~":
            raise _Undecidable(f"unquoted {c!r}")
        else:
            word.append(c)
            in_word = True
        i += 1
    if in_word:
        toks.append("".join(word))
    return toks


def _ids_up(p: Path) -> list:
    """(st_dev, st_ino) of p and of every parent, nearest first."""
    st = p.stat()  # the path itself must be readable: its error propagates
    out = [(st.st_dev, st.st_ino)]
    for q in p.parents:
        try:
            st = q.stat()
        except OSError:
            continue
        out.append((st.st_dev, st.st_ino))
    return out


def _kind(p: Path, root: Path) -> str:
    """'in' when p is inside the project or an ancestor of it, else 'out'.
    Compared by file identity, so firmlinks (/System/Volumes/Data/...) and
    case-insensitive spellings resolve to the same directory."""
    resolved = p.resolve(strict=True)
    root_ids = _ids_up(root.resolve(strict=True))
    target_ids = _ids_up(resolved)
    if root_ids[0] in target_ids or target_ids[0] in root_ids:
        return "in"
    return "out"


def _grep_like_operands(args: list, table: dict) -> tuple:
    """(path operands, pattern files) of one grep/rg call; unknown option -> undecidable."""
    pos: list = []
    pattern_files: list = []
    has_pattern_opt = end_of_opts = False
    skip_into = None  # None, "skip", "pattern_file"
    for a in args:
        if skip_into:
            if skip_into == "pattern_file":
                pattern_files.append(a)
            skip_into = None
            continue
        if end_of_opts or a == "-" or not a.startswith("-"):
            pos.append(a)
        elif a == "--":
            end_of_opts = True
        elif a.startswith("--"):
            name, _, value = a.partition("=")
            if name in table["long_valued"]:
                if name in table["pattern_file_opts"]:
                    if value:
                        pattern_files.append(value)
                    else:
                        skip_into = "pattern_file"
                elif not value and "=" not in a:
                    skip_into = "skip"
            elif name not in table["long_flags"]:
                raise _Undecidable(a)
            has_pattern_opt |= name in table["pattern_opts"]
        else:
            letters = a[1:]
            for idx, ch in enumerate(letters):
                if ch in table["short_valued"]:
                    opt = "-" + ch
                    has_pattern_opt |= opt in table["pattern_opts"]
                    attached = letters[idx + 1:]
                    if opt in table["pattern_file_opts"]:
                        if attached:
                            pattern_files.append(attached)
                        else:
                            skip_into = "pattern_file"
                    elif not attached:
                        skip_into = "skip"
                    break
                if ch not in table["short_flags"]:
                    raise _Undecidable(a)
    if skip_into:
        raise _Undecidable("option without its value")
    return (pos if has_pattern_opt else pos[1:]), pattern_files


def _find_roots(args: list) -> list:
    i = 0
    while i < len(args) and args[i] in _FIND_LEADING_OK:
        i += 1
    roots = []
    while i < len(args) and not args[i].startswith("-") and args[i] != "!":
        roots.append(args[i])
        i += 1
    while i < len(args):
        a = args[i]
        if a in _FIND_UNARY:
            i += 1
        elif a in _FIND_VALUED:
            i += 2  # values are names/patterns/stat references, not traversed
        else:  # -exec, -execdir, -ok, -delete, -L, -follow, -fprint ...
            raise _Undecidable(a)
    if i > len(args):
        raise _Undecidable("find test without its value")
    return roots or ["."]


def _split(toks: list, op: str) -> list:
    parts, cur = [], []
    for t in toks:
        if t == op:
            if not cur:
                raise _Undecidable(f"empty segment before {op}")
            parts.append(cur)
            cur = []
        else:
            cur.append(t)
    if not cur:
        raise _Undecidable(f"empty segment after {op}")
    return parts + [cur]


def _decide(cmd_str: str, root: Path) -> bool:
    cwd = Path.cwd()
    parts = _split(_tokens(cmd_str), "&&")
    if len(parts) > 2:
        raise _Undecidable("more than one &&")
    if len(parts) == 2:
        cd = parts[0]
        if "|" in cd or len(cd) != 2 or cd[0] != "cd":
            raise _Undecidable("only `cd ABS_DIR &&` may precede the search")
        dest = Path(cd[1])
        if not dest.is_absolute() or ".." in dest.parts or not dest.is_dir():
            raise _Undecidable("cd target")
        cwd = dest
    stages = _split(parts[-1], "|")
    for stage in stages[1:]:
        name, args = stage[0], stage[1:]
        if name not in _FILTERS or not all(_FILTER_ARG_RE.match(a) for a in args):
            raise _Undecidable(f"pipe stage {name}")
        if name == "sort" and any(a.startswith(("-o", "--o")) for a in args):
            raise _Undecidable("sort -o")
    name, args = stages[0][0], stages[0][1:]
    if name == "find":
        files = _find_roots(args)
    elif name in _TABLES:
        targets, pattern_files = _grep_like_operands(args, _TABLES[name])
        if "-" in targets or "-" in pattern_files:
            raise _Undecidable("stdin")
        files = (targets or ["."]) + pattern_files  # no operand: searches the cwd
    else:
        raise _Undecidable(name)
    return all(_kind(Path(f) if os.path.isabs(f) else cwd / f, root) == "out"
               for f in files)


def bash_searches_only_out_of_project(cmd_str: str, root: Path) -> bool:
    """True only when the Bash command is the simple search shape above and every
    path it searches or reads is outside `root`. Anything else returns False."""
    if os.name == "nt":
        return False
    try:
        return _decide(cmd_str, root)
    except Exception:  # _Undecidable, OSError, ValueError...: keep nudging
        return False


def grep_tool_path_out_of_project(tool_input: dict, root: Path, is_cwd_relative) -> bool:
    """Grep tool: quiet only for an explicit, non-cwd-relative, existing `path`
    outside `root` and not an ancestor of it (no path means a project-wide search)."""
    try:
        v = str(tool_input.get("path") or "")
        if not v or is_cwd_relative(v):
            return False
        return _kind(Path(v), root) == "out"
    except Exception:  # missing or malformed path: keep nudging
        return False
