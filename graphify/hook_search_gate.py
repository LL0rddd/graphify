"""Project-membership gate for the `hook-guard search` arm (#3882).

The read arm skips out-of-project targets (#1840 (a)); the search arm nudged on
every Bash/Grep search while a graph existed, wherever it pointed. This module
decides whether a search provably touches ONLY paths outside the project root,
so the nudge can stay quiet for it.

Allow-list, not deny-list. The worst failure here is silencing a search over
project code, and shell has too many ways to change what a command touches
(`rg --files`, options that shift the pattern slot, `find -exec`, background or
failed `cd`, `$1`, `${VAR:-x}`, globs that climb back with `..`, symlinks, a
previous command creating the target...). So the gate only ever goes quiet for
command shapes it fully understands:

- grep/egrep/fgrep/zgrep and rg with KNOWN options, and find with plain tests;
  every other search tool (ack, ag, fd, git grep, ...) keeps nudging;
- every other command in the line must be a known read-only one (cat, ls,
  head, date, ...): anything else might create or re-point a search target;
- every search target must exist when the hook runs; globs are expanded and
  each match resolved (symlinks and `..` included); brace expansion is refused;
- cwd changes only via `cd <existing dir>` followed by `;`, `&&` or the end;
- variables only as `$NAME`/`${NAME}` from a standalone assignment earlier in
  the command (plus `$HOME` from the environment: the Bash tool's shell loads
  the user's profile, the hook does not); `$PWD`/`$OLDPWD`/`CDPATH` refused;
- globs are refused with `**` or more than a thousand matches;
- a non-search command that READS a project file makes the line touch the
  project even if the search itself looks elsewhere;
- no substitution, heredoc, subshell, brace group, control keyword, background
  job, or builtin that changes shell state.

Boundary: reading a PROJECT file counts as touching the project even when it is
not the searched corpus (`grep -f app/patterns.txt /elsewhere`, a `cat app/x |
grep`); stat-only references (`find /elsewhere -newer app/x`) do not.

Anything outside that shape is undecidable and keeps the nudge. POSIX shell
semantics only: on Windows the Bash gate stays off.
"""
from __future__ import annotations

import glob as _glob
import os
import re
import shlex
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
    "pattern_file_opts": frozenset({"-f", "--file"}),
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

# Commands that neither change shell state nor create/modify/re-point files by
# themselves (output redirection is handled separately). Anything not listed
# here, and not a modelled search, is undecidable.
_FILE_READERS = frozenset({
    "cat", "head", "tail", "ls", "wc", "cut", "column", "nl", "rev",
})
_READS_CWD_WITHOUT_OPERAND = frozenset({"ls"})  # the others read stdin then
# Their operands are not file contents: names, text, or metadata only (stat/du
# follow the same boundary as `find -newer`). Input redirection (`< file`) is
# still a read for every command.
_NON_READERS = frozenset({
    "echo", "printf", "date", "pwd", "true", "false", "tr", "basename", "dirname",
    "whoami", "id", "uname", "which", "test", "[", "stat", "du",
})
_READ_ONLY = _FILE_READERS | _NON_READERS
# Only HOME is taken from the hook's own environment: Claude Code runs each Bash
# command in a fresh shell initialised from the user's profile, which the hook
# process never loads, so any other variable may hold a different value there.
_ENV_VARS_FROM_PROCESS = frozenset({"HOME"})
_SPECIAL_VARS = frozenset({"PWD", "OLDPWD", "CDPATH"})
_GLOB_LIMIT = 1000  # more matches than this: undecidable (and keeps the hook fast)
_SEPARATORS = frozenset({";", "&&", "||", "|", "&", ";;", "|&"})
_OUTPUT_REDIRECTIONS = frozenset({">", ">>", ">|", "&>", "&>>", ">&", "<&"})
_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_VAR_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
_GLOB_RE = re.compile(r"[*?\[]")


def _expand(tok: str, env: dict) -> str:
    def _sub(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        if name in _SPECIAL_VARS:  # changed by cd; the hook cannot know the value
            raise _Undecidable(m.group(0))
        v = env.get(name)
        if v is None and name in _ENV_VARS_FROM_PROCESS:
            v = os.environ.get(name)
        if v is None:
            raise _Undecidable(m.group(0))
        return v
    out = _VAR_RE.sub(_sub, tok)
    if "$" in out:  # $1, $@, $?, ${VAR:-x} ...: not modelled
        raise _Undecidable(tok)
    if out == "~" or out.startswith("~/"):
        home = env.get("HOME", os.environ.get("HOME"))  # HOME may be reassigned
        if not home:
            raise _Undecidable(tok)
        return home + out[1:]
    return os.path.expanduser(out)  # ~user


def _kind(p: Path, root: Path) -> str:
    resolved = p.resolve()  # its own errors propagate: only relative_to means 'out'
    try:
        resolved.relative_to(root)
    except ValueError:
        return "out"
    return "in"


def _classify(tok: str, cwd: Path, root: Path, *, target: bool) -> str | None:
    """'in' / 'out' of root, or None for a non-target token that is not a path.

    A search TARGET that does not exist yet is undecidable: a previous command,
    or a pattern read in the wrong slot, could make it anything. Globs are
    expanded and every match resolved, so `..` after a wildcard and symlinks
    count; brace expansion is not modelled.
    """
    if "://" in tok:
        return None
    if "{" in tok or "}" in tok:
        raise _Undecidable(tok)
    full = tok if os.path.isabs(tok) else str(cwd / tok)
    if _GLOB_RE.search(tok):
        if "**" in tok:  # recursive in zsh (and bash globstar), not in plain glob
            raise _Undecidable(tok)
        kinds = set()
        for n, match in enumerate(_glob.iglob(full)):
            if n >= _GLOB_LIMIT:
                raise _Undecidable(tok)
            kinds.add(_kind(Path(match), root))
        if not kinds:
            if target:
                raise _Undecidable(tok)  # bash passes the literal on; zsh errors out
            return None
        return "in" if "in" in kinds else "out"
    p = Path(full)
    if not p.exists():
        if target:
            raise _Undecidable(tok)
        return None
    return _kind(p, root)


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
    return roots or ["."]


def _split_redirections(words: list) -> tuple:
    clean, inputs, i = [], [], 0
    while i < len(words):
        w = words[i]
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if w.isdigit() and (nxt in _OUTPUT_REDIRECTIONS or nxt == "<"):
            i += 1
            continue
        if w in _OUTPUT_REDIRECTIONS:
            i += 2
            continue
        if w == "<":
            inputs.append(nxt)
            i += 2
            continue
        clean.append(w)
        i += 1
    return clean, inputs


def _segments(cmd_str: str) -> list:
    """[(words, separator_before, separator_after)] of a simple command list."""
    if any(s in cmd_str for s in ("$(", "`", "<<", "<(", ">(")):
        raise _Undecidable("substitution or heredoc")
    toks: list = []
    for line in cmd_str.replace("\\\n", " ").split("\n"):
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks += list(lex) + [";"]
    if "(" in toks or ")" in toks:
        raise _Undecidable("subshell")
    segs, cur, before = [], [], None
    for t in toks:
        if t in _SEPARATORS:
            if t in ("&", "|&"):
                raise _Undecidable("background job")
            if cur:
                segs.append([cur, before, t])
            cur, before = [], t
        else:
            cur.append(t)
    if cur:
        segs.append([cur, before, None])
    return segs


def _decide(cmd_str: str, root: Path) -> bool:
    env: dict = {}
    cwd = Path.cwd()
    pipe_kinds: list = []  # what the upstream commands of the current pipeline read
    per_search: list = []
    reads: list = []       # every file read by a non-search command in the line
    for words, before, after in _segments(cmd_str):
        from_pipe = before == "|"
        if not from_pipe:
            pipe_kinds = []
        assigns = []
        while words and _ASSIGN_RE.match(words[0]):
            assigns.append(_ASSIGN_RE.match(words[0]).groups())
            words = words[1:]
        if not words:  # standalone assignment: sets shell variables
            for key, val in assigns:
                env[key] = _expand(val, env)
            continue
        # a prefix assignment (`D=x cmd $D`) only reaches the child's environment:
        # $D in this command still expands to the OLD value, so env is untouched.
        words, inputs = _split_redirections(words)
        if not words:
            continue
        name = words[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
        args = words[1:]

        if name == "cd":
            if from_pipe or after in ("|", "||") or len(args) > 1:
                raise _Undecidable("cd")
            dest = _expand(args[0], env) if args else _expand("~", env)
            if dest == "-":
                raise _Undecidable("cd -")
            if not os.path.isabs(dest) and (env.get("CDPATH") or os.environ.get("CDPATH")):
                raise _Undecidable("CDPATH")
            new = Path(dest) if os.path.isabs(dest) else cwd / dest
            if not new.is_dir():  # cd would fail: the next command runs in the old cwd
                raise _Undecidable("cd to a missing dir")
            cwd = new
            continue

        if name in _TABLES or name == "find":
            if name == "find":
                targets, pattern_files = _find_roots(args), []
            else:
                targets, pattern_files = _grep_like_operands(args, _TABLES[name])
            reads_stdin = "-" in targets or (not targets and name != "find")
            files = [t for t in targets if t != "-"] + inputs + pattern_files
            kinds = [k for k in (_classify(_expand(t, env), cwd, root, target=True)
                                 for t in files) if k]
            if reads_stdin and from_pipe:
                kinds += pipe_kinds  # it also reads the upstream commands' output
            if not files and not (reads_stdin and from_pipe):
                kinds = [_kind(cwd, root)]  # no operand: searches the cwd
            if not kinds:
                raise _Undecidable("no decidable target")
            per_search.append(kinds)
            pipe_kinds += kinds
            continue

        if name not in _READ_ONLY:
            raise _Undecidable(name)  # may create, modify or re-point a target
        # reading project code counts as touching the project, whether or not a
        # later search consumes it through the pipe; `< file` is a read for all
        operands = inputs if name in _NON_READERS else args + inputs
        for a in args:  # still refuse what the gate cannot expand
            _expand(a, env)
        read = [k for k in (_classify(_expand(a, env), cwd, root, target=False)
                            for a in operands if not a.startswith("-")) if k]
        if not read and name in _READS_CWD_WITHOUT_OPERAND:
            read = [_kind(cwd, root)]
        reads += read
        pipe_kinds += read
    return (bool(per_search)
            and all(k == "out" for ks in per_search for k in ks)
            and all(k == "out" for k in reads))


def bash_searches_only_out_of_project(cmd_str: str, root: Path) -> bool:
    """True only when every search the Bash command runs provably targets paths
    outside `root`. Any error or uncertainty returns False: keep nudging."""
    if os.name == "nt":
        return False
    try:
        return _decide(cmd_str, root)
    except Exception:  # _Undecidable, shlex ValueError, OSError...
        return False


def grep_tool_path_out_of_project(tool_input: dict, root: Path, is_cwd_relative) -> bool:
    """Grep tool: quiet only for an explicit, non-cwd-relative `path` that
    resolves outside `root` (no path means a project-wide search)."""
    try:
        v = str(tool_input.get("path") or "")
        if not v or is_cwd_relative(v):
            return False
        resolved = Path(v).resolve()
    except Exception:  # includes ValueError from a malformed path: keep nudging
        return False
    try:
        resolved.relative_to(root)
    except ValueError:
        return True
    return False
