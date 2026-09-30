"""The search guard must not nudge on searches that only touch out-of-project paths (#3882).

The read arm skips out-of-project targets (#1840 (a)), but the search arm decided
only "is this a search?" (#3121) and nudged on every Bash/Grep search while a graph
existed - `grep -n foo /some/other/repo/file`, `grep x /etc/hosts`, or the Grep tool
with an absolute `path` elsewhere all got the MANDATORY nudge, although nothing
searched belongs to the graph.

The fix (graphify/hook_search_gate.py) is an allow-list: it goes quiet only when
every search in the command provably targets paths outside the project root AND
the command has a shape the gate fully models. The second half of this file pins
the false-silence traps found while designing it - a whole-command path scan and a
deny-list both silenced real project searches (`grep x /tmp/a; rg y`,
`rg y > /tmp/out`, `rg --files`, unknown options, `find -exec`, background or
failed `cd`, `$1`, `${VAR:-x}`, symlink-following traversal).
"""
import io
import json
import os
import sys

import pytest

from graphify import __main__ as m

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX shell command parsing")


def _invoke(payload, project, monkeypatch):
    monkeypatch.setattr("graphify.paths.GRAPHIFY_OUT", "graphify-out")
    monkeypatch.setattr("graphify.paths.GRAPHIFY_OUT_NAME", "graphify-out")
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    monkeypatch.chdir(project)
    (project / "graphify-out").mkdir(parents=True, exist_ok=True)
    (project / "graphify-out" / "graph.json").write_text("{}", encoding="utf-8")

    class _Stdin:
        def __init__(self, b):
            self.buffer = io.BytesIO(b)

    monkeypatch.setattr(sys, "stdin", _Stdin(json.dumps(payload).encode("utf-8")))
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    m._run_hook_guard("search")
    return buf.getvalue()


@pytest.fixture
def layout(tmp_path):
    """project/ (graphified, with app/models.py) and a sibling elsewhere/ outside it."""
    project = tmp_path / "project"
    (project / "app").mkdir(parents=True)
    (project / "app" / "models.py").write_text("x = 1\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "README.md").write_text("foo\n", encoding="utf-8")
    return project, elsewhere


def _bash(command):
    return {"tool_input": {"command": command}}


# --------------------------------------------------------------------------- #
# Regression: out-of-project searches stay quiet (nudged before the fix)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("template", [
    "grep -n foo {E}/README.md",
    "grep -rn foo {E}",
    "rg foo {E}",
    "S={E}; grep -n foo $S/README.md",
    "grep -e foo {E}/README.md",
    'rg foo "{E}"',
    "find {E} -name '*.md'",
    "find {E} -type f -name '*.md' -print",
    "cat {E}/README.md | grep foo",
    "date; grep foo {E}/README.md",          # external command: no shell state change
    "cd {E} && grep -rn foo .",
    "cd {E}; grep -rn foo .",
    'grep -rn "/api/v1" {E}',
    "grep -rne foo {E}",                    # -e at the end of a cluster takes the pattern
    "grep -m 5 foo {E}/README.md",          # valued option
    "rg -E utf-8 foo {E}",                  # rg -E takes a value (encoding)
    "rg --color never foo {E}",             # rg --color takes a value
    "grep --color=auto foo {E}/README.md",  # grep --color takes it only with =
    "grep foo {E}/*.md",                    # glob expanded; every match is outside
    "find {E} -newer app/models.py -name x",  # -newer reads metadata only
    "echo hi | grep h {E}/README.md",       # echo reads no file
    "ls {E} | grep READ",                   # lists an outside directory
    "stat app/models.py; grep foo {E}/README.md",  # metadata only, like find -newer
    "du -sh app; grep foo {E}/README.md",
    "date -u; grep foo {E}/README.md",                   # options that read nothing
    "printf '%s' x; grep foo {E}/README.md",
    "printf '%-10s %%\\n' x; grep foo {E}/README.md",   # %s with flags/width, literal %%
    "test -f app/models.py; grep foo {E}/README.md",   # test -f is metadata
    "stat -f %z app/models.py; grep foo {E}/README.md",
    'S={E}; grep foo "$S/README.md"',                   # variable value without spaces
    "grep 'foo$' {E}/README.md",                        # a `$` that starts no expansion
    "rg 'foo.*bar' {E}/README.md",                      # glob chars matching no file
    "printf '%d\\n' 1; grep foo {E}/README.md",          # numeric conversion, plain number
    "printf '%b' 'hi\\n'; grep foo {E}/README.md",
    "test -v HOME; grep foo {E}/README.md",             # -v on a plain name only checks
])
def test_out_of_project_search_is_quiet(template, layout, monkeypatch):
    project, elsewhere = layout
    command = template.format(E=elsewhere)
    assert _invoke(_bash(command), project, monkeypatch).strip() == "", command


def test_grep_tool_with_out_of_project_path_is_quiet(layout, monkeypatch):
    project, elsewhere = layout
    payload = {"tool_input": {"pattern": "foo", "path": str(elsewhere)}}
    assert _invoke(payload, project, monkeypatch).strip() == ""


# --------------------------------------------------------------------------- #
# Searches that can touch the project keep nudging - including the traps
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("template", [
    "grep -rn foo app/",
    "grep -rn foo",                                   # no target = cwd = project
    "grep -rn x app/ {E}",                            # mixed targets
    "grep foo {E}/README.md; rg foo",                 # 2nd search runs in the project
    "rg foo > {E}/out.txt",                           # output redirection is not a target
    "cd {E} && rg foo {P}/app/models.py",             # absolute path back into the project
    "cd app && grep -rn x .",                         # cd inside the project
    "cd {E} && cd {P} && grep -rn x",                 # cd back into the project
    "rg foo $UNSET_VAR_3882",                         # unresolved variable
    "rg foo {E}/x ${{UNSET_VAR_3882}}/app",           # unresolved variable among outside paths
    "rg foo $D; D={E}",                               # assignment AFTER use
    "ls {E} | xargs grep foo",                        # xargs feeds unknown paths
    "grep foo $(ls {E})",                             # command substitution
    "(cd {E} && grep foo README.md); grep foo",       # subshell, then project search
    "cd - && grep foo x",                             # unknown cwd
    "cat app/models.py | grep foo",                   # pipe fed from the project
    "git log --oneline | grep fix",                   # pipe with no path: cwd
    "grep foo < app/models.py",                       # input redirection is a target
    'grep -rn "/api/v1"',                             # pattern is not a path
    "find . -name '*.py'",
    "git -C app grep foo",
    "grep -E foo app/ {E}",                           # grep -E takes NO value
    'rg -j 4 "/api/v1"',                              # valued option, pattern looks absolute
    "grep foo /nonexistent-3882/x",                   # absolute target that does not exist
    "sudo grep foo {E}/README.md",                    # wrapper feeding a search
    # allow-list: shapes the gate does not model keep nudging
    "rg --files app {E}",                             # no positional pattern: every operand is a root
    "rg --hyperlink-format default {E}/README.md",    # unknown option may shift the pattern slot
    "find {E} -exec rg foo {P}/app/models.py {{}} +", # -exec runs another search anywhere
    "find {E} -name x -exec grep -l foo {{}} \\;",
    "find -L {E} -name x",                            # follows symlinks during traversal
    "find {E} -follow -name x",
    "rg -L foo {E}",
    "grep -R foo {E}",
    "cd {E} & rg foo app/ {E}/README.md",             # background cd
    "cd missing_3882; rg foo app/ {E}/README.md",     # failed cd: rg runs in the project
    "cd {E} | rg foo app/",                           # cd in a pipeline: subshell
    "cd {E} || rg foo README.md",                     # runs only if cd failed
    "set -- app/; grep foo {E}/README.md $1",         # positional parameters
    "D=app/; grep foo {E}/README.md ${{D:-/tmp}}",    # parameter expansion with default
    "D={E} grep foo $D",                              # prefix assignment: $D is the OLD value
    "export X=1; grep foo {E}/README.md",             # builtin changing shell state
    "git -C {E} grep foo",                            # git grep is not modelled
    "ag -g foo {E}",
    "fd --follow foo {E}",
    "rg --color never {E}/README.md",                 # the path is rg's PATTERN: cwd searched
    "grep foo {E}/../pro*/app/models.py",             # glob climbs back into the project
    "cat app/models.py | grep foo - {E}/README.md",   # `-` reads the project file via the pipe
    "cp app/models.py app/new_3882.py; rg foo app/new_3882.py {E}/README.md",
    "rg foo app/new_3882.py {E}/README.md",           # target does not exist yet
    "HOME={P}; rg foo ~/app/models.py {E}/README.md", # ~ follows the reassigned HOME
    "cd {E}; rg foo $OLDPWD/app/models.py {E}/README.md",
    "grep -f app/models.py {E}/README.md",            # pattern file read from the project
    "rg --file=app/models.py {E}/README.md",
    "grep foo {E}/{{a,b}}.md",                        # brace expansion is not modelled
    "ls | grep foo",                                  # ls with no operand lists the cwd
    "grep foo {E}/**/README.md",                      # recursive glob in zsh
    "head -n 1 app/models.py | grep foo {E}/README.md",  # a pipe stage reads project code
    "cat < app/models.py | grep foo {E}/README.md",
    "cat app/models.py; grep foo {E}/README.md",      # reads project code in the same line
    "file -C -m app/models.py; grep foo {E}/README.md",  # `file` is not read-only
    "tr a b < app/models.py | grep foo {E}/README.md",   # `<` is a read for every command
    "tr a b <>app/models.py | grep foo {E}/README.md",   # `<>` opens read-write
    "< app/models.py; grep foo {E}/README.md",           # redirection without a command
    "cd < app/models.py; grep foo {E}/README.md",
    "du -X app/models.py; grep foo {E}/README.md",       # GNU du reads the exclude file
    "du -sX app/models.py {E}; grep foo {E}/README.md",
    "du --files0-from=app/models.py; grep foo {E}/README.md",
    "du --exclude-f=app/models.py {E}; grep foo {E}/README.md",  # abbreviated long option
    "date -f app/models.py; grep foo {E}/README.md",     # GNU date reads dates from a file
    'D={E}/README.md; printf -v D app/models.py; grep foo "$D"',  # printf -v assigns
    # word splitting: shlex drops quotes, so a value with spaces may be several args
    "O=' -X'; du $O app/models.py; grep foo {E}/README.md",
    "O='--summarize -X'; du $O app/models.py; grep foo {E}/README.md",
    "O=' -f'; date $O app/models.py; grep foo {E}/README.md",
    'D={E}/README.md; O=" -v"; printf $O D app/models.py; grep foo "$D"',
    "O='{E}/README.md app/models.py'; grep foo $O",
    "IFS=/; O=x; grep foo {E}/README.md $O",           # reassigned IFS
    # printf conversions that assign (%n) or evaluate arithmetic (zsh %d, * width)
    'D={E}/README.md; printf "%n" D; grep foo "$D"',
    "D={E}/README.md; printf '%d' 'D=0'; grep foo \"$D\"",
    "D={E}/README.md; printf '%*s' 'D=0' x; grep foo \"$D\"",
    # zsh stat -A/-H and bash test -v 'a[...]' assign variables
    'D={E}/README.md; stat -A D +size /dev/null; grep foo "$D"',
    "a=x; D={E}/README.md; test -v 'a[D=0]'; grep foo \"$D\"",
    "a=x; D={E}/README.md; [ -v 'a[D=0]' ]; grep foo \"$D\"",
    # variables expanded before options/pattern/operands are told apart
    "O=-r; grep $O {E}/README.md",                    # grep -r: the path is the pattern
    "O='--files app'; rg $O {E}/README.md",
    "O='x app/models.py'; grep -e $O {E}/README.md",
    "O='x -o -exec head -n 1 app/models.py ;'; find {E} -name $O",
    "O=; rg $O {E}/README.md",                        # empty value vanishes: path = pattern
    "cd {E}; D=; cd $D; rg -l x .",                   # `cd` with no argument goes HOME
    'F=; D={E}/README.md; printf $F "%n" D; grep foo "$D"',
    # a pattern the shell may glob into extra operands (quotes are invisible)
    "O='*'; rg $O {E}/README.md",
    "rg * {E}/README.md",
    "grep -e {{x,app/models.py}} {E}/README.md",      # brace expansion
])
def test_search_that_can_touch_the_project_nudges(template, layout, monkeypatch):
    project, elsewhere = layout
    command = template.format(E=elsewhere, P=project)
    out = _invoke(_bash(command), project, monkeypatch)
    assert "graphify query" in out, command


def test_process_env_variable_is_not_trusted(layout, monkeypatch):
    """The Bash tool's shell loads the user's profile; the hook process does not.
    A variable the hook sees pointing outside may point into the project there."""
    project, elsewhere = layout
    monkeypatch.setenv("GATE_TEST_DIR_3882", str(elsewhere))
    out = _invoke(_bash("rg foo $GATE_TEST_DIR_3882/README.md"), project, monkeypatch)
    assert "graphify query" in out


@pytest.mark.parametrize("tool_input", [
    {"pattern": "foo"},                     # no path = project-wide
    {"pattern": "foo", "path": "app"},      # relative = cwd-anchored = in project
])
def test_grep_tool_in_project_nudges(tool_input, layout, monkeypatch):
    project, _ = layout
    out = _invoke({"tool_input": tool_input}, project, monkeypatch)
    assert "graphify query" in out, tool_input


def test_grep_tool_absolute_path_inside_project_nudges(layout, monkeypatch):
    project, _ = layout
    payload = {"tool_input": {"pattern": "foo", "path": str(project / "app")}}
    assert "graphify query" in _invoke(payload, project, monkeypatch)


# --------------------------------------------------------------------------- #
# Fail toward nudging: an exception inside the gate never silences
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("payload_fn,attr", [
    (lambda E: {"tool_input": {"pattern": "foo", "path": str(E)}},
     "grep_tool_path_out_of_project"),
    (lambda E: {"tool_input": {"command": f"grep -n foo {E}/README.md"}},
     "bash_searches_only_out_of_project"),
])
def test_gate_exception_keeps_the_nudge(payload_fn, attr, layout, monkeypatch):
    project, elsewhere = layout

    def boom(*a, **k):
        raise RuntimeError("synthetic gate failure")

    monkeypatch.setattr(f"graphify.hook_search_gate.{attr}", boom)
    assert "graphify query" in _invoke(payload_fn(elsewhere), project, monkeypatch)
