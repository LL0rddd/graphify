"""A `NAME=value` prefix whose value contains `/` must not hide the search.

`_bash_invokes_search` (#3121) skipped assignment prefixes by testing for `=` only
after the LAST `/` of the token. For `D=/tmp grep foo x` that tail is `tmp`, so the
token was taken as the executable `tmp`, the segment ended there and the search was
never seen: `PATH=/usr/bin grep -rn x app/` - a search over project code - got no
nudge at all.
"""
import pytest

from graphify.cli import _bash_invokes_search


@pytest.mark.parametrize("cmd", [
    "D=/tmp grep foo x",
    "PATH=/usr/bin grep -rn x app/",
    "D=a/b rg foo src",
    "GREP_COLORS=mt=01 grep -rn x .",       # `=` inside the value
    "A=1 B=/x/y rg foo",                    # several prefixes
    "FOO=1 grep x f",                       # already worked; keep it
])
def test_assignment_prefix_with_slash_still_detects_search(cmd):
    assert _bash_invokes_search(cmd) is True, cmd


@pytest.mark.parametrize("cmd", [
    "D=/tmp make build",                    # prefix + a non-search command
    "./configure --prefix=/usr",            # `=` inside an option, not a prefix
    "/usr/bin/env",                         # a path, not an assignment
])
def test_non_search_commands_with_equals_stay_quiet(cmd):
    assert _bash_invokes_search(cmd) is False, cmd
