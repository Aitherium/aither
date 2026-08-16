"""``adk gobbonet`` — get GobboNet running with keyless search, in one command.

The setup was three commands: clone the UI, start the server, then set an env
var to a URL you had to know. Every one of those is a place to stop, and the
last is the worst — nothing tells you the value is wrong. You get a UI with a
search box that returns nothing, which reads as "search is broken" rather than
"SEARCH_URL is unset".

So this does all of it: finds or clones the UI, starts the server, wires the
search URL, and prints the one line the user actually needs. It is idempotent —
run it again and it reuses the checkout rather than failing on a directory that
already exists.

    adk gobbonet                      # clone if needed, serve, print the URL
    adk gobbonet --ui ./GobboNet      # use a checkout you already have
    adk gobbonet --no-open            # don't open a browser

Deliberately NOT silent about what it did: it prints the checkout path, the
port, and the SEARCH_URL to paste, because a wrapper that hides the underlying
commands leaves the user unable to debug it or run the pieces themselves.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

UPSTREAM = "https://github.com/ElodineOfficial/GobboNet"
DEFAULT_PORT = 11434


def _looks_like_gobbonet(d: Path) -> bool:
    return (d / "chat.html").is_file()


def _find_existing(explicit: str | None) -> Path | None:
    """Locate a GobboNet checkout without cloning."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if _looks_like_gobbonet(p) else None
    for cand in (Path.cwd(), Path.cwd() / "GobboNet", Path.home() / "GobboNet"):
        if _looks_like_gobbonet(cand):
            return cand.resolve()
    return None


def _clone(dest: Path) -> Path:
    if not shutil.which("git"):
        raise SystemExit(
            "git is not installed, and no GobboNet checkout was found.\n"
            f"Either install git, or download {UPSTREAM} and pass --ui <folder>."
        )
    print(f"cloning {UPSTREAM} -> {dest}")
    r = subprocess.run(["git", "clone", "--depth", "1", UPSTREAM, str(dest)])
    if r.returncode != 0:
        raise SystemExit(
            f"clone failed (exit {r.returncode}). Download {UPSTREAM} manually "
            "and pass --ui <folder>."
        )
    if not _looks_like_gobbonet(dest):
        # A clone that "succeeded" into an unexpected layout must not be served
        # as though it were fine — the server would 404 every asset.
        raise SystemExit(f"cloned, but {dest} has no chat.html — layout changed upstream?")
    return dest


def cmd_gobbonet(args) -> int:
    ui = _find_existing(getattr(args, "ui", None))

    if ui is None:
        if getattr(args, "ui", None):
            raise SystemExit(
                f"no chat.html under {args.ui} — point --ui at a GobboNet checkout"
            )
        ui = _clone(Path.cwd() / "GobboNet")
    else:
        print(f"using GobboNet checkout: {ui}")

    requested = int(getattr(args, "port", None) or DEFAULT_PORT)
    host = getattr(args, "host", None) or "127.0.0.1"

    from adk.packs.gobbonet.server import _DefaultEngine, serve

    # BIND FIRST, then print. The requested port is not necessarily the bound
    # one — serve() falls back to an OS-assigned port when the requested one is
    # inside a Windows reserved block — so printing a SEARCH_URL beforehand
    # hands the user a URL nothing is listening on, which is a worse failure
    # than the bind error it was hiding.
    httpd = serve(ui, _DefaultEngine(), host=host, port=requested)
    port = httpd.server_address[1]

    search_url = f"http://{host}:{port}/web_search"
    # Set it for any child process, and print it for the UI's own settings —
    # the server cannot reach into the browser's config, so telling the user
    # is not optional.
    os.environ["SEARCH_URL"] = search_url

    print()
    print(f"  UI          http://{host}:{port}/chat.html")
    print(f"  SEARCH_URL  {search_url}")
    print("              ^ paste this into GobboNet's search settings")
    print()
    print("  keyless DuckDuckGo search - no account, no API key, nothing hosted")
    print()

    if not getattr(args, "no_open", False):
        try:
            webbrowser.open(f"http://{host}:{port}/chat.html")
        except Exception as e:  # noqa: BLE001 - a headless box has no browser
            print(f"  (could not open a browser: {e})")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m adk.packs.gobbonet.launch`."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="adk gobbonet",
        description="Run GobboNet with keyless web search, in one command.",
    )
    ap.add_argument("--ui", help="existing GobboNet checkout (default: find or clone)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    return cmd_gobbonet(ap.parse_args(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
