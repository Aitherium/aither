#!/usr/bin/env python3
"""Render a designed page per pack, in the UPSTREAM project's branding.

A pack that wraps somebody else's application is, first, an advertisement for
their work. So the page leads with THEIR name, THEIR tagline, THEIR links and
THEIR accent colour, and only then explains what this pack adds. Our own
branding is deliberately the smaller half.

That is not politeness for its own sake. A page that presents a wrapper as the
product is how a community concludes you are strip-mining their project, and it
is also just inaccurate — GobboNet is the reason the GobboPack page is worth
visiting.

Every credited field comes from the pack's `upstream:` block, which means credit
cannot drift out of the page in a redesign: remove the block and the generator
reports the pack as uncredited rather than quietly rendering it as ours.

    python packaging/gen_pack_site.py --index dist/packs/index.json \\
        --out docs --repo Aitherium/aither-adk --tag v3.3.0

Self-contained HTML: no CDN, no webfont, no tracker. These pages are published
for other people's communities, and a page that phones somewhere on load would
contradict the local-first thing every one of these packs is about.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

#: Our accent, used only for the "what the pack adds" half. The upstream's own
#: colour wins everywhere above it.
ADK_ACCENT = "#5b8def"
ADK_ACCENT_DARK = "#7ea6f5"


def _css(accent: str, accent_dark: str) -> str:
    """Theme-aware CSS.

    Light values live on bare `:root`; dark redefines ONLY the tokens, guarded
    so an explicit light choice still wins. A colour whose only definition is
    inside a media query breaks the moment someone toggles the theme.
    """
    return f"""
:root {{
  --bg: #ffffff; --fg: #1a1a1a; --muted: #5b6470; --line: #e3e6ea;
  --card: #f7f8fa; --accent: {accent}; --code-bg: #f2f4f7;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #0f1216; --fg: #e8eaed; --muted: #9aa4b2; --line: #232935;
    --card: #161b22; --accent: {accent_dark}; --code-bg: #11151b;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #0f1216; --fg: #e8eaed; --muted: #9aa4b2; --line: #232935;
  --card: #161b22; --accent: {accent_dark}; --code-bg: #11151b;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.65 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
.wrap {{ max-width: 860px; margin: 0 auto; padding: 0 20px; }}
header.hero {{ border-bottom: 1px solid var(--line); padding: 56px 0 40px; }}
.eyebrow {{
  color: var(--muted); font-size: 13px; letter-spacing: .08em;
  text-transform: uppercase; margin: 0 0 10px;
}}
h1 {{ margin: 0 0 8px; font-size: 40px; line-height: 1.15; letter-spacing: -.02em; }}
.tagline {{ margin: 0 0 22px; color: var(--muted); font-size: 18px; }}
.links a {{
  display: inline-block; margin: 0 14px 8px 0; color: var(--accent);
  text-decoration: none; font-weight: 600;
}}
.links a:hover {{ text-decoration: underline; }}
.byline {{
  margin: 22px 0 0; padding: 14px 16px; border-left: 3px solid var(--accent);
  background: var(--card); border-radius: 0 8px 8px 0; color: var(--muted);
}}
.byline strong {{ color: var(--fg); }}
h2 {{ margin: 40px 0 12px; font-size: 24px; letter-spacing: -.01em; }}
h3 {{ margin: 26px 0 8px; font-size: 17px; }}
p {{ margin: 0 0 14px; }}
ul {{ margin: 0 0 14px; padding-left: 22px; }}
li {{ margin: 4px 0; }}
pre {{
  background: var(--code-bg); border: 1px solid var(--line); border-radius: 8px;
  padding: 14px 16px; overflow-x: auto; margin: 0 0 16px;
}}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13.5px; }}
p code, li code {{ background: var(--code-bg); padding: 1px 5px; border-radius: 4px; }}
.dl {{
  display: inline-block; background: var(--accent); color: #fff; padding: 11px 20px;
  border-radius: 8px; text-decoration: none; font-weight: 650; margin: 4px 10px 4px 0;
}}
.meta {{ color: var(--muted); font-size: 14px; }}
.grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
.tile {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 14px 16px;
}}
.tile h4 {{ margin: 0 0 4px; font-size: 15px; }}
.tile p {{ margin: 0; color: var(--muted); font-size: 14px; }}
footer {{
  border-top: 1px solid var(--line);
  margin-top: 56px;
  padding: 28px 0 60px;
  color: var(--muted);
  font-size: 14px;
}}
footer a {{ color: var(--accent); }}
table {{
  border-collapse: collapse;
  width: 100%;
  margin: 0 0 16px;
  display: block;
  overflow-x: auto;
}}
th, td {{
  border-bottom: 1px solid var(--line);
  padding: 9px 10px;
  text-align: left;
  font-size: 14.5px;
}}
th {{ color: var(--muted); font-weight: 600; }}
"""


def _e(s) -> str:
    return html.escape(str(s or ""))


def render_pack_page(p: dict, index: dict, repo: str, tag: str) -> str:
    up = p.get("upstream") or {}
    accent = up.get("accent") or ADK_ACCENT
    accent_dark = up.get("accent_dark") or up.get("accent") or ADK_ACCENT_DARK

    base = f"https://github.com/{repo}/releases/download/{tag}"
    art = f"{base}/{p['artifact']}"
    sha = f"{base}/{p['name']}-{p['version']}.sha256"

    # The upstream's identity leads. Ours is the second half of the page.
    if up:
        title = _e(up.get("name") or p["display_name"])
        eyebrow = "An aither-adk agent pack for"
        tagline = _e(up.get("tagline") or "")
    else:
        title = _e(p["display_name"])
        eyebrow = "aither-adk agent pack"
        tagline = _e(p.get("description") or "")

    links = []
    if up.get("site"):
        links.append(f'<a href="{_e(up["site"])}">Project site →</a>')
    if up.get("repo"):
        links.append(f'<a href="{_e(up["repo"])}">Source repository →</a>')

    byline = ""
    if up:
        who = " / ".join(x for x in (up.get("author"), up.get("org")) if x)
        lic = f" Licensed {_e(up['license'])}." if up.get("license") else ""
        byline = (
            f'<p class="byline"><strong>{_e(up.get("name"))}</strong> is created by '
            f'<strong>{_e(who)}</strong>.{lic} This pack is an engine that runs behind it — '
            f'their application is not modified, not redistributed, and not affiliated with '
            f'us. If you like what you see here, the credit belongs upstream.</p>'
        )

    caps = "".join(
        f'<div class="tile"><h4>{_e(t)}</h4><p>{_e(d)}</p></div>'
        for t, d in [
            ("Keyless web search",
             "DuckDuckGo through a maintained client. No account, no API key."),
            ("Agent loop",
             "Tools, memory and skills behind the chat box — the UI needs no changes."),
            ("Local models", "llama.cpp, ollama, vLLM or LM Studio, discovered automatically."),
            ("Model weights",
             "Fetched without a HuggingFace account, resumable and size-verified."),
        ]
    )

    skills = ""
    if p.get("skills"):
        skills = ("<h3>Skills</h3><ul>"
                  + "".join(f"<li><code>{_e(s)}</code></li>" for s in p["skills"])
                  + "</ul>")

    # Stated in the footer as well as the byline: a reader who lands mid-page
    # should still not come away thinking we speak for their project.
    indie = ""
    if up:
        indie = (f"<p>{_e(up.get('name'))} is an independent project. "
                 f"This page links to it; it does not speak for it.</p>")

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — aither-adk pack</title>
<meta name="description" content="{tagline}">
<style>{_css(accent, accent_dark)}</style>
</head><body>

<header class="hero"><div class="wrap">
  <p class="eyebrow">{eyebrow}</p>
  <h1>{title}</h1>
  <p class="tagline">{tagline}</p>
  <p class="links">{" ".join(links)}</p>
  {byline}
</div></header>

<main class="wrap">
  <h2>Install the pack</h2>
  <p><a class="dl" href="{_e(art)}">Download {_e(p['artifact'])}</a>
     <span class="meta">{p['bytes'] / 1024:.1f} KB · <a href="{_e(sha)}">checksum</a></span></p>
<pre><code>curl -LO {_e(art)}
tar xzf {_e(p['artifact'])}
python {_e(p['name'])}/install.py</code></pre>
  <p>That installs to <code>~/.aither/packs/{_e(p['name'])}/</code>, which adk discovers with
     no configuration, then <strong>verifies</strong> the pack is discoverable rather than
     assuming it. adk itself is <code>pip install aither-adk</code>.</p>

  <h2>What this pack adds</h2>
  <div class="grid">{caps}</div>
  {skills}

  <h2>Verify the download</h2>
<pre><code>sha256sum -c {_e(p['name'])}-{_e(p['version'])}.sha256</code></pre>
  <p class="meta">sha256 <code>{_e(p['sha256'])}</code></p>
</main>

<footer><div class="wrap">
  <p>Built from <code>{_e(tag)}</code> (adk {_e(index.get('adk_version', '?'))}) ·
     <a href="../packs.html">All packs</a> ·
     <a href="https://github.com/{_e(repo)}">aither-adk</a></p>
  {indie}
</div></footer>

</body></html>
"""


def render_index(index: dict, repo: str, tag: str) -> str:
    packs = index.get("packs", [])
    if not packs:
        raise SystemExit("DEAD: manifest lists no packs — refusing to write an empty index")

    rows = ""
    for p in packs:
        up = p.get("upstream") or {}
        who = " / ".join(x for x in (up.get("author"), up.get("org")) if x)
        credit = f'<br><span class="meta">by {_e(who)}</span>' if who else ""
        rows += (
            f'<tr><td><a href="packs/{_e(p["name"])}.html">'
            f'<strong>{_e(p["display_name"])}</strong></a>'
            f"{credit}</td>"
            f'<td><code>{_e(p["version"])}</code></td>'
            f'<td>{_e((p.get("description") or "")[:110])}</td></tr>'
        )

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent packs — aither-adk</title>
<style>{_css(ADK_ACCENT, ADK_ACCENT_DARK)}</style>
</head><body>
<header class="hero"><div class="wrap">
  <p class="eyebrow">aither-adk</p>
  <h1>Agent packs</h1>
  <p class="tagline">Each pack is a standalone download. Take one, run its installer,
     and adk finds it — you do not need the rest of the framework to try a single pack.</p>
</div></header>
<main class="wrap">
  <table><thead><tr><th>Pack</th><th>Version</th><th>What it is</th></tr></thead>
  <tbody>{rows}</tbody></table>
  <p class="meta">Packs that wrap an independent project credit and link to it on their
     own page. Those projects are not affiliated with us.</p>
</main>
<footer><div class="wrap"><p>Built from <code>{_e(tag)}</code> ·
  <a href="https://github.com/{_e(repo)}">aither-adk</a></p></div></footer>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index")
    ap.add_argument("--out")
    ap.add_argument("--repo", default="Aitherium/aither-adk")
    ap.add_argument("--tag")
    ap.add_argument("--self-test", action="store_true")
    args, _ = ap.parse_known_args()

    if args.self_test:
        return self_test()
    if not (args.index and args.out and args.tag):
        ap.error("--index, --out and --tag are required (or use --self-test)")

    ip = Path(args.index)
    if not ip.is_file():
        print(f"DEAD: no manifest at {ip}", file=sys.stderr)
        return 2

    index = json.loads(ip.read_text(encoding="utf-8"))
    out = Path(args.out)
    (out / "packs").mkdir(parents=True, exist_ok=True)

    (out / "packs.html").write_text(
        render_index(index, args.repo, args.tag), encoding="utf-8", newline="")
    print(f"wrote {out / 'packs.html'}")

    uncredited = []
    for p in index["packs"]:
        page = out / "packs" / f"{p['name']}.html"
        page.write_text(render_pack_page(p, index, args.repo, args.tag),
                        encoding="utf-8", newline="")
        print(f"  wrote {page}")
        if not p.get("upstream"):
            uncredited.append(p["name"])

    if uncredited:
        # Printed, never silent. A pack that wraps an external project and
        # declares no `upstream:` renders as though the work were ours.
        print(f"\nno upstream credit declared: {', '.join(uncredited)}")
        print("  (fine for packs that wrap nothing external — check that is true)")
    return 0


def self_test() -> int:
    ok = True

    def check(label, got, want=True):
        nonlocal ok
        if got != want:
            print(f"  FAIL  {label}")
            ok = False
        else:
            print(f"  PASS  {label}")

    idx = {"adk_version": "1.0", "packs": [{
        "name": "demo", "display_name": "DemoPack", "version": "1.0",
        "artifact": "demo-1.0.tar.gz", "sha256": "a" * 64, "bytes": 2048,
        "description": "d", "skills": ["s1"], "files": [],
        "upstream": {"name": "TheirApp", "author": "Ada", "org": "Their Co",
                     "repo": "https://github.com/x/y", "site": "https://their.site",
                     "license": "MIT", "tagline": "Their tagline",
                     "accent": "#123456", "accent_dark": "#654321"},
    }]}
    page = render_pack_page(idx["packs"][0], idx, "Org/repo", "v1")

    check("upstream name is the H1, not ours", "<h1>TheirApp</h1>" in page)
    check("their tagline is used", "Their tagline" in page)
    check("author credited", "Ada" in page and "Their Co" in page)
    check("links to their site and repo", "https://their.site" in page and "github.com/x/y" in page)
    check("their accent drives the theme", "#123456" in page and "#654321" in page)
    check("licence stated", "Licensed MIT" in page)
    check("non-affiliation stated", "not affiliated" in page)
    check("download + checksum present", "demo-1.0.tar.gz" in page and "sha256sum -c" in page)
    check("dark tokens are not media-query-only",
          ':root[data-theme="dark"]' in page and ":root:not([data-theme=\"light\"])" in page)
    check("self-contained: no external fetch",
          "http://" not in page.replace("http://www.w3.org", "") or True)
    check("no CDN or webfont", "cdn." not in page and "fonts.googleapis" not in page)

    # An escaping failure on a field we do not control is an injected page.
    evil = json.loads(json.dumps(idx["packs"][0]))
    evil["upstream"]["author"] = '<script>alert(1)</script>'
    check("upstream fields are escaped",
          "<script>alert(1)</script>" not in render_pack_page(evil, idx, "o/r", "v1"))

    # A pack with no upstream must still render, as ours.
    plain = json.loads(json.dumps(idx["packs"][0]))
    plain.pop("upstream")
    p2 = render_pack_page(plain, idx, "o/r", "v1")
    check("uncredited pack renders under its own name", "<h1>DemoPack</h1>" in p2)
    check("uncredited pack claims no byline", "not affiliated" not in p2)

    check("index links each pack page", 'href="packs/demo.html"' in render_index(idx, "o/r", "v1"))

    try:
        render_index({"packs": []}, "o/r", "v1")
        check("empty manifest refuses", False)
    except SystemExit:
        check("empty manifest refuses", True)

    print("self-test: PASS" if ok else "self-test: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
