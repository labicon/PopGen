"""
Web-based annotation tool for rollout behavior adherence.

Launches a local web server that presents each rollout GIF alongside its
behavior description and asks the annotator to judge whether the rollout
follows the specified behavior (True / False), with an optional reason box
when the answer is False.

Usage:
    python scripts/annotate_rollouts.py <results_path>

Example:
    python scripts/annotate_rollouts.py \
        scripts/population/behavior_policy/results/football__stratified__gpt-5-nano
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import random
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def load_behaviors(results_path: Path) -> list[dict]:
    """Load the behaviors.json from the results directory."""
    behaviors_file = results_path / "behaviors.json"
    if not behaviors_file.exists():
        sys.exit(f"Error: behaviors.json not found in {results_path}")
    return json.loads(behaviors_file.read_text())


def discover_rollouts(results_path: Path, behaviors: list[dict]) -> list[dict]:
    """Walk the results directory and collect every rollout with its metadata.

    Returns a list of dicts, each containing:
        behavior_idx, behavior_name, behavior_description,
        attempt_idx, rollout_idx, gif_path (relative to results_path)
    """
    rollouts: list[dict] = []
    for behavior_dir in sorted(results_path.glob("behavior_*")):
        behavior_idx = int(behavior_dir.name.split("_")[1])
        if behavior_idx >= len(behaviors):
            continue
        behavior = behaviors[behavior_idx]
        for attempt_dir in sorted(behavior_dir.glob("attempt_*")):
            attempt_idx = int(attempt_dir.name.split("_")[1])
            rollouts_dir = attempt_dir / "rollouts"
            if not rollouts_dir.is_dir():
                continue
            for rollout_dir in sorted(rollouts_dir.glob("rollout_*")):
                rollout_idx = int(rollout_dir.name.split("_")[1])
                gif = rollout_dir / "rollout.gif"
                if not gif.exists():
                    continue
                rollouts.append(
                    {
                        "behavior_idx": behavior_idx,
                        "behavior_name": behavior["name"],
                        "behavior_description": behavior["description"],
                        "attempt_idx": attempt_idx,
                        "rollout_idx": rollout_idx,
                        "gif_rel": str(gif.relative_to(results_path)),
                    }
                )
    return rollouts


# ---------------------------------------------------------------------------
# Annotation persistence
# ---------------------------------------------------------------------------

def annotations_file(results_path: Path) -> Path:
    return results_path / "annotations.json"


def load_annotations(results_path: Path) -> dict:
    path = annotations_file(results_path)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_annotations(results_path: Path, annotations: dict) -> None:
    path = annotations_file(results_path)
    path.write_text(json.dumps(annotations, indent=2))


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; background: #f5f5f5; color: #222; }
.container { max-width: 900px; margin: 0 auto; padding: 24px; }
h1 { margin-bottom: 8px; }
.progress { color: #666; margin-bottom: 24px; font-size: 0.95em; }
.progress-bar-outer { background: #ddd; border-radius: 4px; height: 8px; margin-bottom: 24px; }
.progress-bar-inner { background: #4caf50; height: 100%; border-radius: 4px; transition: width 0.3s; }
.card { background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 24px; margin-bottom: 24px; }
.behavior-name { font-size: 1.2em; font-weight: 600; margin-bottom: 8px; }
.behavior-desc { color: #444; line-height: 1.5; margin-bottom: 16px; }
.meta { font-size: 0.85em; color: #888; margin-bottom: 12px; }
.gif-wrapper { text-align: center; margin-bottom: 20px; }
.gif-wrapper img { max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }
.form-group { margin-bottom: 16px; }
label { font-weight: 500; display: block; margin-bottom: 6px; }
.radio-group { display: flex; gap: 16px; }
.radio-group label { font-weight: 400; cursor: pointer; }
textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-family: inherit; font-size: 0.95em; resize: vertical; }
button { background: #1976d2; color: #fff; border: none; padding: 10px 28px; border-radius: 6px; font-size: 1em; cursor: pointer; }
button:hover { background: #1565c0; }
.nav { display: flex; gap: 12px; margin-bottom: 24px; }
.nav a { text-decoration: none; color: #1976d2; font-size: 0.95em; }
.done-msg { text-align: center; padding: 48px 0; }
.done-msg h2 { margin-bottom: 12px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: 600; }
.badge-true { background: #e8f5e9; color: #2e7d32; }
.badge-false { background: #ffebee; color: #c62828; }
.badge-pending { background: #fff3e0; color: #e65100; }
.summary-table { width: 100%; border-collapse: collapse; margin-top: 16px; }
.summary-table th, .summary-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }
.summary-table th { background: #fafafa; font-weight: 600; }
.summary-table tr:hover { background: #f9f9f9; }
"""

def _esc(s: str) -> str:
    return html.escape(s)


def render_annotation_page(
    rollout: dict,
    idx: int,
    total: int,
    annotation: dict | None,
    n_annotated: int,
) -> str:
    checked_true = 'checked' if annotation and annotation.get("adherent") is True else ''
    checked_false = 'checked' if annotation and annotation.get("adherent") is False else ''
    reason_val = _esc(annotation.get("reason", "")) if annotation else ""
    reason_display = "block" if (annotation and annotation.get("adherent") is False) else "none"

    return f"""\
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotate Rollout {idx+1}/{total}</title>
<style>{CSS}</style>
<script>
function toggleReason() {{
    var v = document.querySelector('input[name="adherent"]:checked');
    document.getElementById('reason-box').style.display = (v && v.value === 'false') ? 'block' : 'none';
}}
document.addEventListener('keydown', function(e) {{
    if (e.target.tagName === 'TEXTAREA') return;
    if (e.key === 't' || e.key === 'T') {{
        e.preventDefault();
        document.querySelector('input[name="adherent"][value="true"]').checked = true;
        toggleReason();
        document.querySelector('form').submit();
    }} else if (e.key === 'f' || e.key === 'F') {{
        e.preventDefault();
        document.querySelector('input[name="adherent"][value="false"]').checked = true;
        toggleReason();
        document.querySelector('form').submit();
    }}
}});
</script>
</head><body>
<div class="container">
<h1>Rollout Annotation</h1>
<div class="progress">{n_annotated} / {total} annotated</div>
<div class="progress-bar-outer"><div class="progress-bar-inner" style="width:{100*n_annotated//total if total else 0}%"></div></div>
<div class="nav">
  {"<a href='/?idx="+str(idx-1)+"'>&larr; Previous</a>" if idx > 0 else ""}
  <a href="/summary">Summary</a>
  {"<a href='/?idx="+str(idx+1)+"'>Next &rarr;</a>" if idx < total - 1 else ""}
</div>
<div class="card">
  <div class="meta">Behavior {rollout['behavior_idx']} &middot; Attempt {rollout['attempt_idx']} &middot; Rollout {rollout['rollout_idx']}</div>
  <div class="behavior-name">{_esc(rollout['behavior_name'])}</div>
  <div class="behavior-desc">{_esc(rollout['behavior_description'])}</div>
</div>
<div class="card">
  <div class="gif-wrapper">
    <img src="/gif/{urllib.parse.quote(rollout['gif_rel'])}" alt="rollout gif">
  </div>
</div>
<div class="card">
  <form method="POST" action="/annotate">
    <input type="hidden" name="idx" value="{idx}">
    <div class="form-group">
      <label>Does this rollout follow the specified behavior?</label>
      <div class="radio-group">
        <label><input type="radio" name="adherent" value="true" {checked_true} onclick="toggleReason()" required> True</label>
        <label><input type="radio" name="adherent" value="false" {checked_false} onclick="toggleReason()" required> False</label>
      </div>
    </div>
    <div class="form-group" id="reason-box" style="display:{reason_display}">
      <label>Reason (why does it not follow the behavior?):</label>
      <textarea name="reason" rows="4" placeholder="Describe why this rollout does not follow the behavior...">{reason_val}</textarea>
    </div>
    <button type="submit">Save &amp; Next</button>
  </form>
</div>
</div></body></html>"""


def render_summary_page(rollouts: list[dict], annotations: dict, total: int) -> str:
    n_annotated = len(annotations)
    rows = ""
    for i, r in enumerate(rollouts):
        key = str(i)
        ann = annotations.get(key)
        if ann is None:
            badge = '<span class="badge badge-pending">pending</span>'
            reason = ""
        elif ann["adherent"]:
            badge = '<span class="badge badge-true">True</span>'
            reason = ""
        else:
            badge = '<span class="badge badge-false">False</span>'
            reason = _esc(ann.get("reason", ""))
        rows += f"""\
<tr>
  <td><a href="/?idx={i}">{i+1}</a></td>
  <td>{_esc(r['behavior_name'])}</td>
  <td>A{r['attempt_idx']}/R{r['rollout_idx']}</td>
  <td>{badge}</td>
  <td>{reason}</td>
</tr>"""

    return f"""\
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotation Summary</title>
<style>{CSS}</style>
</head><body>
<div class="container">
<h1>Annotation Summary</h1>
<div class="progress">{n_annotated} / {total} annotated</div>
<div class="progress-bar-outer"><div class="progress-bar-inner" style="width:{100*n_annotated//total if total else 0}%"></div></div>
<div class="nav"><a href="/?idx=0">&larr; Back to annotation</a></div>
<div class="card">
<table class="summary-table">
<thead><tr><th>#</th><th>Behavior</th><th>Attempt/Rollout</th><th>Adherent</th><th>Reason</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
</div></body></html>"""


def render_done_page(total: int) -> str:
    return f"""\
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Done</title>
<style>{CSS}</style>
</head><body>
<div class="container">
<div class="done-msg">
<h2>All {total} rollouts annotated!</h2>
<p>Annotations saved. You can <a href="/summary">view the summary</a> or close this tab.</p>
</div>
</div></body></html>"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class AnnotationHandler(SimpleHTTPRequestHandler):
    results_path: Path
    rollouts: list[dict]
    annotations: dict

    def log_message(self, format, *args):
        # quieter logging
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if parsed.path.startswith("/gif/"):
            self._serve_gif(parsed.path[5:])  # strip "/gif/"
            return

        if parsed.path == "/summary":
            body = render_summary_page(self.rollouts, self.annotations, len(self.rollouts))
            self._respond_html(body)
            return

        # Main annotation page
        idx = int(qs.get("idx", [self._next_unannotated()])[0])
        idx = max(0, min(idx, len(self.rollouts) - 1))
        rollout = self.rollouts[idx]
        ann = self.annotations.get(str(idx))
        n_annotated = len(self.annotations)
        body = render_annotation_page(rollout, idx, len(self.rollouts), ann, n_annotated)
        self._respond_html(body)

    def do_POST(self):
        if self.path != "/annotate":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode()
        params = urllib.parse.parse_qs(raw)
        idx = int(params["idx"][0])
        adherent = params["adherent"][0] == "true"
        reason = params.get("reason", [""])[0].strip()

        self.annotations[str(idx)] = {
            "adherent": adherent,
            "reason": reason if not adherent else "",
            "behavior_idx": self.rollouts[idx]["behavior_idx"],
            "behavior_name": self.rollouts[idx]["behavior_name"],
            "attempt_idx": self.rollouts[idx]["attempt_idx"],
            "rollout_idx": self.rollouts[idx]["rollout_idx"],
        }
        save_annotations(self.results_path, self.annotations)

        # Redirect to next unannotated or summary
        next_idx = self._next_unannotated()
        if next_idx is not None:
            target = f"/?idx={next_idx}"
        elif idx < len(self.rollouts) - 1:
            target = f"/?idx={idx + 1}"
        else:
            target = "/summary"
        self.send_response(303)
        self.send_header("Location", target)
        self.end_headers()

    def _next_unannotated(self) -> int | None:
        for i in range(len(self.rollouts)):
            if str(i) not in self.annotations:
                return i
        return None

    def _serve_gif(self, rel_path: str):
        rel_path = urllib.parse.unquote(rel_path)
        file_path = self.results_path / rel_path
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, f"GIF not found: {rel_path}")
            return
        # Ensure path doesn't escape results directory
        try:
            file_path.resolve().relative_to(self.results_path.resolve())
        except ValueError:
            self.send_error(403, "Forbidden")
            return
        data = file_path.read_bytes()
        content_type = mimetypes.guess_type(str(file_path))[0] or "image/gif"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _respond_html(self, body: str):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Annotate rollout behavior adherence")
    parser.add_argument("results_path", type=Path, help="Path to results directory")
    parser.add_argument("--port", type=int, default=8505, help="Port to serve on (default: 8505)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    results_path = args.results_path.resolve()
    if not results_path.is_dir():
        sys.exit(f"Error: {results_path} is not a directory")

    behaviors = load_behaviors(results_path)
    rollouts = discover_rollouts(results_path, behaviors)
    random.shuffle(rollouts)
    if not rollouts:
        sys.exit(f"Error: no rollouts found in {results_path}")

    annotations = load_annotations(results_path)

    # Inject shared state into the handler class
    AnnotationHandler.results_path = results_path
    AnnotationHandler.rollouts = rollouts
    AnnotationHandler.annotations = annotations

    n_annotated = len(annotations)
    print(f"Found {len(rollouts)} rollouts across {len(behaviors)} behaviors")
    if n_annotated:
        print(f"Resuming: {n_annotated}/{len(rollouts)} already annotated")
    print(f"Serving on http://{args.host}:{args.port}")

    server = HTTPServer((args.host, args.port), AnnotationHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nSaved {len(annotations)}/{len(rollouts)} annotations to {annotations_file(results_path)}")
        server.server_close()


if __name__ == "__main__":
    main()
