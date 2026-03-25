"""Simple HTML viewer for rollout sample trajectories from ROLL training logs.

Usage:
    python jl_patch/view_rollout_samples.py <log_file> [-o output.html]

Example:
    python jl_patch/view_rollout_samples.py output/logs/qwen3_5_4B_agentic_sokoban/qwen3_5_4B_agentic_sokoban/20260318-234454/log_rank_DRIVER_0_1.log
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any


def extract_samples(log_path: str) -> List[Dict[str, Any]]:
    """Extract rollout sample entries from the driver log file."""
    entries = []
    pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*agentic_pipeline\.py \(\d+\)\].*\[\[\{")
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                timestamp = m.group(1)
                json_start = line.index("[[")
                try:
                    data = json.loads(line[json_start:])
                    entries.append({"timestamp": timestamp, "groups": data})
                except json.JSONDecodeError:
                    continue
    return entries


def parse_turns(prompt: str, response: str) -> List[Dict[str, str]]:
    """Split a multi-turn trajectory into individual turns."""
    full_text = prompt + response
    # Split on im_start markers
    parts = re.split(r"<\|im_start\|>", full_text)
    turns = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Extract role (first word before newline)
        newline_idx = part.find("\n")
        if newline_idx == -1:
            role = part.replace("<|im_end|>", "").strip()
            content = ""
        else:
            role = part[:newline_idx].strip()
            content = part[newline_idx + 1:]
        # Clean up
        content = content.replace("<|im_end|>", "").strip()
        # Collapse vision tokens into a short placeholder
        content = re.sub(
            r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>",
            "[IMAGE]",
            content,
        )
        # Extract reasoning_content from <think>...</think> blocks
        reasoning_content = ""
        if role == "assistant" and "</think>" in content:
            reasoning_content = content.split("</think>")[0].split("<think>")[-1].strip()
            content = content.split("</think>")[-1].strip()
        elif role == "assistant" and "<think>" in content:
            # Unclosed think block (truncated)
            reasoning_content = content.split("<think>")[-1].strip()
            content = ""
        turns.append({"role": role, "content": content, "reasoning": reasoning_content})
    return turns


def render_html(entries: List[Dict[str, Any]]) -> str:
    """Render all entries as an HTML page."""
    steps_html = []
    for step_idx, entry in enumerate(entries):
        trajectories_html = []
        for traj_idx, group in enumerate(entry["groups"]):
            item = group[0] if isinstance(group, list) else group
            turns = parse_turns(item["prompt"], item["response"])
            episode_score = item["episode_score"]
            step_scores = item.get("step_score", [])

            # Score color
            if episode_score > 0:
                score_color = "#2e7d32"
            elif episode_score == 0:
                score_color = "#f57f17"
            else:
                score_color = "#c62828"

            turns_html = []
            for turn in turns:
                role = html.escape(turn["role"])
                content = html.escape(turn["content"])
                reasoning = html.escape(turn.get("reasoning", ""))
                if role == "system":
                    cls = "system"
                elif role == "user":
                    cls = "user"
                else:
                    cls = "assistant"
                reasoning_html = ""
                if reasoning:
                    reasoning_html = (
                        f'<details class="thinking" open>'
                        f'<summary>Thinking</summary>'
                        f'<div class="thinking-content">{reasoning}</div>'
                        f'</details>'
                    )
                elif role == "assistant":
                    reasoning_html = '<div class="no-thinking">(no thinking)</div>'
                turns_html.append(
                    f'<div class="turn {cls}">'
                    f'<div class="role">{role}</div>'
                    f'{reasoning_html}'
                    f'<div class="content">{content}</div>'
                    f"</div>"
                )

            # Step scores as a compact bar
            scores_html = ""
            if step_scores:
                score_items = []
                for i, s in enumerate(step_scores):
                    s_val = float(s) if not isinstance(s, (list, dict)) else 0
                    s_color = "#2e7d32" if s_val > 0 else ("#c62828" if s_val < 0 else "#9e9e9e")
                    score_items.append(f'<span class="step-score" style="color:{s_color}" title="Turn {i+1}">{s_val:.2f}</span>')
                scores_html = f'<div class="step-scores">Step scores: {" ".join(score_items)}</div>'

            trajectories_html.append(
                f'<div class="trajectory" id="step{step_idx}-traj{traj_idx}">'
                f'<div class="traj-header">'
                f'<span class="traj-title">Trajectory {traj_idx + 1}</span>'
                f'<span class="episode-score" style="color:{score_color}">Score: {episode_score}</span>'
                f"</div>"
                f"{scores_html}"
                f'<div class="turns">{"".join(turns_html)}</div>'
                f"</div>"
            )

        # Tab buttons for trajectories
        tab_buttons = []
        for traj_idx, group in enumerate(entry["groups"]):
            item = group[0] if isinstance(group, list) else group
            ep_score = item["episode_score"]
            s_color = "#2e7d32" if ep_score > 0 else ("#c62828" if ep_score < 0 else "#f57f17")
            tab_buttons.append(
                f'<button class="tab-btn" onclick="showTraj({step_idx},{traj_idx})" '
                f'id="tab-{step_idx}-{traj_idx}">'
                f'{traj_idx + 1} <small style="color:{s_color}">({ep_score})</small></button>'
            )

        steps_html.append(
            f'<div class="step-section" id="step-section-{step_idx}">'
            f'<h2>Step {step_idx} <small>({entry["timestamp"]})</small></h2>'
            f'<div class="tab-bar">{"".join(tab_buttons)}</div>'
            f'<div class="trajectories">{"".join(trajectories_html)}</div>'
            f"</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rollout Sample Viewer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
  h1 {{ color: #e0e0e0; margin-bottom: 20px; font-size: 1.4em; }}
  h2 {{ color: #b0b0b0; margin-bottom: 10px; font-size: 1.1em; }}
  h2 small {{ color: #666; font-weight: normal; }}

  .step-section {{ margin-bottom: 30px; border: 1px solid #333; border-radius: 8px; padding: 16px; background: #16213e; }}

  .tab-bar {{ display: flex; gap: 4px; margin-bottom: 12px; flex-wrap: wrap; }}
  .tab-btn {{ padding: 6px 12px; border: 1px solid #444; border-radius: 4px; background: #1a1a2e; color: #ccc; cursor: pointer; font-size: 0.85em; }}
  .tab-btn:hover {{ background: #2a2a4e; }}
  .tab-btn.active {{ background: #0f3460; border-color: #4a90d9; color: #fff; }}

  .trajectory {{ display: none; }}
  .trajectory.visible {{ display: block; }}

  .traj-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding: 8px 12px; background: #0f3460; border-radius: 4px; }}
  .traj-title {{ font-weight: bold; }}
  .episode-score {{ font-weight: bold; font-size: 1.1em; }}

  .step-scores {{ margin-bottom: 10px; padding: 6px 12px; background: #1a1a2e; border-radius: 4px; font-size: 0.8em; }}
  .step-score {{ margin-right: 6px; font-weight: bold; }}

  .turns {{ display: flex; flex-direction: column; gap: 8px; }}
  .turn {{ padding: 10px 14px; border-radius: 6px; border-left: 3px solid; }}
  .turn.system {{ background: #1a1a2e; border-color: #666; }}
  .turn.user {{ background: #1a2744; border-color: #4a90d9; }}
  .turn.assistant {{ background: #1a3a2a; border-color: #4caf50; }}
  .role {{ font-size: 0.75em; font-weight: bold; text-transform: uppercase; margin-bottom: 4px; color: #999; }}
  .content {{ white-space: pre-wrap; word-break: break-word; font-size: 0.88em; line-height: 1.5; max-height: 400px; overflow-y: auto; }}

  .thinking {{ margin-bottom: 8px; border: 1px solid #b388ff; border-radius: 4px; background: #1a1a3e; }}
  .thinking summary {{ cursor: pointer; padding: 4px 10px; font-size: 0.78em; font-weight: bold; color: #b388ff; }}
  .thinking-content {{ padding: 6px 10px; white-space: pre-wrap; word-break: break-word; font-size: 0.84em; line-height: 1.4; color: #ce93d8; max-height: 300px; overflow-y: auto; }}
  .no-thinking {{ font-size: 0.75em; color: #666; font-style: italic; margin-bottom: 4px; }}

  .step-nav {{ margin-bottom: 16px; display: flex; gap: 6px; flex-wrap: wrap; }}
  .step-nav-btn {{ padding: 6px 14px; border: 1px solid #444; border-radius: 4px; background: #1a1a2e; color: #ccc; cursor: pointer; }}
  .step-nav-btn:hover {{ background: #2a2a4e; }}
  .step-nav-btn.active {{ background: #0f3460; border-color: #4a90d9; color: #fff; }}
</style>
</head>
<body>
<h1>Rollout Sample Viewer ({len(entries)} steps, {sum(len(e["groups"]) for e in entries)} trajectories)</h1>

<div class="step-nav">
{"".join(f'<button class="step-nav-btn" onclick="showStep({i})" id="step-nav-{i}">Step {i}</button>' for i in range(len(entries)))}
</div>

{"".join(steps_html)}

<script>
function showTraj(stepIdx, trajIdx) {{
  // Hide all trajectories in this step
  const section = document.getElementById('step-section-' + stepIdx);
  section.querySelectorAll('.trajectory').forEach(el => el.classList.remove('visible'));
  section.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  // Show selected
  document.getElementById('step' + stepIdx + '-traj' + trajIdx).classList.add('visible');
  document.getElementById('tab-' + stepIdx + '-' + trajIdx).classList.add('active');
}}

function showStep(stepIdx) {{
  document.querySelectorAll('.step-section').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.step-nav-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('step-section-' + stepIdx).style.display = 'block';
  document.getElementById('step-nav-' + stepIdx).classList.add('active');
}}

// Initialize: show first trajectory of each step, show first step
document.querySelectorAll('.step-section').forEach((section, i) => {{
  showTraj(i, 0);
  if (i > 0) section.style.display = 'none';
}});
if (document.getElementById('step-nav-0')) document.getElementById('step-nav-0').classList.add('active');
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visualize rollout sample trajectories from ROLL training logs")
    parser.add_argument("log_file", help="Path to log_rank_DRIVER_0_1.log")
    parser.add_argument("-o", "--output", default=None, help="Output HTML path (default: <log_dir>/rollout_viewer.html)")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: {log_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {log_path}...")
    entries = extract_samples(str(log_path))
    if not entries:
        print("No rollout sample entries found in log file.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(entries)} steps with samples")

    out_path = Path(args.output) if args.output else log_path.parent / "rollout_viewer.html"
    html_content = render_html(entries)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Viewer written to {out_path}")


if __name__ == "__main__":
    main()
