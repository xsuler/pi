# Train Pi with AReno

This harness trains Pi to generate polished HTML, CSS, and JavaScript pages while AReno supplies the policy model. Pi
uses its built-in read and write tools to create `index.html`, `styles.css`, and `app.js`. Every non-streaming OpenAI response preserves AReno's exact input tokens,
response tokens, and rollout log probabilities in the JSON event stream, which the Python adapter converts into
`AgentTrajectoryTurn` objects.

Build the binary first:

```bash
npm --prefix packages/coding-agent run build:binary
```

The checked-in `web_tasks_4096.jsonl` contains 4096 independently specified product and interface design tasks. It is
the source dataset; no combinatorial dataset generator is included.

Configure the external multimodal judge and train:

```bash
export PI_ARENO_JUDGE_BASE_URL=https://judge.example.com/v1
export PI_ARENO_JUDGE_API_KEY=...
export PI_ARENO_JUDGE_MODEL=vision-model
# Optional when Chromium is not on PATH:
export PI_ARENO_CHROMIUM=/usr/bin/chromium

areno train \
  --ckpt /path/to/checkpoint \
  --dataset-path /path/to/pi/examples/areno/web_tasks_4096.jsonl \
  --dataset-loader-fn /path/to/pi/examples/areno/dataset_loader.py \
  --agent-fn /path/to/pi/examples/areno/run_agent.py \
  --reward-fn-path /path/to/pi/examples/areno/reward.py \
  --algo gspo --world-size 8 --tp-size 4 --n-samples 8
```

Set `PI_ARENO_BINARY` when the compiled binary is stored outside
`packages/coding-agent/dist/pi`. The reward invokes an existing Chromium executable directly; it does not require
Playwright. It renders the HTML, CSS, and JavaScript at 1440x1024 with external network access disabled, wraps the
screenshot in a self-contained SVG, asks the external vision model for four separate scores, and adds a small
efficiency bonus for finishing in fewer assistant turns.

The judge receives both the complete HTML/CSS/JS source and the rendered SVG. Source quality is scored for semantic
structure, responsive behavior, accessibility, validity, and self-containment. Functional completeness is scored
separately by checking whether every requested interaction has real event logic and visible state transitions. The SVG
is scored independently for design-brief alignment and visual aesthetics.
