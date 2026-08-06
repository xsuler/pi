# Train Pi to generate SVG animation frames with AReno

This example trains the real Pi coding-agent loop to create a seamless one-second animation as eight consecutive,
self-contained SVG files: `frame-00.svg` through `frame-07.svg`. Pi owns the complete tool loop. It creates, reads, and
edits frames until all files satisfy the prompt, SVG constraints, adjacent-frame continuity, and the frame-07 to
frame-00 loop boundary.

Each frame must be a static `512x512` SVG with `viewBox="0 0 512 512"`. Scripts, SMIL, CSS animation, external assets,
embedded raster images, and cross-frame references are prohibited. The adapter only validates the final workspace and
converts Pi events into AReno trajectories; it does not generate or copy frames.

## Dataset

`svg_animation_tasks_4096.jsonl` contains 4096 fully expanded animation briefs. Every record includes a concrete
subject, scene, motion narrative, art direction, loop requirement, frame count, FPS, and output-file contract. The
checked-in JSONL is the training dataset; no runtime combinatorial generator is used.

Use `dataset_loader.py`, which verifies exactly 4096 unique records and supplies the isolated workspace README.

## Reward

Install the Python rasterizer dependency in the training environment:

```bash
pip install -r examples/areno/requirements.txt
```

`reward.py` validates every SVG, uses CairoSVG to rasterize all eight frames to ordered `512x512` PNGs, and sends the
prompt followed by frames 00-07 to an OpenAI-compatible multimodal judge. It requests separate rubric-item scores for
SVG quality, motion continuity, prompt alignment, and visual aesthetics, then aggregates them locally with weakest-area
penalty and a small turn-efficiency component. Rendered SVG/PNG pairs are retained under `/tmp/areno_animation`.

Configure the judge:

```bash
export PI_ARENO_JUDGE_BASE_URL=https://judge.example.com/v1
export PI_ARENO_JUDGE_API_KEY=...
export PI_ARENO_JUDGE_MODEL=vision-model
```

For a deterministic curriculum reward that does not call a vision model, use `reward_stage1.py`. It checks all eight
files, XML validity, the 512 viewBox, visible vector elements, self-containment, and frame diversity.

## Train

Build Pi or point the adapter to an existing binary:

```bash
npm --prefix packages/coding-agent run build:binary
export PI_ARENO_BINARY=/path/to/pi
```

Then run:

```bash
areno train \
  --ckpt /path/to/checkpoint \
  --dataset-path /path/to/pi/examples/areno/svg_animation_tasks_4096.jsonl \
  --dataset-loader-fn /path/to/pi/examples/areno/dataset_loader.py \
  --agent-fn /path/to/pi/examples/areno/run_agent.py \
  --reward-fn-path /path/to/pi/examples/areno/reward.py \
  --algo gspo --world-size 8 --tp-size 4 --n-samples 8
```

`PI_MAX_TURNS` defaults to 20. The adapter enforces it at complete Pi `turn_end` boundaries, after tool execution, so
an in-progress SVG edit is not truncated.

## Diagnostics

Run one rollout:

```bash
export ARENO_BASE_URL=http://127.0.0.1:PORT/v1
export ARENO_API_KEY=areno-agentic
python examples/areno/test_rollout.py --raw
```

Run the concurrent batch-shaped diagnostic:

```bash
python examples/areno/concurrent_rollout.py \
  --base-url "$ARENO_BASE_URL" \
  --binary "$PI_ARENO_BINARY"
```

Run the live reward against an existing eight-frame workspace:

```bash
python examples/areno/test_reward_live.py \
  --workspace /tmp/my-animation \
  --prompt "Animate a bookstore order moving from intake to shelf."
```
