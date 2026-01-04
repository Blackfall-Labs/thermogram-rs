# Engineering Log Template

Use `scripts/eng_log.py` to generate timestamped engineering logs.

## Quick Usage

```bash
# Simple entry
python scripts/eng_log.py \
    --subject "Your Subject" \
    --achievement "What you did" \
    --rationale "Why it matters"

# Full JSON entry via stdin
echo '{"subject": "...", "achievement": "...", "rationale": "..."}' | python scripts/eng_log.py --from-stdin

# See template
python scripts/eng_log.py --template
```

## JSON Template for Claude

When documenting engineering work, use this JSON structure:

```json
{
    "subject": "<Brief title of the engineering work>",
    "achievement": "<What was accomplished - be specific about code changes>",
    "rationale": "<Why this change matters - link to research, architectural benefits>",
    "tags": ["training", "architecture", "optimization", "bugfix"],
    "related_files": [
        "crates/astromind/src/path/to/file.rs"
    ],
    "metrics": {
        "params": "<model parameter count>",
        "epochs": "<training epochs>",
        "wall_clock": "<training time e.g. '5m 23s'>",
        "accuracy": "<final accuracy e.g. '98.5%'>",
        "failure_mode": "<what fails, if anything>",
        "before": {
            "accuracy": "<metric before>",
            "convergence": "<epochs to converge>",
            "stability": "<description>"
        },
        "after": {
            "accuracy": "<metric after>",
            "convergence": "<epochs to converge>",
            "stability": "<description>"
        },
        "delta": "<summary of improvement e.g. '+15% accuracy, 2x faster convergence'>"
    },
    "commits": ["<commit hash>", "<commit hash>"],
    "references": ["<papers, docs, or prior logs>"],
    "conclusions": [
        "<Key finding 1 with specific numbers>",
        "<Key finding 2 with specific numbers>",
        "<Actionable insight for future work>"
    ],
    "next_steps": ["<follow-up work identified>"]
}
```

## Required Metrics for Training Experiments

Every training-related log MUST include:

| Field | Description | Example |
|-------|-------------|---------|
| `params` | Model parameter count | `"164,212"` |
| `epochs` | Training epochs run | `"500"` |
| `wall_clock` | Total training time | `"4m 56s"` |
| `accuracy` | Final accuracy achieved | `"98.05%"` |
| `failure_mode` | What doesn't work | `"Diverges after epoch 200 without projection"` |
| `delta` | Before/after summary | `"+12% accuracy, stable to 24 unroll steps"` |
| `commits` | Related git commits | `["abc123", "def456"]` |

## Conclusions Format

Conclusions should be **specific and quantified**:

```json
"conclusions": [
    "mHC projection reduced variance across recursive depth by 47%",
    "Orthogonal init allowed stable unroll from 12 → 24 steps",
    "Virtual Width achieved equivalent accuracy with 25% fewer params",
    "Training converged in 150 epochs vs 400 without orthogonal penalty"
]
```

**Bad conclusions** (too vague):
- "Performance improved"
- "Training was more stable"
- "Model works better"

**Good conclusions** (specific):
- "Action accuracy: 87% → 100% (+13%) after orthogonal init"
- "Subject accuracy plateaued at 97.98% - limited by category granularity"
- "Divergence eliminated: 0/50 runs failed vs 12/50 baseline"

## Categories/Tags

Use consistent tags for filtering:

- `architecture` - Structural changes to TRMs/meshes
- `training` - Training pipeline, data generation, optimization
- `performance` - Speed/memory improvements
- `accuracy` - Model accuracy improvements
- `bugfix` - Bug fixes
- `refactor` - Code refactoring
- `research` - Research findings/experiments
- `mhc` - Manifold-constrained hyperconnections work
- `integration` - Integration with other systems

## Full Example Entry

```json
{
    "subject": "mHC Orthogonal Projection for CodeGenMesh Training",
    "achievement": "Integrated orthogonal_init and orthogonal_penalty into CodeGenMesh v2 training. Added manifold projection every 100 steps to maintain identity-like residual behavior.",
    "rationale": "Per DeepSeek mHC paper, identity-preserving residuals are critical for stable training of recursive/compositional models. CodeGenMesh uses shared embeddings across action/subject/modifier predictors - orthogonal constraints ensure these don't interfere.",
    "tags": ["training", "mhc", "accuracy"],
    "related_files": [
        "crates/astromind-cli/src/bin/train_codegen_mesh_full.rs",
        "crates/astromind/src/training/orthogonal.rs"
    ],
    "metrics": {
        "params": "164,212",
        "epochs": "500",
        "wall_clock": "4m 56s",
        "accuracy": "Action: 100%, Subject: 98.5%",
        "failure_mode": "Subject accuracy plateaus ~98% - category granularity limit",
        "before": {
            "accuracy": "Action: 100%, Subject: 97.98%",
            "convergence": "300 epochs",
            "stability": "Occasional gradient spikes"
        },
        "after": {
            "accuracy": "Action: 100%, Subject: 98.5%",
            "convergence": "200 epochs",
            "stability": "Smooth throughout"
        },
        "delta": "+0.5% subject accuracy, 33% faster convergence, eliminated gradient spikes"
    },
    "commits": ["abc1234"],
    "references": [
        "DeepSeek mHC paper (https://huggingface.co/papers/2512.24880)",
        "engineering/20260102-001720-mhc-paper-analysis.md"
    ],
    "conclusions": [
        "Orthogonal init eliminated gradient spikes in first 50 epochs",
        "Manifold projection every 100 steps maintained W^T*W ≈ I (penalty < 0.01)",
        "Convergence accelerated from 300 → 200 epochs (33% faster)",
        "Subject accuracy ceiling appears to be category design, not training stability"
    ],
    "next_steps": [
        "Expand subject categories from 16 → 24 for finer granularity",
        "Test Virtual Width layer for embedding expansion",
        "Apply same techniques to GenerativeMesh training"
    ]
}
```
