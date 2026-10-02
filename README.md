# LOGIC-X

LOGIC-X is a framework for tool-integrated logical reasoning in large
language models. It couples **LogicEval-X**, a benchmark of 944
expert-verified multiple-choice logical reasoning problems (averaging 8.31
entities, 12.47 attributes, and 18.39 clues per problem), with **T-MCTS**
(Tool-Integrated Monte Carlo Tree Search), a post-training framework in
which solver-guided search produces verified tool-use trajectories for
supervised fine-tuning and process-level rewards for reinforcement
learning. Training with T-MCTS yields the **LOGIC-X-8B** and **LOGIC-X-14B**
models, which solve logical problems by formulating them symbolically and
delegating complex inference to external symbolic solvers: **Pyke**
(Datalog rule inference), **Prover9** (first-order theorem proving), **Z3**
(SMT solving), and **MiniZinc** (finite-domain constraint solving).

## Repository layout

```
LOGIC-X/
├── tmcts/                     core library
│   ├── tools.py               solver registry + sandboxed execution
│   ├── prompts.py             system / generation / repair / interpretation prompts
│   ├── pipeline.py            autonomous reasoning loop (inference & linear generation)
│   ├── search.py              T-MCTS: selection / expansion / rollout / backpropagation
│   ├── tree.py                search tree with UCB selection
│   ├── rewards.py             solver-grounded rewards and verification
│   ├── exporter.py            ShareGPT export for SFT
│   └── llm/                   OpenAI-compatible endpoint pool
├── tools/datalog_engine.py    pure-Python Datalog engine (Pyke backend)
├── LogicEval-X/
│   ├── data/logiceval_x.jsonl  the benchmark (944 problems)
│   └── run_inference.py        inference + evaluation harness
└── T-MCTS/
    ├── generate_trajectories.py  trajectory generation (--search linear | tree)
    ├── train_data/               training data (LOGIC-X-8B / LOGIC-X-14B)
    ├── sft/                      parquet conversion + verl SFT configs
    └── rl/                       RL rollout sampling, filtering, training loop
```

## Evaluation data: LogicEval-X

`LogicEval-X/data/logiceval_x.jsonl` — one JSON object per problem:

```json
{
  "id": "logiceval-0080",
  "question": {
    "context": "On a certain morning, each of six presenters—Feinberg, Guzman, Harrison, Jansen, Kim, and Mackey—will give a presentation for exactly one hour. Each presentation will take place in either the conference room or the auditorium, with exactly one presenter at a time presenting in each of these two rooms. The presentations must be given in a manner consistent with the following conditions: Exactly three of the presentations are given in each room, the first beginning at precisely 8 A.M. on that morning, the second at precisely 9 A.M., and the third at precisely 10 A.M. Feinberg's presentation must begin at the same time as or later than Guzman's. Neither Jansen's presentation nor Mackey's presentation begins at the same time as Feinberg's. Harrison's presentation begins earlier than Feinberg's. Jansen and Mackey, not necessarily in that order, present in the auditorium.",
    "question": "Each of the following is a pair of presenters whose presentations could begin at the same time as each other EXCEPT:",
    "options": [
      "A) Feinberg and Guzman",
      "B) Feinberg and Kim",
      "C) Guzman and Jansen",
      "D) Guzman and Kim",
      "E) Guzman and Mackey"
    ]
  },
  "ground_truth_tool": "z3",
  "ground_truth_answer": "D"
}
```

Every problem carries its ground-truth tool (which solver class it is
designed for) and the correct option. The tool distribution is
pyke 337 / prover9 336 / z3 170 / minizinc 101.

## Evaluation

`run_inference.py` runs a model over the benchmark in the autonomous
protocol: the model selects a solver, writes the symbolic program, and
decides on its own whether to revise, reselect, or answer.

```bash
# Local checkpoint (auto-launches vLLM)
python LogicEval-X/run_inference.py \
    --input LogicEval-X/data/logiceval_x.jsonl \
    --output outputs/eval \
    --local your/path/to/LOGIC-X-8B \
    --gpus 0,1 --port 8009

# Or an OpenAI-compatible endpoint configured in .env
python LogicEval-X/run_inference.py \
    --input LogicEval-X/data/logiceval_x.jsonl \
    --output outputs/eval --online
```

Each run writes per-attempt traces (`trace_internal.jsonl`), failed cases
(`failed_samples.jsonl`), and `metrics.json` (accuracy, tool-selection
accuracy vs gold, execution success rate, revision / reselection rates).

## Training data

`T-MCTS/train_data/` is organized by target model:

| Path | Content | Records |
|---|---|---|
| `LOGIC-X-8B/sft_trajectories.jsonl.gz` | complete tool-use trajectories | 11,774 |
| `LOGIC-X-8B/rl_prompts.jsonl` | RL question set | 10,249 |
| `LOGIC-X-14B/sft_atomic_tasks.jsonl.gz` | single-turn skill items (phase 1) | 22,005 |
| `LOGIC-X-14B/sft_trajectories.jsonl.gz` | complete tool-use trajectories (phase 2) | 11,774 |
| `LOGIC-X-14B/rl_prompts.jsonl` | RL question set | 10,249 |

Large training files are stored gzip-compressed; all loaders in this
repository read `*.jsonl.gz` transparently, or use `gunzip`/`zcat`.

Each trajectory is a complete multi-turn conversation in ShareGPT format
(`human` → `function_call` → `observation` → … → `gpt`). The excerpt below
shows a solver-verified entailment trajectory: an initial Prover9
formulation, two feedback-based revisions after parse failures, a solver
reselection, and a final proved theorem (elided for brevity):

```json
{
  "id": "logicx-traj-00987",
  "conversations": [
    { "from": "human",
      "value": "Context: Crete is an island in Greece. In August 2005, a wildfire swept across Crete. Homes were destroyed in the August 2005 wildfire in Crete. …" },
    { "from": "function_call",
      "value": "<think> This is a straightforward entailment task, so prover9 is the best fit. I formalize: Crete is in Greece; a wildfire occurred in Crete in August 2005; … </think>\n<tool_call> {\"name\": \"prover9\", \"arguments\": {\"code\": \"formulas(assumptions). …\"}} </tool_call>" },
    { "from": "observation",
      "value": "============================== Prover9 =============================== … SEARCH FAILED …" },
    { "from": "function_call",
      "value": "<think> The prover9 run failed because variables in facts were implicitly universally quantified … </think>\n<tool_call> … </tool_call>" },
    { "from": "observation", "value": "ParseError: Cannot parse line 9: …" },
    { "from": "function_call",
      "value": "<think> The repeated parse errors show the chosen syntax is not working … so further tool analysis is needed with a different tool. … </think>\n<tool_call> … </tool_call>" },
    { "from": "observation", "value": "============================== Prover9 =============================== … THEOREM PROVED …" },
    { "from": "gpt",
      "value": "<think> The latest prover9 run succeeded and proved the target theorem. … </think>\n<answer> … The tool's goal was `disaster_in_year(greece,year2005)`. Prover9 returned `THEOREM PROVED` … The final answer is A </answer>" }
  ],
  "system": "You are a tool-integrated symbolic reasoning assistant. …",
  "tools": "[… OpenAI function schemas for the four solvers …]",
  "metadata": { "gold_tool": "prover9", "selected_tool": "prover9",
                "ground_truth": "A", "final_answer": "A", "repair_count": 2 }
}
```

## Trajectory generation with T-MCTS

T-MCTS explores tool-use trajectories with four operations. Each tree node
is a tool-use state — the problem, the current symbolic representation, the
tool-use history, and the latest solver feedback; each action is a
tool-integrated operation (symbolic formulation, solver selection, solver
invocation, feedback-based revision, or answering):

- **Selection** — promising actions are chosen by the upper confidence
  bound `argmax [Q(s,a) + c·sqrt(log N(s) / N(s,a))]`;
- **Expansion** — the selected state is expanded by sampling candidate
  symbolic formulations, solver choices, invocations, and continuations;
- **Tool-grounded rollout** — solvers are invoked along the trajectory and
  their feedback is incorporated into subsequent reasoning;
- **Solver-based backpropagation** — each completed trajectory receives the
  solver-grounded return
  `R_T = clip₁(0.8·R_ans + 0.2·R_exec − 0.05·n_switch − 0.05·n_repair)`,
  which rewards correct answers and executable symbolic programs while
  penalizing unnecessary solver reselection and repeated revision; the
  return is backpropagated through the visited nodes.

A trajectory is **solver-verified** when its final answer is correct and
its final symbolic program executes on the selected solver. For supervised
fine-tuning, one random solver-verified trajectory is retained per
training instance, preserving diverse successful tool-use behaviour.

```bash
python T-MCTS/generate_trajectories.py \
    --input your/path/to/questions.jsonl \
    --output outputs/generate \
    --search tree \
    --workers 64
```

Search budgets (runs per case, rollouts per run, search depth) and reward
weights are configured through `.env` (`TMCTS_*` variables). The linear
guided loop remains available with `--search linear`.

## SFT training

Trajectories are converted to verl parquet and used for full-parameter
fine-tuning (two-phase curriculum for LOGIC-X-14B: single-turn skill
items, then complete trajectories):

```bash
# Convert ShareGPT jsonl -> verl parquet
python T-MCTS/sft/convert_to_verl_parquet.py \
    --input  your/path/to/sft_trajectories.jsonl \
    --output your/path/to/sft_trajectories.parquet

# Optional: validate through the Qwen3 chat template
python T-MCTS/sft/validate_verl_data.py \
    --parquet your/path/to/sft_trajectories.parquet \
    --model your/path/to/qwen3-8b

# Train (or: MODEL=LOGIC-X-8B MODEL_PATH=your/path/to/qwen3-8b bash T-MCTS/sft/run_sft.sh)
torchrun --nproc_per_node 8 -m verl.trainer.sft_trainer \
    --config-path your/path/to/T-MCTS/sft \
    --config-name verl_sft_LOGIC-X-8B
```

## RL training

After SFT, model-specific RL sets are constructed: the SFT model answers
every training instance five times; cases correct in four or five runs
(stable-correct) and cases never correct (stable-incorrect) are removed,
keeping the cases solved in one to three runs. Optimization uses the
solver-grounded return `R_T` as the training signal — no value model is
trained — with solver calls executed online during rollout generation and
reward computation.

```bash
# 1) Sample five rollouts per question with the current checkpoint
#    and keep the 1..3-of-5 correct cases
MODEL=LOGIC-X-8B CKPT=your/path/to/sft/checkpoint \
    bash T-MCTS/rl/run_rl_rollouts.sh

# 2) Retrain on the kept cases (rl_keep.jsonl) via the SFT recipe above
```

## Environment

```bash
pip install -e .
cp .env.example .env     # fill in your endpoint credentials
```

Solver backends: `pip install z3-solver`; install the
[Prover9/LADR](https://github.com/LaurentClaessens/ladr) binaries, a
[MiniZinc bundle](https://www.minizinc.org/software.html) with Gecode, and
place them under `tools_bin/` (or point `TMCTS_TOOLS_BIN` at them). Pyke
uses the bundled pure-Python Datalog engine.

## Citation

```bibtex
@inproceedings{logicx2027,
  title     = {Towards Symbolic Tool-Integrated Logical Reasoning in Large Language Models},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2027}
}
```
