"""
Shared Memory for E-SPL — ACE-inspired (arXiv:2510.04618).

After each RL step all programs' rollouts are reviewed to extract
*globally* useful insights that don't belong to any single strategy.
These bullets are stored in SharedMemory and injected into the
PrincipleUpdater's self-reflection prompt so programs can reference
the global knowledge while updating their own per-strategy principles.

ACE design:
  - Itemised bullets with unique IDs and helpful/harmful counters
  - Delta-based incremental updates (Generator → Reflector → Curator)
  - Bullets pruned when score (helpful - harmful) drops below threshold
  - Long-term reflection accumulates across many steps (unlike per-step
    per-strategy mutation which is short-term)

ReflectivePrompt connection (arXiv:2508.18870):
  - SharedMemory corresponds to the *long-term reflection* buffer:
    knowledge updated each epoch based on the current population.
  - Per-strategy PrincipleUpdater mutation is the *short-term reflection*:
    it operates within one generation and sees shared memory as context.
"""

import copy
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local LLM helpers (avoids circular import with system_prompt_learning_rl)
# ---------------------------------------------------------------------------

def _llm_chat(conv, renderer, sampling_client, sampling_params) -> str:
    model_input = renderer.build_generation_prompt(conv)
    future = sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=sampling_params)
    result = future.result()
    sampled_tokens = result.sequences[0].tokens
    parsed_message, _ = renderer.parse_response(sampled_tokens)
    return parsed_message["content"]


def _get_operations_from_json(response: str) -> list[dict]:
    import re as _re
    json_block = response.split("```json")[-1].split("```")[0]
    # strip single-line comments like // ...
    json_block = _re.sub(r"//[^\n]*", "", json_block)
    # fix unescaped backslashes
    import json as _json
    try:
        return _json.loads(json_block)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryBullet:
    bullet_id: str          # e.g. "M0", "M1"
    content: str            # the insight text
    helpful_count: int = 0  # times it was explicitly marked useful
    harmful_count: int = 0  # times it was explicitly marked misleading
    created_at: int = -1    # RL step when created

    @property
    def score(self) -> int:
        return self.helpful_count - self.harmful_count

    def to_dict(self) -> dict:
        return {
            "bullet_id": self.bullet_id,
            "content": self.content,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryBullet":
        return cls(
            bullet_id=d["bullet_id"],
            content=d["content"],
            helpful_count=d.get("helpful_count", 0),
            harmful_count=d.get("harmful_count", 0),
            created_at=d.get("created_at", -1),
        )


class SharedMemory:
    """
    Global knowledge store shared across all Programs in the EvolutionPool.

    Bullets are indexed by ID (e.g. "M0").  Each bullet is a short insight
    (<= 32 words) that applies universally across strategies.
    """

    PRUNE_SCORE_THRESHOLD = -2  # remove bullets whose score drops this low

    def __init__(self, max_bullets: int = 50):
        self.max_bullets = max_bullets
        self.bullets: dict[str, MemoryBullet] = {}
        self._next_id: int = 0

    # --- mutation ---

    def add_bullet(self, content: str, step: int = -1) -> str:
        bid = f"M{self._next_id}"
        self._next_id += 1
        self.bullets[bid] = MemoryBullet(bullet_id=bid, content=content, created_at=step)
        logger.debug(f"SharedMemory: added {bid}: {content[:60]}")
        return bid

    def mark_helpful(self, bullet_id: str):
        if bullet_id in self.bullets:
            self.bullets[bullet_id].helpful_count += 1

    def mark_harmful(self, bullet_id: str):
        if bullet_id in self.bullets:
            self.bullets[bullet_id].harmful_count += 1

    def prune(self):
        """Remove low-scoring bullets and enforce max_bullets cap."""
        to_remove = [
            bid for bid, b in self.bullets.items()
            if b.score <= self.PRUNE_SCORE_THRESHOLD
        ]
        for bid in to_remove:
            logger.debug(f"SharedMemory: pruning {bid} (score={self.bullets[bid].score})")
            del self.bullets[bid]

        if len(self.bullets) > self.max_bullets:
            sorted_bullets = sorted(
                self.bullets.values(),
                key=lambda b: (b.score, b.created_at),
            )
            for b in sorted_bullets[: len(self.bullets) - self.max_bullets]:
                del self.bullets[b.bullet_id]

    # --- serialisation ---

    def state_dict(self) -> dict:
        return {
            "max_bullets": self.max_bullets,
            "_next_id": self._next_id,
            "bullets": {bid: b.to_dict() for bid, b in self.bullets.items()},
        }

    def load_state_dict(self, state: dict):
        self.max_bullets = state.get("max_bullets", self.max_bullets)
        self._next_id = state.get("_next_id", 0)
        self.bullets = {
            bid: MemoryBullet.from_dict(d)
            for bid, d in state.get("bullets", {}).items()
        }

    # --- formatting ---

    def get_context_string(self) -> str:
        if not self.bullets:
            return "None"
        return "\n".join(
            f"[{bid}] {b.content}" for bid, b in sorted(self.bullets.items())
        )

    def __len__(self) -> int:
        return len(self.bullets)

    def __repr__(self) -> str:
        return f"SharedMemory({len(self.bullets)} bullets)"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SHARED_MEMORY_UPDATE_TEMPLATE = """Multiple independent problem-solving strategies were evaluated on the same batch of math problems.
Each strategy uses its own per-strategy principles, but we want to identify insights that are useful ACROSS ALL strategies.

Strategy performance summary:
{strategy_summaries}

Current shared memory (global insights already captured):
{existing_memory}

Your task: Update the shared memory by proposing operations based on cross-strategy evidence.

New bullets to ADD must:
  1. Apply to MULTIPLE strategies (not specific to one approach)
  2. NOT duplicate anything already in shared memory
  3. Be at most 32 words, concrete, and focused on strategic thinking
  4. Describe failure modes, verification strategies, or universally helpful heuristics

Return a JSON list of operations (at most 3 total, or an empty list if no clear cross-strategy insight):
```json
[
    {{"operation": "add", "content": "globally useful insight across all strategies"}},
    {{"operation": "mark_helpful", "bullet_id": "M3"}},
    {{"operation": "mark_harmful", "bullet_id": "M7"}}
]
```
Return only operations that are strongly supported by cross-strategy evidence."""


SINGLE_QUERY_CRITIQUE_WITH_MEMORY_TEMPLATE = """An agent system is provided with a set of principles, and has tried to solve problems with those principles. Some suggestions have been made to improve the existing principles.
Your task is to collect and review these individual suggestions, and come up with a final Revision Plan for the existing principles.

Each final principle (produced after the final Revision Plan) must satisfy the following requirements:
1. It must be clear, generalizable lessons for this case, with no more than 32 words
2. Begin with general background with several words in the principle
3. Focus on strategic thinking patterns, not specific calculations
4. Emphasize decision points that could apply to similar problems
5. Avoid repeating similar sayings or ideas across multiple principles

<shared_memory>
{shared_memory}
</shared_memory>

NOTE: The shared memory above contains universal insights already available to ALL strategies.
When proposing updates to THIS strategy's principles, focus on strategy-specific improvements
that complement but do NOT duplicate the shared memory bullets.

<existing_principles>
{principles}
</existing_principles>

<suggested_updates>
{updates}
</suggested_updates>

Please provide reasoning in each suggestion, and think about how to update the existing principles.
You can use two types of operations: ["modify", "merge"]
* "modify": You can modify an existing principle to make it more helpful
* "merge": You can merge similar principles into a more general form to reduce duplication

After generating the step-by-step reasoning, you will finish by providing the final Revision Plan as a list of operations in the following JSON format:
```json
[
    {{
        "operation": "modify",
        "principle": "the modified principle",
        "modified_from": "C1"
    }},
    {{
        "operation": "merge",
        "principle": "the merged principle",
        "merged_from": ["C1", "C3", "S4"]
    }},
    ...
]
```"""


# ---------------------------------------------------------------------------
# SharedMemoryUpdater
# ---------------------------------------------------------------------------

class SharedMemoryUpdater:
    """
    After each RL step, aggregates rollouts from ALL programs and extracts
    cross-strategy insights to update SharedMemory (long-term reflection).

    This mirrors ACE's Curator role: it runs Generator (strategy summaries)
    → Reflector (LLM analysis) → Curator (delta operations) → Merge.
    """

    def __init__(self, renderer, sampling_client, sampling_params, max_new_bullets: int = 3):
        self.renderer = renderer
        self.sampling_client = sampling_client
        self.sampling_params = sampling_params
        self.max_new_bullets = max_new_bullets

    def update(
        self,
        rollouts_all_programs: list[list[dict]],
        program_lists,
        shared_memory: SharedMemory,
        step: int,
        save_dir: str,
    ) -> SharedMemory:
        """
        Args:
            rollouts_all_programs: list of rollout lists, one per parallel program
            program_lists: list of Program objects (same order)
            shared_memory: SharedMemory instance to update in-place
            step: current training step (for bullet timestamps)
            save_dir: directory for saving intermediate results
        Returns:
            Updated SharedMemory (same object, modified in place)
        """
        if len(rollouts_all_programs) < 2:
            logger.info("SharedMemoryUpdater: need ≥ 2 programs, skipping")
            return shared_memory

        summary = self._build_strategy_summaries(rollouts_all_programs, program_lists)
        if not summary:
            return shared_memory

        prompt = SHARED_MEMORY_UPDATE_TEMPLATE.format(
            strategy_summaries=summary,
            existing_memory=shared_memory.get_context_string(),
        )

        try:
            conversation = [{"role": "user", "content": prompt}]
            response = _llm_chat(conversation, self.renderer, self.sampling_client, self.sampling_params)
            operations = _get_operations_from_json(response)
            self._apply_operations(operations, shared_memory, step)
            shared_memory.prune()

            # Save result for debugging
            result_path = os.path.join(save_dir, "shared_memory_update.json")
            with open(result_path, "w") as f:
                json.dump({
                    "prompt": prompt,
                    "response": response,
                    "operations": operations,
                    "shared_memory_after": shared_memory.state_dict(),
                }, f, indent=2)

            logger.info(f"SharedMemory updated: {len(shared_memory)} bullets total")

        except Exception as e:
            logger.warning(f"SharedMemoryUpdater failed: {e}")

        return shared_memory

    def _build_strategy_summaries(self, rollouts_all_programs, program_lists) -> str:
        """Summarise performance of each strategy on the batch."""
        summaries = []
        for prog_idx, (rollouts, program) in enumerate(zip(rollouts_all_programs, program_lists)):
            if not rollouts:
                continue
            rewards = [r.get("reward", 0.0) for r in rollouts]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            n_problems = len(set(r["problem"] for r in rollouts if "problem" in r))
            principles_preview = list(program.principles.values())[:3] if program.principles else []
            label = chr(ord("A") + prog_idx)
            summaries.append(
                f"Strategy {label} (program_id={program.program_id}, "
                f"avg_reward={avg_reward:.3f}, problems={n_problems}):\n"
                f"  Top principles: {principles_preview}"
            )
        return "\n\n".join(summaries)

    def _apply_operations(self, operations: list[dict], shared_memory: SharedMemory, step: int):
        new_count = 0
        for op in operations:
            try:
                if op.get("operation") == "add" and new_count < self.max_new_bullets:
                    content = op.get("content", "").strip()
                    if content:
                        shared_memory.add_bullet(content, step=step)
                        new_count += 1
                elif op.get("operation") == "mark_helpful":
                    shared_memory.mark_helpful(op.get("bullet_id", ""))
                elif op.get("operation") == "mark_harmful":
                    shared_memory.mark_harmful(op.get("bullet_id", ""))
            except Exception as e:
                logger.debug(f"SharedMemoryUpdater op error: {op} | {e}")
