"""
Search tree for T-MCTS.

A node is a tool-use state: the full message history contains the problem
(x), the current symbolic representation (z), the tool-use history (h) and
the latest solver feedback (f).  Edges are tool-use actions (a candidate
tool call or a candidate answer).  Node statistics follow the upper
confidence bound of Eq. (1):

    a* = argmax_a [ Q_T(s, a) + c * sqrt( log N(s) / N(s, a) ) ]

Q_T(s, a) is stored on the child node (incremental mean of backpropagated
returns), N(s) is the parent's visit count and N(s, a) the child's.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def action_key(action: dict) -> str:
    """Stable dedup key for a candidate action."""
    if action["type"] == "answer":
        return f"answer::{action['answer']}"
    if action["type"] == "execute":
        return f"execute::{action['tool_name']}"
    return f"call::{action['tool_name']}::{hashlib.md5(action['code'].encode()).hexdigest()}"


@dataclass
class TreeNode:
    messages: List[Dict[str, str]]          # tool-use state (conversation)
    parent: Optional["TreeNode"] = None
    action: Optional[dict] = None            # edge from parent: {type, tool_name, code, answer}
    depth: int = 0                           # assistant decisions so far
    children: List["TreeNode"] = field(default_factory=list)
    untried_keys: set = field(default_factory=set)     # keys of known-but-unexpanded candidates
    untried: List[dict] = field(default_factory=list)  # pending candidate actions
    n_visits: int = 0
    q_value: float = 0.0
    exhausted: bool = False                  # no candidates can be expanded here

    @property
    def key(self) -> Optional[str]:
        return action_key(self.action) if self.action else None

    def add_candidate(self, action: dict) -> None:
        """Register a candidate action for later expansion (deduplicated)."""
        k = action_key(action)
        if k in self.untried_keys:
            return
        if self.action is not None and action_key(self.action) == k:
            return  # never re-expand the action that produced this node
        if any(action_key(c.action) == k for c in self.children):
            return
        self.untried_keys.add(k)
        self.untried.append(action)

    def pop_candidate(self) -> Optional[dict]:
        if self.untried:
            action = self.untried.pop(0)
            self.untried_keys.discard(action_key(action))
            return action
        return None

    def expand(self, action: dict, response_text: str) -> "TreeNode":
        """Create a child node by appending the candidate assistant turn."""
        child = TreeNode(
            messages=self.messages + [{"role": "assistant", "content": response_text}],
            parent=self,
            action=action,
            depth=self.depth + 1,
        )
        self.children.append(child)
        return child


def ucb_select(node: TreeNode, c: float) -> Optional[TreeNode]:
    """Select the child maximising the UCB of Eq. (1)."""
    if not node.children:
        return None
    log_n = math.log(max(1, node.n_visits))

    def score(child: TreeNode) -> float:
        if child.n_visits == 0:
            return float("inf")
        return child.q_value + c * math.sqrt(log_n / child.n_visits)

    return max(node.children, key=score)


def backup(node: TreeNode, ret: float) -> None:
    """Backpropagate a trajectory return through the visited nodes."""
    cur = node
    while cur is not None:
        cur.n_visits += 1
        cur.q_value += (ret - cur.q_value) / cur.n_visits
        cur = cur.parent


def select_expandable(root: TreeNode, c: float) -> Optional[TreeNode]:
    """Phase 1 (Selection): descend by UCB until a node with pending
    candidates is found.  Dead-end nodes are skipped."""
    node = root
    while node.untried or node.children:
        if node.untried:
            return node
        child = ucb_select(node, c)
        if child is None:
            node.exhausted = True
            return None
        node = child
    return None if node.exhausted else node


def tree_stats(root: TreeNode) -> dict:
    """Aggregate node statistics for logging."""
    nodes = []

    def walk(n: TreeNode) -> None:
        nodes.append(n)
        for ch in n.children:
            walk(ch)

    walk(root)
    return {
        "nodes": len(nodes),
        "expanded": sum(1 for n in nodes if n.children or n.action),
        "visited": sum(1 for n in nodes if n.n_visits > 0),
        "max_depth": max((n.depth for n in nodes), default=0),
    }
