# Financial Domain RL Reward Functions — Paper Survey

*Compiled: 2026-04-20. Papers focused on RL for tool calling, financial analysis, and trading.*

---

## Papers with Exact Reward Functions

### Directly Relevant to Our Use Case

| Paper | Year | Domain | Key Reward Innovation | arXiv |
|-------|------|--------|----------------------|-------|
| **ToolRL** | Apr 2025 | Tool calling | Fine-grained decomposition: tool name + param name + param value | [2504.13958](https://arxiv.org/abs/2504.13958) |
| **ToolRLA** | Mar 2026 | **Financial advisory** | Multiplicative veto + compliance penalty (λ=10) | [2603.01620](https://arxiv.org/abs/2603.01620) |
| **Trading-R1** | Sep 2025 | **Equity research** | 3-stage curriculum: structure → evidence grounding → decision | [2509.11420](https://arxiv.org/abs/2509.11420) |
| **Trade-R1** | Jan 2026 | **Financial trading** | Triangular consistency (evidence ↔ reasoning ↔ decision) | [2601.03948](https://arxiv.org/abs/2601.03948) |
| **Fin-PRM** | Aug 2025 | **Financial reasoning** | Knowledge base verification + concept coverage | [2508.15202](https://arxiv.org/abs/2508.15202) |
| **OTC** | Apr 2025 | Tool efficiency | Cosine decay penalty for excess calls; 73% reduction | [2504.14870](https://arxiv.org/abs/2504.14870) |
| **PORTool** | Oct 2025 | Tool calling | Step-level reward with granular format scoring (5 sub-checks) | [2510.26020](https://arxiv.org/abs/2510.26020) |
| **Tool-R1** | Sep 2025 | Tool calling | LLM-as-judge (partial credit: 1.0/0.5/0.0) + parse + exec ratio | [2509.12867](https://arxiv.org/abs/2509.12867) |

### Broader Financial RL

| Paper | Year | Domain | Key Pattern | arXiv |
|-------|------|--------|------------|-------|
| Risk-Aware RL | Jun 2025 | Trading | 4-component modular: return, downside risk, alpha, Treynor | [2506.04358](https://arxiv.org/abs/2506.04358) |
| FinRL-DeepSeek | Feb 2025 | Trading | LLM-scored sentiment/risk as reward modifiers | [2502.07393](https://arxiv.org/abs/2502.07393) |
| RLMF | 2024 | Trading | Actual market returns as reward signal | [OpenReview](https://openreview.net/forum?id=y3W1TVuJii) |
| GDPO (NVIDIA) | Jan 2026 | Multi-reward RL | Decoupled normalization for multi-component rewards (fixes GRPO collapse) | [2601.05242](https://arxiv.org/abs/2601.05242) |

---

## Key Takeaways for the Talk

1. **Multiplicative > Additive** for tool correctness. ToolRLA proved in production: multiplicative veto prevents trading tool-selection errors against parameter accuracy (+7pp improvement in ablation).

2. **Fine-grained > Coarse** reward granularity. ToolRL: decomposing into tool name + param key + param value consistently outperforms binary correct/incorrect.

3. **Format is a gate, not a score.** Nearly every paper treats format as binary prerequisite. PORTool rescales format to [-0.25, 0.25] so it never overrides correctness.

4. **Evidence grounding is the financial differentiator.** Trading-R1's Stage II curriculum specifically rewards citing data from retrieved sources. Maps directly to our "memo should reference actual tool outputs" requirement.

5. **Efficiency penalties work.** OTC reduced tool calls by 68% while maintaining accuracy. Key: make efficiency multiplicative with correctness so the model doesn't learn to never call tools.

6. **Compliance penalty dominates by design.** ToolRLA's λ=10 ensures any compliant trajectory scores higher than any non-compliant one. For finance: never fabricate data.

7. **GDPO fixes multi-reward GRPO collapse.** Standard GRPO normalization collapses distinct reward signals. NVIDIA's GDPO is a drop-in fix for multi-component rewards.
