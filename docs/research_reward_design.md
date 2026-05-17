# Reward Engineering — What We Learned and What to Teach

*Compiled: 2026-04-20. Restructured for practical understanding, not literature review.*

---

## The Core Learning

**Your reward function IS your product specification.** It defines exactly what "good" means. Get it wrong and the model optimizes the wrong thing (reward hacking). Get it right and a 7B model beats Opus 4.5 on your task.

Three lessons from the literature:

1. **Start simple, add complexity only when needed.** DeepSeek-R1 used just accuracy + format and got emergent reasoning. ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948))
2. **Multiplicative beats additive for correctness.** If the model picks the wrong tool, nothing else should save it. ToolRLA proved this in production — additive rewards let the model compensate bad tool selection with good parameters. ([arXiv:2603.01620](https://arxiv.org/abs/2603.01620))
3. **Dense beats sparse.** The model should always get partial credit. Binary 0/1 rewards give almost no gradient signal. ([arXiv:2504.13958](https://arxiv.org/abs/2504.13958))

---

## Two Reward Functions: Simple and Composite

### Reward A: Simple (Explain in 60 Seconds)

Four components, all rule-based. No LLM judges. This is what you'd build first.

```python
def reward_simple(messages, valid_tools):
    """
    Simple reward for tool-calling RL. Range: [-2, 4].
    
    Sources:
      - Format check: ToolRL (arXiv:2504.13958)
      - Coverage with veto: ToolRLA (arXiv:2603.01620)  
      - Completion + efficiency: OTC (arXiv:2504.14870)
    """
    tool_calls = extract_tool_calls(messages)
    used_names = {tc["name"] for tc in tool_calls}
    
    # 1. FORMAT: Are all tool calls valid JSON? (0 or 1)
    #    From ToolRL — binary format gate
    all_valid = all(is_valid_json(tc["arguments"]) for tc in tool_calls)
    r_format = 1.0 if (all_valid and tool_calls) else 0.0
    
    # 2. TOOL COVERAGE: What fraction of the 4 tools did you call? (0 to 2)
    #    From ToolRL — Jaccard similarity of tool name sets
    #    PLUS ToolRLA multiplicative veto — hallucinated tool = zero
    coverage = len(used_names & valid_tools) / len(valid_tools)
    hallucinated = used_names - valid_tools
    r_coverage = 0.0 if hallucinated else coverage * 2.0
    
    # 3. COMPLETION: Did you produce a final research memo? (0 or 1)
    #    From ToolRL — binary completion check
    r_completion = 1.0 if has_final_memo(messages) else 0.0
    
    # 4. EFFICIENCY: Penalty for excess tool calls (-N per extra)
    #    From OTC — penalize over-calling (they reduced calls by 73%)
    excess = max(0, len(tool_calls) - 5)
    r_efficiency = -1.0 * excess
    
    return r_format + r_coverage + r_completion + r_efficiency
```

**How to read it:**
- Perfect trajectory (4 valid calls, all real tools, writes memo): 1 + 2 + 1 + 0 = **4.0**
- Decent but inefficient (7 valid calls, 3 of 4 tools, writes memo): 1 + 1.5 + 1 + (-2) = **1.5**
- Hallucinated tool (calls fake tool, writes memo): 1 + **0** + 1 + 0 = **2.0** (veto kills coverage)
- Total failure (invalid JSON, no memo): 0 + 0 + 0 + 0 = **0.0**

**This is already better than our current `compute_reward` because of the multiplicative veto** — hallucinated tools can't be compensated by other good behavior.

---

### Reward B: Composite (Production-Grade, 7 Components)

Adds quality signals: evidence grounding (does the memo cite tool data?) and compliance (no fabricated numbers). These are the signals SFT can't teach — this is where RL earns its keep.

```python
def reward_composite(messages, valid_tools, ticker, judge_client=None):
    """
    Production reward for equity research tool-calling. Range: [-2, 6].
    
    Sources:
      - Format: ToolRL + PORTool (arXiv:2510.26020)
      - Tool correctness (multiplicative): ToolRLA (arXiv:2603.01620) 
      - Parameter accuracy: ToolRL fine-grained decomposition (arXiv:2504.13958)
      - Analysis quality (LLM-as-judge): Tool-R1 (arXiv:2509.12867)
                                         + ART RULER (art.openpipe.ai)
      - Compliance penalty: ToolRLA (lambda=5)
    """
    tool_calls = extract_tool_calls(messages)
    used_names = {tc["name"] for tc in tool_calls}
    final_memo = get_final_response(messages)
    tool_outputs = get_tool_outputs(messages)
    
    # ═══ TIER 1: FORMAT (gate) ═══
    # From PORTool — granular format scoring
    r_format = sum([
        0.3 if all(is_valid_json(tc["arguments"]) for tc in tool_calls) else 0,
        0.2 if len(tool_calls) > 0 else 0,
        0.3 if has_thinking(messages) else 0,
        0.2 if (final_memo and len(final_memo) > 100) else 0,
    ])
    
    # ═══ TIER 2: TOOL CORRECTNESS (multiplicative veto) ═══
    # From ToolRLA — S_name × S_comp. Hallucinated tool = zero.
    hallucinated = used_names - valid_tools
    s_name = 0.0 if hallucinated else 1.0           # veto
    s_comp = len(used_names & valid_tools) / len(valid_tools)  # coverage
    r_correctness = s_name * s_comp * 2.0            # [0, 2]
    
    # From ToolRL — parameter accuracy (did you pass the right ticker?)
    correct_ticker_calls = sum(
        1 for tc in tool_calls
        if tc["name"] in valid_tools and ticker.upper() in tc["arguments"]
    )
    r_params = correct_ticker_calls / max(len(tool_calls), 1)  # [0, 1]
    
    # ═══ TIER 3: ANALYSIS QUALITY (LLM-as-judge — the "true RL" signal) ═══
    # From Tool-R1 + ART RULER
    # An LLM judge reads the FULL trajectory (tool calls + outputs + memo)
    # and scores the analysis quality. This is what SFT cannot teach.
    #
    # The judge sees: "Given these tool outputs, does the memo contain
    # specific, actionable analysis? Or is it generic boilerplate?"
    #
    # Scoring: 0.0 (poor/generic), 0.5 (adequate), 1.0 (specific & insightful)
    
    if judge_client:
        r_quality = llm_judge_analysis(
            trajectory=messages,
            tool_outputs=tool_outputs,
            final_memo=final_memo,
            client=judge_client,
            rubric="""Score 0.0 to 1.0:
            - Does the memo reference specific data points from tool results?
            - Are claims supported by evidence, not generic statements?
            - Does it contain actionable insights a portfolio manager could use?
            - Is the bull/bear case grounded in the actual financial data retrieved?
            Score 0.0 if generic ("revenue is growing"), 0.5 if adequate,
            1.0 if specific ("revenue grew 12% YoY to $416B, driven by Services")."""
        )
    else:
        # Heuristic fallback: section coverage (from Fin-PRM concept coverage)
        sections = ["metric", "development", "financial", "price",
                     "bull", "bear", "bottom line"]
        found = sum(1 for s in sections if s in final_memo.lower())
        r_quality = found / len(sections)
    
    r_quality *= 2.0  # Scale to [0, 2] — quality should be the dominant signal
    
    # ═══ TIER 4: COMPLIANCE (hard penalty) ═══
    # From ToolRLA (lambda=5). Any compliant trajectory beats non-compliant.
    # The judge also flags fabrication — did the model invent data
    # that doesn't appear in any tool output?
    r_compliance = 0.0
    if judge_client:
        has_fabrication = llm_judge_fabrication(
            final_memo=final_memo,
            tool_outputs=tool_outputs,
            client=judge_client,
        )
        r_compliance = -5.0 if has_fabrication else 0.0
    
    # ═══ TOTAL ═══
    total = r_format + r_correctness + r_params + r_quality + r_compliance
    
    return {
        "format": round(r_format, 2),          # [0, 1]   - gate
        "correctness": round(r_correctness, 2), # [0, 2]   - multiplicative veto
        "params": round(r_params, 2),           # [0, 1]   - accuracy  
        "quality": round(r_quality, 2),         # [0, 2]   - LLM judge (the key signal)
        "compliance": r_compliance,             # {-5, 0}  - hard penalty
        "total": round(total, 2),               # [-5, 6]
    }
```

**What changed from the naive version (and why):**

| Removed | Why |
|---------|-----|
| Naive number-matching "evidence grounding" | Matching raw numbers without context is meaningless. "1" in tool output and "1" in memo doesn't mean the model cited evidence. |
| Arbitrary efficiency penalty (>N calls = bad) | Complex questions legitimately need more tool calls. Penalizing >5 calls is arbitrary — an agent researching a conglomerate might need 8 calls. |
| Number-based "compliance" check | Same problem as evidence grounding — no context, just string matching. |

| Added | Why | Source |
|-------|-----|--------|
| **LLM-as-judge for quality** | A judge model reads the full trajectory and scores whether the analysis is specific and data-grounded vs generic boilerplate. This is a real quality signal that can't be gamed with string matching. | [Tool-R1](https://arxiv.org/abs/2509.12867), [ART RULER](https://art.openpipe.ai/fundamentals/ruler) |
| **LLM-as-judge for fabrication** | The judge checks whether claims in the memo are actually supported by tool outputs — with full semantic understanding, not string matching. | [ToolRLA](https://arxiv.org/abs/2603.01620) compliance principle |

**Critical caveat: LLM judges cluster scores.**
[DeepLearning.AI found](https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/) that LLM judges on a 1-10 scale cluster around 0.7-0.9. They avoid being "wrong" by giving generous scores. This kills GRPO because advantages need variance. Solutions:
- Force categorical scores (0.0 / 0.5 / 1.0) — the judge must commit
- Or use a **quiz-based alternative**: generate factual questions from tool outputs ("What was AAPL's TTM revenue?"), test if the memo answers them. This naturally produces varied scores.
- Always verify `reward_std > 0` during training — if the judge gives everything 0.8, you get zero gradient.

**How the LLM-as-judge works in practice:**

```python
async def llm_judge_analysis(trajectory, tool_outputs, final_memo, client, rubric):
    """
    From Tool-R1 (arXiv:2509.12867): LLM-as-judge with partial credit.
    Returns: 1.0 (specific & insightful), 0.5 (adequate), 0.0 (poor/generic)
    
    The judge sees the FULL context: what tools were called, what they
    returned, and what the model wrote. It can tell whether "revenue grew 12%"
    came from the tool output or was fabricated.
    """
    response = await client.chat.completions.create(
        model="gpt-4.1-mini",  # cheap, fast, good enough for judging
        messages=[{
            "role": "user",
            "content": f"""Score this equity research memo 0.0 to 1.0.

TOOL OUTPUTS THE MODEL RECEIVED:
{tool_outputs}

MEMO THE MODEL WROTE:
{final_memo}

RUBRIC:
{rubric}

Return ONLY a number: 0.0, 0.5, or 1.0."""
        }],
        max_tokens=5,
    )
    score = float(response.choices[0].message.content.strip())
    return max(0.0, min(1.0, score))
```

**How to read the Composite reward:**
- Great analysis (format + all tools + right ticker + judge says 1.0): 1 + 2 + 1 + 2 + 0 = **6.0**
- Generic boilerplate (format ok, tools ok, judge says 0.0): 1 + 2 + 1 + 0 + 0 = **4.0**
- Fabricates data (judge catches it): 1 + 2 + 1 + 0.5 + **(-5)** = **-0.5**

**The progression from Simple → Composite shows the audience:**
- Simple = format + correctness (rule-based, what SFT already learns)
- Composite adds LLM-as-judge for quality + compliance (signals beyond what rules can capture)
- "The quality judgment is where RL earns its keep — and it requires a judge, not a regex"

---

## Reward Hacking — What Goes Wrong

**Show the audience a hacked completion:**

> "This trajectory scored 4.0/4.0 on our Simple reward. It called all 4 tools with valid JSON, produced a memo, correct ticker everywhere. Perfect score. Now read the memo..."
>
> *"AAPL is a company. The financials show numbers. The price has moved. Analysts have opinions. In conclusion, AAPL is a stock."*
>
> "That's reward hacking. The Simple reward can't tell the difference between this and real analysis. This is why we need the Composite reward with an LLM judge — a model that actually READS the memo and scores whether it says anything useful."

### Documented Hacking Patterns

From [Lilian Weng (2024)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) and [Anthropic (2025)](https://arxiv.org/abs/2511.18397):

| Pattern | What the model does | Which reward it exploits |
|---------|-------------------|------------------------|
| Tool spam | Calls every tool to maximize coverage | Coverage reward |
| Empty thinking | Outputs `<think>ok</think>` to check the box | Thinking reward |
| Parrot memo | Copies tool output verbatim as "analysis" | Completion reward |
| Exact-N calls | Always makes exactly N calls regardless of need | Efficiency reward |
| Number fabrication | Invents plausible-looking statistics | Quality/grounding reward |

**Mitigations:**
- Multiplicative veto (ToolRLA) — wrong tool zeros everything, can't compensate
- Competing incentives — efficiency vs completeness create natural tension
- Evidence grounding — memo must cite specific data from actual tool outputs
- Compliance penalty (-10) — fabrication is catastrophic, overwhelms all positive rewards
- Human spot-checks — periodically audit high-reward completions

---

## The Evolution of Reward Design (For the "How We Got Here" Narrative)

| Era | Approach | Example | Problem |
|-----|----------|---------|---------|
| 2024 | Binary correct/incorrect | "Did you call any tool?" → 0 or 1 | Almost no gradient signal |
| Early 2025 | Additive composite | Our current `compute_reward` — 6 components summed | Model compensates bad tool selection with good formatting |
| Mid 2025 | Fine-grained decomposition | ToolRL — tool name + param key + param value separately | Better signal, but still additive |
| Late 2025 | Multiplicative + compliance | ToolRLA — wrong tool = zero correctness, plus heavy compliance penalty | Production-proven in finance |
| 2026 | Curriculum + evidence grounding | Trading-R1 — format first, then evidence, then decision quality | Stage-by-stage, each building on previous |

---

## Citation Index

| Paper | Key Innovation | arXiv |
|-------|---------------|-------|
| DeepSeek-R1 | Simplicity: accuracy + format only → emergent reasoning | [2501.12948](https://arxiv.org/abs/2501.12948) |
| ToolRL | Fine-grained: tool name + param name + param value | [2504.13958](https://arxiv.org/abs/2504.13958) |
| ToolRLA | Multiplicative veto + compliance; **financial advisory production** | [2603.01620](https://arxiv.org/abs/2603.01620) |
| Trading-R1 | 3-stage curriculum; **equity research with evidence grounding** | [2509.11420](https://arxiv.org/abs/2509.11420) |
| Trade-R1 | Triangular consistency (evidence ↔ reasoning ↔ decision) | [2601.03948](https://arxiv.org/abs/2601.03948) |
| Fin-PRM | Knowledge verification + concept coverage for financial reasoning | [2508.15202](https://arxiv.org/abs/2508.15202) |
| OTC | Efficiency optimization — 73% fewer tool calls, same accuracy | [2504.14870](https://arxiv.org/abs/2504.14870) |
| PORTool | Granular format scoring (5 sub-checks), correctness dominates | [2510.26020](https://arxiv.org/abs/2510.26020) |
| Tool-R1 | LLM-as-judge with partial credit (1.0 / 0.5 / 0.0) | [2509.12867](https://arxiv.org/abs/2509.12867) |
| GDPO (NVIDIA) | Fixes GRPO reward normalization collapse for multi-component rewards | [2601.05242](https://arxiv.org/abs/2601.05242) |
| Lilian Weng | Reward hacking survey and mitigations | [blog](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) |
| Anthropic | Emergent misalignment from reward hacking in production | [2511.18397](https://arxiv.org/abs/2511.18397) |
