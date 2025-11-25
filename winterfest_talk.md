# Agentic Systems in Practice: Applying AI to Investments
**❄️ Data Science Festival WinterFest 2025**

<img src="docs/speaker_flyer.jpg" alt="Kristof Rabay" height="400">

*Kristof Rabay - Applied AI @ The Carlyle Group*  
*November 27, 2025*

---

> **⚠️ Disclaimer:**  
> The views and opinions expressed in this presentation are those of the speaker alone and do not reflect the official policy, position, or views of The Carlyle Group or its affiliates.

---

## 📋 Agenda

1.  **The Objective:** Why are we here?
2.  **The Timeline:** From Chatbots to Autonomous Agents (2022-2025).
3.  **Stage 1:** Building our Research Agent (Prototyping).
4.  **Stage 2:** Creating our Custom Analyst Model.
5.  **Conclusion:** The Future of Applied AI Architectures.

---

## 🎯 Objective

**To demonstrate how modern AI capabilities enable rapid prototyping of complex workflows, and how one can leverage Context Engineering and Fine-Tuning to turn generic models into specialized enterprise tools.**

---

## ⏳ The Acceleration Era: A 3-Year Journey

We have witnessed a Cambrian explosion in AI capabilities. In just 36 months, we moved from simple text completion to autonomous agents capable of deep research and reasoning.

### 2022-2023: The "API" Era 👶
*   **[Nov 2022](https://openai.com/index/chatgpt/)** **gpt-3.5-turbo** launches. The world wakes up to AI.
*   **[June 2023](https://openai.com/blog/function-calling-and-other-api-updates):** **Function Calling**. LLMs get "hands" to touch external tools.
    *   *Impact:* We moved from brittle `regex` parsing to native tool execution.

### 2024: The "Agentic" Shift 🧠
*   **[Sept 2024](https://openai.com/index/learning-to-reason-with-llms/):** **OpenAI o1-preview**. The first "Reasoning" model.
    *   *Shift:* Models start "thinking" before speaking.
*   **[Oct 2024](https://github.com/openai/swarm):** **Agents SDK (Swarm)**. Orchestration becomes a first-class citizen.
*   **[Nov 2024](https://www.anthropic.com/news/model-context-protocol):** **Anthropic MCP**. A universal standard for connecting AI to data.
    *   *Impact:* No more writing custom integrations for every data source.

### 2025: Deep Research, Autonomy & Fine-tuning 🚀
*   **[Jan 2025](https://github.com/deepseek-ai/DeepSeek-R1):** **DeepSeek R1**. Open-source reasoning matches proprietary models.
*   **[Feb 2025](https://openai.com/index/introducing-deep-research/):** **OpenAI Deep Research**.
    *   *Capability:* A snapshot of an o3-fine-tune to navigate the web
*   **[Jun 2025](https://www.anthropic.com/engineering/multi-agent-research-system)**: **Anthropic Deep Research**.
    *   *Capability:* A planner - worker multi-agent team setup for deep research
*   **[Sept-Nov 2025](https://openai.com/index/introducing-upgrades-to-codex/)**: Coding-specific snapshots of 'base' models
    *   *Capability:* Autonomous coding agents that can maintain context for days.
*   **[Nov 2025](https://openai.com/index/gpt-5-1-codex-max/)** **GPT-5.1-Codex-Max**.
    *   *Capability:* **The 24-Hour Agent.** Can work independently on complex tasks for over a day without human intervention.

---

## 🏗️ Stage 1: The Research Agent (Prototyping)

To solve our Equity Research problem, we started with **Prototyping**. We built a "General Purpose" Research Agent using the best proprietary models available today.

### The Stack
*   **Model:** `GPT-5.1` (High Intelligence, High Cost)
*   **Connectivity:** **Model Context Protocol (MCP)**
    *   *Why MCP?* It allowed us to plug in Yahoo Finance, Internal Docs, and Web Search standardly.

### 🛠️ Tools in Action
The agent autonomously decides which tool to use:
1.  **🌐 Web Search:** For real-time competitor news.
2.  **📂 Internal Docs (RAG):** For proprietary investment mandates.
3.  **📈 Stock Market MCP:** For live price history and financials.
4.  **🐍 Code Interpreter:** For calculating valuation metrics on the fly.

<img src="docs/research_agent.png" alt="Agent Setup" height="200">

### 🔧 Context Engineering: Optimizing Tool Use
One of the biggest challenges in agentic systems is "context management."

*   **The Problem:** Overloading the context window with too long prompts and too many tool definitions takes away space for analysis and more importantly - confuses the model.
*   **The Solution:** **[Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use).**
    *   *Strategy:* Don't just dump all tools. Use a "Router" or "Planner" step to select *only* the relevant tools for the current step.
    *   *Optimization:* Combine reasoning and action. Instead of `Reason -> Tool Call -> Reason -> Tool Call`, allow the model to write a **script** that executes multiple steps at once.

> "By enabling the model to write execution scripts across tools, we reduce the Reasoning-Action chain latency and cost." — *Inspired by Anthropic Engineering*

<img src="https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Ff359296f770706608901eadaffbff4ca0b67874c-1999x1125.png&w=3840&q=75" alt="Advanced Tool Use Visualization" width="600">

---

## 🧠 Stage 2: The Analyst Model (Customization)

While the Research Agent is powerful, it is **generic** and **expensive**. To scale, we needed a specialist.

### The Pivot: Prototyping ➡️ Production
We didn't just want "an answer"; we wanted **OUR** answer.

*   **Context Engineering:** We utilized the Research Agent to generate thousands of synthetic "Reasoning Traces" (Chain of Thought).
*   **Fine-Tuning:** We distilled this intelligence into a smaller, faster, open-source model (`Qwen3-4B-Thinking`).

### The Comparison
| Feature | 🕵️‍♂️ Research Agent (Prototype) | 🧠 Analyst Model (Production) |
| :--- | :--- | :--- |
| **Model** | GPT-5.1 (Proprietary) | Qwen3-4B (Open Source) |
| **Cost** | $$$ per run | ¢ per run |
| **Style** | Generic Helpful Assistant | **Ruthless Investment Analyst** |
| **Speed** | Slow (Reasoning API) | **Lightning Fast** |

<!-- SUGGESTED IMAGE: The Training Loss Curve image showing how the model learned to think like an analyst -->

---

## 💡 Conclusion

We are no longer just "prompting" models. We are **architecting** systems.

1.  **Use Frontier Models** (GPT-5.1) to **explore** and **prototype** (The Research Agent).
2.  **Use Synthetic Data** to **capture** that intelligence.
3.  **Fine-Tune SLMs** (Small Language Models) to **specialize** and **scale** (The Analyst Model).

*Welcome to the future of Applied AI.* ❄️
