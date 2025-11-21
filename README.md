# Winterfest Applied AI: Equity Investment Opportunity sourcing

This project implements an end-to-end AI system for automated public equity investment research and analysis. It combines a powerful **Research Agent** capable of gathering real-time internal and external data with a custom **Fine-Tuned Analyst Model** that reasons through the data to provide an investment verdict.

## 🏗️ Architecture

The system operates in a two-stage pipeline:

```text
   [ Target Company ]
           │
           ▼
 ┌───────────────────────┐
 │  🕵️‍♂️ Research Agent    │
 │ (Web, Files, Stocks)  │
 └───────────┬───────────┘
             │
             ▼
   [ 📝 Research Memo ]
             │
             ▼
 ┌───────────────────────┐
 │   🧠 Analyst Model    │
 │  (Fine-Tuned Qwen)    │
 └───────────┬───────────┘
             │
             ▼
  [ 💡 Investment Verdict ]
```

## 1. The Research Agent 🕵️‍♂️

Located in `nb/agent.ipynb`, the Equity Research Agent is built using a proprietary agentic framework powered by `gpt-5.1`. It acts as the initial gatherer of intelligence, compiling a comprehensive "memo-like" writeup on a target company.

![Research Agent](docs/research_agent.png)

### Capabilities & Tools
The agent is equipped with a suite of tools to ensure exhaustive coverage:

- **🌐 Web Search Tool:** Scours the internet for the latest news, competitor analysis, and market trends.
- **📂 File Search Tool:** RAG-enabled search over internal documents (e.g., PDF text files, proprietary reports) stored in `data/docs`.
- **💻 Code Interpreter:** Executes Python code for on-the-fly data analysis and visualization.
- **📈 Stock Market Tools (MCP):** 
  - Connects to a local **Model Context Protocol (MCP)** server (`tools/mcp/stock_server.py`).
  - Provides real-time financial data via Yahoo Finance integration.
  - Capabilities: `get_financials`, `get_price_history`, `get_stock_news`, `get_recommendations`.

**Output:** A structured markdown report covering **Competition**, **Customers**, **Financials**, and **Growth Opportunities**.

## 2. The Investment Analyst 🧠

The second stage involves a custom Small Language Model (SLM) fine-tuned specifically to mimic the reasoning process of a senior investment analyst.

### The Model
- **Base Model:** `unsloth/Qwen3-4B-Thinking` (Qwen 3, 4B parameters).
- **Architecture:** 4-bit Quantized, Fine-tuned using LoRA (Low-Rank Adaptation) via **Unsloth**.
- **Objective:** To take a raw research report, perform "thinking-aloud" (Chain-of-Thought) analysis, and output a structured verdict (`Strong Yes`, `Questionable`, `Strong No`).

### Training Process
1.  **Synthetic Data Generation** (`nb/training_data_generator.ipynb`):
    -   We generated **~5,000** synthetic training examples using `gpt-4.1-mini`.
    -   Each example consists of a fictional company report and a corresponding expert-level investment verdict with reasoning.
    -   Data is stored in `data/training_data_examples_all.jsonl`.

2.  **Fine-Tuning Recipe** (`nb/training_recipe.ipynb`):
    -   **Library:** Unsloth (2x faster training, 60% less memory).
    -   **Technique:** Supervised Fine-Tuning (SFT) on the generated dataset.
    -   **Format:** Trained to output a `<think>` block followed by the verdict, encouraging the model to "reason" before deciding.
    -   **Performance:** Achieved minimal overfitting (Train/Eval loss gap ~0.01) and strong generalization on holdout sets.

![Training Loss](docs/loss_curve.png)

## 📂 Project Structure

```text
winterfest_applied_ai_pe/
├── data/
│   ├── docs/                   # Internal documents for RAG (PDFs, etc.)
│   ├── output/                 # Generated research reports
│   ├── training_data_*.jsonl   # Synthetic datasets for fine-tuning
│   └── ...    
├── docs/              
├── nb/
│   ├── helpers/                # Utility scripts (LLM streaming, etc.)  
│   ├── agent.ipynb             # Stage 1: Research Agent implementation
│   ├── training_data_generator.ipynb # Generates synthetic training data
│   └── training_recipe.ipynb   # Stage 2: Model Fine-tuning workflow
├── tools/
│   └── mcp/
│       └── stock_server.py     # MCP Server for Yahoo Finance data
├── main.py                     # Main entry point
└── README.md                   # This file
```

## 🚀 Getting Started

1.  **Install Dependencies:** Ensure you have `uv` or `pip` installed and the environment set up.
2.  **Start MCP Server:** Run the stock market tools server.
    ```bash
    python tools/mcp/stock_server.py
    ```
3.  **Run the Agent:** Open `nb/agent.ipynb` to generate research reports on target companies.
4.  **Train/Inference:** Use `nb/training_recipe.ipynb` to fine-tune the model or run inference on the generated reports.

