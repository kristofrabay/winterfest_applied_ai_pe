
Search

Get app
Write
Sign up

Sign in



Towards AI
Towards AI
We build Enterprise AI. We teach what we learn. Join 100K+ AI practitioners on Towards AI Academy. Free: 6-day Agentic AI Engineering Email Guide: https://email-course.towardsai.net/

Follow publication

Fine-Tuning Open Source Models for Function Calling: A Complete Guide with Unsloth and Docker
Sharath Kumar
Sharath Kumar

Follow
10 min read
·
Oct 6, 2025
1




Press enter or click to view image in full size

The rise of AI agents and tool-enabled applications has made function calling a critical capability for language models. While proprietary models like GPT-4 excel at function calling out of the box, open-source alternatives require specialized fine-tuning to achieve comparable performance. In this comprehensive guide, we’ll explore how to leverage Unsloth’s Docker environment to fine-tune Llama 3.1 8B on the Hermes Function Calling V1 dataset. Let’s create a powerful, cost-effective function calling model!

Why Function Calling Matters?
Function calling enables language models to interact with external systems, APIs, and tools by generating structured JSON outputs that can be reliably parsed and executed. This capability is essential for:

AI Agents and Chatbots: Enabling dynamic interactions with databases, APIs, and services
Enterprise Applications: Automating complex workflows and business processes
Data Extraction: Converting natural language queries into structured API calls
Multi-step Reasoning: Coordinating multiple tools to solve complex problems
The Power of Open Source: Why Llama 3.1 8B?
Based on recent benchmarks and community feedback, Llama 3.1 8B-Instruct stands out as the optimal choice for function calling fine-tuning. Here’s why:

Technical Advantages:

128K context window: Handles complex multi-tool scenarios
Advanced architecture: Grouped-Query Attention (GQA) for efficient inference
Strong baseline performance: Competitive with larger models in many scenarios
Active community: Extensive documentation and support ecosystem
Practical Benefits:

Resource efficiency: Runs effectively on single GPUs (8GB+ VRAM)
Cost-effective: Significantly cheaper than proprietary alternatives
Customizable: Full control over fine-tune-ing data and model behavior
Privacy-focused: On-premise deployment without data sharing concerns
Understanding the Hermes Function Calling V1 Dataset
The Hermes Function Calling V1 dataset from NousResearch represents a gold standard for function calling training data. This comprehensive dataset includes:

Dataset Composition:

Single-turn function calling: Direct query-to-function mappings
Multi-turn conversations: Complex interactions with multiple function calls
Structured JSON outputs: Template-based response formatting
Diverse domains: IoT, e-commerce, data analysis, and more scenarios
Data Structure:

Each conversation follows the ShareGPT format with specific roles:

{
 "id": "unique-identifier",
 "conversations": [
 {"from": "system", "value": "Function calling instructions with tool definitions"},
 {"from": "human", "value": "Natural language query"},
 {"from": "model", "value": "JSON-formatted function call response"}
 ],
 "category": "Domain classification",
 "subcategory": "Specific use case",
 "task": "Task description"
}
Training Format Advantages:

Standardized prompt structure: Consistent <tools> and <tool_call> formatting
Schema validation: Proper JSON structure enforcement
Error handling: Examples of edge cases and error recovery
Real-world scenarios: Practical applications across industries
Now that we have a decent idea of what we’re working with, let’s set up the unsloth docker environment.

Setting Up Unsloth with Docker: The Complete Environment
Unsloth’s Docker image provides a stable, dependency-managed environment that eliminates the complexity of local setup while delivering 2x faster training with 70% less VRAM usage.

Prerequisites and Installation
System Requirements:

NVIDIA GPU with 8GB+ VRAM
Docker installed on your system
NVIDIA Container Toolkit
Step 1: Install NVIDIA Container Toolkit

export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1

sudo apt-get update && sudo apt-get install -y \
  nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
Step 2: Launch the Unsloth Container

First, we need to have Docker set up on our PC. I’m using Windows and Docker Desktop for the same. Ensure your Windows system meets the requirements for Docker Desktop, including a 64-bit processor, sufficient RAM, and enabled hardware virtualization (Hyper-V or WSL 2). Next, download Docker desktop installer and run it. Follow the on-screen prompts and ensure the option to use the WSL 2 based engine (recommended for performance) or Hyper-V is selected during installation. Restart your computer and run Docker Desktop. Once you have it up and running, run the following command to ensure its set up correctly.

docker run hello-world
If you already have Docker set up, run the following to launch the Unsloth container:

docker run -d -e JUPYTER_PASSWORD="mypassword" \
  -p 8888:8888 -p 2222:22 \
  -v $(pwd)/work:/workspace/work \
  --gpus all \
  unsloth/unsloth
What does the above command do?

docker run: This is the main Docker command to create and start a new container from an image.
-d: Runs the container in detached mode, meaning the container runs in the background, freeing your terminal for other tasks.
-e JUPYTER_PASSWORD="mypassword": Sets an environment variable inside the container. Here it sets the password for accessing the Jupyter notebook interface.
-p 8888:8888: Maps port 8888 on your local machine to port 8888 inside the container, letting you access Jupyter notebooks via http://localhost:8888.
-p 2222:22: Maps local port 2222 to port 22 inside the container, enabling SSH access on port 2222 of your machine.
-v $(pwd)/work:/workspace/work: Mounts your current working directory's work folder into /workspace/work inside the container. This keeps your work persistent outside the container.
--gpus all: Grants the container access to all available NVIDIA GPUs on your machine, which is essential for accelerated model training.
unsloth/unsloth: Specifies the Docker image to run, which contains the Unsloth environment preconfigured for fast fine-tuning tasks.
Summary: This command starts the Unsloth environment containers in the background, sets up password-protected Jupyter notebook access, forwards necessary ports for Jupyter and SSH, mounts your project folder for persistent storage, and enables GPU support . Making sure we’re ready for AI model training and experimentation.

Fine-Tuning Implementation: Step-by-Step Guide
Now the fun part, let’s start the model fine-tuning process. Do note that we’re fine tuning an open source model for tool-calling purposes. This will enable it to be used in an agentic workflow efficiently.

Step 1: Dataset Preparation
First, download and prepare the Hermes Function Calling V1 dataset:

from datasets import load_dataset
import json

# Load the Hermes Function Calling V1 dataset
dataset = load_dataset("NousResearch/hermes-function-calling-v1")

# Examine the dataset structure
print("Dataset features:", dataset['train'].features)
print("Total examples:", len(dataset['train']))

# Sample data inspection
sample = dataset['train'][0]
print("Sample conversation structure:")
print(json.dumps(sample['conversations'], indent=2))
Step 2: Model Loading and LoRA Configuration
In this step, we load the base language model and adapt it for efficient fine-tuning using LoRA, a powerful technique that modifies the model without updating all its parameters, making training faster and less resource-intensive.

What is LoRA?

A brief detour on what LoRA is, skip this if you already know. LoRA (Low-Rank Adaptation) is a method designed to fine-tune large language models by learning small low-rank matrices that adjust the original model’s weights during training rather than updating every single parameter. This helps:

Reduce memory usage: Only a few extra parameters are trained, so VRAM requirements drop drastically.
Speed up training: Less computation is needed because the original weights stay frozen.
Make fine-tuning accessible: Enables training large models on smaller GPUs.
How LoRA Works in Model Modification

Get Sharath Kumar’s stories in your inbox
Join Medium for free to get updates from this writer.

Enter your email
Subscribe

Remember me for faster sign in

The technique inserts trainable low-rank update matrices into specific layers of the model (commonly the query, key, value, and output projection matrices in attention layers). During inference:

The original model weights remain fixed.
LoRA layers produce small adjustments that are added to the original weights dynamically.
This clever factorization allows the model to adapt to new tasks — like understanding and generating function calls without extensive retraining.

Practical Impact for Your Fine-Tuning

When you run the code to get a LoRA-adapted model, you:

Load the base Llama 3.1 8B Instruct model, which is well-suited for instruction-following and function calling.
Specify target modules (e.g., “q_proj”, “k_proj”, “v_proj”, etc.) where LoRA will inject trainable parameters.
Set hyperparameters such as rank (r), alpha scaling, and dropout rate to control how much and how robustly your updates influence the model.
Optionally enable Rank-Stabilized LoRA (RSLORA) and gradient checkpointing for better convergence and memory efficiency.
Overall, this step prepares a fine-tuning-friendly model that trains faster, uses less GPU memory, and focuses on learning the nuances of the function calling task effectively.

import torch
from unsloth import FastLanguageModel

# Configuration parameters
max_seq_length = 4096  # Extended for complex function calling scenarios
dtype = None  # Auto-detect optimal dtype
load_in_4bit = True  # Memory optimization

# Load Llama 3.1 8B Instruct
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # Use your HF token if needed
    # token="hf_...",
)

# Configure LoRA for function calling optimization
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Higher rank for complex function calling patterns
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,  # Matching the rank for optimal performance
    lora_dropout=0.05,  # Slight dropout for regularization
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=True,  # Rank-stabilized LoRA for better convergence
)
Step 3: Chat Template and Data Formatting
from unsloth.chat_templates import get_chat_template

# Configure the Llama 3.1 chat template for function calling
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    map_eos_token=True,
)

def format_function_calling_prompt(examples):
    """
    Format the Hermes dataset for function calling fine-tuning
    """
    texts = []
    
    for conversation in examples['conversations']:
        # Convert the conversation format to Llama 3.1 structure
        formatted_conversation = []
        
        for message in conversation:
            if message['from'] == 'system':
                formatted_conversation.append({
                    "role": "system",
                    "content": message['value']
                })
            elif message['from'] == 'human':
                formatted_conversation.append({
                    "role": "user", 
                    "content": message['value']
                })
            elif message['from'] == 'gpt':
                formatted_conversation.append({
                    "role": "assistant",
                    "content": message['value']
                })
        
        # Apply the chat template
        text = tokenizer.apply_chat_template(
            formatted_conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    return {"text": texts}

# Apply formatting to the dataset
formatted_dataset = dataset.map(
    format_function_calling_prompt,
    batched=True,
    remove_columns=dataset['train'].column_names
)
Step 4: Training Configuration and Execution
from transformers import TrainingArguments
from trl import SFTTrainer

# Optimized training configuration for function calling
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size of 16 (4x4)
    warmup_steps=100,
    num_train_epochs=3,  # Conservative for function calling
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="./llama-3.1-8b-function-calling",
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="steps" if "validation" in dataset else "no",
    eval_steps=500 if "validation" in dataset else None,
    load_best_model_at_end=True if "validation" in dataset else False,
    report_to="none",  # Disable wandb for this example
)

# Initialize the SFT trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset['train'],
    eval_dataset=formatted_dataset.get('validation'),
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Important for function calling format preservation
    args=training_args,
)

# Execute training
trainer_stats = trainer.train()
print("Training completed!")
print(f"Training loss: {trainer_stats.training_loss:.4f}")
And that’s it! We’ve executed the training step. The next step is to evaluate how our model does on the test data we set aside.

Model Evaluation and Testing
After training, it’s crucial to evaluate the model’s function calling capabilities:

# Test the fine-tuned model
FastLanguageModel.for_inference(model)

def test_function_calling(query, tools_schema):
    """Test function calling with a specific query"""
    
    system_prompt = f"""You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query.

<tools>
{tools_schema}
</tools>

For each function call return a json object with function name and arguments within <tool_call> </tool_call> tags with the following schema:
<tool_call>
{{'arguments': <args-dict>, 'name': <function-name>}}
</tool_call>"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        use_cache=True,
        temperature=0.1,
        do_sample=True,
    )
    
    response = tokenizer.batch_decode(outputs)[0]
    return response

# Example test
weather_tools = """[{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get current weather for a location', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'City name'}, 'units': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['location']}}}]"""

result = test_function_calling("What's the weather like in San Francisco?", weather_tools)
print(result)
Model Export and Deployment
After successfully fine-tuning and testing your model on the function calling task, the next important step is exporting it so you can easily load and deploy it for inference in different environments or integrate it with your applications.

Why Export the Model?

Portability: Save the trained model weights and tokenizer so you can transfer or share your function calling model.
Deployment: Load the fine-tuned model seamlessly into production servers, cloud platforms, or edge devices.
Version Control: Keep snapshots of different training stages or experiment versions.
Inference Optimization: You can merge LoRA adapters with the base model and apply quantization for smaller models that serve faster with lower resource needs.
Exporting the Model: What to Save?
When exporting a LoRA-fine-tuned model, wegenerally save:

Base model weights — The pretrained Llama 3.1 8B parameters.
LoRA adapters — Low-rank matrices with fine-tuning updates.
Tokenizer files — Vocabulary and tokenization rules matching the model.
Merged Model (Optional) — The combined model after merging LoRA weights into base weights.
Quantized Model (Optional) — Reduced precision model for faster, smaller deployments.
How to Export
# Save the LoRA adapters
model.save_pretrained("./llama-3.1-8b-function-calling-lora")
tokenizer.save_pretrained("./llama-3.1-8b-function-calling-lora")

# Merge LoRA with base model for deployment
merged_model = FastLanguageModel.merge_and_unload(model)
merged_model.save_pretrained("./llama-3.1-8b-function-calling-merged")

# Export to GGUF for efficient inference
model.save_pretrained_gguf("./function-calling-model", tokenizer, quantization_method="q4_k_m")
Loading the Model Elsewhere
You can load your exported models in any Python environment or platform that supports the transformers or Unsloth libraries:

Loading LoRA-Enabled Model for Further Training or Evaluation
from unsloth import FastLanguageModel

# Load base model with LoRA adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    "./llama-3.1-8b-function-calling-lora",
    load_lora=True
)
2. Loading Merged Model for Inference

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "./llama-3.1-8b-function-calling-merged"
)

# Put model in inference mode
model.eval()
Deployment Options
Cloud services: Deploy on AWS, Azure, or Google Cloud with GPU instances.
Containerized environments: Use Docker containers for reproducible deployments.
API serving: Integrate your fine-tuned model into REST or gRPC APIs using frameworks like FastAPI or Flask.
Edge devices: Compress models with quantization to run on edge GPUs or CPUs.
Troubleshooting Common Issues
Memory Management:

Use gradient checkpointing for longer sequences
Implement dynamic batching for variable-length inputs
Monitor GPU memory usage during training
Training Stability:

Start with lower learning rates (1e-5 to 2e-4)
Use warmup steps to stabilize early training
Implement gradient clipping if needed
Function Calling Quality:

Validate JSON format during training
Include negative examples (when not to call functions)
Test with diverse function schemas
Future Directions and Enhancements
The field of function calling continues to evolve rapidly. Consider these emerging trends:

Advanced Capabilities:

Multi-modal function calling: Integrating vision and text inputs
Chain-of-thought reasoning: Enhanced planning for complex tasks
Dynamic tool discovery: Runtime function registration and usage
Performance Improvements:

Speculative decoding: Faster inference for structured outputs
Knowledge distillation: Transfer capabilities from larger models
Reinforcement learning: RLHF for better function-calling behavior
Conclusion: Building the Future of AI Agents
Press enter or click to view image in full size

Fine-tuning open-source models for function calling represents a pivotal step toward democratizing AI agent capabilities. By combining Unsloth’s efficient training framework with high-quality datasets like Hermes Function Calling V1, developers can create powerful, customizable models that rival proprietary alternatives.

LLM
Llm Agent
Finetune Llm
Docker
Unsloth
1



Towards AI
Published in Towards AI
118K followers
·
Last published just now
We build Enterprise AI. We teach what we learn. Join 100K+ AI practitioners on Towards AI Academy. Free: 6-day Agentic AI Engineering Email Guide: https://email-course.towardsai.net/


Follow
Sharath Kumar
Written by Sharath Kumar
16 followers
·
16 following
Senior Data Scientist focused on NLP and Large Language Models. Fascinated by the interplay of human behavior, physics, and philosophy.


Follow
No responses yet

Write a response

What are your thoughts?

Cancel
Respond
More from Sharath Kumar and Towards AI
The Ultimate Guide to LLM Evaluation: A Practical Framework for Real-World AI Systems
Sharath Kumar
Sharath Kumar

The Ultimate Guide to LLM Evaluation: A Practical Framework for Real-World AI Systems
Large Language Models have transformed from academic curiosities to mission-critical components powering everything from customer service…
Oct 14, 2025
52
Close-up of crumpled resignation letter reading “The world is in peril” on glass desk
Towards AI
In

Towards AI

by

MohamedAbdelmenem

They Built the AI. Now They’re Quitting. Here’s What They Saw.
Matt Shumer’s essay hit 75M views. xAI’s co-founder quit citing “recursive self-improvement.” Microsoft’s AI chief says 18 months. Here’s…

Feb 15
6.5K
273
I’ve Been Recommending DeepSeek & Kimi for Months. Then Anthropic Published This.
Towards AI
In

Towards AI

by

Adham Khaled

I’ve Been Recommending DeepSeek & Kimi for Months. Then Anthropic Published This.
A breakdown of the most explosive AI security report of 2026 — and what it honestly means for everyone using Chinese AI tools.

Feb 24
1.7K
84
What If Parenting Doesn’t Shape Kids As Much As We Think?
Sharath Kumar
Sharath Kumar

What If Parenting Doesn’t Shape Kids As Much As We Think?
When most of us think about how children develop, we typically imagine that parents are the architects — molding personalities, guiding…
Apr 4, 2025
See all from Sharath Kumar
See all from Towards AI
Recommended from Medium
9 RAG Architectures Every AI Developer Must Know: A Complete Guide with Examples
Towards AI
In

Towards AI

by

Divy Yadav

9 RAG Architectures Every AI Developer Must Know: A Complete Guide with Examples
Architectures beyond Naive Rag to build reliable production AI Systems

Dec 19, 2025
2.4K
42
Building the 7 Layers of a Production-Grade Agentic AI System
Level Up Coding
In

Level Up Coding

by

Fareed Khan

Building the 7 Layers of a Production-Grade Agentic AI System
Service Layer, Middleware, Context Management and more

Dec 19, 2025
1.8K
30
AI Agents: Complete Course
Data Science Collective
In

Data Science Collective

by

Marina Wyss

AI Agents: Complete Course
From beginner to intermediate to production.

Dec 6, 2025
4.97K
183
Full Fine-Tuning vs PEFT vs RLHF vs DPO: Which LLM Tuning Method Is Right for You?
Martin Keywood
Martin Keywood

Full Fine-Tuning vs PEFT vs RLHF vs DPO: Which LLM Tuning Method Is Right for You?

Jan 23
4
Stanford Just Killed Prompt Engineering With 8 Words (And I Can’t Believe It Worked)
Generative AI
In

Generative AI

by

Adham Khaled

Stanford Just Killed Prompt Engineering With 8 Words (And I Can’t Believe It Worked)
ChatGPT keeps giving you the same boring response? This new technique unlocks 2× more creativity from ANY AI model — no training required…

Oct 20, 2025
25K
683
A Lawyer Just Beat 500 Developers at Anthropic’s Hackathon
AI Advances
In

AI Advances

by

Jing Hu

A Lawyer Just Beat 500 Developers at Anthropic’s Hackathon
The most valuable skill in AI right now isn’t code — and you probably already have it.

Feb 24
3.4K
46
See more recommendations
Help

Status

About

Careers

Press

Blog

Privacy

Rules

Terms

Text to speech