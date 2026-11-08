# LLM Agent Assignment 1

A local LLM-based agent for solving reasoning problems using Ollama.

## Requirements

- Python 3.9 or newer
- Ollama installed and running
- The evaluation model downloaded in Ollama

## Installation

Install the Python dependency:

```bash
pip install -r requirements.txt
```

Download the model specified in the assignment:

```bash
ollama pull qwen3.8:27b
```

Make sure Ollama is running before executing the program.

## Input format

Create an `input.json` file:

```json
{
    "id": "test_001",
    "problem": "What is 17 multiplied by 23?"
}
```

## Run

```bash
python main.py --input input.json --output output.json
```

## Output format

The program writes the result to the specified output file:

```json
{
    "id": "test_001",
    "answer": "391"
}
```

The `answer` field contains only the final answer. If input validation,
model execution, or final verification fails, the answer field is empty.

## Agent architecture

1. **Planner:** creates a concise plan for the problem.
2. **Solver:** solves the problem using the plan and returns a JSON answer.
3. **Verifier:** checks the proposed answer.
4. **Revision:** if the answer is rejected, the agent attempts one correction
   and verifies the corrected answer again.

All model inference is performed locally through Ollama. No cloud LLM API is used.
Each Ollama request has a 120-second timeout.

## GitHub submission

Create a GitHub repository with this exact name:

```text
LLM_Agent_Assignment_1_4041241016
```

Upload `main.py`, `requirements.txt`, and `README.md` to the repository.
Submit only the repository URL in the course group, as instructed.
