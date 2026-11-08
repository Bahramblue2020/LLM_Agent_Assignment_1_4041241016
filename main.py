import argparse
import json
import sys

import ollama


# Model specified in the assignment PDF for final evaluation.
MODEL_NAME = "qwen3.8:27b"

# Maximum wait time for each local Ollama request, in seconds.
OLLAMA_TIMEOUT = 120

client = ollama.Client(timeout=OLLAMA_TIMEOUT)


def create_plan(problem):
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a planning agent. "
                    "Create a concise solution plan for the problem. "
                    "Identify the problem type, important constraints, "
                    "and the steps needed to solve it. "
                    "Do not calculate the final answer. "
                    "Return only the plan."
                ),
            },
            {
                "role": "user",
                "content": problem,
            },
        ],
        options={"temperature": 0},
    )

    return response["message"]["content"].strip()


def solve_problem(problem, plan):
    response = client.chat(
        model=MODEL_NAME,
        format="json",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a problem-solving agent. "
                    "Use the provided plan to solve the problem carefully. "
                    'Return a valid JSON object with exactly one key: "answer". '
                    "The answer value must contain only the final answer, "
                    "without explanations or Markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem}\n\n"
                    f"Plan:\n{plan}\n\n"
                    'Return JSON in this format: {"answer": "FINAL_ANSWER"}'
                ),
            },
        ],
        options={"temperature": 0},
    )

    result = json.loads(response["message"]["content"])

    if not isinstance(result, dict) or "answer" not in result:
        raise ValueError("Solver returned an invalid JSON response.")

    answer = result["answer"]

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Solver returned an empty or invalid answer.")

    return answer.strip()


def verify_answer(problem, answer):
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a verification agent. "
                    "Check whether the proposed answer correctly solves "
                    "the given problem. Respond with exactly one word: "
                    "VALID or INVALID. Do not include any explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem}\n\n"
                    f"Proposed Answer:\n{answer}"
                ),
            },
        ],
        options={"temperature": 0},
    )

    result = response["message"]["content"].strip().upper()

    if result == "VALID":
        return "VALID"

    return "INVALID"


def revise_answer(problem, plan, previous_answer):
    response = client.chat(
        model=MODEL_NAME,
        format="json",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a problem-solving agent. "
                    "The previous answer was judged incorrect. "
                    "Reconsider the problem using the plan, correct any "
                    "mistakes, and return only the final answer as JSON. "
                    'Use exactly one key: "answer". '
                    "Do not include explanations or Markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem}\n\n"
                    f"Plan:\n{plan}\n\n"
                    f"Previous answer:\n{previous_answer}\n\n"
                    'Return JSON in this format: {"answer": "FINAL_ANSWER"}'
                ),
            },
        ],
        options={"temperature": 0},
    )

    result = json.loads(response["message"]["content"])

    if not isinstance(result, dict) or "answer" not in result:
        raise ValueError("Revision agent returned an invalid JSON response.")

    answer = result["answer"]

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Revision agent returned an empty or invalid answer.")

    return answer.strip()


def run_agent(problem):
    plan = create_plan(problem)
    answer = solve_problem(problem, plan)

    verification = verify_answer(problem, answer)

    if verification != "VALID":
        answer = revise_answer(problem, plan, answer)
        verification = verify_answer(problem, answer)

    # Do not submit an answer that failed the final verification.
    if verification != "VALID":
        return ""

    return answer


def main():
    parser = argparse.ArgumentParser(
        description="Local LLM Agent for Assignment 1"
    )
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    record_id = ""
    answer = ""

    try:
        with open(args.input, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Input must be a JSON object.")

        if "id" not in data or "problem" not in data:
            raise ValueError("Input must contain 'id' and 'problem'.")

        if not isinstance(data["id"], str):
            raise ValueError("'id' must be a string.")

        record_id = data["id"].strip()

        if not record_id:
            raise ValueError("ID cannot be empty.")

        if not isinstance(data["problem"], str):
            raise ValueError("'problem' must be a string.")

        problem = data["problem"].strip()

        if not problem:
            raise ValueError("Problem cannot be empty.")

        answer = run_agent(problem)

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        answer = ""

    result = {
        "id": record_id,
        "answer": answer,
    }

    try:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=4)
    except Exception as error:
        print(f"Output error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
