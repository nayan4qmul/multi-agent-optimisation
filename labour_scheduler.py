from autogen import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
    GroupChat,
)
from ortools.sat.python import cp_model
import pandas as pd
import json
import random
from typing import Dict


# ======================
# 1. Data Preparation
# ======================
def generate_employees(n=100) -> pd.DataFrame:
    """Generate synthetic employee data with preferences"""
    roles = ["Nurse"] * 60 + ["Doctor"] * 20 + ["Admin"] * 20
    random.shuffle(roles)

    data = []
    for i in range(n):
        role = roles[i]
        wage = (
            random.randint(15, 25)
            if role == "Nurse"
            else random.randint(30, 40) if role == "Doctor" else random.randint(20, 30)
        )

        data.append(
            {
                "id": f"EMP_{i:03d}",
                "role": role,
                "wage": wage,
                "max_hours": random.choice([20, 30, 40]),
                "preferred_shifts": random.sample(
                    ["Morning", "Evening", "Night"], k=random.randint(1, 3)
                ),
                "unavailable_days": random.sample(
                    ["Mon", "Tue", "Wed", "Thu", "Fri"], k=random.randint(0, 2)
                ),
            }
        )
    return pd.DataFrame(data)


employees = generate_employees(100)
employees.to_csv("employees.csv", index=False)

# ======================
# 2. Configuration
# ======================
config_list = [
    {
        "model": "llama3.2:3b",  # "mistral:latest",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
]

llm_config = {"config_list": config_list, "temperature": 0.1}


# ======================
# 3. Agent Definitions
# ======================
class ScheduleValidator:
    @staticmethod
    def validate(schedule: Dict) -> Dict:
        """Programmatic constraint validation"""
        errors = []
        total_hours = {emp_id: 0 for emp_id in schedule.keys()}

        # Validate shift coverage
        shift_counts = {"Morning": 0, "Evening": 0, "Night": 0}
        for emp_id, shifts in schedule.items():
            for shift in shifts:
                shift_time = shift.split("_")[1]
                shift_counts[shift_time] += 1
                total_hours[emp_id] += 8  # Assuming 8-hour shifts

        # Check max hours
        for emp_id, hours in total_hours.items():
            emp_data = employees[employees["id"] == emp_id].iloc[0]
            if hours > emp_data["max_hours"]:
                errors.append(f"{emp_id} exceeds max hours ({emp_data['max_hours']})")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "coverage": shift_counts,
            "required_coverage": {"Morning": 10, "Evening": 8, "Night": 5},
        }


# Core Agents
manager = AssistantAgent(
    name="Manager",
    system_message="""Coordinate scheduling. Your tasks:
    1. Collect proposals from Proposer
    2. Route to Validator for checks
    3. Incorporate feedback from reviewers
    4. Finalize schedule in JSON format:
       {"EMP_001": ["Mon_Morning", "Tue_Evening"], ...}
    Terminate with 'FINAL_SCHEDULE' when done.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

proposer = AssistantAgent(
    name="Proposer",
    system_message="""Generate shift schedules considering:
    - Employee preferences (preferred_shifts in employees.csv)
    - Wage costs (minimize total)
    - Fair shift distribution
    Output MUST be JSON format.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

validator = AssistantAgent(
    name="Validator",
    system_message="""Validate schedules programmatically.
    You have access to the ScheduleValidator tool.
    Reply with validation results ONLY.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Reviewer Agents
cost_reviewer = AssistantAgent(
    name="Cost_Reviewer",
    system_message="""Analyze schedule cost efficiency:
    1. Calculate total payroll cost
    2. Identify potential savings
    Output format: {"total_cost": X, "suggestions": []}""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

fairness_reviewer = AssistantAgent(
    name="Fairness_Reviewer",
    system_message="""Evaluate schedule fairness:
    1. Count shifts per employee
    2. Check preference alignment
    Output format: {"score": 0-100, "issues": []}""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)


# ======================
# 4. OR-Tools Fallback
# ======================
def ortools_solve() -> Dict:
    """Mathematical optimization fallback"""
    model = cp_model.CpModel()

    # Create variables
    emp_vars = {
        row["id"]: model.NewIntVar(0, row["max_hours"], row["id"])
        for _, row in employees.iterrows()
    }

    # Shift coverage constraints
    shift_requirements = {"Morning": 10, "Evening": 8, "Night": 5}
    for shift, req in shift_requirements.items():
        model.Add(sum(emp_vars[emp["id"]] for _, emp in employees.iterrows()) >= req)

    # Objective: Minimize cost
    model.Minimize(
        sum(emp["wage"] * emp_vars[emp["id"]] for _, emp in employees.iterrows())
    )

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        return {
            emp["id"]: solver.Value(emp_vars[emp["id"]])
            for _, emp in employees.iterrows()
        }
    raise RuntimeError("OR-Tools failed to find solution")


# ======================
# 5. Execution Workflow
# ======================
def run_scheduling() -> Dict:
    # Initialize group chat for core problem-solving
    problem_solving_group = GroupChat(
        agents=[manager, proposer, validator],
        messages=[],
        max_round=8,
        speaker_selection_method="round_robin",
    )
    group_manager = GroupChatManager(
        groupchat=problem_solving_group, llm_config=llm_config
    )

    # Start the conversation
    user_proxy = UserProxyAgent(
        name="Admin", human_input_mode="NEVER", code_execution_config=False
    )
    user_proxy.initiate_chat(
        group_manager,
        message=f"Generate optimal schedule for {len(employees)} employees. Constraints:\n"
        f"- Minimum coverage: Morning=10, Evening=8, Night=5\n"
        f"- No employee exceeds max_hours\n"
        f"- Prefer employee shift preferences where possible",
    )

    # Extract final proposal
    final_message = problem_solving_group.messages[-1]["content"]

    # Phase 2: Expert Reviews
    if "FINAL_SCHEDULE" in final_message:
        try:
            schedule = json.loads(final_message.replace("FINAL_SCHEDULE", "").strip())

            # Parallel reviews
            cost_report = cost_reviewer.initiate_chat(
                recipient=manager, message=json.dumps(schedule), max_turns=1
            ).last_message()

            fairness_report = fairness_reviewer.initiate_chat(
                recipient=manager, message=json.dumps(schedule), max_turns=1
            ).last_message()

            # Final consolidation
            refined = manager.initiate_chat(
                recipient=manager,
                message=f"Cost Report:\n{cost_report}\n\nFairness Report:\n{fairness_report}",
                max_turns=2,
            ).last_message()

            return json.loads(refined)

        except json.JSONDecodeError:
            print("Failed to parse agent output")

    # Fallback to OR-Tools if agents fail
    print("Agent negotiation failed. Using OR-Tools fallback...")
    return ortools_solve()


# ======================
# 6. Main Execution
# ======================
if __name__ == "__main__":
    # Register tools
    validator.register_function(
        function_map={"validate_schedule": ScheduleValidator.validate}
    )

    # Run optimization
    final_schedule = run_scheduling()

    # Save results
    with open("final_schedule.json", "w") as f:
        json.dump(final_schedule, f, indent=2)

    # Calculate metrics
    total_cost = sum(
        employees[employees["id"] == emp_id]["wage"].values[0] * hours
        for emp_id, hours in final_schedule.items()
    )

    print(f"\n=== Optimization Complete ===")
    print(f"Total payroll cost: ${total_cost:,.2f}")
    print(f"Schedule saved to final_schedule.json")
