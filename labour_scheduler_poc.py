import json
import pandas as pd
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from ortools.sat.python import cp_model

employees = pd.read_csv("employees.csv")
shifts = ["Morning", "Evening", "Night"]
min_coverage = {"Morning": 10, "Evening": 8, "Night": 5}

llm_config = {
    "config_list": [
        {
            "model": "llama3.2:3b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        }
    ],
    "temperature": 0.1,
}

manager = AssistantAgent(
    name="Manager",
    system_message="""
    Coordinate shift scheduling. Finalize the schedule by summarizing in JSON format like: 
    {"EMP_001": 20, "EMP_002": 30, ...}. If consensus fails after 10 rounds, say 'FALLBACK'.
    """,
    llm_config=llm_config,
)

proposer = AssistantAgent(
    name="Proposer",
    system_message=f"""
    Propose schedules in JSON format.
    Hard constraints: 
    1) No employee exceeds max_hours. 
    2) Coverage >= {min_coverage}.
    Soft preferences: 
    1) Honor 'prefers_evenings' where possible.

    After each proposal, reflect:
    1. Did I cover all shifts adequately?
    2. Did any employee exceed max_hours?
    3. Are evening preferences honored?
    4. Is total cost minimized?
    """,
    llm_config=llm_config,
)

validator = AssistantAgent(
    name="Validator",
    system_message="""
    Validate the proposed schedule. Reply ONLY with:
    {"valid": true/false, "errors": []}
    """,
    llm_config=llm_config,
)

refiner = AssistantAgent(
    name="Refiner",
    system_message="""
    Adjust schedules to minimize cost. Use employee wages to minimize total cost.
    Return refined schedule in JSON.
    """,
    llm_config=llm_config,
)

critic = AssistantAgent(
    name="Critic",
    system_message="""
    Flag fairness issues. Check:
    1. Are high-wage employees overworked?
    2. Are preferences ignored?
    Reply with {'needs_refinement': true/false, 'issues': []}
    """,
    llm_config=llm_config,
)

groupchat = GroupChat(
    agents=[manager, proposer, validator, refiner, critic],
    messages=[],
    max_round=10,
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)


def ortools_fallback():
    model = cp_model.CpModel()
    emp_vars = {
        emp["id"]: model.NewIntVar(0, emp["max_hours"], emp["id"])
        for _, emp in employees.iterrows()
    }

    for shift in shifts:
        model.Add(
            sum(emp_vars[emp["id"]] for _, emp in employees.iterrows())
            >= min_coverage[shift]
        )

    model.Minimize(
        sum(emp["wage"] * emp_vars[emp["id"]] for _, emp in employees.iterrows())
    )

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    return {
        emp["id"]: solver.Value(emp_vars[emp["id"]]) for emp in employees.iterrows()
    }


def run_scheduler():
    user_proxy = UserProxyAgent(
        name="Admin", human_input_mode="NEVER", code_execution_config=False
    )
    user_proxy.initiate_chat(
        manager,
        message=f"""Schedule {len(employees)} employees with:
        - Roles: {employees['role'].unique()}
        - Shifts: {shifts}
        - Min coverage: {min_coverage}
        - Preferences: {sum(employees['prefers_evenings'])} prefer evenings""",
    )

    final_message = groupchat.messages[-1]["content"]
    if "FALLBACK" in final_message:
        print("Agent consensus failed. Using OR-Tools fallback...")
        schedule = ortools_fallback()
    else:
        try:
            schedule = json.loads(final_message)
        except json.JSONDecodeError:
            print("Failed to parse agent output. Using OR-Tools...")
            schedule = ortools_fallback()

    with open("schedule.json", "w") as f:
        json.dump(schedule, f, indent=2)


if __name__ == "__main__":
    run_scheduler()
