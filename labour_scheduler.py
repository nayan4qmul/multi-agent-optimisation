from autogen import AssistantAgent
from ortools.sat.python import cp_model
import random
from datetime import time
import json

config_list = [
    {
        "model": "mistral:latest",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
]

llm_config = {"config_list": config_list, "temperature": 0.0}

task = """
Create an optimized 1-week labor schedule for a medium-sized retail store with:
- Staff: 15 employees (5 full-time, 10 part-time)
- Store hours: 9AM-9PM weekdays, 8AM-10PM weekends
- Peak times: 11AM-2PM and 5PM-8PM daily
- Special requirements: 
  - Must have 2 managers on duty at all times
  - At least 4 staff during peak, 2 during off-peak
  - No employee should work more than 5 consecutive days
  - Part-timers max 30 hrs/week
  - Full-timers 38-40 hrs/week
- Budget: Minimize overtime while meeting coverage needs
"""


def generate_employee_data():
    employees = []
    for i in range(5):
        is_manager = i < 2
        employees.append(
            {
                "id": f"FT{i+1}",
                "name": f"Full-Time {'Manager' if is_manager else 'Staff'} {i+1}",
                "type": "full-time",
                "role": "manager" if is_manager else "staff",
                "min_hours": 38,
                "max_hours": 40,
                "availability": generate_availability("full-time"),
                "preferred_shifts": generate_preferred_shifts(),
                "skills": ["management"] if is_manager else ["cashier", "stocking"],
            }
        )

    for i in range(10):
        employees.append(
            {
                "id": f"PT{i+1}",
                "name": f"Part-Time Staff {i+1}",
                "type": "part-time",
                "role": "staff",
                "min_hours": 10,
                "max_hours": 30,
                "availability": generate_availability("part-time"),
                "preferred_shifts": generate_preferred_shifts(),
                "skills": ["cashier", "customer_service"],
            }
        )

    return employees


def generate_availability(emp_type):
    availability = {}
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    for day in days:
        if emp_type == "full-time":
            if random.random() < 0.8:
                if day in ["Saturday", "Sunday"]:
                    availability[day] = (
                        ("08:00", "22:00") if random.random() < 0.7 else None
                    )
                else:
                    availability[day] = (
                        ("09:00", "21:00") if random.random() < 0.9 else None
                    )
            else:
                availability[day] = None
        else:
            if random.random() < 0.6:
                if day in ["Saturday", "Sunday"]:
                    availability[day] = (
                        ("08:00", "22:00") if random.random() < 0.5 else None
                    )
                else:
                    availability[day] = (
                        ("09:00", "21:00") if random.random() < 0.7 else None
                    )
            else:
                availability[day] = None

    return availability


def generate_preferred_shifts():
    preferences = []
    shift_types = ["morning", "afternoon", "evening"]
    preferred = random.sample(shift_types, random.randint(1, 2))
    return preferred


def ortools_scheduler(employees, requirements):
    model = cp_model.CpModel()

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    shifts = {
        "morning": (time(9, 0), time(14, 0)),
        "afternoon": (time(14, 0), time(19, 0)),
        "evening": (time(19, 0), time(22, 0)),
    }
    shift_durations = {
        "morning": 5 * 60,
        "afternoon": 5 * 60,
        "evening": 3 * 60,
    }

    shifts_dict = {}
    for emp in employees:
        for day in days:
            for shift_name in shifts:
                available = False
                if emp["availability"][day] is not None:
                    avail_start, avail_end = emp["availability"][day]
                    avail_start = time(*map(int, avail_start.split(":")))
                    avail_end = time(*map(int, avail_end.split(":")))
                    shift_start, shift_end = shifts[shift_name]

                    if avail_start <= shift_start and avail_end >= shift_end:
                        available = True

                var_name = f"{emp['id']}_{day}_{shift_name}"
                shifts_dict[(emp["id"], day, shift_name)] = model.NewBoolVar(var_name)

                if not available:
                    model.Add(shifts_dict[(emp["id"], day, shift_name)] == 0)

    for day in days:
        for shift_name in shifts:
            min_staff = 4 if shift_name in ["morning", "afternoon"] else 2
            if day in ["Saturday", "Sunday"]:
                min_staff += 1

            managers_on_shift = [
                shifts_dict[(emp["id"], day, shift_name)]
                for emp in employees
                if emp["role"] == "manager"
            ]
            model.Add(sum(managers_on_shift) >= 2)

            staff_on_shift = [
                shifts_dict[(emp["id"], day, shift_name)] for emp in employees
            ]
            model.Add(sum(staff_on_shift) >= min_staff)

    for emp in employees:
        total_minutes = 0
        for day in days:
            for shift_name in shifts:
                total_minutes += (
                    shifts_dict[(emp["id"], day, shift_name)]
                    * shift_durations[shift_name]
                )

        min_minutes = emp["min_hours"] * 60
        max_minutes = emp["max_hours"] * 60
        model.Add(total_minutes >= min_minutes)
        model.Add(total_minutes <= max_minutes)

        for start_day in range(len(days) - 5):
            consecutive_days = []
            for i in range(6):
                day = days[start_day + i]
                worked = []
                for shift_name in shifts:
                    worked.append(shifts_dict[(emp["id"], day, shift_name)])
                day_worked = model.NewBoolVar(f"{emp['id']}_consec_{start_day}_{i}")
                model.AddMaxEquality(day_worked, worked)
                consecutive_days.append(day_worked)
            model.Add(sum(consecutive_days) <= 5)

    preference_score = 0
    for emp in employees:
        for day in days:
            for shift_name in shifts:
                if shift_name in emp["preferred_shifts"]:
                    preference_score += shifts_dict[(emp["id"], day, shift_name)]

    model.Maximize(preference_score)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        schedule = {day: {shift: [] for shift in shifts} for day in days}
        employee_hours = {emp["id"]: 0 for emp in employees}

        for emp in employees:
            for day in days:
                for shift_name in shifts:
                    if solver.Value(shifts_dict[(emp["id"], day, shift_name)]) == 1:
                        schedule[day][shift_name].append(emp["id"])
                        employee_hours[emp["id"]] += shift_durations[shift_name] / 60

        return {
            "status": "optimal" if status == cp_model.OPTIMAL else "feasible",
            "schedule": schedule,
            "employee_hours": employee_hours,
            "preference_score": solver.ObjectiveValue(),
            "solver_stats": {
                "conflicts": solver.NumConflicts(),
                "branches": solver.NumBranches(),
                "wall_time": solver.WallTime(),
            },
        }
    else:
        return {"status": "infeasible", "message": "No solution found"}


scheduler = AssistantAgent(
    name="Scheduler",
    system_message="""
    You are an expert retail workforce scheduling AI that creates optimized labor schedules.
    
    Your responsibilities:
    1. Create weekly schedules that meet all business requirements
    2. Balance employee availability, labor laws, and business needs
    3. Ensure adequate coverage for all shifts, especially peak periods
    4. Distribute shifts fairly among employees
    5. Minimize overtime costs while meeting operational needs
    6. Accommodate special employee requests when possible
    
    Input requirements:
    - Store operating hours
    - Employee roster with availability and contract types
    - Peak/off-peak period definitions
    - Minimum staffing requirements
    - Any special constraints (e.g., manager coverage)
    
    Output format:
    1. Weekly overview showing total hours and coverage
    2. Day-by-day schedule with shift times and roles
    3. Employee summary showing hours distribution
    4. Coverage analysis highlighting any gaps
    5. Budget impact summary
    
    Always verify:
    - No labor law violations
    - All business requirements met
    - Fair distribution of shifts
    - Optimal use of labor budget
    """,
    llm_config=llm_config,
)

personalization_agent = AssistantAgent(
    name="Staffing-Profile-Agent",
    llm_config=llm_config,
    system_message="""
    You ensure the schedule matches the specific staffing profile and needs.
    
    Required profile information (request if missing):
    - Employee types (full-time/part-time/casual)
    - Individual availability constraints
    - Skill sets/qualifications (cashier, manager, etc.)
    - Preferred hours if any
    - Contractual limitations
    
    Your checks:
    1. Verify all shifts are covered by qualified staff
    2. Check compliance with individual contracts
    3. Flag any availability conflicts
    4. Ensure fair distribution of desirable/undesirable shifts
    5. Validate against union rules if applicable
    
    Output format:
    Staffing Profile Check:
    Met requirements: [list]
    Needs adjustment: [specific suggestions]
    Missing elements: [request for information]
    """,
)

labor_law_compliance = AssistantAgent(
    name="Labor-Law-Expert",
    llm_config=llm_config,
    system_message="""
    You specialize in labor law compliance for retail scheduling.
    
    Check for:
    1. Break requirements (minimum rest periods between shifts)
    2. Maximum daily/weekly hours
    3. Overtime rules
    4. Weekend/holiday pay requirements
    5. Minor employment laws if applicable
    6. Predictive scheduling laws
    
    Reference typical regulations (specify jurisdiction if known):
    - Minimum 11 hours between shifts
    - Maximum 48-hour work week
    - 20-minute breaks every 6 hours
    
    Flag any violations with specific citations.
    Begin with 'Compliance Check:'.
    """,
)

coverage_analyzer = AssistantAgent(
    name="Coverage-Analyzer",
    llm_config=llm_config,
    system_message="""
    You analyze whether the schedule meets operational coverage needs.
    
    Evaluate:
    1. Adequate staffing during peak periods
    2. Minimum coverage during all open hours
    3. Appropriate skill mix on each shift
    4. Overstaffing risks
    5. Ability to handle unexpected absences
    
    Provide coverage heatmap showing:
    - Strong/weak coverage periods
    - Risk areas
    - Suggested adjustments
    
    Begin with 'Coverage Analysis:'.
    """,
)

budget_optimizer = AssistantAgent(
    name="Budget-Optimizer",
    llm_config=llm_config,
    system_message="""
    You optimize the labor budget while maintaining service levels.
    
    Analyze:
    1. Total labor cost projection
    2. Overtime hours/costs
    3. Premium pay situations
    4. Underutilized staff
    5. Opportunities for shift adjustments to save costs
    
    Provide:
    - Cost breakdown by employee type
    - Comparison to labor budget
    - Savings opportunities
    - Risk/cost tradeoff analysis
    
    Begin with 'Budget Analysis:'.
    """,
)

employee_fairness = AssistantAgent(
    name="Fairness-Reviewer",
    llm_config=llm_config,
    system_message="""
    You ensure fair and equitable shift distribution.
    
    Check:
    1. Balance of desirable vs. undesirable shifts
    2. Weekend/holiday rotation
    3. Equal opportunity for overtime
    4. Consistency with employee preferences
    5. No apparent favoritism
    
    Provide fairness metrics:
    - Shift distribution by employee
    - Weekend/holiday counts
    - Prime vs. off-peak shifts
    
    Begin with 'Fairness Review:'.
    """,
)

scenario_planner = AssistantAgent(
    name="Scenario-Planner",
    llm_config=llm_config,
    system_message="""
    You generate alternative scenarios for comparison.
    
    Create variations considering:
    1. Different staffing mixes
    2. Varying peak coverage levels
    3. Alternative shift patterns
    4. What-if scenarios (absences, surges)
    
    Present 2-3 alternatives with:
    - Coverage differences
    - Cost impacts
    - Operational implications
    
    Begin with 'Scenario Options:'.
    """,
)

visual_scheduler = AssistantAgent(
    name="Visual-Scheduler",
    llm_config=llm_config,
    system_message="""
    You transform schedules into visual formats.
    
    Output options:
    1. **Weekly Gantt Chart**:
       - Color-coded by employee type/role
       - Visual coverage heatmap
    
    2. **Daily Coverage View**:
       Hour-by-hour staffing levels
    
    3. **Employee Calendar**:
       Individual view of assigned shifts
    
    4. **Budget Pie Chart**:
       Cost distribution by category
    
    Always include:
    - Clear time markers
    - Color coding for roles/statuses
    - Highlighted critical periods
    - Summary statistics
    
    Output in markdown format for easy rendering.
    """,
)

feedback_prioritizer = AssistantAgent(
    name="Feedback-Prioritizer",
    llm_config=llm_config,
    system_message="""
    You resolve conflicts between optimization priorities.
    
    Prioritization rules:
    1. Legal compliance > All other factors
    2. Minimum coverage > Cost savings
    3. Employee fairness > Perfect optimization
    4. Business needs > Individual preferences
    
    Output:
    MUST FIX: [Critical issues]
    SHOULD ADDRESS: [Important improvements]
    NICE TO HAVE: [Optional optimizations]
    TRADEOFF DECISIONS: [Your resolution of conflicts]
    """,
)


def generate_optimized_schedule(task):
    employees = generate_employee_data()
    requirements = {
        "min_managers": 2,
        "min_staff_peak": 4,
        "min_staff_offpeak": 2,
        "max_consecutive_days": 5,
        "store_hours": {"weekdays": ("09:00", "21:00"), "weekends": ("08:00", "22:00")},
    }

    result = {"method": "failed", "error": "Initialization failed"}

    try:
        # Try agent-based approach first
        draft = scheduler.initiate_chat(
            recipient=personalization_agent, message=task, max_turns=1
        )

        reviewed = scheduler.initiate_chat(
            recipient=feedback_prioritizer, message=draft.summary, max_turns=2
        )

        if "❌" in reviewed.summary or "MUST FIX" in reviewed.summary:
            raise ValueError("Agent-generated schedule failed validation")

        visual = visual_scheduler.initiate_chat(
            recipient=scheduler, message=reviewed.summary, max_turns=1
        )

        result = {
            "method": "agent-based",
            "schedule": reviewed.summary,
            "visual": visual.summary,
            "analysis": {
                "compliance": labor_law_compliance.last_message(),
                "coverage": coverage_analyzer.last_message(),
                "budget": budget_optimizer.last_message(),
                "fairness": employee_fairness.last_message(),
            },
        }

    except Exception as e:
        print(f"Agent-based scheduling failed: {str(e)}")
        print("Attempting OR-Tools optimization...")

        try:
            ortools_result = ortools_scheduler(employees, requirements)

            if ortools_result["status"] in ["optimal", "feasible"]:
                schedule_summary = "=== OR-TOOLS GENERATED SCHEDULE ===\n"
                for day in ortools_result["schedule"]:
                    schedule_summary += f"\n{day}:\n"
                    for shift, staff in ortools_result["schedule"][day].items():
                        schedule_summary += f"  {shift}: {', '.join(staff)}\n"

                employee_summary = "\n=== EMPLOYEE HOURS ===\n"
                for emp_id, hours in ortools_result["employee_hours"].items():
                    employee_summary += f"{emp_id}: {hours:.1f} hours\n"

                full_summary = schedule_summary + employee_summary

                reviewed = scheduler.initiate_chat(
                    recipient=feedback_prioritizer, message=full_summary, max_turns=2
                )

                visual = visual_scheduler.initiate_chat(
                    recipient=scheduler, message=full_summary, max_turns=1
                )

                result = {
                    "method": "or-tools",
                    "schedule": full_summary,
                    "visual": visual.summary,
                    "analysis": {
                        "compliance": labor_law_compliance.last_message(),
                        "coverage": coverage_analyzer.last_message(),
                        "budget": budget_optimizer.last_message(),
                        "fairness": employee_fairness.last_message(),
                    },
                    "ortools_stats": ortools_result.get("solver_stats", {}),
                }
            else:
                result = {
                    "method": "failed",
                    "error": "OR-Tools could not find a feasible solution",
                    "ortools_result": ortools_result,
                    "suggestions": [
                        "Relax some constraints (e.g., reduce minimum staff requirements)",
                        "Increase employee availability",
                        "Hire additional part-time staff",
                        "Adjust store hours during low-traffic periods",
                    ],
                }

        except Exception as e:
            result = {
                "method": "failed",
                "error": f"Both methods failed: {str(e)}",
                "suggestions": [
                    "Check if constraints are too restrictive",
                    "Verify employee availability data",
                    "Consider adding more flexible part-time staff",
                    "Review minimum staffing requirements",
                ],
            }

    return result


def reflection_message(recipient, messages, sender, config):
    return f"""Review this schedule and provide detailed feedback:
            \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"""


review_chats = [
    {
        "recipient": personalization_agent,
        "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "max_turns": 1,
    },
    {
        "recipient": labor_law_compliance,
        "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "max_turns": 1,
    },
    {
        "recipient": coverage_analyzer,
        "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "max_turns": 1,
    },
    {
        "recipient": budget_optimizer,
        "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "max_turns": 1,
    },
    {
        "recipient": employee_fairness,
        "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "max_turns": 1,
    },
    {
        "recipient": scenario_planner,
        "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "max_turns": 1,
    },
    {
        "recipient": feedback_prioritizer,
        "message": "Organize and prioritize all feedback",
        "max_turns": 1,
    },
]

result = generate_optimized_schedule(task)

with open("optimized_schedule_result.json", "w") as f:
    json.dump(result, f, indent=2)

print("=== SCHEDULE GENERATION RESULT ===")
print(f"Method used: {result.get('method', 'N/A').upper()}")

if result["method"] == "failed":
    print("\n=== SCHEDULING FAILED ===")
    print(f"Error: {result.get('error', 'Unknown error')}")
    print("\nSuggested actions:")
    for suggestion in result.get("suggestions", []):
        print(f"- {suggestion}")
else:
    print("\n=== OPTIMIZED SCHEDULE ===")
    print(result.get("schedule", "No schedule generated"))

    print("\n=== VISUAL REPRESENTATION ===")
    print(result.get("visual", "No visual available"))

    print("\n=== ANALYSIS REPORTS ===")
    for category, report in result.get("analysis", {}).items():
        print(f"\n{category.upper()}:\n{report}")

    if "ortools_stats" in result:
        print("\n=== OR-TOOLS SOLVER STATS ===")
        for stat, value in result["ortools_stats"].items():
            print(f"{stat.replace('_', ' ').title()}: {value}")

print("\n=== ADDITIONAL DIAGNOSTICS ===")
print("Employee count:", len(generate_employee_data()))
print(
    "Manager count:",
    len([e for e in generate_employee_data() if e["role"] == "manager"]),
)
