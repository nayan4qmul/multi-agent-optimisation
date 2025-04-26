from autogen import AssistantAgent, UserProxyAgent
import random
from faker import Faker
from typing import List, Dict
import json


class LaborSchedulingSystem:
    def __init__(self):
        # Initialize Faker for test data generation
        self.fake = Faker()

        # Load LLM configuration
        self.config_list = [
            {
                "model": "mistral:latest",
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
            },
        ]

        # Initialize agents
        self._initialize_agents()

        # Generate test data
        self.employees = self.generate_employee_data(50)
        self.requirements = self.generate_business_requirements()

    def _initialize_agents(self):
        """Initialize the AutoGen agents"""
        self.scheduler_agent = AssistantAgent(
            name="Scheduler",
            system_message="You are an expert labor scheduler. Create optimized work schedules considering business needs, employee preferences, and labor laws. Return schedules in clear table format.",
            llm_config={"config_list": self.config_list},
        )

        self.reflection_agent = AssistantAgent(
            name="Reflector",
            system_message="You are a scheduling quality analyst. Provide detailed feedback on schedules considering: 1) Coverage of all shifts 2) Fair distribution of hours 3) Compliance with labor laws 4) Employee preferences. Be specific about improvements needed.",
            llm_config={"config_list": self.config_list},
        )

        self.user_proxy = UserProxyAgent(
            name="Admin",
            human_input_mode="NEVER",
            code_execution_config=False,
            default_auto_reply="Continue with the scheduling process.",
        )

        # Register reflection function
        self.user_proxy.register_function(
            function_map={"reflect_on_schedule": self.reflection_message}
        )

    def reflection_message(self, recipient, messages, sender, config):
        """Reflection framework to analyze and improve schedules"""
        last_message = recipient.chat_messages_for_summary(sender)[-1]["content"]
        return f"""Review this schedule and provide detailed feedback:
                1. Check for coverage gaps in all time slots
                2. Verify compliance with labor regulations (max hours, breaks, etc.)
                3. Assess fairness in shift distribution
                4. Identify any conflicts with employee preferences
                5. Suggest specific improvements
                
                Current Schedule:
                \n\n{last_message}"""

    def generate_employee_data(self, num_employees: int = 50) -> List[Dict]:
        """Generate realistic employee data for testing"""
        ROLES = [
            "Cashier",
            "Stock Clerk",
            "Customer Service",
            "Shift Supervisor",
            "Barista",
            "Cook",
            "Server",
            "Janitor",
            "Manager",
        ]
        SKILLS = [
            "Cash Handling",
            "Inventory",
            "Food Prep",
            "Customer Service",
            "POS System",
            "Cleaning",
            "Leadership",
            "Multilingual",
        ]
        WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        WEEKEND = ["Saturday", "Sunday"]

        employees = []

        for _ in range(num_employees):
            available_days = random.sample(WEEKDAYS, random.randint(3, 5))
            if random.random() > 0.7:
                available_days.extend(random.sample(WEEKEND, random.randint(0, 2)))

            morning_shift = f"{random.randint(6, 8)}:00-{random.randint(12, 15)}:00"
            evening_shift = f"{random.randint(12, 15)}:00-{random.randint(18, 22)}:00"

            if random.random() > 0.6:
                availability = f"{', '.join(available_days)} {morning_shift}"
            elif random.random() > 0.3:
                availability = f"{', '.join(available_days)} {evening_shift}"
            else:
                availability = (
                    f"{', '.join(available_days)} {morning_shift} or {evening_shift}"
                )

            employee = {
                "name": self.fake.name(),
                "availability": availability,
                "skills": random.sample(SKILLS, random.randint(1, 4)),
                "max_hours": random.choice([20, 25, 30, 35, 40]),
                "preferences": self._generate_preferences(),
                "role": random.choice(ROLES),
                "employee_id": self.fake.uuid4()[:8],
                "hourly_rate": round(random.uniform(10.0, 25.0), 2),
            }
            employees.append(employee)

        return employees

    def _generate_preferences(self) -> str:
        """Generate realistic employee preferences"""
        preferences = []

        if random.random() > 0.5:
            preferences.append(
                random.choice(["morning shifts", "evening shifts", "weekends off"])
            )

        if random.random() > 0.7:
            preferences.append(f"max {random.randint(3, 5)} consecutive days")

        if random.random() > 0.6:
            preferences.append(
                random.choice(
                    [
                        "prefer cashier station",
                        "avoid cleaning duties",
                        "prefer customer interaction",
                    ]
                )
            )

        return ", ".join(preferences) if preferences else "None"

    def generate_business_requirements(self) -> Dict:
        """Generate sample business requirements"""
        return {
            "operating_hours": "Mon-Fri 8:00-22:00, Sat-Sun 9:00-20:00",
            "shifts": [
                {
                    "time": "Weekdays 8:00-12:00",
                    "min_staff": 3,
                    "required_skills": ["Cash Handling"],
                },
                {
                    "time": "Weekdays 12:00-16:00",
                    "min_staff": 4,
                    "required_skills": ["Customer Service"],
                },
                {
                    "time": "Weekdays 16:00-20:00",
                    "min_staff": 5,
                    "required_skills": ["Cash Handling", "Customer Service"],
                },
                {
                    "time": "Weekdays 20:00-22:00",
                    "min_staff": 2,
                    "required_skills": ["Cleaning"],
                },
                {
                    "time": "Weekends 9:00-13:00",
                    "min_staff": 3,
                    "required_skills": ["Cash Handling"],
                },
                {
                    "time": "Weekends 13:00-17:00",
                    "min_staff": 4,
                    "required_skills": ["Customer Service"],
                },
                {
                    "time": "Weekends 17:00-20:00",
                    "min_staff": 3,
                    "required_skills": ["Cash Handling"],
                },
            ],
            "budget": "1000 staff-hours/week",
            "constraints": [
                "No employee should work more than 5 consecutive days",
                "Minimum 12 hours between shifts",
                "30-minute break required for shifts >5 hours",
            ],
        }

    def _build_scheduling_prompt(self) -> str:
        """Construct the initial scheduling prompt"""
        shifts = "\n".join(
            [
                f"- {s['time']}: {s['min_staff']} staff needed (Skills: {', '.join(s['required_skills'])})"
                for s in self.requirements["shifts"]
            ]
        )

        employee_info = "\n".join(
            [
                f"{e['name']} ({e['role']}): Available {e['availability']}, Skills: {', '.join(e['skills'])}, "
                f"Max hours/week: {e['max_hours']}, Preferences: {e['preferences']}"
                for e in self.employees[
                    :10
                ]  # Show sample of employees to avoid huge prompt
            ]
        )

        return f"""Create an optimal labor schedule with these requirements:
                
                BUSINESS NEEDS:
                Operating Hours: {self.requirements['operating_hours']}
                Shifts:
                {shifts}
                Total weekly labor budget: {self.requirements['budget']}
                Constraints: {'; '.join(self.requirements['constraints'])}
                
                EMPLOYEES (showing 10 of {len(self.employees)}):
                {employee_info}
                
                FORMATTING REQUIREMENTS:
                - Provide the schedule in a clear table format with columns: Day, Shift, Employee, Role, Hours
                - Include a summary section with total hours per employee
                - Highlight any constraint violations
                """

    def has_conflicts(self, schedule_text: str) -> bool:
        """Analyze schedule text for conflicts and violations"""
        conflict_checks = [
            "violation",
            "conflict",
            "missing coverage",
            "understaffed",
            "overworked",
            "constraint not met",
            "break violation",
        ]

        # Convert to lowercase for case-insensitive check
        schedule_lower = schedule_text.lower()

        # Check if any conflict terms appear in the schedule
        return any(keyword in schedule_lower for keyword in conflict_checks)

    def optimize_schedule(self, max_iterations: int = 3) -> str:
        """
        Generate and refine a labor schedule through iterative optimization
        Returns optimized schedule in table format
        """
        # Initial schedule generation
        prompt = self._build_scheduling_prompt()
        self.user_proxy.initiate_chat(self.scheduler_agent, message=prompt)
        current_schedule = self.scheduler_agent.last_message()["content"]

        if not self.has_conflicts(current_schedule):
            return current_schedule

        # Reflection and improvement loop
        for _ in range(max_iterations):  # Two refinement iterations
            print(f"\n=== Refinement Iteration {iteration + 1}/{max_iterations} ===")
            self.user_proxy.initiate_chat(
                self.reflection_agent,
                message=self.reflection_message,
                recipient=self.scheduler_agent,
            )

            # Incorporate feedback
            self.user_proxy.initiate_chat(
                self.scheduler_agent,
                message="Please revise the schedule based on this feedback: "
                + self.reflection_agent.last_message()["content"],
            )

            current_schedule = self.scheduler_agent.last_message()["content"]

            # Early termination check
            if not self.has_conflicts(current_schedule):
                print(
                    f"\nEarly termination at iteration {iteration + 1} - conflict-free solution found"
                )
                break

        return current_schedule

    def save_test_data(self, filename: str = "scheduling_test_data.json"):
        """Save the generated test data to a file"""
        data = {"employees": self.employees, "requirements": self.requirements}
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Test data saved to {filename}")

    def save_schedule(self, schedule: str, filename: str = "optimized_schedule.txt"):
        """Save the optimized schedule to a text file"""
        with open(filename, "w") as f:
            f.write(schedule)
        print(f"\nOptimized schedule saved to {filename}")

    def run(self):
        """Execute the complete scheduling process"""
        print("Generating test data for 50 employees...")
        print(
            f"Sample employee: {self.employees[0]['name']} ({self.employees[0]['role']})"
        )

        print("\nGenerating optimized schedule...")
        optimized_schedule = self.optimize_schedule()

        print("\nFINAL OPTIMIZED SCHEDULE:")
        print(optimized_schedule)

        self.save_test_data()
        self.save_schedule(optimized_schedule)


if __name__ == "__main__":
    # Run the complete system
    system = LaborSchedulingSystem()
    system.run()
