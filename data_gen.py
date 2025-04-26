import pandas as pd
import random

def generate_employees(n=100):
    data = []
    for i in range(n):
        role = random.choice(["Engineer", "Manager", "Admin"])
        wage = random.randint(15, 40) if role != "Manager" else random.randint(50, 100)
        max_hours = random.choice([20, 30, 40])
        prefers_evenings = random.random() > 0.7
        data.append({
            "id": f"EMP_{i:03d}",
            "role": role,
            "wage": wage,
            "max_hours": max_hours,
            "prefers_evenings": prefers_evenings
        })
    return pd.DataFrame(data)

employees = generate_employees(100)
employees.to_csv("employees.csv", index=False)
print(employees)