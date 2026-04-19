import yaml
from agents.baseline import run_baseline_evaluation

if __name__ == "__main__":
    # Run baseline evaluation
    run_baseline_evaluation(
        initial_capital=100_000,
        seed=42,
    )

    # Load config
    try:
        with open("openenv.yaml", "r") as f:
            config = yaml.safe_load(f)
        print(config.get("name", "No name found in config"))
    except FileNotFoundError:
        print("openenv.yaml not found")

#inputs
while True:
    cmd = input("Enter command (step/run/exit): ")

    if cmd == "step":
        action = input("Enter allocation vector: ")
        # parse and run step()

    elif cmd == "run":
        # run full simulation

    elif cmd == "exit":
        break
