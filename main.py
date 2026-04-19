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