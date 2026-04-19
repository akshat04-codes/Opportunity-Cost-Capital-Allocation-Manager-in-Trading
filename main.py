import yaml
import numpy as np
from agents.baseline import run_baseline_evaluation
from env.env import TradingAllocationEnv


def load_config():
    # Load optional config file
    try:
        with open("openenv.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        print("Config:", cfg.get("name", "N/A"))
    except:
        print("Config not found")


def parse(x, n):
    # Convert user input into normalized allocation vector
    try:
        arr = np.array(list(map(float, x.split())))
        if len(arr) != n:
            print("Wrong size")
            return None
        return arr / arr.sum() if arr.sum() > 0 else arr
    except:
        print("Invalid input")
        return None


def get_initial_capital():
    # Take initial capital from user
    try:
        cap = float(input("Enter initial capital: "))
        if cap <= 0:
            print("Capital must be positive. Using default 100000.")
            return 100000
        return cap
    except:
        print("Invalid input. Using default 100000.")
        return 100000


def main():
    print("Commands: run | step | config | exit")

    # User-defined initial capital
    initial_capital = get_initial_capital()

    # Initialize environment with user capital
    env = TradingAllocationEnv(initial_capital=initial_capital, seed=42)
    state = env.reset()

    load_config()

    while True:
        cmd = input("\n> ").strip().lower()

        # Exit
        if cmd == "exit":
            break

        # Run baseline evaluation
        elif cmd == "run":
            run_baseline_evaluation(initial_capital, seed=42)

        # Show config
        elif cmd == "config":
            load_config()

        # Manual trading step
        elif cmd == "step":

            # state[0] = capital ratio → convert to actual capital
            print("Capital:", state[0] * env.initial_capital)

            x = input("Alloc: ")

            action = parse(x, len(env.opportunities))
            if action is None:
                continue

            # simulate step
            state, reward, done, _ = env.step(action)

            print("Reward:", reward)
            print("Capital:", state[0] * env.initial_capital)

            # reset if episode ends
            if done:
                print("Episode finished. Resetting...\n")
                state = env.reset()

        else:
            print("Invalid command")


if __name__ == "__main__":
    main()
