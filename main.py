import random
import time

import matplotlib.pyplot as plt

# ============================================================
# 1. ARTIFICIAL BEE COLONY ALGORITHM (ABC)
# ============================================================


def abc_knapsack(items, capacity, bees=20, iterations=50):
    """
    Solve knapsack using Artificial Bee Colony algorithm.
    Returns: (best_solution, best_value)
    """
    n = len(items)

    # Initialize random solutions (binary: 0=not included, 1=included)
    solutions = [[random.randint(0, 1) for _ in range(n)] for _ in range(bees)]

    # Track best solution
    best_solution = None
    best_value = 0
    convergence_history = []  # Track convergence over iterations

    print("\n" + "=" * 60)
    print("ARTIFICIAL BEE COLONY (ABC) - Metaheuristic Algorithm")
    print("=" * 60)
    print(f"Items: {n}, Capacity: {capacity}, Bees: {bees}, Iterations: {iterations}")

    start = time.perf_counter()  # More precise timing

    for iteration in range(iterations):
        # Evaluate all solutions
        for i in range(bees):
            # Fix solution if overweight
            solutions[i] = fix_overweight(solutions[i], items, capacity)

            # Calculate value
            value = calculate_value(solutions[i], items, capacity)

            # Update best
            if value > best_value:
                best_value = value
                best_solution = solutions[i].copy()

        convergence_history.append(best_value)  # Record best value at each iteration

        # Generate new solutions by modifying existing ones
        for i in range(bees):
            new_sol = solutions[i].copy()

            # Flip 1-2 random bits
            for _ in range(random.randint(1, 2)):
                pos = random.randint(0, n - 1)
                new_sol[pos] = 1 - new_sol[pos]

            # Fix if needed
            new_sol = fix_overweight(new_sol, items, capacity)

            # Keep new solution if better
            if calculate_value(new_sol, items, capacity) > calculate_value(
                solutions[i], items, capacity
            ):
                solutions[i] = new_sol

        if (iteration + 1) % 20 == 0:
            print(f"Iteration {iteration + 1}: Best value = {best_value}")

    elapsed = time.perf_counter() - start

    print(f"\nCompleted in {elapsed:.6f} seconds")
    print(f"Best value: {best_value}")
    print_solution(best_solution, items, capacity)

    return best_solution, best_value, elapsed, convergence_history


def fix_overweight(solution, items, capacity):
    """Remove items until solution is valid"""
    sol = solution.copy()
    weight = sum(items[i][0] * sol[i] for i in range(len(items)))

    while weight > capacity:
        # Remove a random selected item
        selected = [i for i in range(len(items)) if sol[i] == 1]
        if not selected:
            break
        remove = random.choice(selected)
        sol[remove] = 0
        weight -= items[remove][0]

    return sol


def calculate_value(solution, items, capacity):
    """Calculate total value if valid, 0 if overweight"""
    weight = sum(items[i][0] * solution[i] for i in range(len(items)))
    value = sum(items[i][1] * solution[i] for i in range(len(items)))
    return value if weight <= capacity else 0


# ============================================================
# 2. BACKTRACKING ALGORITHM (Exact Solution)
# ============================================================


def backtracking_knapsack(items, capacity):
    """
    Solve knapsack using backtracking with branch & bound.
    Returns: (best_solution, best_value)
    """
    n = len(items)
    best = {"solution": [0] * n, "value": 0, "nodes": 0}

    print("\n" + "=" * 60)
    print("BACKTRACKING - Exact Algorithm (Optimal)")
    print("=" * 60)
    print(f"Items: {n}, Capacity: {capacity}")

    start = time.perf_counter()  # More precise timing

    def backtrack(i, weight, value, solution):
        """Recursive backtracking"""
        best["nodes"] += 1

        # Base case: all items considered
        if i == n:
            if value > best["value"]:
                best["value"] = value
                best["solution"] = solution.copy()
            return

        # Prune: check if we can possibly beat current best
        remaining_value = sum(items[j][1] for j in range(i, n))
        if value + remaining_value <= best["value"]:
            return  # Prune this branch

        # Try including item i
        if weight + items[i][0] <= capacity:
            solution[i] = 1
            backtrack(i + 1, weight + items[i][0], value + items[i][1], solution)
            solution[i] = 0

        # Try excluding item i
        backtrack(i + 1, weight, value, solution)

    backtrack(0, 0, 0, [0] * n)

    elapsed = time.perf_counter() - start

    print(f"\nCompleted in {elapsed:.6f} seconds")
    print(f"Nodes explored: {best['nodes']}")
    print(f"Optimal value: {best['value']}")
    print_solution(best["solution"], items, capacity)

    return best["solution"], best["value"], elapsed


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def print_solution(solution, items, capacity):
    """Print the selected items"""
    weight = sum(items[i][0] * solution[i] for i in range(len(items)))
    print(f"Weight used: {weight}/{capacity}")
    print("Selected items:")
    for i, selected in enumerate(solution):
        if selected:
            print(f"  Item {i+1}: weight={items[i][0]}, value={items[i][1]}")


def plot_analysis(abc_val, abc_time, bt_val, bt_time, convergence_history):
    """Create visualization of results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Knapsack Problem: ABC vs Backtracking Analysis", fontsize=16, fontweight="bold"
    )

    # 1. Time Comparison
    ax1 = axes[0, 0]
    algorithms = ["ABC\n(Metaheuristic)", "Backtracking\n(Exact)"]
    times = [abc_time, bt_time]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax1.bar(
        algorithms, times, color=colors, alpha=0.7, edgecolor="black", linewidth=2
    )
    ax1.set_ylabel("Time (seconds)", fontweight="bold", fontsize=11)
    ax1.set_title("Execution Time Comparison", fontweight="bold", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{time_val:.4f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Add speedup annotation
    speedup = bt_time / abc_time if abc_time > 0 else 0
    ax1.text(
        0.5,
        max(times) * 0.5,
        f"ABC is {speedup:.1f}× faster!",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
    )

    # 2. Value Comparison
    ax2 = axes[0, 1]
    algorithms = ["ABC", "Backtracking\n(Optimal)"]
    values = [abc_val, bt_val]
    colors = ["#3498db", "#9b59b6"]
    bars = ax2.bar(
        algorithms, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2
    )
    ax2.set_ylabel("Total Value", fontweight="bold", fontsize=11)
    ax2.set_title("Solution Quality Comparison", fontweight="bold", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Add accuracy annotation
    accuracy = (abc_val / bt_val * 100) if bt_val > 0 else 0
    gap = abs(bt_val - abc_val)
    ax2.text(
        0.5,
        max(values) * 0.5,
        f"Accuracy: {accuracy:.1f}%\nGap: {gap}",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    # 3. ABC Convergence History
    ax3 = axes[1, 0]
    iterations = list(range(1, len(convergence_history) + 1))
    ax3.plot(
        iterations,
        convergence_history,
        color="#e67e22",
        linewidth=2.5,
        marker="o",
        markersize=4,
        markevery=5,
        label="Best Value",
    )
    ax3.axhline(
        y=bt_val, color="#9b59b6", linestyle="--", linewidth=2, label="Optimal Value"
    )
    ax3.set_xlabel("Iteration", fontweight="bold", fontsize=11)
    ax3.set_ylabel("Best Value Found", fontweight="bold", fontsize=11)
    ax3.set_title("ABC Convergence Over Time", fontweight="bold", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right", fontsize=10)
    ax3.fill_between(iterations, convergence_history, alpha=0.3, color="#e67e22")

    # 4. Performance Metrics Summary
    ax4 = axes[1, 1]
    ax4.axis("off")

    metrics_text = f"""
    PERFORMANCE METRICS
    {'='*40}
    
    Artificial Bee Colony (ABC):
       • Time: {abc_time:.6f} seconds
       • Value: {abc_val}
       • Type: Metaheuristic (Approximate)
    
    Backtracking (Branch & Bound):
       • Time: {bt_time:.6f} seconds
       • Value: {bt_val} (OPTIMAL)
       • Type: Exact Algorithm
    
    Comparison:
       • Speed Ratio: {speedup:.2f}x faster (ABC)
       • Accuracy: {accuracy:.2f}%
       • Value Gap: {gap}
    
    Conclusion:
       ABC found {accuracy:.1f}% optimal solution
       in just {(abc_time/bt_time*100):.2f}% of the time!
    """

    ax4.text(
        0.1,
        0.5,
        metrics_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig("knapsack_analysis.png", dpi=300, bbox_inches="tight")
    print("\nAnalysis graph saved as 'knapsack_analysis.png'")
    plt.show()


# ============================================================
# MAIN PROGRAM
# ============================================================


if __name__ == "__main__":
    random.seed(42)

    print("\n" + "=" * 70)
    print("KNAPSACK PROBLEM: ABC vs BACKTRACKING")
    print("=" * 70)

    # Example problem: (weight, value) for each item - LARGE INSTANCE
    items = [
        (10, 60),
        (20, 100),
        (30, 120),
        (15, 70),
        (25, 90),
        (5, 30),
        (12, 50),
        (18, 80),
        (22, 85),
        (8, 45),
        (14, 65),
        (28, 110),
        (16, 75),
        (11, 55),
        (19, 88),
        (7, 40),
        (24, 95),
        (13, 60),
        (21, 92),
        (9, 48),
        (17, 78),
        (26, 105),
        (6, 35),
        (23, 98),
        (27, 115),
    ]
    capacity = 150

    print(f"\nProblem: {len(items)} items, capacity = {capacity}")
    print("\nItems (weight, value):")
    for i, (w, v) in enumerate(items, 1):
        print(f"  Item {i}: weight={w}, value={v}")

    # Solve with Artificial Bee Colony
    abc_sol, abc_val, abc_time, convergence = abc_knapsack(
        items, capacity, bees=20, iterations=50
    )

    # Solve with Backtracking (Exact)
    bt_sol, bt_val, bt_time = backtracking_knapsack(items, capacity)

    # Comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"ABC (Metaheuristic):  Value = {abc_val}")
    print(f"Backtracking (Exact): Value = {bt_val} (OPTIMAL)")
    print(f"Difference: {abs(bt_val - abc_val)}")
    if bt_val > 0:
        print(f"ABC Accuracy: {(abc_val / bt_val * 100):.1f}%")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("• ABC is FAST but approximate (good for large problems)")
    print("• Backtracking is EXACT but slower (guaranteed optimal)")
    print("• Use ABC for 100+ items, Backtracking for < 30 items")
    print("=" * 70)

    # Generate analysis graphs
    plot_analysis(abc_val, abc_time, bt_val, bt_time, convergence)
