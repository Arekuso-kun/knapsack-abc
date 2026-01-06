import os
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


def plot_analysis(abc_val, abc_time, bt_val, bt_time, convergence_history, size=None):
    """Create visualization of results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f"Problema Rucsacului: Analiza ABC vs Backtracking"
    if size is not None:
        title += f" ({size} obiecte)"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # 1. Time Comparison
    ax1 = axes[0, 0]
    algorithms = ["ABC\n(Metaeuristic)", "Backtracking\n(Exact)"]
    times = [abc_time, bt_time]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax1.bar(
        algorithms, times, color=colors, alpha=0.7, edgecolor="black", linewidth=2
    )
    ax1.set_ylabel("Timp (secunde)", fontweight="bold", fontsize=11)
    ax1.set_title("Comparație Timp de Execuție", fontweight="bold", fontsize=12)
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
    if speedup >= 1:
        speedup_text = f"ABC este de {speedup:.2f}x mai rapid!"
        box_color = "yellow"
    else:
        inverse_speedup = abc_time / bt_time if bt_time > 0 else 0
        speedup_text = f"ABC este de {inverse_speedup:.2f}x mai lent!"
        box_color = "lightcoral"

    ax1.text(
        0.5,
        max(times) * 0.5,
        speedup_text,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.5),
    )

    # 2. Value Comparison
    ax2 = axes[0, 1]
    algorithms = ["ABC", "Backtracking\n(Optim)"]
    values = [abc_val, bt_val]
    colors = ["#3498db", "#9b59b6"]
    bars = ax2.bar(
        algorithms, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2
    )
    ax2.set_ylabel("Valoare Totală", fontweight="bold", fontsize=11)
    ax2.set_title("Comparație Calitate Soluții", fontweight="bold", fontsize=12)
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
        f"Acuratețe: {accuracy:.1f}%\nDiferență: {gap}",
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
        label="Cea Mai Bună Valoare",
    )
    ax3.axhline(
        y=bt_val, color="#9b59b6", linestyle="--", linewidth=2, label="Valoare Optimă"
    )
    ax3.set_xlabel("Iterație", fontweight="bold", fontsize=11)
    ax3.set_ylabel("Cea Mai Bună Valoare Găsită", fontweight="bold", fontsize=11)
    ax3.set_title("Convergența ABC în Timp", fontweight="bold", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right", fontsize=10)
    ax3.fill_between(iterations, convergence_history, alpha=0.3, color="#e67e22")

    # 4. Performance Metrics Summary
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Determine which algorithm is faster
    if speedup >= 1:
        speed_comparison = f"Raport Viteză: ABC este {speedup:.2f}x mai rapid"
        time_percentage = f"ABC în doar {(abc_time/bt_time*100):.2f}% din timp!"
    else:
        inverse_speedup = abc_time / bt_time if bt_time > 0 else 0
        speed_comparison = f"Raport Viteză: ABC este {inverse_speedup:.2f}x mai lent"
        time_percentage = f"ABC ia {(abc_time/bt_time*100):.2f}% mai mult timp!"

    metrics_text = f"""
    METRICI DE PERFORMANȚĂ
    {'='*40}
    
    Colonie Artificială de Albine (ABC):
       • Timp: {abc_time:.6f} secunde
       • Valoare: {abc_val}
       • Tip: Metaeuristic (Aproximativ)
    
    Backtracking (Branch & Bound):
       • Timp: {bt_time:.6f} secunde
       • Valoare: {bt_val} (OPTIM)
       • Tip: Algoritm Exact
    
    Comparație:
       • {speed_comparison}
       • Acuratețe: {accuracy:.2f}%
       • Diferență Valoare: {gap}
    
    Concluzie:
       ABC a găsit o soluție {accuracy:.1f}% optimă
       {time_percentage}
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

    # Create results folder if it doesn't exist
    os.makedirs("results", exist_ok=True)

    filename = (
        f"results/knapsack_analysis_size_{size}.png"
        if size is not None
        else "results/knapsack_analysis.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\nAnalysis graph saved as '{filename}'")
    plt.close()  # Close to avoid showing multiple windows


# ============================================================
# MAIN PROGRAM
# ============================================================


def generate_random_items(n_items, max_weight=50, max_value=100):
    """Generate random knapsack items"""
    items = []
    for _ in range(n_items):
        weight = random.randint(5, max_weight)
        value = random.randint(20, max_value)
        items.append((weight, value))
    # Set capacity to ~50% of total weight
    total_weight = sum(w for w, v in items)
    capacity = total_weight // 2
    return items, capacity


def test_multiple_sizes():
    """Test both algorithms on multiple problem sizes"""
    print("\n" + "=" * 80)
    print("MULTI-SIZE PERFORMANCE TEST: ABC vs BACKTRACKING")
    print("=" * 80)

    test_sizes = [10, 15, 20, 25, 30, 35]
    results = []

    for size in test_sizes:
        print(f"\n{'='*80}")
        print(f"TEST: {size} ITEMS")
        print(f"{'='*80}")

        # Generate problem
        items, capacity = generate_random_items(size)
        print(f"Generated {size} items, capacity = {capacity}")

        # Test ABC
        print(f"\nRunning ABC...")
        abc_sol, abc_val, abc_time, convergence = abc_knapsack(
            items, capacity, bees=30, iterations=100
        )

        # Test Backtracking
        print(f"\nRunning Backtracking...")
        bt_sol, bt_val, bt_time = backtracking_knapsack(items, capacity)
        accuracy = (abc_val / bt_val * 100) if bt_val > 0 else 0
        speedup = bt_time / abc_time if abc_time > 0 else 0

        results.append(
            {
                "size": size,
                "abc_val": abc_val,
                "abc_time": abc_time,
                "bt_val": bt_val,
                "bt_time": bt_time,
                "accuracy": accuracy,
                "speedup": speedup,
            }
        )

        # Print comparison
        print(f"\n{'-'*80}")
        print(f"SIZE {size} RESULTS:")
        print(f"  ABC: value={abc_val}, time={abc_time:.6f}s")
        if bt_val is not None:
            print(f"  Backtracking: value={bt_val}, time={bt_time:.6f}s")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Speedup: {speedup:.2f}x")

            # Generate detailed analysis for this size
            print(f"\nGenerating detailed analysis graph for size {size}...")
            plot_analysis(abc_val, abc_time, bt_val, bt_time, convergence, size)
        print(f"{'-'*80}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(
        f"{'Size':<8} {'ABC Value':<12} {'ABC Time':<12} {'BT Value':<12} {'BT Time':<12} {'Accuracy':<12} {'Speedup'}"
    )
    print(f"{'-'*80}")

    for r in results:
        bt_val_str = str(r["bt_val"]) if r["bt_val"] is not None else "N/A"
        bt_time_str = f"{r['bt_time']:.4f}s" if r["bt_time"] is not None else "N/A"
        acc_str = f"{r['accuracy']:.2f}%" if r["accuracy"] is not None else "N/A"
        speedup_str = f"{r['speedup']:.2f}x" if r["speedup"] is not None else "N/A"

        print(
            f"{r['size']:<8} {r['abc_val']:<12} {r['abc_time']:.4f}s{'':<6} {bt_val_str:<12} {bt_time_str:<12} {acc_str:<12} {speedup_str}"
        )

    print(f"{'='*80}")

    # Plot scaling behavior
    plot_scaling_analysis(results)

    return results


def plot_scaling_analysis(results):
    """Create visualization of scaling behavior"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Analiză de Scalare: ABC vs Backtracking", fontsize=16, fontweight="bold"
    )

    sizes = [r["size"] for r in results]
    abc_times = [r["abc_time"] for r in results]
    bt_times = [r["bt_time"] for r in results if r["bt_time"] is not None]
    bt_sizes = [r["size"] for r in results if r["bt_time"] is not None]

    # 1. Time Scaling
    ax1 = axes[0, 0]
    ax1.plot(
        sizes, abc_times, "o-", color="#2ecc71", linewidth=2, markersize=8, label="ABC"
    )
    if bt_times:
        ax1.plot(
            bt_sizes,
            bt_times,
            "s-",
            color="#e74c3c",
            linewidth=2,
            markersize=8,
            label="Backtracking",
        )
    ax1.set_xlabel(
        "Dimensiunea Problemei (nr. obiecte)", fontweight="bold", fontsize=11
    )
    ax1.set_ylabel("Timp (secunde)", fontweight="bold", fontsize=11)
    ax1.set_title(
        "Timp de Execuție vs Dimensiunea Problemei", fontweight="bold", fontsize=12
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Speedup
    ax2 = axes[0, 1]
    speedups = [r["speedup"] for r in results if r["speedup"] is not None]
    speedup_sizes = [r["size"] for r in results if r["speedup"] is not None]
    if speedups:
        ax2.plot(
            speedup_sizes, speedups, "o-", color="#9b59b6", linewidth=2, markersize=8
        )
        ax2.set_xlabel(
            "Dimensiunea Problemei (nr. obiecte)", fontweight="bold", fontsize=11
        )
        ax2.set_ylabel(
            "Factor de Accelerare (timp BT / timp ABC)", fontweight="bold", fontsize=11
        )
        ax2.set_title(
            "Accelerarea ABC față de Backtracking", fontweight="bold", fontsize=12
        )
        ax2.grid(True, alpha=0.3)

    # 3. Accuracy
    ax3 = axes[1, 0]
    accuracies = [r["accuracy"] for r in results if r["accuracy"] is not None]
    acc_sizes = [r["size"] for r in results if r["accuracy"] is not None]
    if accuracies:
        ax3.plot(
            acc_sizes, accuracies, "o-", color="#3498db", linewidth=2, markersize=8
        )
        ax3.axhline(y=100, color="red", linestyle="--", linewidth=1, label="Optim")
        ax3.set_xlabel(
            "Dimensiunea Problemei (nr. obiecte)", fontweight="bold", fontsize=11
        )
        ax3.set_ylabel("Acuratețe (%)", fontweight="bold", fontsize=11)
        ax3.set_title("Calitatea Soluției ABC", fontweight="bold", fontsize=12)
        ax3.set_ylim([90, 105])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Time Comparison (Log Scale)
    ax4 = axes[1, 1]
    ax4.semilogy(
        sizes, abc_times, "o-", color="#2ecc71", linewidth=2, markersize=8, label="ABC"
    )
    if bt_times:
        ax4.semilogy(
            bt_sizes,
            bt_times,
            "s-",
            color="#e74c3c",
            linewidth=2,
            markersize=8,
            label="Backtracking",
        )
    ax4.set_xlabel(
        "Dimensiunea Problemei (nr. obiecte)", fontweight="bold", fontsize=11
    )
    ax4.set_ylabel("Timp (secunde, scară logaritmică)", fontweight="bold", fontsize=11)
    ax4.set_title("Scalare Timp (Logaritmic)", fontweight="bold", fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    # Create results folder if it doesn't exist
    os.makedirs("results", exist_ok=True)

    plt.savefig("results/knapsack_scaling_analysis.png", dpi=300, bbox_inches="tight")
    print("\nScaling analysis graph saved as 'results/knapsack_scaling_analysis.png'")
    plt.show()


if __name__ == "__main__":
    random.seed(42)

    # Run multi-size test
    test_multiple_sizes()
