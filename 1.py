import random
import math

import numpy as np
import matplotlib.pyplot as plt

# Визначення функції Сфери
def sphere_function(x):
    return sum(xi ** 2 for xi in x)


# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    step_size = 0.1

    current = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
    current_value = func(current)

    trajectory = [(current.copy(), current_value)]

    for _ in range(iterations):
        neighbors = [
            [current[j] - step_size for j in range(len(bounds))],
            [current[j] + step_size for j in range(len(bounds))]
        ]

        next_point = None
        next_value = float("inf")
        for neighbor in neighbors:
            if all(bounds[i][0] <= neighbor[i] <= bounds[i][1] for i in range(len(bounds))):
                value = func(neighbor)
                if value < next_value:
                    next_point = neighbor
                    next_value = value

        if next_value >= current_value or abs(next_value - current_value) < epsilon:
            break

        current, current_value = next_point, next_value
        trajectory.append((current.copy(), current_value))

    return current, current_value, trajectory


# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    step_size = 0.5

    current = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
    current_value = func(current)

    trajectory = [(current.copy(), current_value)]

    for _ in range(iterations):
        next_point = [current[j] + random.uniform(-step_size, step_size) for j in range(len(bounds))]
        if any(next_point[j] < bounds[j][0] or next_point[j] > bounds[j][1] for j in range(len(bounds))):
            continue  # Skip evaluation if out of bounds

        next_value = func(next_point)

        if next_value < current_value or random.random() < 0.1: # Маленький ймовірнісний фактор
            current, current_value = next_point, next_value
            trajectory.append((current.copy(), current_value))

        if current_value < epsilon:
            break

    return current, current_value, trajectory


# Simulated Annealing
def simulated_annealing(func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6):
    current = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
    current_value = func(current)

    trajectory = [(current.copy(), current_value)]

    while temp > 0.001 and iterations > 0:
        next_point = [current[j] + random.uniform(-1, 1) for j in range(len(bounds))]
        # Ensure next_point is within bounds
        if any(next_point[j] < bounds[j][0] or next_point[j] > bounds[j][1] for j in range(len(bounds))):
            iterations -= 1
            continue  # Skip evaluation if out of bounds

        next_value = func(next_point)
        delta_value = next_value - current_value

        if delta_value < 0 or random.random() < math.exp(-delta_value / temp):
            current = next_point
            current_value = next_value
            trajectory.append((current.copy(), current_value))

        if abs(delta_value) < epsilon:
            break

        temp *= cooling_rate
        iterations -= 1

    return current, current_value, trajectory


# Single Window 3D Visualization Function for All Trajectories
def visualize_trajectories_3d(bounds, trajectories, labels, colors):
    fig = plt.figure(figsize=(18, 6))

    x_range = np.linspace(bounds[0][0], bounds[0][1], 100)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[sphere_function([x, y]) for x in x_range] for y in y_range])

    for i, trajectory in enumerate(trajectories):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.6)

        traj_x = [p[0][0] for p in trajectory]
        traj_y = [p[0][1] for p in trajectory]
        traj_z = [p[1] for p in trajectory]

        # Plot the trajectory
        ax.plot(traj_x, traj_y, traj_z, marker='o', linestyle='dashed', color=colors[i], label=labels[i])

        # Mark start and end points
        ax.scatter(traj_x[0], traj_y[0], traj_z[0], color='green', s=100, label="Start Point")
        ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], color='red', s=100, label="Final Point")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Sphere Function Value")
        ax.set_title(f"{labels[i]} Trajectory")
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Межі для функції
    bounds = [(-5, 5), (-5, 5)]

    # Виконання алгоритмів
    print("Hill Climbing:")
    hc_solution, hc_value, hc_trajectory = hill_climbing(sphere_function, bounds)
    print("Розв'язок:", hc_solution, "Значення:", hc_value)
    # visualize_trajectory_3d(bounds, hc_trajectory)

    print("\nRandom Local Search:")
    rls_solution, rls_value, rls_trajectory = random_local_search(sphere_function, bounds)
    print("Розв'язок:", rls_solution, "Значення:", rls_value)
    # visualize_trajectory_3d(bounds, rls_trajectory)

    print("\nSimulated Annealing:")
    sa_solution, sa_value, sa_trajectory = simulated_annealing(sphere_function, bounds)
    print("Розв'язок:", sa_solution, "Значення:", sa_value)
    # visualize_trajectory_3d(bounds, sa_trajectory)

    visualize_trajectories_3d(
        bounds,
        trajectories=[hc_trajectory, rls_trajectory, sa_trajectory],
        labels=["Hill Climbing", "Random Local Search", "Simulated Annealing"],
        colors=["black", "blue", "orange"]
    )
