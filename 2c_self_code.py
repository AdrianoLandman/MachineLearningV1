# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:30:30 2026

@author: A3ano
"""

import numpy as np
import matplotlib.pyplot as plt

# Data: positions, timesteps, and initial position

positions = np.array([
    [ 2.00,  0.00,  1.00],
    [ 1.08,  1.68,  2.38],
    [-0.83,  1.82,  2.49],
    [-1.97,  0.28,  2.15],
    [-1.31, -1.51,  2.59],
    [ 0.57, -1.91,  4.32]
], dtype=float)

times = np.array([1, 2, 3, 4, 5, 6])

p0 = positions[0]   # P(1)

#gradient descent solver
def gradient_descent(initial_guess_v_and_a, gradient, learn_rate, max_iter, tol):
    
    params = initial_guess_v_and_a
    
    for iteration in range(max_iter):
        grad = gradient(params, positions, times)
        diff = learn_rate * grad

        if np.linalg.norm(diff) < tol:
            break
        params = params - diff

    return params

def gradient_of_error(params, positions, times):
    vx, vy, vz, ax, ay, az = params
    v = np.array([vx, vy, vz])
    a = np.array([ax, ay, az])

    dv = np.zeros(3)
    da = np.zeros(3)

    #gradient update in gradient descent solver
    for i in range(len(times)):
        t = times[i]
        predicted = p0 + v * t + 0.5 * a * (t**2)
        residual = positions[i] - predicted  
        
        dv += -2.0 * t * residual
        da += -(t**2) * residual

    return np.concatenate([dv, da])

optimized_params = gradient_descent(
    initial_guess_v_and_a=np.zeros(6),
    gradient=gradient_of_error,
    learn_rate=0.001,
    max_iter= 20000,
    tol=0.000001)

vx, vy, vz, ax, ay, az = optimized_params

def sse_function(params, positions, times):
    vx, vy, vz, ax, ay, az = params
    v = np.array([vx, vy, vz])
    a = np.array([ax, ay, az])

    predicted = np.zeros_like(positions)
    for i in range(len(times)):
        predicted[i] = p0 + v * times[i] + 0.5 * a * (times[i]**2)
        
   
    sse = np.sum((positions - predicted) ** 2)
    return sse
    
optimized_params = gradient_descent(
    initial_guess_v_and_a=np.zeros(6),
    gradient=gradient_of_error,
    learn_rate=0.001,
    max_iter= 200000,
    tol=0.00001)

final_sse = sse_function(optimized_params, positions, times)

print("Estimated velocity v =", np.array([vx, vy, vz]))
print("Estimated acceleration a =", np.array([ax, ay, az]))
print("Final residual error (SSE) =", final_sse)

"""
Continueing with question 2c
"""

t_predict = 7
predicted_position = p0 + np.array([vx, vy, vz]) * t_predict + 0.5 * np.array([ax, ay, az]) * (t_predict**2)

# -----------------------------
# Plotting the positions
plt.figure(figsize=(8, 6))

# Plot the actual positions
plt.plot(times, positions[:, 0], 'ro-', label="Actual x-position")
plt.plot(times, positions[:, 1], 'go-', label="Actual y-position")
plt.plot(times, positions[:, 2], 'bo-', label="Actual z-position")

# Plot the predicted position at t = 7
plt.plot(t_predict, predicted_position[0], 'r^', label=f"Predicted x-position at t=7", markersize=10)
plt.plot(t_predict, predicted_position[1], 'g^', label=f"Predicted y-position at t=7", markersize=10)
plt.plot(t_predict, predicted_position[2], 'b^', label=f"Predicted z-position at t=7", markersize=10)

# Add labels and legend
plt.xlabel('Time (seconds)')
plt.ylabel('Position')
plt.title('Drone Position vs Time (Constant Acceleration Model)')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# Print the predicted position at t=7
print(f"Predicted position at t=7: {predicted_position}")

# 3D Plotting of positions

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the actual positions at t = 1, 2, 3, 4, 5, 6
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', label="Actual positions")

# Plot the predicted position at t = 7
ax.scatter(predicted_position[0], predicted_position[1], predicted_position[2], 
           color='blue', label=f"Predicted position at t=7", s=100, marker='^')

# Optional: label each point with its time
for (x, y, z), t in zip(positions, times):
    ax.text(x, y, z, f"t={t}", fontsize=9)

# Set labels and title
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Drone Trajectory (Constant Acceleration Model)')

# Display the legend
ax.legend()

# Show the plot
plt.show()