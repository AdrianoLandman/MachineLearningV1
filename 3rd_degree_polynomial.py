# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:47:30 2026

@author: A3ano
"""
import numpy as np
import matplotlib.pyplot as plt

# Data: positions, timesteps
positions = np.array([
    [ 2.00,  0.00,  1.00],
    [ 1.08,  1.68,  2.38],
    [-0.83,  1.82,  2.49],
    [-1.97,  0.28,  2.15],
    [-1.31, -1.51,  2.59],
    [ 0.57, -1.91,  4.32]
], dtype=float)

times = np.array([1, 2, 3, 4, 5, 6])

def gradient_descent(initial_guess_p0_v_and_a, gradient, learn_rate, max_iter, tol):
    params = initial_guess_p0_v_and_a
    for iteration in range(max_iter):
        grad = gradient(params, positions, times)
        diff = learn_rate * grad
        if np.linalg.norm(diff) < tol:
            break
        params = params - diff
    return params

#Gradient of error with 3rd degree polynomial
def gradient_of_error(params, positions, times):
    x0, y0, z0, vx, vy, vz, ax, ay, az, bx, by, bz = params  
    p0 = np.array([x0, y0, z0])
    v = np.array([vx, vy, vz])
    a = np.array([ax, ay, az])
    b = np.array([bx, by, bz]) 
    
    dp0 = np.zeros(3)
    dv = np.zeros(3)
    da = np.zeros(3)
    db = np.zeros(3) 

    for i in range(len(times)):
        t = times[i]
        predicted = p0 + v * t + 0.5 * a * (t**2) + (1/6) * b * (t**3)
        residual = positions[i] - predicted  
        
        dp0 += -2 * residual
        dv += -2.0 * t * residual
        da += -(t**2) * residual
        db += -(t**3) * residual  # Gradient for cubic terms

    return np.concatenate([p0, dv, da, db])  


optimized_params = gradient_descent(
    initial_guess_p0_v_and_a=np.zeros(12), 
    gradient=gradient_of_error,
    learn_rate=0.0001,
    max_iter=2000000,
    tol=0.000001)


x0, y0, z0, vx, vy, vz, ax, ay, az, bx, by, bz = optimized_params


def predicted_position_at_time(t):
    return np.array([x0, y0, z0]) + np.array([vx, vy, vz]) * t + 0.5 * np.array([ax, ay, az]) * (t**2) + (1/6) * np.array([bx, by, bz]) * (t**3)

#store predicted positions in array
predicted_positions = np.array([predicted_position_at_time(t) for t in times])


#X-direction
plt.figure(figsize=(8, 6))
plt.plot(times, positions[:, 0], 'ro-', label="Actual x-position")  # Actual x-positions
plt.plot(times, predicted_positions[:, 0], 'r--', label="Predicted x-position")  # Predicted x-positions
plt.xlabel('Time (seconds)')
plt.ylabel('X Position')
plt.title('Drone x-Position vs Time')
plt.legend()
plt.grid(True)
plt.show()

#Y-direction
plt.figure(figsize=(8, 6))
plt.plot(times, positions[:, 1], 'go-', label="Actual y-position")  
plt.plot(times, predicted_positions[:, 1], 'g--', label="Predicted y-position")  
plt.xlabel('Time (seconds)')
plt.ylabel('Y Position')
plt.title('Drone y-Position vs Time')
plt.legend()
plt.grid(True)
plt.show()

#Z-direction
plt.figure(figsize=(8, 6))
plt.plot(times, positions[:, 2], 'bo-', label="Actual z-position")  
plt.plot(times, predicted_positions[:, 2], 'b--', label="Predicted z-position")  
plt.xlabel('Time (seconds)')
plt.ylabel('Z Position')
plt.title('Drone z-Position vs Time')
plt.legend()
plt.grid(True)
plt.show()

def sse_function(params, positions, times):
    x0, y0, z0, vx, vy, vz, ax, ay, az, bx, by, bz = params
    v = np.array([vx, vy, vz])
    a = np.array([ax, ay, az])
    b = np.array([bx, by, bz])  
    p0 = np.array([x0, y0, z0])

    predicted = np.zeros_like(positions)
    for i in range(len(times)):
        predicted[i] = p0 + v * times[i] + 0.5 * a * (times[i]**2) + (1/6) * b * (times[i]**3)  # Add cubic terms
        
    sse = np.sum((positions - predicted) ** 2)
    return sse

final_sse = sse_function(optimized_params, positions, times)

print("Estimated velocity v =", np.array([vx, vy, vz]))
print("Estimated acceleration a =", np.array([ax, ay, az]))
print("Estimated cubic coefficients b =", np.array([bx, by, bz]))
print("Final residual error (SSE) =", final_sse)

fig = plt.figure(figsize=(10, 8))
axes_3d = fig.add_subplot(111, projection='3d')

#Actual positions (t = 1 to t = 6)
axes_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', label="Actual positions")

#Predicted positions (t = 1 to t = 6)
axes_3d.plot(predicted_positions[:, 0], predicted_positions[:, 1], predicted_positions[:, 2], label="Predicted trajectory", linewidth=3)

predicted_position_t7 = predicted_position_at_time(7)
axes_3d.scatter(predicted_position_t7[0], predicted_position_t7[1], predicted_position_t7[2], 
                color='orange', label=f"Predicted position at t=7", s=100, marker='o')


for (x, y, z), t in zip(positions, times):
    axes_3d.text(x, y, z, f"t={t}", fontsize=9)

axes_3d.set_xlabel('X Position')
axes_3d.set_ylabel('Y Position')
axes_3d.set_zlabel('Z Position')
axes_3d.set_title('Drone Actual and Predicted Trajectory (Constant Acceleration Model)')

axes_3d.legend()

plt.show()