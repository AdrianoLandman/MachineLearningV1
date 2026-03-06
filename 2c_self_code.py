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

#gradient descent solver
def gradient_descent(initial_guess_p0_v_and_a, gradient, learn_rate, max_iter, tol):
    
    params = initial_guess_p0_v_and_a
    
    for iteration in range(max_iter):
        grad = gradient(params, positions, times)
        diff = learn_rate * grad

        if np.linalg.norm(diff) < tol:
            break
        params = params - diff

    return params

def gradient_of_error(params, positions, times):
    x0, y0, z0, vx, vy, vz, ax, ay, az = params
    p0 = np.array([x0, y0, z0])
    v = np.array([vx, vy, vz])
    a = np.array([ax, ay, az])
    
    dp0 = np.zeros(3)
    dv = np.zeros(3)
    da = np.zeros(3)

    #gradient update in gradient descent solver
    for i in range(len(times)):
        t = times[i]
        predicted = p0 + v * t + 0.5 * a * (t**2)
        residual = positions[i] - predicted  
        
        dp0 += -2 * residual
        dv += -2.0 * t * residual
        da += -(t**2) * residual

    return np.concatenate([p0, dv, da])

optimized_params = gradient_descent(
    initial_guess_p0_v_and_a=np.zeros(9),
    gradient=gradient_of_error,
    learn_rate=0.0001,                        #LEARNING RATE, CHANGE VALUE HERE
    max_iter= 2000000,                        #MAX NUMBER OF ITERATIONS, CHANGE VALUE HERE
    tol=0.000001)                             #TOLERANCE, CHANGE VALUE HERE

x0, y0, z0, vx, vy, vz, ax, ay, az = optimized_params

def sse_function(params, positions, times):
    x0, y0, z0, vx, vy, vz, ax, ay, az = params
    v = np.array([vx, vy, vz])
    a = np.array([ax, ay, az])
    p0 = np.array([x0, y0, z0])

    predicted = np.zeros_like(positions)
    for i in range(len(times)):
        predicted[i] = p0 + v * times[i] + 0.5 * a * (times[i]**2)
        
   
    sse = np.sum((positions - predicted) ** 2)
    return sse

final_sse = sse_function(optimized_params, positions, times)

print("Estimated velocity v =", np.array([vx, vy, vz]))
print("Estimated acceleration a =", np.array([ax, ay, az]))
print("Final residual error (SSE) =", final_sse)

"""
Continueing with question 2c
"""

t_predict = 7
predicted_position = np.array([x0, y0, z0]) + np.array([vx, vy, vz]) * t_predict + 0.5 * np.array([ax, ay, az]) * (t_predict**2)

#plotting predicted position, with actual positions for first 6 timesteps in 3D space

plt.figure(figsize=(8, 6))

#Actual positions
plt.plot(times, positions[:, 0], 'ro-', label="Actual x-position")
plt.plot(times, positions[:, 1], 'go-', label="Actual y-position")
plt.plot(times, positions[:, 2], 'bo-', label="Actual z-position")

#plot t=7
plt.plot(t_predict, predicted_position[0], 'r^', label=f"Predicted x-position at t=7", markersize=10)
plt.plot(t_predict, predicted_position[1], 'g^', label=f"Predicted y-position at t=7", markersize=10)
plt.plot(t_predict, predicted_position[2], 'b^', label=f"Predicted z-position at t=7", markersize=10)

plt.xlabel('Time (seconds)')
plt.ylabel('Position')
plt.title('Drone Position vs Time (Constant Acceleration Model)')
plt.legend()

plt.grid(True)
plt.show()

# Print the predicted position at t=7
print(f"Predicted position at t=7: {predicted_position}")

# 3D Plotting of positions
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

#Actual positions in 3D space
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', label="Actual positions")

#Predicted position at t = 7 in 3D space
ax.scatter(predicted_position[0], predicted_position[1], predicted_position[2], 
           color='orange', label=f"Predicted position at t=7", s=100, marker='o')

for (x, y, z), t in zip(positions, times):
    ax.text(x, y, z, f"t={t}", fontsize=9)

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Drone Trajectory (Constant Acceleration Model)')

ax.legend()
plt.show()

