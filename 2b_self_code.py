
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
    learn_rate=0.0001,
    max_iter= 2000000,
    tol=0.000001)

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





