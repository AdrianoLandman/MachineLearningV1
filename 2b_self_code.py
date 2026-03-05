# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 15:14:45 2026

@author: A3ano
"""
# -*- coding: utf-8 -*-

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



