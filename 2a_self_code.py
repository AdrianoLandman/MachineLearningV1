# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:08:58 2026

@author: A3ano
"""
import numpy as np

#Drone tracking data, respectively x, y and z
positions = np.array([
    [2.00, 0.00, 1.00],
    [1.08, 1.68, 2.38],
    [-0.83, 1.82, 2.49],
    [-1.97, 0.28, 2.15],
    [-1.31, -1.51, 2.59],
    [0.57, -1.91, 4.32]
])

times = np.array([1, 2, 3, 4, 5, 6])  #Timesteps

#Own gradient descent implementation (similar to example in course site)
def gradient_descent(initial_guess_v, gradient, learn_rate, max_iter, tol):

    params = initial_guess_v  #Start from initial guess for velocity
    
    for iteration in range(max_iter):
        grad = gradient(params, positions, times)  #Calculate the gradient (slope of error)
        diff = learn_rate * grad  #Calculate the step size
        
        if np.linalg.norm(diff) < tol:  #Compare normation of vector (euclidian distance), to tolerance. np.linalf.norm used to make vector into single scalar value
            break #If step size is small enough, stop
        
        params = params - diff  #Update the parameters (velocity)
    
    return params

#gradient function assuming initial position is fixed at P(1)
def gradient_of_error(params, positions, times):  

    vx, vy, vz = params  #Velocity params
    p0 = positions[0]  #Initial position, which is set at P(t=1)
    v = np.array([vx, vy, vz])  #velocity vector

    # Initial gradient vector for velocity
    dv = np.zeros(3)  # Gradient with respect to velocity

    #Loop over each time step to compute the gradients
    for i in range(len(times)):
        predicted_position = p0 + v * times[i]  #Predicted position at time t
        residual = positions[i] - predicted_position  #residuals

        dv += -2 * times[i] * residual  #Gradient with respect to velocity (so essentially derivative of error function)

    return dv  


#Below gradient descent is run to find the 'optimal' constant velocity (initial position is fixed at P(1))
optimized_constant_velocity = gradient_descent(
    initial_guess_v= np.zeros(3),                    # Starting guess for velocity
    gradient=gradient_of_error, #Gradient function
    learn_rate=0.001,         # Learning rate
    max_iter=20000,          # Maximum number of iterations
    tol=0.001                      # Tolerance for stopping
)

#the constant velocity
vx, vy, vz = optimized_constant_velocity

print('Constant velocity for, respectively, vx, vy and vz:', vx, vy, vz)

#Error function assuming initial position is fixed at P(1)
def error_function(params, positions, times):
  
    vx, vy, vz = params  #Only velocity to estimate
    p0 = positions[0]  #Initial position is now the position at t = 1
    v = np.array([params])  #Velocity vector

    #Predicted positions array with the same dimensions as positions array
    predicted_positions = np.zeros_like(positions)

    #Loop over each time step to calculate the predicted position
    for i in range(len(times)):
        predicted_positions[i] = p0 + v * times[i]  #Predicted position at time t = P(1) + v * t

    #Sum of squared errors (SSE) between actual and predicted positions
    error = np.sum((positions - predicted_positions) ** 2)
    return error



# Calculate the final residual error (sum of squared errors)
final_error = error_function(optimized_constant_velocity, positions, times)
print('Final error =', final_error)