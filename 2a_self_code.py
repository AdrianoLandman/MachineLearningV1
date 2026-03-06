
import numpy as np

#Drone tracking data for x, y and z
positions = np.array([
    [2.00, 0.00, 1.00],
    [1.08, 1.68, 2.38],
    [-0.83, 1.82, 2.49],
    [-1.97, 0.28, 2.15],
    [-1.31, -1.51, 2.59],
    [0.57, -1.91, 4.32]
])

times = np.array([1, 2, 3, 4, 5, 6])  #Timesteps

#Own gradient descent implementation (in a similar as shown in the example code)
def gradient_descent(initial_guess_p0_and_v, gradient, learn_rate, max_iter, tol):

    params = initial_guess_p0_and_v  #Start from initial guess for p0 and velocity
    
    for iteration in range(max_iter):
        grad = gradient(params, positions, times)  #Calculate the gradient with gradient of error function defined below
        diff = learn_rate * grad  #Calculate the step size
        
        if np.linalg.norm(diff) < tol:  #Compare vector (normation of vector) to tolerance. np.linalf.norm used to make vector into single scalar value
            break #If stepsize is smaller than tolerance than algorithm converges
        
        params = params - diff  #If algorithm did not converge, than params is updated and process is repeated
    
    return params

#gradient function assuming initial position is fixed at P(1)
def gradient_of_error(params, positions, times):  

    x0, y0, z0, vx, vy, vz = params  
    p0 = np.array([x0, y0,z0]) #initial position vector
    v = np.array([vx, vy, vz])  #velocity vector

    #Initial gradient vector for velocity and p0
    dv = np.zeros(3) 
    dp0 = np.zeros(3)

    #Compute gradient (loop over each timestep)
    for i in range(len(times)):
        predicted_position = p0 + v * times[i]  
        residual = positions[i] - predicted_position  #residuals

        dp0 += -2 * residual
        dv += -2 * times[i] * residual  
        
    return np.concatenate([dp0, dv])  


#run gradient descent below
optimized_constant_velocity = gradient_descent(
    initial_guess_p0_and_v= np.zeros(6),       #Starting guess for velocity
    gradient=gradient_of_error,                #Gradient function
    learn_rate=0.0001,                         #LEARNING RATE, CHANGE VALUE TO TUNE LEARNING RATE
    max_iter=200000,                           #MAX. NUMBER OF ITER., CHANGE VALUE TO TUNE MAX NUMBER OF ITERATIONS
    tol=0.000001                               #TOLERANCE, CHANGE VALUE TO TUNE TOLERANCE
)

#the constant velocity
x0, y0, z0, vx, vy, vz = optimized_constant_velocity

print('Constant velocity for, respectively, vx, vy and vz:', vx, vy, vz)

#Error function, 
def error_function(params, positions, times):
  
    x0, y0, z0, vx, vy, vz = params  
    p0 = np.array([x0, y0, z0])
    v = np.array([vx, vy, vz])  

    #Predicted positions array with the same dimensions as positions array
    predicted_positions = np.zeros_like(positions)

    #Loop over each time step to calculate the predicted position
    for i in range(len(times)):
        predicted_positions[i] = p0 + v * times[i]  #Predicted position at time t = p0 + v * t

    #Sum of squared errors (SSE) between actual and predicted positions
    error = np.sum((positions - predicted_positions) ** 2)
    return error



# Calculate the final residual error (sum of squared errors)
final_error = error_function(optimized_constant_velocity, positions, times)
print('Final error =', final_error)

