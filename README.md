# Simulation and Visualization of Artificial Intelligence Application in Robotics

## Introduction
This project is about various robotic algorithms used in mobile robots. A mobile robot is a machine capable of moving in any environment, either by being propelled by wheels, legs, or any other means. A mobile robot is a machine controlled by software that use sensors and other technology to identify its surroundings and move around its environment. Mobile robots function using a combination of artificial intelligence (AI) and physical robotic elements, such as wheels, tracks and legs. Mobile robots are becoming increasingly popular across different business sectors. They are used to assist with work processes and even accomplish tasks that are impossible or dangerous for human workers. It is designed to perform tasks without human intervention. In the context of this project, a mobile robot is represented by a bicycle/car model.

A bicycle/car model is a simplified model of a two bicycle that is used to represent a mobile robot in many robotics simulations and control algorithms. The model assumes that the robot has four wheels and moves in a plane. It is a commonly used model for studying the control and localization of mobile robots. You can see the model below:

![car-like](/docs/images/car-like.png "Mobile Robot Model")

In this project, we will examine several algorithms for localization, tracking, search, PID control, and SLAM in the context of the bicycl/car model. Through animations and simulations, we will explore how these algorithms can be used to control the behavior of a mobile robot.

## Localization and Tracking
In this project, we will be working with various algorithms for mobile robot localization and tracking. The algorithms include the Histogram Filter, Kalman Filter, and Particle Filter. The aim is to localize the robot's position within an environment and track its movements over time. These algorithms provide solutions for the robot's perception of its surroundings and its ability to navigate in a controlled manner. Understanding and implementing these algorithms is crucial for enabling mobile robots to operate autonomously and perform tasks effectively in dynamic and unstructured environments. In the following sections, we will provide an in-depth explanation of each of these algorithms.

![localization-algos](/docs/images/localization-recap.png "anim hist one")

### Histogram Filter
The Histogram Filter, also known as Bayes Filter, is a probabilistic algorithm used for localization and tracking. The basic idea of the histogram filter is to use a probability distribution to model the state of a robot. In this filter, the state of the robot is represented by a histogram which represents the probabilities of the robot being in different states. The algorithm uses sensor measurements to update the histogram and update the probabilities of the state of the robot. The Histogram Filter is suitable for problems where the robot's state can be divided into a finite set of discrete states.

The algorithm starts with an initial probability distribution for the state of the robot. This probability distribution is updated at each time step based on the sensor measurements and the motion model of the robot. The motion model describes how the state of the robot changes over time. The sensor model describes how the measurements are related to the state of the robot. The final probability distribution gives us the most likely state of the robot.

In conclusion, the Histogram Filter is a simple and efficient algorithm for problems where the state of the robot can be represented by a discrete set of states. It can provide a good approximation of the state of the robot in real-time, making it useful for various robotic applications such as localization and tracking.

#### Main Histogram Filter Code:

##### The World
```python
p=[0.2, 0.2, 0.2, 0.2, 0.2]
world=['green', 'red', 'red', 'green', 'green']
measurements = ['red', 'green']
motions = [1,1]
pHit = 0.6
pMiss = 0.2
pExact = 0.8
pOvershoot = 0.1
pUndershoot = 0.1
``` 

##### Funtions
```python
def sense(p, Z):
    q=[]
    for i in range(len(p)):
        hit = (Z == world[i])
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))
    s = sum(q)
    for i in range(len(q)):
        q[i] = q[i] / s
    return q

def move(p, U):
    #
    q = []
    for i in range(len(p)):
        q.append(p[i-U % len(p)])
    return q

```

##### Apply
```python
for k in range(len(measurements)):
    p = sense(p,measurements[k])
    p = move(p,motions[k])
print(p)
>>> [0.21157894736842103, 0.1515789473684211, 0.08105263157894739, 0.16842105263157897, 0.3873684210526316]
``` 

The code above implements the histogram filter algorithm for robot localization. The algorithm is used to estimate the position of a robot in a given environment based on its measurements and movements.

The initial probability distribution of the robot's position is given as p = [0.2, 0.2, 0.2, 0.2, 0.2]. This means that the robot is equally likely to be at any of the five possible positions.

The environment is represented by the world list, which contains the colors of the different positions. In this example, the environment is ['green', 'red', 'red', 'green', 'green'].

The measurements taken by the robot are given as measurements = ['red', 'green']. These measurements are used to update the probability distribution of the robot's position.

The movements of the robot are given as motions = [1, 1]. These movements are used to update the probability distribution of the robot's position.

The probability of hitting the correct color, given a measurement, is given as pHit = 0.6. The probability of missing the correct color, given a measurement, is given as pMiss = 0.2.

The function sense(p, Z) implements the update of the probability distribution based on the measurements. It takes the current probability distribution and the current measurement as inputs and returns the updated probability distribution.

The function move(p, U) implements the update of the probability distribution based on the movements. It takes the current probability distribution and the current movement as inputs and returns the updated probability distribution.

The for loop in the code iterates through the measurements and movements and updates the probability distribution accordingly. After all the iterations, the final probability distribution is printed and represents the estimated position of the robot.

#### Animation
##### One Dimensional Example
![hist-one-d](/docs/images/hist-localization_1D.gif "anim hist one")
##### Two Dimensional Example
![hist-two-d](/docs/images/hist-localization_2D.gif "anim hist two")

### Kalman Filters
Kalman Filter is an algorithm that is widely used in the field of robotics for estimation of the state of a system, especially in the context of localization and tracking. The main aim of the Kalman Filter is to estimate the state of the system and to minimize the estimation error. The algorithm operates in a two-step process. In the first step, the algorithm predicts the next state of the system based on the previous state and the control inputs. In the second step, the algorithm updates the predicted state by taking into account the measurement information.

Kalman Filter is particularly useful in the presence of noisy measurements or uncertain dynamic models. It provides an optimal estimate by taking into account the estimated measurement noise and system model noise. The algorithm also has the ability to handle multi-dimensional systems, making it a suitable choice for mobile robots.

To implement Kalman Filter for a mobile robot, we need to have a mathematical model of the robot and its environment. The robot model should include the robot's kinematic equations, the measurement model, and the process noise. Then, we can use the Kalman Filter algorithm to estimate the state of the robot based on the measurements and control inputs. The algorithm can be implemented using linear algebra and matrix operations, making it computationally efficient for real-time applications.

The Kalman Filter provides a unimodal prediction of the robot's position. It uses a single Gaussian distribution to represent the state of the system, rather than multiple distributions. The algorithm uses the measurement and process models to update the mean and covariance of this Gaussian distribution, providing a single prediction for the state of the system.

In conclusion, the Kalman Filter is a powerful algorithm that provides optimal estimation of the state of a system. Its ability to handle multi-dimensional systems, noisy measurements, and uncertain models make it a popular choice for robotic applications, particularly for localization and tracking.

#### Main Kalman Filter Code:

##### Kalman Filter Function

```python
def kalman_filter(x, P):
    
    for n in range(len(measurements)):
        global u, F, H, R, I
        
        # measurement update
        Z = matrix([[measurements[n]]])
        y = Z - (H*x)
        S = H* P * H.transpose() + R
        K = P * H.transpose() * S.inverse()
        x = x + (K * y)
        P = (I - (K * H)) * P
        
        # prediction
        x = (F * x) + u
        P = F* P *F.transpose()
    return x,P

```

##### Test

```python
measurements = [1, 2, 3]

x = matrix([[0.], [0.]]) # initial state (location and velocity)
P = matrix([[1000., 0.], [0., 1000.]]) # initial uncertainty
u = matrix([[0.], [0.]]) # external motion
F = matrix([[1., 1.], [0, 1.]]) # next state function
H = matrix([[1., 0.]]) # measurement function
R = matrix([[1.]]) # measurement uncertainty
I = matrix([[1., 0.], [0., 1.]]) # identity matrix

result = kalman_filter(x, P)
print('x: ', result[0])
print('P: ', result[1])
# output should be:
# x: [[3.9996664447958645], [0.9999998335552873]]
# P: [[2.3318904241194827, 0.9991676099921091], [0.9991676099921067, 0.49950058263974184]]
```

The main function kalman_filter takes two inputs: x and P. x represents the initial state of the robot, which includes its location and velocity, and P represents the initial uncertainty about the state of the robot.

The function first initializes several variables, including the state transition function F, the measurement function H, the measurement uncertainty R, and the identity matrix I. Then, for each measurement, the function performs a measurement update and a prediction.

In the measurement update step, the function uses the measurement and the measurement function H to calculate the difference between the measurement and the current estimate of the robot's state. This difference is used to calculate the Kalman gain K, which is used to update the state estimate and its uncertainty.

In the prediction step, the function uses the state transition function F to predict the next state of the robot based on its current state and the external motion u. The uncertainty of the predicted state is also updated.

The result of the kalman_filter function is a tuple containing the final estimate of the robot's state and its uncertainty.

The code also includes a test case, which sets the initial state, the initial uncertainty, and the measurement data, and calls the kalman_filter function. The output should be the final estimate of the robot's state and its uncertainty.

#### Animation
##### One Dimensional Example
![kal-one-d](/docs/images/kalman-localization_1D.gif "anim kal one")
##### Two Dimensional Example
![kal-two-d](/docs/images/kalman-localization_2D.gif "anim kal two")

### Particle Filters
## Motion Planning/Search Algorithms
### Breadth First Search Algorithm
### A* Search Algorithm
### Dynamic Programming
## PID Control
### Smoothing
## SLAM
