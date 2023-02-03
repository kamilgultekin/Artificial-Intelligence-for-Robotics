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
### Particle Filters
## Motion Planning/Search Algorithms
### Breadth First Search Algorithm
### A* Search Algorithm
### Dynamic Programming
## PID Control
### Smoothing
## SLAM
