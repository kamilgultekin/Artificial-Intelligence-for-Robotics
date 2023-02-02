# Questions

1. You roll two fair 6-sided dice. One of the dice shows 6, but the other falls off the table and you can not see it. What is the probability that both dice show 6?
    
    - Given that the first die shows 6, the probability that the second die also shows 6 is 1/6.

2. After years of observations, you have found out that if it rains on a given day, there is a 60 % chance that it will rain on the next day too. If it is not raining, the chance of rain on the next day is only 25 %. The weather forecast for Friday predicts the chance of rain is 75 %. What is the probability that at least one day of the weekend will have no rain? "Weekend" in this case refers to Saturday and Sunday - do not include Friday's weather.
    
    - To find the probability that at least one day of the weekend will have no rain, we need to find the probability that both Saturday and Sunday will have rain, and then subtract that from 1.

    - Let's assume that the probability of rain on Saturday is P(rain on Saturday). If it is raining on Friday, then the probability of rain on Saturday is 60%. If it is not raining on Friday, then the probability of rain on Saturday is 25%. We can use Bayes' rule to calculate P(rain on Saturday | raining on Friday) as follows:

    - P(rain on Saturday | raining on Friday) = P(raining on Friday | rain on Saturday) * P(rain on Saturday) / P(raining on Friday) where P(raining on Friday) = 0.75. We know that P(raining on Friday | rain on Saturday) = 0.6. To calculate P(rain on Saturday), we can use the law of total probability:

    - P(rain on Saturday) = P(rain on Saturday | raining on Friday) * P(raining on Friday) + P(rain on Saturday | not raining on Friday) * P(not raining on Friday) where P(not raining on Friday) = 1 - P(raining on Friday) = 0.25. We also know that P(rain on Saturday | not raining on Friday) = 0.25.

    - Substituting these values into the formula, we get:

    - P(rain on Saturday) = 0.6 * 0.75 + 0.25 * 0.25 = 0.55

    - So, the probability of rain on Saturday is 55%. We can calculate the probability of rain on Sunday using the same method, by assuming that the probability of rain on Sunday is P(rain on Sunday).

    - The probability that both Saturday and Sunday will have rain is P(rain on Saturday) * P(rain on Sunday | rain on Saturday). Since we already calculated P(rain on Saturday), we can calculate P(rain on Sunday | rain on Saturday) by using Bayes' rule and the fact that the probability of rain on the next day given that it is raining is 0.6:

    - P(rain on Sunday | rain on Saturday) = 0.6

    - So, P(rain on Saturday) * P(rain on Sunday | rain on Saturday) = 0.55 * 0.6 = 0.33.

    - Finally, the probability that at least one day of the weekend will have no rain is 1 - P(both Saturday and Sunday will have rain) = 1 - 0.33 = 0.67.

    - So, there is a 67% chance that at least one day of the weekend will have no rain.

1. You have the following world: 'green-green-red-green-red' and you start with a uniform distribution. Update the probablities to reflect a measurement of 'red' if there is a measurement error of 0.1. 

    ```python
    p=[0.2, 0.2, 0.2, 0.2, 0.2]
    world=['green', 'green', 'red', 'green', 'red']
    Z = 'red'
    pHit = 0.9
    pMiss = 0.1

    def sense(p, Z):
        #
        q = []
        for i in range(len(p)):
            hit = (Z == world[i])
            q.append(round(p[i]*(hit * pHit + (1-hit) * pMiss),3))
        return q

    q = sense(p,Z)
    print('Probality:\n',q)
    >>> Probality:
    >>> [0.02, 0.02, 0.18, 0.02, 0.18]
    ################################################################
    sum = 0
    for i in range(len(q)):
        sum = sum + q[i]
    for i in range(len(q)):
        q[i] = round(q[i]/sum,3)
    print('Normalized Probability:\n',q)
    >>>> Normalized Probability:
    >>>> [0.048, 0.048, 0.429, 0.048, 0.429]
    ```

2. You have the following world: 'green-green-red-green-red' and you start with a uniform distribution. In the world robot senses 'red' the moves right and senses again. Whr world is not cyclic, so if the robor hits a wall, it satys in the same position. Measurement error is 0.1, move is exact. What is the probablity that it will sense red again.
    - 10.9/21.0

3. The robots location is characterised by a gaussian distribution which the values mean = 1 and var=1.You perform a measurement with a variance 1 and the value of the measurement is 3. What are the values of the resulting mean and variance. 

    - Let's assume that the initial mean and variance of the robot's location are represented by mu and sigma^2, respectively. The measurement with variance R and value z can be incorporated into the estimate of the robot's location by using Bayes' rule to update the mean and variance of the Gaussian distribution.

    The updated mean is given by:

    ```scss
    mu_bar = (sigma^2 * z + R * mu) / (sigma^2 + R)
    ```

    And the updated variance is given by:

    ```scss
    sigma_bar^2 = 1 / (1 / sigma^2 + 1 / R)
    ```

    Substituting the given values into these equations, we get:

    ```scss
    mu_bar = (1 * 3 + 1 * 1) / (1 + 1) = 2
    sigma_bar^2 = 1 / (1 / 1 + 1 / 1) = 1/2
    ```

    So the resulting mean is 2 and the variance is 1/2.

4. You can model a car as a Gaussian distribution. If the car undergoes motion, which of the following statements are true:

    - [x] The expected location, mu, changes
    - [ ] The variance, sigma^2, gets smaller
    - [x] The variance, sigma^2, gets larger
    - [ ] none of above

5. Check the following statements about particle filter if the are true:

    - [ ] They can only be applied to discrete state space
    - [ ] they can scale linearly with dimensionality of the state space
    - [x] they can represent multi modal distributions
    - [x] they are usually easy to implement
    - [ ] they can not be used for more than 2 dimensions
    - [ ] none of above

6. | p1 - green |  green |
    | - | - |
    |p2 - yellow |p2 - yello |

P(sense green) = 0.8, P(sense yellow) = 0.2, P(sense green) = 0.3, P(sense yellow) = 0.7

Probablity to sense 'green' on top row is 0.8, Probability to sense 'yellow' on botom row is 0.7
There are 3 particles p1, p2 and p3. You observe yellow.

    What is the normalized weight of p1 = 0.125
    What is the normalized weight of p3 = 0.4375

1. | S | 6 | 5  | 4 | 3 |
    | - | - | -- | - | - |
    | 6 | - | 4 | 3 | 3 |
    | 5 | - | 3 | 2 | 1 |
    | 4 | 3 | 2 | 1 | G |

You need to get from start S to goal G. There is no cost of turnig. Is the heuristic admissble if:
    
- [ ] the cost of motion is 1 and diagonal move is not allowed.
- [ ] the cost of motion is 1 and diagonal move is allowed.
- [x] the cost of motion is 2 or more
- [ ] none of above

2. |  | C |   | A |  |
    | - | - | -- | - | - |
    |  | x |  | x |  |
    |  | B |  | D |  |
    | x | x |  | x | x |
    | x | x | Car | x | x |

You have a Car, facing north, a road (empty cell) and obstacles (x). 4 goal states A,B,C,D for the car. The car can move forward or turn right, which both have a cost of 1, or it can turn left which has a cost of 14. For example, the cost to move to goal state D is 4. When the car turns, it stays in the same cell ad it can not turn several times in a row. It is not allowed to drive into obstacles. What is the cost of the optimal policy:

    - for goal state A = 6
    - for goal state A = 14
    - for goal state A = 19

3. You are developing a PID controller for controlling the speed of your car. Imagine that you drive on a busy highway and suddenly notice that the car in front of you has stopped. You car decides to break and stop too. MArk the most likely outcome:

- [ ] You may crach in the car in front of if P is too big
- [x] You may crach in the car in front of if P is too small
- [x] car behind may crach in you if P is too big
- [ ] car behind may crach in you if P is too small
- [ ] all will be fine if P is too big
- [ ] all will be fine if P is too small