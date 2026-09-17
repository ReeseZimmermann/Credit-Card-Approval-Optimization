# Credit-Card-Approval-Optimization

A Python-based artificial intelligence project comparing **hill-climbing** and **genetic algorithms** for optimizing a simple credit-card approval prediction model.

The project explores how two different search strategies navigate the same solution space and minimize prediction error over a credit-card approval dataset.

## Overview

The goal of this project is to find an optimal six-dimensional weight vector for a simple linear prediction model.

Each candidate solution contains six weights:

```text
w = [w1, w2, w3, w4, w5, w6]
```

where every weight is restricted to:

```text
{-1, +1}
```

The model evaluates each credit-card applicant using six features and searches for the weight combination that minimizes **Mean Squared Error (MSE)** between predicted and actual approval values.

Two optimization techniques are compared:

* **Hill Climbing**
* **Genetic Algorithm**

This project demonstrates fundamental artificial intelligence concepts including local search, evolutionary optimization, fitness evaluation, mutation, crossover, and solution-space exploration.

---

## Dataset

The dataset contains **339 credit-card applicant records**.

The six features used by the optimization algorithms are:

| Feature       | Encoding / Type            |
| ------------- | -------------------------- |
| Gender        | Male = 1, Female = 0       |
| CarOwner      | Yes = 1, No = 0            |
| PropertyOwner | Yes = 1, No = 0            |
| #Children     | Integer                    |
| WorkPhone     | Integer / binary indicator |
| Email_ID      | Integer / binary indicator |

The prediction target is:

```text
CreditApprove
```

The applicant identifier, `Ind_ID`, is retained in the dataset but is not used as a prediction feature.

---

## Prediction Model

For an applicant feature vector `x` and candidate weight vector `w`, the model calculates:

```text
f(x) = w · x
```

or:

```text
f(x) = Σ(wj × xj)
```

The quality of a candidate weight vector is evaluated using **Mean Squared Error**:

```text
er(w) = (1/n) Σ(f(xi) - yi)²
```

where:

* `w` = candidate weight vector
* `xi` = applicant feature vector
* `yi` = actual credit approval value
* `f(xi)` = predicted value
* `n` = number of applicant records

The optimization objective is to:

```text
Minimize er(w)
```

---

## Hill-Climbing Algorithm

The hill-climbing implementation performs a local search over the possible weight configurations.

### Process

1. Begin with an initial six-element weight vector.
2. Generate every neighbor that differs by exactly one weight.
3. Calculate the MSE of each neighbor.
4. Move to the neighbor with the lowest error if it improves the current solution.
5. Repeat until no single weight change produces a better result.

A neighbor is created by flipping exactly one value:

```text
Current:
[-1, -1, -1, -1, -1, -1]

Example neighbor:
[ 1, -1, -1, -1, -1, -1]
```

Because there are six weights, each solution has six **Hamming-distance-one neighbors**.

The algorithm terminates when:

* no neighboring solution improves the error, or
* the maximum number of rounds is reached.

### Hill-Climbing Result

Starting from:

```text
[-1, -1, -1, -1, -1, -1]
```

the algorithm reduced the error as follows:

| Round |   Best Error |
| ----: | -----------: |
|     0 |     8.150442 |
|     1 |     3.654867 |
|     2 |     1.790560 |
|     3 |     1.365782 |
|     4 | **1.271386** |

Final weight vector:

```text
[1, -1, 1, -1, 1, 1]
```

Final MSE:

```text
1.2713864306784661
```

---

## Genetic Algorithm

The genetic algorithm uses an evolutionary approach rather than following a single local-search path.

Each six-element weight vector represents a **chromosome**.

### Genetic Algorithm Process

```text
Initialize Population
        ↓
Evaluate Fitness
        ↓
Roulette-Wheel Selection
        ↓
Midpoint Crossover
        ↓
Mutation
        ↓
Create Next Generation
        ↓
Repeat
```

### Fitness

Because the goal is to minimize error, fitness is calculated as:

```text
fitness(w) = e^(-er(w))
```

A lower MSE therefore produces a higher fitness value.

### Selection

The program uses **fitness-proportionate roulette-wheel selection**.

Candidate solutions with greater fitness have a higher probability of being selected as parents.

### Crossover

The six-element chromosome is divided at its midpoint.

Example:

```text
Parent 1:
[A, B, C | D, E, F]

Parent 2:
[G, H, I | J, K, L]

Children:
[A, B, C | J, K, L]
[G, H, I | D, E, F]
```

### Mutation

Each gene has a probability of changing signs:

```text
1 → -1

-1 → 1
```

Mutation helps maintain diversity within the population and allows the algorithm to explore new areas of the solution space.

### Configuration Used

The current implementation runs with:

```text
Population Size: 8
Generations: 50
Mutation Rate: 0.20
Random Seed: 2025
Elitism: 0
```

### Genetic Algorithm Result

The genetic algorithm reached the same best solution:

```text
[1, -1, 1, -1, 1, 1]
```

with:

```text
MSE = 1.2713864306784661
```

The best-so-far error reached approximately `1.271386` by **generation 5** and remained at that value through the remainder of the 50-generation run.

---

## Algorithm Comparison

| Characteristic      | Hill Climbing         | Genetic Algorithm               |
| ------------------- | --------------------- | ------------------------------- |
| Search Strategy     | Local search          | Population-based search         |
| Candidate Solutions | One current solution  | Multiple solutions              |
| Neighbor Generation | Flip one weight       | Crossover + mutation            |
| Selection           | Lowest-error neighbor | Roulette-wheel selection        |
| Randomness          | Initial state / seed  | Population, selection, mutation |
| Main Risk           | Local optimum         | Additional computational search |
| Best MSE Found      | **1.271386**          | **1.271386**                    |
| Best Weight Vector  | `[1,-1,1,-1,1,1]`     | `[1,-1,1,-1,1,1]`               |

Both approaches ultimately identified the same best weight configuration for this search space, but they reached the result using fundamentally different optimization strategies.

---

## Project Structure

```text
Credit-Card-Approval-Optimization/
│
├── README.md
├── .gitignore
│
├── CreditCard.csv
│
├── hw2_local.py
├── hw2_genetic.py
│
├── hill_climb_results.csv
└── genetic_errors.csv
```

### Files

**`CreditCard.csv`**
Dataset containing applicant features and credit-approval outcomes.

**`hw2_local.py`**
Implements the hill-climbing optimization algorithm.

**`hw2_genetic.py`**
Implements the genetic algorithm.

**`hill_climb_results.csv`**
Stores the error after each accepted hill-climbing improvement.

**`genetic_errors.csv`**
Stores the best-so-far error for each genetic-algorithm generation.

---

## Running the Project

### Requirements

* Python 3
* No external Python libraries are required.

The implementation uses Python standard-library modules including:

```text
csv
math
random
```

### Clone the Repository

```bash
git clone https://github.com/ReeseZimmermann/Credit-Card-Approval-Optimization.git
```

Navigate into the project:

```bash
cd Credit-Card-Approval-Optimization
```

### Run Hill Climbing

```bash
python hw2_local.py
```

The program prints the best weight vector and MSE and generates:

```text
hill_climb_results.csv
```

### Run the Genetic Algorithm

```bash
python hw2_genetic.py
```

The program prints the best weight vector and MSE and generates:

```text
genetic_errors.csv
```

---

## Key Concepts Demonstrated

This project provided hands-on experience with:

* Artificial intelligence search algorithms
* Hill climbing
* Genetic algorithms
* Local vs. population-based optimization
* Fitness functions
* Roulette-wheel selection
* Crossover
* Mutation
* Hamming-distance neighborhoods
* Mean Squared Error
* Feature encoding
* CSV data processing
* Reproducible randomized algorithms
* Algorithm performance comparison
* Convergence tracking

---

## What I Learned

This project helped demonstrate how different AI optimization strategies can solve the same search problem in very different ways.

Hill climbing provides a straightforward and computationally efficient method for improving a single candidate solution, but its search is limited by the quality of neighboring solutions.

The genetic algorithm instead maintains a population of candidates and introduces variation through crossover and mutation, allowing it to explore the solution space differently.

Implementing both approaches from scratch also provided experience working directly with the mechanics behind optimization algorithms rather than relying on a machine-learning library to perform the search automatically.

---

## Limitations

This project is an **educational demonstration of AI optimization techniques**, not a production credit-risk or lending system.

The model intentionally uses a small feature set, a constrained `{-1, +1}` weight space, and a simplified linear prediction function to make the behavior of the optimization algorithms easier to analyze.

Real-world credit decision systems require substantially more rigorous modeling, validation, fairness analysis, regulatory compliance, security, and risk controls.

---

## Future Improvements

Potential extensions include:

* Compare multiple random hill-climbing starting points
* Experiment with different genetic-algorithm population sizes
* Compare mutation rates and crossover strategies
* Enable and evaluate elitism
* Plot convergence curves for both algorithms
* Add automated unit tests
* Separate data, source code, and result files into dedicated directories
* Compare the custom optimization methods with additional search techniques
* Evaluate the model using additional performance metrics

---

## Author

**Reese Zimmermann**

Computer Science

University of Oklahoma
