# Computational Intelligence 2024 Project - CI2024_project-work
## Symbolic Regression
This project addresses a symbolic regression task using Genetic Programming (GP). The objective is to automatically evolve mathematical expressions that accurately fit a set of input-output data points.  
    
_Notes:_  
   - _Logs of generations will not be present in the github folder because I used kaggle-servers to run all_  
   - _Problem 8 was very slow so i decided to set less generations, but it was still improving_

## Strategy
An initial population of random formulas is generated and iteratively improved over several generations using genetic operations and other strategies. In detail:
1. Generate initial population
2. Evaluate the fitness for each formula, then select with Elitism the best formulas
3. Parents are selected via tournament selection
4. Finally we apply our genetic operators:
   - Crossover ( alone, with 30% of probability )
   - Mutation ( alone, with 60% of probabilty )
   - Both Crossover, then mutation ( 10% of probability )
5. The algorithm also ensure that depth limit and the presence of all variables are guaranteed by auxiliary functions
6. Random injection is performed each (n_generations/10) generations, to avoid fast convergence
7. The algorithm repeat the process untill a new population with the same size of population is obtained, then copy into population and iterates over the remaining generations

## Some details about specific decision made during the design

In this section i want to report details for some function, starting from the class Node to the Genetic Program design aspects.

### Class Node - evaluate function

In evaluate, is_invalid handle invalid results (NaN, inf), returning np.inf that will be then convert into a very large score (1e100) in the GP **evaluate_fitness** function.  
Moreover, there are extra control to avoid overflow with most of operators:
- close to zero right-value for divisions
- clipping value for exp() before 700 ( since is close to the max value before overflow )
- domain check for sqrt and log  
Even if this strategy makes the algorithm slower, it also increase robustness to strange mutations

```
def evaluate(self, variable_values):
        def is_invalid(val):
            return not np.isfinite(val)
        if self.value in operators:
            left_value = self.left.evaluate(variable_values) if self.left else 0
            if is_invalid(left_value):
                return np.inf
            right_value = self.right.evaluate(variable_values) if self.right else 0
            if is_invalid(right_value):
                return np.inf
            if self.value == '+':
                result = left_value + right_value
            elif self.value == '-':
                result = left_value - right_value
            elif self.value == '*':
                result = left_value * right_value
            elif self.value == '/':
                if abs(right_value) < 1e-10: 
                    return np.inf
                result = left_value / right_value
            return result if np.isfinite(result) else np.inf
        elif self.value in functions:
            f_value = self.right.evaluate(variable_values) if self.right else 0
            if is_invalid(f_value):
                return np.inf
            try:
                if self.value == 'np.sin':
                    result = np.sin(f_value)
                elif self.value == 'np.cos':
                    result = np.cos(f_value)
                elif self.value == 'np.exp':
                    if f_value > 700:
                        return np.inf
                    result = np.exp(f_value)
                elif self.value == 'np.log':
                    if f_value <= 0:
                        return np.inf
                    result = np.log(f_value)
                elif self.value == 'np.sqrt':
                    if f_value < 0:
                        return np.inf
                    result = np.sqrt(f_value)
                elif self.value == 'np.abs':
                    result = np.abs(f_value)
                else:
                    return np.inf
                return result if np.isfinite(result) else np.inf
            except (OverflowError, ValueError, RuntimeWarning):
                return np.inf
        elif self.value.startswith('x'):
            try:     
                idx = int(self.value[2:-1])
                val = variable_values.get(f"x[{idx}]", np.inf)
                return val if np.isfinite(val) else np.inf
            except (ValueError, KeyError):
                return np.inf
        else:
            try:
                val = float(self.value)
                return val if np.isfinite(val) else np.inf
            except ValueError:
                return np.inf
```

### Genetic Program functions - mutation function

If mutation happens (20% chance), the approach is subtree mutation or 'point' mutation:
- subtree mutation: we select a random node of the three and we generate a new formula with the same depth (to avoid excessive growth), then we replace the random node with the new one
- point mutation: we replace only the 'top' node ( operator, function or leaf )

```
def mutation(formula, n_var, mutation_rate = 0.2):
    if random.random() < mutation_rate:
        all_nodes = get_all_nodes(formula)
        if random.random() < 0.5:
            node_to_replace = random.choice(all_nodes)
            new_sub_tree = generate_formula(random.randint(0, node_to_replace.depth()), n_var)
            node_to_replace.value, node_to_replace.left, node_to_replace.right = new_sub_tree.value, new_sub_tree.left, new_sub_tree.right
        else:
            node_to_mutate = random.choice(all_nodes)
            if node_to_mutate.is_operator():
                new_operator = random.choice(operators)
                node_to_mutate.value = new_operator
            elif node_to_mutate.is_function():
                new_function = random.choice(functions)
                node_to_mutate.value = new_function
            elif node_to_mutate.is_leaf():
                if random.random() < 0.5:
                    index = random.randint(0, n_var - 1)
                    node_to_mutate.value = f"x[{index}]"
                else:
                    node_to_mutate.value = str(round(random.uniform(-10, 10), 3))
    return formula
```