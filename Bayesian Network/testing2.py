import operator
from collections import defaultdict
from typing import List, Set

from bnetbase import Variable, Factor, BN
import csv
import itertools
import functools
from collections import defaultdict, Counter
DEBUG = False


# def test_factor(f: Factor):
#     # if DEBUG:
#     # if len(f.scope) != 0:
#     #     s = functools.reduce(operator.mul, [v.domain_size() for v in f.scope])
#     #         # print(f)
#     #     assert len(f.values) <= s
#     pass

# def multiply(factors):
#     """
#     Multiplies a list of Factor objects into a single Factor.
#
#     :param factors: List of Factor objects to be multiplied.
#     :return: A new Factor object representing the product of all input factors.
#     """
#     all_variables = []
#     for factor in factors:
#         for variable in factor.scope:
#             if variable not in all_variables:
#                 all_variables.append(variable)
#     variable_domains = [variable.domain() for variable in all_variables]
#     new_factor_entries = []
#     for values in itertools.product(*variable_domains):
#         assignment = dict(zip(all_variables, values))
#         product_value = 1
#         for factor in factors:
#             factor_assignment = tuple(assignment[var] for var in factor.scope)
#             factor_value = factor.get_value(factor_assignment)
#             product_value *= factor_value
#         new_factor_entries.append(list(values) + [product_value])
#     new_factor_name = ' * '.join([factor.name for factor in factors])
#     result_factor = Factor(f'({new_factor_name})', all_variables)
#     result_factor.add_values(new_factor_entries)
#     return result_factor



def multiply(factor_list) -> Factor:
    """
    Multiply a list of factors together.
    Do not modify any of the input factors.

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    """
    combined_f = factor_list[0]
    for factor in factor_list[1:]:
        combined_f = multiply2(combined_f, factor)
    return combined_f


def multiply2(factor_one: Factor, factor_two: Factor) -> Factor:
    """
    Multiply two factors together.

    :param factor_one: the first Factor object.
    :param factor_two: the second Factor object.
    :return: a new Factor object resulting from multiplying the two input factors.
    """
    # Merge the scopes
    new_scope = list(factor_one.scope)
    for var in factor_two.scope:
        if var not in new_scope:
            new_scope.append(var)
    domains = [var.domain() for var in new_scope]
    new_assignments = []
    for assignment in itertools.product(*domains):
        assignment_dict = dict(zip(new_scope, assignment))
        f1_assignment = tuple(assignment_dict[var] for var in factor_one.scope)
        f2_assignment = tuple(assignment_dict[var] for var in factor_two.scope)
        f1_value = factor_one.get_value(f1_assignment)
        f2_value = factor_two.get_value(f2_assignment)
        multiplied_value = f1_value * f2_value
        new_assignments.append(list(assignment) + [multiplied_value])
    combined_factor = Factor(f'({factor_one.name} * {factor_two.name})', new_scope)
    combined_factor.add_values(new_assignments)

    return combined_factor



# def restrict(f: Factor, var: Variable, value) -> Factor:
#     """
#     f is a factor, var is a Variable, and value is a value from var.domain.
#     Return a new factor that is the restriction of f by this var = value.
#     Don't change f! If f has only one variable its restriction yields a
#     constant factor.
#     @return a factor
#     """
#     values, scope = f.values, f.get_scope()
#     var_idx = scope.index(var)
#     value_idx = var.value_index(value)
#     # if len(scope) == 1:
#     #     restricted_f = Factor(f'r({f.name}, {var.name}={value})', scope)
#     #     restricted_f.values = [f.values[value_idx]]
#     #     return restricted_f
#     # units
#     unit = 1
#     for i in range(var_idx):
#         unit *= scope[i].domain_size()
#     unit_size = len(values) // unit
#     # gap between values of var
#     gap = 1
#     for i in range(var_idx + 1, len(scope)):
#         gap *= scope[i].domain_size()
#     start = value_idx * gap
#     new_values = []
#     for i in range(0, len(values), unit_size):
#         new_values.extend(values[i + j] for j in range(start, start + gap))
#     scope.pop(var_idx)
#     restricted_f = Factor(f'r({f.name}, {var.name}={value})', scope)
#     restricted_f.values = new_values
#     # if DEBUG:
#     #     test_factor(restricted_f)
#     return restricted_f

# def restrict(factor: Factor, variable: Variable, value) -> Factor:
#     """
#     Returns a new factor where the specified variable is restricted to a given value.
#     The new factor's scope excludes the restricted variable.
#     """
#     # Get the scope and values from the original factor
#     original_scope = factor.get_scope()
#     original_values = factor.values
#
#     # Find the index of the variable to restrict
#     var_index = original_scope.index(variable)
#     # Get the index of the value in the variable's domain
#     value_index = variable.value_index(value)
#
#     # Remove the restricted variable from the scope to create the new scope
#     new_scope = original_scope[:var_index] + original_scope[var_index + 1:]
#
#     # Calculate the sizes needed to navigate through the factor's values
#     # Compute the number of consecutive values before the variable changes
#     pre_var_size = 1
#     for var in original_scope[:var_index]:
#         pre_var_size *= var.domain_size()
#
#     # Compute the number of values to skip after each value of the variable
#     post_var_size = 1
#     for var in original_scope[var_index + 1:]:
#         post_var_size *= var.domain_size()
#
#     # Compute the total number of values in one complete cycle of the variable
#     var_cycle_size = variable.domain_size() * post_var_size
#
#     # Create a list to hold the new values after restriction
#     restricted_values = []
#
#     # Iterate over the values, selecting those where variable == value
#     for start in range(0, len(original_values), var_cycle_size):
#         # Calculate the index range corresponding to the desired value
#         value_start = start + value_index * post_var_size
#         value_end = value_start + post_var_size
#         # Append the relevant values to the restricted values
#         restricted_values.extend(original_values[value_start:value_end])
#
#     # Create the new restricted factor
#     restricted_factor = Factor(f"r({factor.name}, {variable.name}={value})", new_scope)
#     restricted_factor.values = restricted_values
#
#     return restricted_factor


# def sum_out(f: Factor, var: Variable) -> Factor:
#     """
#     f is a factor, var is a Variable.
#     Return a new factor that is the result of summing var out of f, by summing
#     the function generated by the product over all values of var.
#     @return a factor
#     """
#     values, scope = f.values, f.get_scope()
#     var_idx = scope.index(var)
#     # gap between values of var
#     gap = 1
#     for i in range(var_idx + 1, len(scope)):
#         gap *= scope[i].domain_size()
#     new_values = []
#     offset = 0
#     while offset < len(values):
#         # sum out var for one unit
#         new_values.extend(
#             sum(values[i + j * gap] for j in range(var.domain_size()))
#             for i in range(offset, gap + offset)
#         )
#         offset += gap * var.domain_size()
#     scope.pop(var_idx)
#     new_f = Factor(f's({f.name}, {var.name})', scope)
#     new_f.values = new_values
#     # if DEBUG:
#     #     test_factor(new_f)
#     return new_f

def restrict(factor: Factor, variable: Variable, value) -> Factor:
    """
    Returns a new factor where the specified variable is fixed to a given value.
    The new factor's scope excludes the restricted variable.
    """
    original_scope = factor.get_scope()
    new_scope = [var for var in original_scope if var != variable]
    new_assignments = []
    assignments = itertools.product(*[var.domain() for var in original_scope])
    for assignment in assignments:
        assignment_dict = dict(zip(original_scope, assignment))
        if assignment_dict[variable] == value:
            new_assignment = tuple(assignment_dict[var] for var in new_scope)
            factor_value = factor.get_value(assignment)
            new_assignments.append(list(new_assignment) + [factor_value])
    restricted_factor = Factor(f"r({factor.name}, {variable.name}={value})", new_scope)
    restricted_factor.add_values(new_assignments)
    return restricted_factor

def sum_out(factor: Factor, variable: Variable) -> Factor:
    """
    Returns a new factor with the specified variable summed out (marginalized).
    The new factor's scope excludes the variable being summed out.
    """
    original_scope = factor.get_scope()
    new_scope = [var for var in original_scope if var != variable]
    sum_table = {}
    assignments = itertools.product(*[var.domain() for var in original_scope])
    for assignment in assignments:
        assignment_dict = dict(zip(original_scope, assignment))
        new_assignment = tuple(assignment_dict[var] for var in new_scope)
        factor_value = factor.get_value(assignment)
        if new_assignment in sum_table:
            sum_table[new_assignment] += factor_value
        else:
            sum_table[new_assignment] = factor_value
    new_assignments = [list(assignment) + [value] for assignment, value in sum_table.items()]
    summed_factor = Factor(f"s({factor.name}, {variable.name})", new_scope)
    summed_factor.add_values(new_assignments)
    return summed_factor



def normalize(factor):
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object.
    :return: a new Factor object resulting from normalizing factor.
    '''
    # raise NotImplementedError
    # total_sum = sum(factor.values)
    # normalized_values = [value / total_sum for value in factor.values]
    # new_factor = Factor(factor.name, factor.scope)
    # new_factor.values = normalized_values
    # return new_factor
    normal = sum(factor.values)
    nums = []
    for num in factor.values:
        nums.append(num / normal)
    factor2 = Factor(factor.name, factor.scope)
    factor2.values = nums
    return factor2

def ve(bayes_net, query_var, evidence_vars):
    """
    Performs variable elimination on the given Bayesian network to compute
    the distribution over the query variable given the evidence.

    Args:
        bayes_net: The Bayesian network object.
        query_var: The variable whose distribution we want to compute.
        evidence_vars: A list of Variable objects with evidence set.

    Returns:
        A normalized Factor representing the distribution over the query variable.
    """
    factor_list = bayes_net.factors()
    factor_list = apply_evidence(factor_list, evidence_vars)
    hidden_vars = get_hidden_variables(factor_list, query_var, evidence_vars)
    elim_order = get_elimination_order(factor_list, hidden_vars)
    for var in elim_order:
        factor_list = eliminate_var(factor_list, var)
    result_factor = multiply(factor_list)
    normalized_factor = normalize(result_factor)

    return normalized_factor

def apply_evidence(factors, evidence_vars):
    """
    Restricts factors based on the provided evidence.
    """
    updated_factors = []
    for factor in factors:
        restricted_factor = factor
        for ev_var in evidence_vars:
            if ev_var in factor.get_scope():
                ev_value = ev_var.get_evidence()
                restricted_factor = restrict(restricted_factor, ev_var, ev_value)
        updated_factors.append(restricted_factor)
    return updated_factors

def get_hidden_variables(factors, query_var, evidence_vars):
    """
    Identifies hidden variables that need to be eliminated.
    """
    all_vars = set(var for factor in factors for var in factor.get_scope())
    hidden_vars = all_vars - {query_var} - set(evidence_vars)
    return hidden_vars

def get_elimination_order(factors, hidden_vars):
    """
    Determines the elimination order using the min-degree heuristic.
    """
    elim_order = []
    scopes = [set(factor.get_scope()) for factor in factors]

    while hidden_vars:
        var_degrees = {}
        for var in hidden_vars:
            degree = sum(1 for scope in scopes if var in scope)
            var_degrees[var] = degree
        min_var = min(var_degrees, key=var_degrees.get)
        elim_order.append(min_var)
        hidden_vars.remove(min_var)
        related_scopes = [scope for scope in scopes if min_var in scope]
        new_scope = set().union(*related_scopes) - {min_var}
        scopes = [scope for scope in scopes if min_var not in scope]
        if new_scope:
            scopes.append(new_scope)

    return elim_order

def eliminate_var(factors, variable):
    """
    Eliminates a variable from the list of factors.
    """
    factors_with_var = [f for f in factors if variable in f.get_scope()]
    if not factors_with_var:
        return factors
    combined_factor = multiply(factors_with_var)
    summed_factor = sum_out(combined_factor, variable)
    remaining_factors = [f for f in factors if f not in factors_with_var]
    remaining_factors.append(summed_factor)
    return remaining_factors

# def ve(bayes_net: BN, query_variable: Variable, evidence_variables: List[Variable]) -> Factor:
#     """
#     Performs variable elimination on the given Bayesian network to compute
#     the distribution over the query variable given evidence.
#
#     Args:
#         bayes_net: The Bayesian network object.
#         query_variable: The variable whose distribution we want to compute.
#         evidence_variables: A list of Variable objects with evidence set.
#
#     Returns:
#         A normalized Factor representing the distribution over the query variable.
#     """
#     # Step 1: Retrieve all factors from the Bayesian network
#     factors = bayes_net.factors()
#
#     # Step 2: Restrict factors based on the evidence
#     for evidence_var in evidence_variables:
#         evidence_value = evidence_var.get_evidence()
#         for idx in range(len(factors)):
#             factor = factors[idx]
#             if evidence_var in factor.get_scope():
#                 # Restrict the factor to the evidence value
#                 factors[idx] = restrict(factor, evidence_var, evidence_value)
#
#     # Step 3: Determine the elimination order using the min-fill heuristic
#     # Collect all variables in the factors' scopes, excluding the query variable
#     scopes = [set(factor.get_scope()) for factor in factors]
#     all_variables = set().union(*scopes)
#     hidden_variables = all_variables - {query_variable}
#
#     elimination_order = []
#
#     while hidden_variables:
#         min_fill_variable = None
#         min_fill_scope = None
#         min_fill_size = float('inf')
#         min_related_scopes = []
#
#         # Evaluate each hidden variable to find the one with minimal fill-in
#         for var in hidden_variables:
#             # Find all scopes containing the variable
#             related_scopes = [scope for scope in scopes if var in scope]
#             # Compute the union of these scopes (excluding the variable to eliminate)
#             combined_scope = set().union(*related_scopes) - {var}
#             fill_size = len(combined_scope)
#
#             # Select the variable with the smallest resulting scope
#             if fill_size < min_fill_size:
#                 min_fill_variable = var
#                 min_fill_scope = combined_scope
#                 min_fill_size = fill_size
#                 min_related_scopes = related_scopes
#
#         # Update the elimination order and remove the variable from hidden_variables
#         elimination_order.append(min_fill_variable)
#         hidden_variables.remove(min_fill_variable)
#
#         # Update the scopes by removing related scopes and adding the new combined scope
#         scopes = [scope for scope in scopes if scope not in min_related_scopes]
#         scopes.append(min_fill_scope)
#
#     # Step 4: Eliminate hidden variables according to the elimination order
#     for eliminate_var in elimination_order:
#         # Identify factors that include the variable to eliminate
#         factors_to_multiply = [factor for factor in factors if eliminate_var in factor.get_scope()]
#
#         if not factors_to_multiply:
#             continue  # No factors to process for this variable
#
#         # Multiply all relevant factors together
#         product_factor = multiply(factors_to_multiply)
#
#         # Sum out the variable to eliminate
#         summed_factor = sum_out(product_factor, eliminate_var)
#
#         # Update the factors list: remove old factors and add the new one
#         factors = [factor for factor in factors if factor not in factors_to_multiply]
#         factors.append(summed_factor)
#
#     # Step 5: Multiply the remaining factors
#     final_factor = multiply(factors)
#
#     # Step 6: Normalize the final factor
#     normalized_factor = normalize(final_factor)
#
#     return normalized_factor



# def ve(bn: BN, query_var: Variable, evidence_vars: List[Variable]) -> Factor: #real
#     """
#     Performs variable elimination on the given Bayesian network to compute
#     the distribution over the query variable given evidence.
#
#     Args:
#         bn: The Bayesian network object.
#         query_var: The variable whose distribution we want to compute.
#         evidence_vars: A list of Variable objects with evidence set.
#
#     Returns:
#         A normalized Factor representing the distribution over the query variable.
#     """
#     factor_list = bn.factors()
#     for ev_var in evidence_vars:
#         ev_value = ev_var.get_evidence()
#         for i in range(len(factor_list)):
#             current_factor = factor_list[i]
#             if ev_var in current_factor.get_scope():
#
#                 factor_list[i] = restrict(current_factor, ev_var, ev_value)
#
#     scopes = [set(f.get_scope()) for f in factor_list]
#     all_vars = set().union(*scopes)
#     vars_to_eliminate = all_vars - {query_var}
#
#     elimination_order = []
#
#     while vars_to_eliminate:
#         best_var = None
#         best_scope = None
#         min_fill = float('inf')
#         scopes_to_merge = []
#         for var in vars_to_eliminate:
#             related_scopes = [s for s in scopes if var in s]
#             combined_scope = set().union(*related_scopes) - {var}
#             fill_in_size = len(combined_scope)
#             if fill_in_size < min_fill:
#                 best_var = var
#                 best_scope = combined_scope
#                 min_fill = fill_in_size
#                 scopes_to_merge = related_scopes
#         elimination_order.append(best_var)
#         vars_to_eliminate.remove(best_var)
#         scopes = [s for s in scopes if s not in scopes_to_merge]
#         scopes.append(best_scope)
#     for elim_var in elimination_order:
#         factors_to_process = [f for f in factor_list if elim_var in f.get_scope()]
#         if not factors_to_process:
#             continue
#         multiplied_factor = multiply(factors_to_process)
#         new_factor = sum_out(multiplied_factor, elim_var)
#         factor_list = [f for f in factor_list if f not in factors_to_process]
#         factor_list.append(new_factor)
#     if factor_list:
#         final_factor = multiply(factor_list)
#     else:
#         final_factor = Factor('Unit', [query_var])
#         final_factor.add_values([[value, 1.0] for value in query_var.domain()])
#     normalized_result = normalize(final_factor)
#     return normalized_result



def naive_bayes_model(s) -> BN:
    """
    NaiveBayesModel returns a BN that is a Naive Bayes model that
    represents the joint distribution of value assignments to
    variables in the Adult Dataset from UCI.  Remember a Naive Bayes model
    assumes P(X1, X2,.... XN, Class) can be represented as
    P(X1|Class)*P(X2|Class)* .... *P(XN|Class)*P(Class).
    When you generated your Bayes Net, assume that the values
    in the SALARY column of the dataset are the CLASS that we want to predict.
    @return a BN that is a Naive Bayes model and which represents the Adult
    Dataset.
    """
    # READ IN THE DATA
    input_data = []
    with open(s, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)  # skip header row
        # each row is a list of str
        for row in reader:
            input_data.append(row)

    # DOMAIN INFORMATION REFLECTS ORDER OF COLUMNS IN THE DATA SET
    variable_domains = {
        "Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
        "Education": [
            '<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors',
            'Masters', 'Doctorate'],
        "Occupation": [
            'Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service',
            'Professional'],
        "MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
        "Relationship": [
            'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative',
            'Unmarried'],
        "Race": [
            'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
            'Other'],
        "Gender": ['Male', 'Female'],
        "Country": [
            'North-America', 'South-America', 'Europe', 'Asia', 'Middle-East',
            'Carribean'],
        "Salary": ['<50K', '>=50K']
    }
    # create variables
    WORK = Variable('work', variable_domains['Work'])
    EDUCATION = Variable('education', variable_domains['Education'])
    MARTIAL_STATUS = Variable('martial_status',
                              variable_domains['MaritalStatus'])
    OCCUPATION = Variable('occupation', variable_domains['Occupation'])
    RELATIONSHIP = Variable('relationship', variable_domains['Relationship'])
    RACE = Variable('race', variable_domains['Race'])
    GENDER = Variable('gender', variable_domains['Gender'])
    COUNTRY = Variable('country', variable_domains['Country'])
    SALARY = Variable('salary', variable_domains['Salary'])

    # create factors
    f_work = Factor('work', [WORK, SALARY])
    f_education = Factor('education', [EDUCATION, SALARY])
    f_martial_status = Factor('martial_status', [MARTIAL_STATUS, SALARY])
    f_occupation = Factor('occupation', [OCCUPATION, SALARY])
    f_relationship = Factor('relationship', [RELATIONSHIP, SALARY])
    f_race = Factor('race', [RACE, SALARY])
    f_gender = Factor('gender', [GENDER, SALARY])
    f_country = Factor('country', [COUNTRY, SALARY])
    f_salary = Factor('salary', [SALARY])
    factors = [f_work, f_education, f_martial_status, f_occupation,
               f_relationship, f_race, f_gender, f_country]

    # salary count
    n = len(input_data)
    salary_count = {'<50K': 0., '>=50K': 0.}
    for w, e, m, o, re, ra, g, c, s in input_data:
        salary_count[s] += 1
    # other
    other_count = [defaultdict(float) for _ in range(8)]
    for w, e, m, o, re, ra, g, c, s in input_data:
        other_count[0][(w, s)] += 1
        other_count[1][(e, s)] += 1
        other_count[2][(m, s)] += 1
        other_count[3][(o, s)] += 1
        other_count[4][(re, s)] += 1
        other_count[5][(ra, s)] += 1
        other_count[6][(g, s)] += 1
        other_count[7][(c, s)] += 1
    for i in range(8):
        values = [[key[0], key[1], value / salary_count[key[1]]] for key, value in
                  other_count[i].items()]
        factors[i].add_values(values)
    # init salary factor
    f_salary.add_values([[key, value / n] for key, value in salary_count.items()])
    factors.append(f_salary)
    # create BN
    bn = BN('bayes_net',
            [WORK, EDUCATION, MARTIAL_STATUS, OCCUPATION, RELATIONSHIP,
             RACE, GENDER, COUNTRY, SALARY],
            factors)
    return bn




def explore(bayes_net: BN, question_number: int) -> float:
    """
    Computes specific statistics based on queries over a Bayesian network and test data.

    Args:
        bayes_net: A Bayesian network object.
        question_number: An integer representing the question to be computed (1 to 6).

    Returns:
        A float representing the percentage calculated for the given question.
    """
    # Unpack variables from the Bayesian network
    (WORK_CLASS, EDUCATION_LEVEL, MARITAL_STATUS, OCCUPATION_TYPE,
     RELATIONSHIP_STATUS, RACE_CATEGORY, GENDER_TYPE, COUNTRY_ORIGIN,
     SALARY_RANGE) = bayes_net.variables()

    # Define evidence variable groups
    EVIDENCE_E1 = [WORK_CLASS, OCCUPATION_TYPE, EDUCATION_LEVEL, RELATIONSHIP_STATUS]
    EVIDENCE_E2 = EVIDENCE_E1 + [GENDER_TYPE]

    # Read test data from CSV file
    test_data = []
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)  # Skip header row
        for row in reader:
            test_data.append(row)

    # Count the number of males and females in the test data
    num_males = sum(1 for row in test_data if row[6] == 'Male')
    num_females = sum(1 for row in test_data if row[6] == 'Female')

    # Define helper function to set evidence
    def set_evidence(variables, values):
        for variable, value in zip(variables, values):
            variable.set_evidence(value)

    # Helper function to compute probability with given evidence
    def compute_probability(evidence_vars, evidence_values):
        set_evidence(evidence_vars, evidence_values)
        return ve(bayes_net, SALARY_RANGE, evidence_vars).values[1]

    # Determine gender and total count based on the question
    if question_number in [1, 3, 5]:
        target_gender = 'Female'
        total_count = num_females
    elif question_number in [2, 4, 6]:
        target_gender = 'Male'
        total_count = num_males
    else:
        raise ValueError("Question number must be between 1 and 6.")

    # Initialize counters
    count = 0
    total = 0

    # Process each row in the test data
    for row in test_data:
        (work_class, education_level, marital_status, occupation_type,
         relationship_status, race_category, gender_type, country_origin,
         salary_range) = row

        if gender_type != target_gender:
            continue

        # Evidence values for E1 and E2
        evidence_values_E1 = [work_class, occupation_type, education_level, relationship_status]
        evidence_values_E2 = evidence_values_E1 + [gender_type]

        if question_number in [1, 2]:
            # Questions 1 & 2: Compare P(Salary|E1) and P(Salary|E2)
            prob_E1 = compute_probability(EVIDENCE_E1, evidence_values_E1)
            prob_E2 = compute_probability(EVIDENCE_E2, evidence_values_E2)
            if prob_E1 > prob_E2:
                count += 1

        elif question_number in [3, 4]:
            # Questions 3 & 4: Percentage with P(Salary|E1) > 0.5 who actually earn >=50K
            prob_salary = compute_probability(EVIDENCE_E1, evidence_values_E1)
            if prob_salary > 0.5:
                total += 1
                if salary_range == '>=50K':
                    count += 1

        elif question_number in [5, 6]:
            # Questions 5 & 6: Percentage assigned P(Salary|E1) > 0.5
            prob_salary = compute_probability(EVIDENCE_E1, evidence_values_E1)
            if prob_salary > 0.5:
                count += 1

    # Calculate and return the percentage
    if question_number in [3, 4]:
        return (count / total * 100) if total > 0 else 0.0
    else:
        return (count / total_count * 100) if total_count > 0 else 0.0

# def explore(bayes_net: BN, question_number: int) -> float:
#     """
#     Computes specific statistics based on queries over a Bayesian network and test data.
#
#     Args:
#         bayes_net: A Bayesian network object.
#         question_number: An integer representing the question to be computed (1 to 6).
#
#     Returns:
#         A float representing the percentage calculated for the given question.
#     """
#     import csv
#
#     # Extract variables from the Bayesian network
#     variables = bayes_net.variables()
#     VAR_WORK_CLASS, VAR_EDUCATION, VAR_MARITAL_STATUS, VAR_OCCUPATION, \
#     VAR_RELATIONSHIP, VAR_RACE, VAR_GENDER, VAR_COUNTRY, VAR_SALARY = variables
#
#     # Read test data from CSV file
#     try:
#         with open('data/adult-test.csv', newline='') as csvfile:
#             csv_reader = csv.reader(csvfile)
#             headers = next(csv_reader)  # Skip header row
#             data_samples = [row for row in csv_reader]
#     except FileNotFoundError:
#         raise FileNotFoundError("Data file 'adult-test.csv' not found.")
#
#     # Initialize counts
#     gender_counts = {'Male': 0, 'Female': 0}
#     for sample in data_samples:
#         gender = sample[6]
#         if gender in gender_counts:
#             gender_counts[gender] += 1
#
#     # Initialize counters
#     match_count = 0
#     total_relevant = 0
#
#     # Handle each question separately
#     if question_number == 1 or question_number == 2:
#         # Questions 1 & 2: Compare P(Salary|E1) and P(Salary|E2)
#         target_gender = 'Female' if question_number == 1 else 'Male'
#         total_count = gender_counts[target_gender]
#
#         for sample in data_samples:
#             if sample[6] != target_gender:
#                 continue
#
#             # Set evidence for E1
#             evidence_vars_E1 = [VAR_WORK_CLASS, VAR_OCCUPATION, VAR_EDUCATION, VAR_RELATIONSHIP]
#             evidence_vals_E1 = [sample[0], sample[3], sample[1], sample[4]]
#             for var, val in zip(evidence_vars_E1, evidence_vals_E1):
#                 var.set_evidence(val)
#
#             # Compute P(Salary|E1)
#             factor_E1 = ve(bayes_net, VAR_SALARY, evidence_vars_E1)
#             prob_E1 = factor_E1.values[1]
#
#             # Set evidence for E2 (includes gender)
#             evidence_vars_E2 = evidence_vars_E1 + [VAR_GENDER]
#             evidence_vals_E2 = evidence_vals_E1 + [sample[6]]
#             for var, val in zip(evidence_vars_E2, evidence_vals_E2):
#                 var.set_evidence(val)
#
#             # Compute P(Salary|E2)
#             factor_E2 = ve(bayes_net, VAR_SALARY, evidence_vars_E2)
#             prob_E2 = factor_E2.values[1]
#
#             # Compare probabilities
#             if prob_E1 > prob_E2:
#                 match_count += 1
#
#         # Calculate percentage
#         percentage = (match_count / total_count) * 100 if total_count > 0 else 0.0
#
#     elif question_number == 3 or question_number == 4:
#         # Questions 3 & 4: Percentage with P(Salary|E1) > 0.5 who actually earn >=50K
#         target_gender = 'Female' if question_number == 3 else 'Male'
#
#         for sample in data_samples:
#             if sample[6] != target_gender:
#                 continue
#
#             # Set evidence for E1
#             evidence_vars_E1 = [VAR_WORK_CLASS, VAR_OCCUPATION, VAR_EDUCATION, VAR_RELATIONSHIP]
#             evidence_vals_E1 = [sample[0], sample[3], sample[1], sample[4]]
#             for var, val in zip(evidence_vars_E1, evidence_vals_E1):
#                 var.set_evidence(val)
#
#             # Compute P(Salary|E1)
#             factor_E1 = ve(bayes_net, VAR_SALARY, evidence_vars_E1)
#             prob_E1 = factor_E1.values[1]
#
#             if prob_E1 > 0.5:
#                 total_relevant += 1
#                 if sample[8] == '>=50K':
#                     match_count += 1
#
#         # Calculate percentage
#         percentage = (match_count / total_relevant) * 100 if total_relevant > 0 else 0.0
#
#     elif question_number == 5 or question_number == 6:
#         # Questions 5 & 6: Percentage assigned P(Salary|E1) > 0.5
#         target_gender = 'Female' if question_number == 5 else 'Male'
#         total_count = gender_counts[target_gender]
#
#         for sample in data_samples:
#             if sample[6] != target_gender:
#                 continue
#
#             # Set evidence for E1
#             evidence_vars_E1 = [VAR_WORK_CLASS, VAR_OCCUPATION, VAR_EDUCATION, VAR_RELATIONSHIP]
#             evidence_vals_E1 = [sample[0], sample[3], sample[1], sample[4]]
#             for var, val in zip(evidence_vars_E1, evidence_vals_E1):
#                 var.set_evidence(val)
#
#             # Compute P(Salary|E1)
#             factor_E1 = ve(bayes_net, VAR_SALARY, evidence_vars_E1)
#             prob_E1 = factor_E1.values[1]
#
#             if prob_E1 > 0.5:
#                 match_count += 1
#
#         # Calculate percentage
#         percentage = (match_count / total_count) * 100 if total_count > 0 else 0.0
#
#     else:
#         raise ValueError("Question number must be between 1 and 6.")
#
#     return percentage
#
# def explore(bayes_net: BN, question_num: int) -> float:
#     """
#     Computes specific statistics based on queries over a Bayesian network and test data.
#
#     Args:
#         bayes_net: A Bayesian network object.
#         question_num: An integer representing the question to be computed (1 to 6).
#
#     Returns:
#         A float representing the percentage calculated for the given question.
#     """
#     import csv
#
#     # Extract variables from the Bayesian network
#     variables = bayes_net.variables()
#     WORK_CLASS, EDUCATION, MARITAL_STATUS, OCCUPATION, RELATIONSHIP, RACE, GENDER, COUNTRY, SALARY = variables
#
#     # Define evidence variables
#     E1 = [WORK_CLASS, OCCUPATION, EDUCATION, RELATIONSHIP]
#     E2 = E1 + [GENDER]
#
#     # Read test data
#     try:
#         with open('data/adult-test.csv', newline='') as csvfile:
#             reader = csv.reader(csvfile)
#             next(reader)  # Skip header
#             data = [row for row in reader]
#     except FileNotFoundError:
#         raise FileNotFoundError("Data file 'adult-test.csv' not found.")
#
#     # Count total males and females
#     total_males = sum(1 for row in data if row[6] == 'Male')
#     total_females = sum(1 for row in data if row[6] == 'Female')
#
#     # Determine target gender and total count based on the question number
#     if question_num in [1, 3, 5]:
#         target_gender = 'Female'
#         total_count = total_females
#     elif question_num in [2, 4, 6]:
#         target_gender = 'Male'
#         total_count = total_males
#     else:
#         raise ValueError("Question number must be between 1 and 6.")
#
#     count = 0
#     total = 0  # Used for questions 3 and 4
#
#     # Process each sample
#     for row in data:
#         if row[6] != target_gender:
#             continue
#
#         # Extract sample values
#         sample_evidence = [row[0], row[3], row[1], row[4]]  # Corresponds to E1 variables
#         sample_gender = row[6]
#         sample_salary = row[8]
#
#         # Set evidence for E1
#         for var, val in zip(E1, sample_evidence):
#             var.set_evidence(val)
#
#         # Compute P(Salary|E1)
#         factor_E1 = ve(bayes_net, SALARY, E1)
#         prob_E1 = factor_E1.values[1]
#
#         if question_num in [1, 2]:
#             # Set evidence for E2
#             E2_values = sample_evidence + [sample_gender]
#             for var, val in zip(E2, E2_values):
#                 var.set_evidence(val)
#
#             # Compute P(Salary|E2)
#             factor_E2 = ve(bayes_net, SALARY, E2)
#             prob_E2 = factor_E2.values[1]
#
#             # Compare probabilities
#             if prob_E1 > prob_E2:
#                 count += 1
#
#         elif question_num in [3, 4]:
#             if prob_E1 > 0.5:
#                 total += 1
#                 if sample_salary == '>=50K':
#                     count += 1
#
#         elif question_num in [5, 6]:
#             if prob_E1 > 0.5:
#                 count += 1
#
#     # Calculate percentage
#     if question_num in [3, 4]:
#         percentage = (count / total) * 100 if total > 0 else 0.0
#     else:
#         percentage = (count / total_count) * 100 if total_count > 0 else 0.0
#
#     return percentage


if __name__ == '__main__':
    nb = naive_bayes_model('data/adult-train.csv')
    for i in range(1, 7):
        print("explore(nb,{}) = {}".format(i, explore(nb, i)))