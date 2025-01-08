from bnetbase import Variable, Factor, BN
import csv
import itertools


def normalize(factor):
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object.
    :return: a new Factor object resulting from normalizing factor.
    '''
    normal = sum(factor.values)
    nums = []
    for num in factor.values:
        nums.append(num / normal)
    factor2 = Factor(factor.name, factor.scope)
    factor2.values = nums
    return factor2


def restrict(factor, variable, value):
    '''
    Restrict a factor by assigning value to variable.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to restrict.
    :param value: the value to restrict the variable to
    :return: a new Factor object resulting from restricting variable to value.
             This new factor no longer has variable in it.

    '''
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

def sum_out(factor, variable):
    '''
    Sum out a variable variable from factor factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''
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

def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors.

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''
    combined_f = factor_list[0]
    for factor in factor_list[1:]:
        combined_f = multiply2(combined_f, factor)
    return combined_f

def multiply2(factor_one, factor_two):
    """
    Multiply two factors together.
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

def ve(bayes_net, var_query, EvidenceVars):
    '''

    Execute the variable elimination algorithm on the Bayesian network bayes_net
    to compute a distribution over the values of var_query given the
    evidence provided by EvidenceVars.

    :param bayes_net: a BN object.
    :param var_query: the query variable. we want to compute a distribution
                     over the values of the query variable.
    :param EvidenceVars: the evidence variables. Each evidence variable has
                         its evidence set to a value from its domain
                         using set_evidence.
    :return: a Factor object representing a distribution over the values
             of var_query. that is a list of numbers, one for every value
             in var_query's domain. These numbers sum to 1. The i-th number
             is the probability that var_query is equal to its i-th value given
             the settings of the evidence variables.

    For example, assume that
        var_query = A with Dom[A] = ['a', 'b', 'c'],
        EvidenceVars = [B, C], and
        we have called B.set_evidence(1) and C.set_evidence('c'),
    then VE would return a list of three numbers, e.g. [0.5, 0.24, 0.26].
    These numbers would mean that
        Pr(A='a'|B=1, C='c') = 0.5,
        Pr(A='a'|B=1, C='c') = 0.24, and
        Pr(A='a'|B=1, C='c') = 0.26.

    '''
    factor_list = bayes_net.factors()
    for ev_var in EvidenceVars:
        ev_value = ev_var.get_evidence()
        for i in range(len(factor_list)):
            current_factor = factor_list[i]
            if ev_var in current_factor.get_scope():
                factor_list[i] = restrict(current_factor, ev_var, ev_value)

    scopes = [set(f.get_scope()) for f in factor_list]
    all_vars = set().union(*scopes)
    vars_to_eliminate = all_vars - {var_query}

    elimination_order = []

    while vars_to_eliminate:
        best_var = None
        best_scope = None
        min_fill = float('inf')
        scopes_to_merge = []
        for var in vars_to_eliminate:
            related_scopes = [s for s in scopes if var in s]
            combined_scope = set().union(*related_scopes) - {var}
            fill_in_size = len(combined_scope)
            if fill_in_size < min_fill:
                best_var = var
                best_scope = combined_scope
                min_fill = fill_in_size
                scopes_to_merge = related_scopes
        elimination_order.append(best_var)
        vars_to_eliminate.remove(best_var)
        scopes = [s for s in scopes if s not in scopes_to_merge]
        scopes.append(best_scope)
    for elim_var in elimination_order:
        factors_to_process = [f for f in factor_list if elim_var in f.get_scope()]
        if not factors_to_process:
            continue
        multiplied_factor = multiply(factors_to_process)
        new_factor = sum_out(multiplied_factor, elim_var)
        factor_list = [f for f in factor_list if f not in factors_to_process]
        factor_list.append(new_factor)
    if factor_list:
        final_factor = multiply(factor_list)
    else:
        final_factor = Factor('Unit', [var_query])
        final_factor.add_values([[value, 1.0] for value in var_query.domain()])
    normalized_result = normalize(final_factor)
    return normalized_result


def naive_bayes_model(data_file, variable_domains = {"Work": ['Not Working', 'Government', 'Private', 'Self-emp'], "Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'], "Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'], "MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'], "Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'], "Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], "Gender": ['Male', 'Female'], "Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'], "Salary": ['<50K', '>=50K']}, class_var = Variable("Salary", ['<50K', '>=50K'])):
    '''
   NaiveBayesModel returns a BN that is a Naive Bayes model that
   represents the joint distribution of value assignments to
   variables in the Adult Dataset from UCI.  Remember a Naive Bayes model
   assumes P(X1, X2,.... XN, Class) can be represented as
   P(X1|Class)*P(X2|Class)* .... *P(XN|Class)*P(Class).
   When you generated your Bayes bayes_net, assume that the values
   in the SALARY column of the dataset are the CLASS that we want to predict.
   @return a BN that is a Naive Bayes model and which represents the Adult Dataset.
    '''
    ### READ IN THE DATA
    input_data = []
    with open(data_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None) #skip header row
        for row in reader:
            input_data.append(row)

    ### DOMAIN INFORMATION REFLECTS ORDER OF COLUMNS IN THE DATA SET
    #variable_domains = {
    #"Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
    #"Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
    #"Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
    #"MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
    #"Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    #"Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    #"Gender": ['Male', 'Female'],
    #"Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
    #"Salary": ['<50K', '>=50K']
    #}

#variable list
    variables = [Variable('Work', variable_domains['Work']), Variable('Education', variable_domains['Education']), Variable('MaritalStatus', variable_domains['MaritalStatus']),
                 Variable('Occupation', variable_domains['Occupation']), Variable('Relationship', variable_domains[
            'Relationship']), Variable('Race', variable_domains['Race']), Variable('Gender', variable_domains['Gender']), Variable('Country', variable_domains['Country']),
                 Variable('Salary', variable_domains['Salary'])]
    factor_salary = Factor('Salary', [variables[8]])
#list of the factors
    factors = [Factor('Work', [variables[0], variables[8]]), Factor('Education', [variables[1], variables[8]]),
               Factor('MaritalStatus', [variables[2], variables[8]]),
               Factor('Occupation', [variables[3], variables[8]]), Factor('Relationship', [variables[4],
                                                                                           variables[8]]),
               Factor('Race', [variables[5], variables[8]]),
               Factor('Gender', [variables[6], variables[8]]), Factor('Country', [variables[7], variables[8]])]

    class_counts = {'<50K': 0.0, '>=50K': 0.0}
    for data_row in input_data:
        class_counts[data_row[-1]] += 1  #salary last column

    conditional_counts = [{} for _ in range(8)]
    for data_row in input_data:
        work, edu, mar, occ, rel, race, gen, country, sal = data_row
        conditional_counts[0][(work, sal)] = conditional_counts[0].get((work, sal), 0) + 1
        conditional_counts[1][(edu, sal)] = conditional_counts[1].get((edu, sal), 0) + 1
        conditional_counts[2][(mar, sal)] = conditional_counts[2].get((mar, sal), 0) + 1
        conditional_counts[3][(occ, sal)] = conditional_counts[3].get((occ, sal), 0) + 1
        conditional_counts[4][(rel, sal)] = conditional_counts[4].get((rel, sal), 0) + 1
        conditional_counts[5][(race, sal)] = conditional_counts[5].get((race, sal), 0) + 1
        conditional_counts[6][(gen, sal)] = conditional_counts[6].get((gen, sal), 0) + 1
        conditional_counts[7][(country, sal)] = conditional_counts[7].get((country, sal), 0) + 1

    for idx, factor in enumerate(factors):
        values = []
        for (feature_value, class_value), count in conditional_counts[idx].items():
            probability = count / class_counts[class_value]
            values.append([feature_value, class_value, probability])
        factor.add_values(values)

    total_samples = len(input_data)
    class_probabilities = []
    for class_value, count in class_counts.items():
        probability = count / total_samples
        class_probabilities.append([class_value, probability])
    factor_salary.add_values(class_probabilities)
    factors.append(factor_salary)

    return BN('naive_bayes_model', variables, factors)


def explore(bayes_net, question):
    '''    Input: bayes_net---a BN object (a Bayes bayes_net)
           question---an integer indicating the question in HW4 to be calculated. Options are:
           1. What percentage of the women in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
           2. What percentage of the men in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
           3. What percentage of the women in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
           4. What percentage of the men in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
           5. What percentage of the women in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
           6. What percentage of the men in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
           @return a percentage (between 0 and 100)
    '''
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

    num_males = sum(1 for row in test_data if row[6] == 'Male')
    num_females = sum(1 for row in test_data if row[6] == 'Female')
    def set_evidence(variables, values):
        for variable, value in zip(variables, values):
            variable.set_evidence(value)

    def compute_probability(evidence_vars, evidence_values):
        set_evidence(evidence_vars, evidence_values)
        return ve(bayes_net, SALARY_RANGE, evidence_vars).values[1]

    if question in [1, 3, 5]:
        target_gender = 'Female'
        total_count = num_females
    elif question in [2, 4, 6]:
        target_gender = 'Male'
        total_count = num_males
    else:
        raise ValueError("Question number must be between 1 and 6.")
    count = 0
    total = 0

    for row in test_data:
        (work_class, education_level, marital_status, occupation_type,
         relationship_status, race_category, gender_type, country_origin,
         salary_range) = row

        if gender_type != target_gender:
            continue

        evidence_values_E1 = [work_class, occupation_type, education_level, relationship_status]
        evidence_values_E2 = evidence_values_E1 + [gender_type]

        if question in [1, 2]:
            prob_E1 = compute_probability(EVIDENCE_E1, evidence_values_E1)
            prob_E2 = compute_probability(EVIDENCE_E2, evidence_values_E2)
            if prob_E1 > prob_E2:
                count += 1

        elif question in [3, 4]:
            prob_salary = compute_probability(EVIDENCE_E1, evidence_values_E1)
            if prob_salary > 0.5:
                total += 1
                if salary_range == '>=50K':
                    count += 1

        elif question in [5, 6]:
            prob_salary = compute_probability(EVIDENCE_E1, evidence_values_E1)
            if prob_salary > 0.5:
                count += 1
    if question in [3, 4]:
        if total > 0:
            return (count / total * 100)
        else:
            return 0.0
    else:
        if total_count > 0:
            return (count / total_count * 100)
        else:
            return 0.0


if __name__ == '__main__':
    nb = naive_bayes_model('data/adult-train.csv')
    for i in range(1,7):
        print("explore(nb,{}) = {}".format(i, explore(nb, i)))