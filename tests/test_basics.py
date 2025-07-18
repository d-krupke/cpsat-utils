from cpsat_utils.testing import AssertModelFeasible, AssertModelInfeasible

def test_basic_feasibility():
    with AssertModelFeasible() as model:
        x = model.new_bool_var(name='x')
        y = model.new_bool_var(name='y')
        model.add(x + y == 1)
        model.minimize(x + y)

def test_basic_infeasibility():
    with AssertModelInfeasible() as model:
        x = model.new_bool_var(name='x')
        y = model.new_bool_var(name='y')
        model.add(x + y == 3)  # This is infeasible since x and y are bool vars
        model.minimize(x + y)
    