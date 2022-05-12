from sklearn.metrics import accuracy_score

from mlting.utils.model import BaseModel
from mlting.utils.pandas import get_features
from mlting.utils.tester import BaseTester


def test_biased_performance_across_category(
    model, test_data, category, target, runner, evaluation_function=accuracy_score, threshold=0.1
):
    """
    Calculates various performance over values in a category and tests for
    a difference in performance inferior to the threshold value.

    TODO:
    - More complex threshold function
    - Modify cardinality

    To add to output report:
    - Show distributions of category variables
    - Performance plots of results
    """
    model = BaseModel(model)
    categories = test_data[category].unique()
    if len(categories) > 20:
        raise Exception("Cardinality too high for performance test.")
    result = {}
    for value in categories:
        filtered_data = test_data.query(f"`{category}`=='{value}'")
        predictions = model.predict(filtered_data.loc[:, get_features(filtered_data, target)])
        result[value] = evaluation_function(filtered_data[target], predictions)
    max_performance_difference = max(result.values()) - min(result.values())
    BaseTester(max_performance_difference, threshold).assertion(type="less", runner=runner)
    return result


def test_biased_positive_outcome():
    """
    is the positive (survived in titanic UC) outcome of the
    model more likely to happen to a specific group in a single column?
    """
    return None
