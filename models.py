from sklearn.tree import DecisionTreeClassifier


def get_model(model_name, random_state=42):
    """
    Return a model object for the project.
    """

    if model_name == "decision_tree":
        return DecisionTreeClassifier(max_depth=2, random_state=random_state)

    if model_name == "adaboost":
        # TODO(team_member_adaboost):
        # Implement AdaBoost here.

        raise NotImplementedError(
            "AdaBoost model is intentionally left for a teammate to implement."
        )

    if model_name == "gradient_boosting":
        # TODO(team_member_gradient_boosting):
        # Implement GradientBoostingClassifier here.

        raise NotImplementedError(
            "Gradient Boosting model is intentionally left for a teammate to implement."
        )

    raise ValueError(f"Unknown model name: {model_name}")


def get_default_model_names():
    return [
        "decision_tree",
    ]
