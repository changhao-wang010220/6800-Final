from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def get_model(model_name, random_state=42):
    """
    Return a model object for the project.
    """

    if model_name == "decision_tree":
        return DecisionTreeClassifier(max_depth=2, random_state=random_state)


    '''
    AdaBoost 概述：
    AdaBoost 是把很多很弱的模型组合起来，变成一个比较强的分类模型

    训练过程中 AdaBoost 会逐步调整资料的权重
    让后面的模型去关注前一轮分类错误的样本
    所以模型会慢慢把重点放在比较难分的资料上
    因此整体的预测效果会一轮一轮变好

    用简单的决策树当作弱学习器
    虽然单一模型本身能力不强，但能透过不断组合这些简单模型，来得到不错的分类效果
    '''
    if model_name == "adaboost":
        # max_depth=1 为设定一个简单的 decision tree 弱分类器來当 AdaBoost 的弱模型
        # 设定 random_state 可以让每次跑出来的结果一样，方便后续做实验比较
        base_tree = DecisionTreeClassifier(max_depth=1, random_state=random_state)

        # 迭代训练 100 个弱学习器，最终的预测结果是这 100 次的加权结果
        # learning_rate 设定可以控制每个弱分类器对最后结果的影响程度
        # 0.5 为中等偏大的数值设定，可以加快模型学习速度，但仍维持稳定性
        return AdaBoostClassifier(
            estimator=base_tree,
            n_estimators=100,
            learning_rate=0.5,
            random_state=random_state,
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
