from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
        

    '''
    Gradient Boosting 概述：
    Gradient Boosting 是一种逐步提升模型预测能力的集成学习方法

    训练过程中，模型会依序建立多棵较简单的决策树
    每一棵新树都会尝试弥补前面模型的预测不足
    也就是说，后续模型会重点学习前面模型尚未处理好的错误或残差

    透过不断累加这些简单模型的预测结果
    Gradient Boosting 能够逐步降低整体误差
    因此可以得到比单一决策树更稳定、预测能力更强的分类效果
    '''
    if model_name == "gradient_boosting":
        # Gradient Boosting 会逐步构建由多棵浅层决策树组成的加法集成模型。
        return GradientBoostingClassifier(
            n_estimators=100,
            #控制 boosting 的迭代轮数，也就是弱学习器数量。
            learning_rate=0.1,
            #控制每棵树对最终模型的贡献强度。
            max_depth=3,
            #控制每棵单独决策树的复杂度。
            random_state=random_state,
        )


    raise ValueError(f"Unknown model name: {model_name}")


def get_default_model_names():
    return [
        "decision_tree",
        "adaboost",
        "gradient_boosting",
    ]

