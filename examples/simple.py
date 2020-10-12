from sklearn import datasets
from sklearn.model_selection import train_test_split

from xgboost_ray import RayDMatrix, train


def main():
    # Load dataset
    data, labels = datasets.load_breast_cancer(return_X_y=True)
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=0.25)

    train_set = RayDMatrix(train_x, train_y)
    test_set = RayDMatrix(test_x, test_y)

    # Set config
    config = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "max_depth": 3,
    }

    # Train the classifier
    bst, evals = train(
        config,
        train_set,
        evals=[(test_set, "eval")],
        verbose_eval=False)

    bst.save_model('simple.xgb')
    print("Final validation error: {:.4f}".format(evals["eval"]["error"][-1]))


if __name__ == "__main__":
    main()
