def readme_sklearn_api():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    from xgboost_ray import RayParams, RayXGBClassifier

    seed = 42

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.25, random_state=42
    )

    clf = RayXGBClassifier(
        n_jobs=4, random_state=seed  # In XGBoost-Ray, n_jobs sets the number of actors
    )

    # scikit-learn API will automatically conver the data
    # to RayDMatrix format as needed.
    # You can also pass X as a RayDMatrix, in which case
    # y will be ignored.

    clf.fit(X_train, y_train)

    pred_ray = clf.predict(X_test)
    print(pred_ray)

    pred_proba_ray = clf.predict_proba(X_test)
    print(pred_proba_ray)

    # It is also possible to pass a RayParams object
    # to fit/predict/predict_proba methods - will override
    # n_jobs set during initialization

    clf.fit(X_train, y_train, ray_params=RayParams(num_actors=2))

    pred_ray = clf.predict(X_test, ray_params=RayParams(num_actors=2))
    print(pred_ray)


if __name__ == "__main__":
    import ray

    ray.init(num_cpus=5)

    print("Readme: scikit-learn API example")
    readme_sklearn_api()
