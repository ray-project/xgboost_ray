Roadmap
=======

API improvements
----------------

- __sklearn API__: Implementing the sklearn API for 
  `XGBClassifier` and `XGBRegressor` is a natural next step.

Potential speedups
------------------

- __Distributed data set__: Modin could be implemented as a 
  distributed data set for larger than memory datasets.

- __Distributed data reading__: The `RayDMatrix` currently uses a
  pandas process to read all data before sharding.
  Distributing these reads could speed up data loading.

