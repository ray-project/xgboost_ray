Roadmap
=======

Potential speedups
------------------

- __Distributed data reading__: The `RayDMatrix` currently uses a
  single-threaded pandas process to read all data before sharding.
  Distributing these reads will speed up data loading. This seems to
  be the main performance overhead compared to Dask.

- __Distributed data sharding__: The `RayDMatrix` could store the final
  dataframe (`self._df`) in the Ray object store. The workers can then
  access this dataframe via shared memory and collect their respective
  shards themselves instead of querying the shared object.
