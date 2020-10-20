Roadmap
=======

Potential speedups
------------------
- __Distributed data sharding__: The `RayDMatrix` could store the final
  dataframe (`self._df`) in the Ray object store. The workers can then
  access this dataframe via shared memory and collect their respective
  shards themselves instead of querying the shared object.