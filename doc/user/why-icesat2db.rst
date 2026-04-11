.. currentmodule:: icesat2db

.. _whyicesat2db:

Overview: Why icesat2DB?
========================

icesat2DB is a scalable Python package built to simplify working with **ICESat-2 (Ice, Cloud, and land Elevation Satellite 2)** data. It offers intuitive modules for processing, querying, and analyzing ICESat-2 data stored in **tileDB databases**.

The motivation behind icesat2DB
-------------------------------

Working with ICESat-2 data in its raw HDF5 format can be challenging due to:

 - **Complex data structure**: ICESat-2 files are organized by orbit, making it inefficient for users interested in specific regions.
 - **High redundancy**: Users often need only a few metrics from across different products for each footprint, yet each HDF5 file contains extensive redundant information, leading to excessive disk and network load.
 - **Filter challenges**: When working with raw ICESat-2 HDF5 files, researchers encounter a large volume of data, including many low-quality shots that are not suitable for scientific analysis. Although the raw HDF5 files contain various quality-related flags and variables, these filters are not pre-applied.

icesat2DB was designed to address these issues by providing an efficient, pre-filtered **tileDB** database system for the ICESat-2 ATL08 land and vegetation product.

What icesat2DB enables
----------------------

By overcoming ICESat-2’s high dimensionality and spatial complexities, icesat2DB offers powerful capabilities that simplify data access and analysis, including:

 - **Efficient, region-specific querying**: Quickly filter data by regions, variables, and time intervals for targeted analysis.
 - **Advanced geospatial querying**: Harness **tileDB** for spatially enabled data retrieval within specified boundaries.
 - **Distributed processing**: Leverage parallel engines to parallelize and scale data processing, ensuring large-scale ICESat-2 datasets are handled efficiently.

By abstracting the complexity of raw ICESat-2 HDF5 files, icesat2DB helps researchers to focus on their scientific objectives without data management bottlenecks.

What does processing mean in icesat2DB?
---------------------------------------

Processing within icesat2DB involves the following steps:

 - **Data Transformation**: Conversion of raw HDF5 granules into TileDB arrays for efficient storage and querying.
 - **Spatial and Temporal Restructuring**: Reorganizing the data from orbit-based granules into a spatially and temporally indexed format to facilitate region-specific and time-based analyses.
 - **Filtering**: Applying user-defined filters, such as quality flags or exclusion criteria, to reduce data size and focus on relevant observations.
 - **Metadata Enhancement**: Adding metadata that improves dataset usability, such as variable descriptions and dataset provenance information.
 
It is important to note that icesat2DB maintains the scientific integrity of the original ICESat-2 measurements. No temporal aggregation, spatial binning, or correction factors are applied unless explicitly requested by the user.

ICESat-2 data structure and icesat2DB’s solution
------------------------------------------------

ICESat-2’s multi-dimensional data—spanning time, space, and height—presents unique challenges in processing and interpretation. icesat2DB simplifies these complexities by aligning data dimensions and providing intuitive modules for accessing and manipulating data. Users can:

 - **Filter ICESat-2 data by time and space**: Retrieve data within specified geographic or temporal ranges.
 - **Merge and unify ICESat-2 products**: Integrate multiple ICESat-2 products for smooth, consolidated analyses.
 - **Perform spatial operations**: Execute custom spatial queries based on user-defined boundaries.

Core components of icesat2DB
----------------------------

icesat2DB's two primary modules facilitate data processing and access:

1. :py:class:`icesat2db.IceSat2Processor`: This component manages data processing tasks, ensuring efficient handling and integrity across large ICESat-2 datasets. It includes features for:
   - Transforming orbit-based HDF5 data into spatially and temporally indexed TileDB arrays.
   - Filtering and validating data during processing.

2. :py:class:`icesat2db.IceSat2Provider`: The high-level module for querying ICESat-2 data stored in **tileDB** arrays. It retrieves data as **Pandas** DataFrames or **xarray** Datasets, enabling users to specify variables, apply spatial filters, and set time ranges.

These modules provide structured access to ICESat-2 data, preserving relationships and metadata between datasets for comprehensive analysis.

Goals and aspirations
---------------------

icesat2DB's primary objective is to create an efficient, scalable platform that meets the needs of various research fields, including:

 - **Geosciences**: Facilitating research on forest structure, canopy height, and biomass.
 - **Remote Sensing**: Enabling cross-referencing ICESat-2 data with other remote sensing products for ecosystem studies.
 - **Data Science & Machine Learning**: Supporting developers in integrating ICESat-2 data into data pipelines for modeling and large-scale analyses.

With a robust foundation for querying and processing ICESat-2 data, icesat2DB aims to be a useful tool for conducting analyses of ecosystem dynamics as inferred from ICESat-2 observations.

---

A collaborative project
=======================

icesat2DB was developed by Felix Dombrowski, Simon Besnard, Mikhail Urbazaev, and Amelia Holcomb from the `Global Land Monitoring Group <https://www.gfz.de/en/section/remote-sensing-and-geoinformatics/topics/global-land-monitoring>`_ at the Helmholtz Centre Potsdam GFZ German Research Centre for Geosciences. It was built on design principles established in the GEDI-focused `gedidb <https://github.com/simonbesnard1/gedidb>`_ package and adapted to handle the ICESat-2 ATL08 product at global scale. The transition to a production-ready tool was driven by the need to handle large datasets and complex queries effectively. The project remains open-source and welcomes contributions from the research community to support its growth and adaptability.

---


