Data Model
==========

Analysing the results from simple climate models involves a lot of timeseries handling, including:

* filtering
* plotting
* resampling
* serialization/deserialisation
* computation

As a result, **scmdata**'s approach to data handling focusses on efficient handling of timeseries.


The **ScmRun** class
--------------------

The :class:`scmdata.ScmRun` class represents a collection of timeseries data including metadata and provides methods for manipulating the data.
Internally, :class:`ScmRun <scmdata.run.ScmRun>` stores the timeseries data in a single :class:`pandas.DataFrame` and the timeseries metadata
:class:`pandas.MultiIndex` of type `pandas.Categorical`, for efficient indexing.

This class is the primary way of handling timeseries data within the **scmdata** package.
For example, the :class:`scmdata.timeseries.TimeSeries`'s can be filtered to only find the :class:`TimeSeries <scmdata.timeseries.TimeSeries>`  which have a ``"scenario"`` metadata label equal to ``"green"`` (see :meth:`scmdata.ScmRun.filter <scmdata.run.ScmRun.filter>` for full details).
Other operations include grouping, setting and (basic) plotting.


The complete set of manipulation features can be found in the documentation pages of :class:`ScmRun <scmdata.run.ScmRun>`.

:class:`ScmRun <scmdata.run.ScmRun>` has three key properties and one key method, which allow the user to quickly access their data in more standard formats:

* :attr:`values <scmdata.run.ScmRun.values>` returns all of the timeseries as a single :obj:`numpy.ndarray` without any metadata or indication of the time axis.
* :attr:`meta <scmdata.run.ScmRun.meta>` returns all of the timeseries' metadata as a single :obj:`pandas.DataFrame`. This allows users to quickly have an overview of the timeseries held by :class:`scmdata.ScmRun` without having to also view the data itself.
* :attr:`metadata <scmdata.run.ScmRun.metadata` stores run-specific metadata, i.e. metadata which isn't tied to any timeseries specifically.
* :meth:`timeseries() <scmdata.run.ScmRun.timeseries>` combines :attr:`values <scmdata.run.ScmRun.values>` and :attr:`meta <scmdata.run.ScmRun.meta>` to form a :obj:`pandas.DataFrame` whose index is equal to :attr:`scmdata.ScmRun.meta <scmdata.run.ScmRun.meta>` and whose values are equal to :attr:`scmdata.ScmRun.values <scmdata.run.ScmRun.values>`. The columns of the output of :meth:`timeseries() <scmdata.run.ScmRun.timeseries>` are the time axis of the data.


Metadata handling
~~~~~~~~~~~~~~~~~

**scmdata** can store any kind of metadata about the timeseries, without restriction.
This combination allows it to be a high performing, yet flexible library for timeseries data.

However, to do this it must make assumptions about the type of data it holds and these assumptions come with tradeoffs.
In particular, **scmdata** cannot hold metadata at a level finer than a complete timeseries.
For example, it couldn't handle a case where one point in a timeseries needed to be labelled with an 'erroneous' label.
In such a case the entire timeseries would have to be labelled 'erroneous' (or a new timeseries made with just that data point, which may not be very performant).
If behaviour of this type is required, we suggest trying another data handling approach.
