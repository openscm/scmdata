Data Model
==========

The **ScmRun** class
--------------------

The :class:`scmdata.ScmRun` class holds a collection of timeseries along with their metadata.
:class:`scmdata.ScmRun` holds a collection (in the form of a list) of :class:`scmdata.TimeSeries` and enables simplified manipulation of that collection.
For example, the :class:`scmdata.TimeSeries`'s can be filtered to only find the :class:`scmdata.TimeSeries` which have a ``"scenario"`` metadata label equal to ``"green"`` (see :meth:`scmdata.ScmRun.filter <scmdata.run.ScmRun.filter>` for full details).
Other operations include grouping, setting and (basic) plotting.
The complete set of manipulation features can be found in the documentation pages of :class:`scmdata.ScmRun <scmdata.run.ScmRun>`.

:class:`scmdata.ScmRun` has two key properties and one key method, which allow the user to quickly access their data in more standard formats.
The first property, :attr:`scmdata.ScmRun.values <scmdata.run.ScmRun.values>`, returns all of the timeseries as a single :obj:`numpy.ndarray` without any metadata or indication of the time axis.
The second property, :attr:`scmdata.ScmRun.meta <scmdata.run.ScmRun.meta>`, returns all of the timeseries' metadata as a single :obj:`pandas.DataFrame`.
This allows users to quickly have an overview of the timeseries held by :class:`scmdata.ScmRun` without having to also view the data itself at the same time.
The key method is :meth:`scmdata.ScmRun.timeseries() <scmdata.run.ScmRun.timeseries>`.
This method combines the two key properties to return a :obj:`pandas.DataFrame` whose index is equal to :attr:`scmdata.ScmRun.meta <scmdata.run.ScmRun.meta>` and whose values are equal to :attr:`scmdata.ScmRun.values <scmdata.run.ScmRun.values>`.
The columns of the output of :meth:`scmdata.ScmRun.timeseries() <scmdata.run.ScmRun.timeseries>` are the time axis of the data.
As all the underlying :class:`scmdata.TimeSeries` might not have the same time axis, it is quite common for :obj:`numpy.nan` to appear in the output of :meth:`scmdata.ScmRun.timeseries() <scmdata.run.ScmRun.timeseries>`.


The **TimeSeries** class
------------------------

**scmdata**'s approach to data handling focusses on timeseries.
Each :class:`scmdata.TimeSeries` instance has three key properties.
The first is :attr:`values <scmdata.timeseries.TimeSeries.values>`.
This property contains the values of the timeseries as a :obj:`numpy.ndarray`.
The second is :attr:`time_points <scmdata.timeseries.TimeSeries.time_points>`.
This property returns the data's time axis as a :class:`scmdata.time.TimePoints <scmdata.time.TimePoints>` instance (which provides simplified handling of time points).
The third is :attr:`meta <scmdata.timeseries.TimeSeries.meta>`.
This property contains all of the metadata about the timeseries, as a dictionary.
The combination of these three properties provides complete information about the timeseries.


Metadata handling
~~~~~~~~~~~~~~~~~

The key feature of **scmdata** is that its smallest discrete unit is a timeseries.
Via the [TODO update to meta rather than metadata] :attr:`meta <scmdata.timeseries.TimeSeries.meta>` attribute, **scmdata** can store any kind of metadata about the timeseries, without restriction.
This combination allows it to be a high performing, yet flexible library for timeseries data.
However, to do this it must make assumptions about the type of data it holds and these assumptions come with tradeoffs.
In particular, **scmdata** cannot hold metadata at a level finer than a complete timeseries.
For example, it couldn't handle a case where one point in a timeseries needed to be labelled with an 'erroneous' label.
In such a case the entire timeseries would have to be labelled 'erroneous' (or a new timeseries made with just that data point, which may not be very performant).
If behaviour of this type is required, we suggest trying another data handling approach.
