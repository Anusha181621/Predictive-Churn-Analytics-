"""Customer segmentation.

**This package is intentionally empty.** The value/risk/behaviour segmentation it was reserved for
is implemented in :mod:`src.retention.segments`, which defines the brief's twelve segments --
Champions, High-Value At Risk, Discount-Driven At Risk, Seasonal, Dormant, High-Return and the
rest -- as overlapping flags, so a customer can carry several analytical dimensions rather than one
rigid label.

It lives there rather than here because segmentation is not a standalone step: assigning a segment
needs the churn probability and the expected-revenue projection, and the segments exist to be
consumed by the prioritisation and recommendation logic sitting beside them. Splitting the two
across packages would have meant a circular dependency, or an interface whose only caller is one
module away.

The directory is kept because the brief's project layout names it, and because an empty package
that says where the work went is more useful to a reader than a missing one.
"""
