#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""RDF data transformation utilities."""

import pydash as py_
from rdflib import RDF, Graph, URIRef


def mut_graph_field_str(
    graph: Graph, subject: URIRef, predicate: URIRef, default: str = None
) -> str | None:
    """Return the first string literal for ``(subject, predicate, ?)``.

    Args:
        graph (Graph): RDF graph.

        subject (URIRef): Subject URI.

        predicate (URIRef): Predicate URI.

        default (str): Default value if object is None.

    Returns:
        str | None: First string literal for ``(subject, predicate, ?)``.
    """
    # Get first object
    obj = py_.head(list(graph.objects(subject, predicate)))

    # Return string literal if object is not None
    return str(obj) if obj is not None else default


def mut_graph_field_float(
    graph: Graph, subject: URIRef, predicate: URIRef, default: float = None
) -> float | None:
    """Return the first float literal for ``(subject, predicate, ?)``.

    Args:
        graph (Graph): RDF graph.

        subject (URIRef): Subject URI.

        predicate (URIRef): Predicate URI.

        default (float): Default value if object is None.

    Returns:
        float | None: First float literal for ``(subject, predicate, ?)``.
    """
    # Get first object
    obj = py_.head(list(graph.objects(subject, predicate)))

    # Return None if object is None
    if obj is None:
        return default

    # Return float literal if object is a float
    try:
        return float(obj)

    except (ValueError, TypeError):
        return default


def mut_graph_field_str_list(
    graph: Graph, subject: URIRef, predicate: URIRef, default: list[str] = None
) -> list[str]:
    """Return all string literals for ``(subject, predicate, ?)``.

    Args:
        graph (Graph): RDF graph.

        subject (URIRef): Subject URI.

        predicate (URIRef): Predicate URI.

        default (list[str]): Default value if objects are None.

    Returns:
        list[str]: All string literals for ``(subject, predicate, ?)``.
    """
    # Return all string literals
    return py_.map_(list(graph.objects(subject, predicate)), str) or default


def mut_extract_uri_tail(uri_str: str) -> str:
    """Extract the trailing segment of a URI after the last ``.`` or ``/``.

    Args:
        uri_str (str): URI string.

    Returns:
        str: Trailing segment of the URI.
    """
    # Extract trailing segment
    sep = "." if "." in uri_str else "/"

    # Return trailing segment
    return uri_str.rsplit(sep, maxsplit=1)[-1]


def mut_build_label_map(
    graph: Graph,
    rdf_type: URIRef,
    value_predicate,
    fallback_predicate: URIRef | None = None,
) -> dict[str, str]:
    """Build a ``{URI: label}`` lookup from all subjects of the given RDF type.

    Args:
        graph (Graph): RDF graph.

        rdf_type (URIRef): RDF type.

        value_predicate (URIRef): Value predicate.

        fallback_predicate (URIRef | None): Fallback predicate.

    Returns:
        dict[str, str]: ``{URI: label}`` lookup.
    """
    # Build label map
    result = {}

    # Iterate over subjects
    for uri in graph.subjects(RDF.type, rdf_type):
        # Get value
        val = mut_graph_field_str(graph, uri, value_predicate)

        # If value is None
        if val is None and fallback_predicate:
            # Use fallback predicate
            val = mut_graph_field_str(graph, uri, fallback_predicate)

        # If value is not None
        if val is not None:
            result[str(uri)] = val

    # Return result
    return result


def mut_resolve_uri(
    graph: Graph,
    subject: URIRef,
    predicate: URIRef,
    labels: dict[str, str],
) -> str:
    """Resolve an optional URI reference to its label.

    Args:
        graph (Graph): RDF graph.

        subject (URIRef): Subject URI.

        predicate (URIRef): Predicate URI.

        labels (dict[str, str]): ``{URI: label}`` lookup.

    Returns:
        str: Label if found, otherwise ``""``.
    """
    # Get first object
    uri_obj = py_.head(list(graph.objects(subject, predicate)))

    # Return!
    return labels.get(str(uri_obj), "") if uri_obj else ""
