"""MongoDB storage support for *httk-data*.

The package itself is importable without PyMongo.  The backend modules are
loaded lazily so applications that do not use MongoDB do not need the optional
``httk-data[mongodb]`` dependency installed.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .database import MongoDatabase, TransactionsUnavailableError
    from .documents import RecordTooLargeError
    from .entry_provider import StoreEntryProvider, auto_definition
    from .fsck import FsckCollectionSummary, FsckSummary
    from .leases import StoreLockedError, clear_stale_lock
    from .optimade import optimade_filter_searcher
    from .results import MongoResultSet
    from .searcher import MongoExpression, MongoField, MongoSearcher, MongoVariable
    from .store import MongoStore
    from .stored_properties import (
        MongoStoredPropertyCandidateStream,
        MongoStoredPropertyConfigurationError,
        MongoStoredPropertyPlan,
        stored_property_mongo_plan,
    )

__all__ = [
    "FsckCollectionSummary",
    "FsckSummary",
    "MongoDatabase",
    "MongoExpression",
    "MongoField",
    "MongoResultSet",
    "MongoSearcher",
    "MongoStore",
    "MongoStoredPropertyCandidateStream",
    "MongoStoredPropertyConfigurationError",
    "MongoStoredPropertyPlan",
    "MongoVariable",
    "RecordTooLargeError",
    "StoreEntryProvider",
    "StoreLockedError",
    "TransactionsUnavailableError",
    "auto_definition",
    "clear_stale_lock",
    "optimade_filter_searcher",
    "stored_property_mongo_plan",
]

_MONGO_EXPORTS = {
    "FsckCollectionSummary": ".fsck",
    "FsckSummary": ".fsck",
    "MongoDatabase": ".database",
    "MongoExpression": ".searcher",
    "MongoField": ".searcher",
    "MongoResultSet": ".results",
    "StoreEntryProvider": ".entry_provider",
    "auto_definition": ".entry_provider",
    "MongoSearcher": ".searcher",
    "MongoStore": ".store",
    "MongoStoredPropertyCandidateStream": ".stored_properties",
    "MongoStoredPropertyConfigurationError": ".stored_properties",
    "MongoStoredPropertyPlan": ".stored_properties",
    "MongoVariable": ".searcher",
    "optimade_filter_searcher": ".optimade",
    "stored_property_mongo_plan": ".stored_properties",
    "RecordTooLargeError": ".documents",
    "StoreLockedError": ".leases",
    "TransactionsUnavailableError": ".database",
    "clear_stale_lock": ".leases",
}


def __getattr__(name: str) -> Any:
    """Load a MongoDB-backed export on first access.

    :param name: The module attribute to import.
    :return: The requested MongoDB-layer export.
    :raises AttributeError: If ``name`` is not exported by this package.
    :raises ImportError: If PyMongo is unavailable for the requested export.
    """
    module_name = _MONGO_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name, __name__)
    except ModuleNotFoundError as error:
        if error.name is not None and error.name.partition(".")[0] in {"pymongo", "bson"}:
            raise ImportError(
                f"{__name__}.{name} needs pymongo; install the 'httk-data[mongodb]' extra to use the MongoDB layer"
            ) from error
        raise
    return getattr(module, name)
