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
    from .fsck import FsckCollectionSummary, FsckSummary
    from .leases import StoreLockedError, clear_stale_lock
    from .optimade import optimade_filter_searcher
    from .results import MongoResultSet
    from .searcher import MongoExpression, MongoField, MongoSearcher, MongoVariable
    from .store import MongoStore

__all__ = [
    "FsckCollectionSummary",
    "FsckSummary",
    "MongoDatabase",
    "MongoExpression",
    "MongoField",
    "MongoResultSet",
    "MongoSearcher",
    "MongoStore",
    "MongoVariable",
    "RecordTooLargeError",
    "StoreLockedError",
    "TransactionsUnavailableError",
    "clear_stale_lock",
    "optimade_filter_searcher",
]

_MONGO_EXPORTS = {
    "FsckCollectionSummary": ".fsck",
    "FsckSummary": ".fsck",
    "MongoDatabase": ".database",
    "MongoExpression": ".searcher",
    "MongoField": ".searcher",
    "MongoResultSet": ".results",
    "MongoSearcher": ".searcher",
    "MongoStore": ".store",
    "MongoVariable": ".searcher",
    "optimade_filter_searcher": ".optimade",
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
