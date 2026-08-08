"""MongoDB client lifecycle and transaction-mode probing."""

from types import TracebackType
from typing import Any, Self

from pymongo import MongoClient
from pymongo.read_preferences import ReadPreference

__all__ = ["MongoDatabase", "TransactionsUnavailableError"]


class TransactionsUnavailableError(RuntimeError):
    """An explicit transaction was requested without replica-set support."""


class MongoDatabase:
    """A named MongoDB database reached through a wrapped PyMongo client.

    The wrapper performs one ``hello`` probe when it is constructed.  A
    replica-set name in that reply enables transaction mode unless the caller
    explicitly pins degraded mode.

    :param client: The PyMongo client that owns the connection pool.
    :param database: The database name.
    :param transactions: ``"auto"``, ``"require"``, or ``"never"``.
    :raises ValueError: If ``transactions`` is invalid or required transactions
        are unavailable.
    """

    def __init__(self, client: MongoClient, database: str, *, transactions: str = "auto") -> None:
        if transactions not in {"auto", "require", "never"}:
            raise ValueError("transactions must be one of 'auto', 'require', or 'never'")
        self._client = client
        # Store-level handles must never inherit a caller's secondary-preferred
        # default: dedup and metadata observations are primary-authoritative.
        self._database = client.get_database(database, read_preference=ReadPreference.PRIMARY)
        hello = client.admin.command("hello")
        replica_set = bool(hello.get("setName"))
        if transactions == "require" and not replica_set:
            raise ValueError("transactions='require' needs a MongoDB replica set")
        self._supports_transactions = replica_set and transactions != "never"
        self._transactions_mode = transactions

    @classmethod
    def connect(cls, uri: str, *, database: str, transactions: str = "auto") -> Self:
        """Connect to a MongoDB URI with durable majority defaults.

        :param uri: MongoDB connection URI.
        :param database: Database name to expose through the wrapper.
        :param transactions: ``"auto"``, ``"require"``, or ``"never"``.
        :return: A connected database wrapper.
        :raises ValueError: If the requested transaction mode is unavailable.
        """
        client: MongoClient[dict[str, Any]] = MongoClient(uri, w="majority", journal=True, readConcernLevel="majority")
        try:
            return cls(client, database, transactions=transactions)
        except BaseException:
            client.close()
            raise

    @property
    def client(self) -> MongoClient:
        """Return the wrapped PyMongo client.

        :return: The client that owns this database connection pool.
        """
        return self._client

    @property
    def database(self) -> Any:
        """Return the selected PyMongo database handle.

        :return: The PyMongo database handle.
        """
        return self._database

    @property
    def supports_transactions(self) -> bool:
        """Whether this wrapper is configured to use MongoDB transactions.

        :return: ``True`` when the mode probe found a replica set and the mode
            was not pinned to ``"never"``.
        """
        return self._supports_transactions

    def dispose(self) -> None:
        """Close the client and its connection pool.

        :return: None.
        """
        self._client.close()

    def __enter__(self) -> Self:
        """Enter a context that owns this client's connection pool.

        :return: This database wrapper.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the client when leaving the context.

        :param exc_type: The exception class raised in the context, if any.
        :param exc_value: The exception instance raised in the context, if any.
        :param traceback: The traceback for the context exception, if any.
        :return: None.
        """
        self.dispose()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._database.name!r})"
