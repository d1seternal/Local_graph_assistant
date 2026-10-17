import sqlite3


ACCESS_LEVELS = {
    "read": 1,
    "write": 2,
    "manage": 3,
}


def can_access_collection(
    connection: sqlite3.Connection,
    user_id: str,
    collection_id: str,
    required_level: str,
) -> bool:
    if required_level not in ACCESS_LEVELS:
        return False

    user = connection.execute(
        """
        SELECT role, is_active
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    ).fetchone()

    if user is None or not user["is_active"]:
        return False

    collection = connection.execute(
        """
        SELECT kind, owner_id
        FROM collections
        WHERE id = ?
        """,
        (collection_id,),
    ).fetchone()

    if collection is None:
        return False

    if user["role"] == "developer" and collection["kind"] != "test":
        return False

    if collection["kind"] == "personal":
        return collection["owner_id"] == user_id

    permission = connection.execute(
        """
        SELECT access_level
        FROM collection_permissions
        WHERE user_id = ? AND collection_id = ?
        """,
        (user_id, collection_id),
    ).fetchone()

    if permission is None:
        return False

    granted = ACCESS_LEVELS[permission["access_level"]]
    required = ACCESS_LEVELS[required_level]

    return granted >= required