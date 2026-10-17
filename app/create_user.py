import argparse
from getpass import getpass
import json
import sqlite3

from app.core.auth import create_user
from app.storage.database import initialize_database


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("username")
    parser.add_argument(
        "--role",
        choices=["administrator", "developer", "user"],
        default="user",
    )
    args = parser.parse_args()

    password = getpass("Пароль: ")
    confirmation = getpass("Повторите пароль: ")

    if password != confirmation:
        parser.error("Пароли не совпадают.")

    initialize_database()

    try:
        user = create_user(args.username, password, args.role)
    except ValueError as error:
        parser.error(str(error))
    except sqlite3.IntegrityError:
        parser.error("Не удалось создать пользователя: проверьте имя и БД.")

    print(json.dumps(user, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
