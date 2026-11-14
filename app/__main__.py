import argparse
from pathlib import Path
import shutil
import tempfile

from app.core.config import PROJECT_ROOT, load_config
from app.storage.database import DATABASE_PATH, initialize_database

E5_FILES = (
    "*.json", "sentencepiece.bpe.model", "model.safetensors",
    "1_Pooling/config.json",
)


def _download_e5(parser):
    settings = load_config()["embeddings"]
    model_id = settings["model_id"]

    path = PROJECT_ROOT / settings["model_path"]
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        parser.error(f"Каталог уже не пуст: {path}. Существующие веса не перезаписываются. "
                     "Для новой загрузки задайте пустой путь в [embeddings].model_path.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.error("Установите зависимости из requirements.txt.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="e5-", dir=path.parent) as tmp:
            staged = Path(tmp) / "snapshot"
            snapshot_download(repo_id=model_id, local_dir=str(staged), allow_patterns=E5_FILES,)
            for name in ("config.json", "modules.json", "model.safetensors", "tokenizer.json", "1_Pooling/config.json"):
                if not (staged / name).is_file():
                    raise ValueError(f"Не получен обязательный файл: {name}")
            cache = staged / ".cache"
            if cache.is_dir():
                shutil.rmtree(cache)
            staged.rename(path)
    except Exception as error:
        parser.error(f"Не удалось скачать E5: {error}")
    print(f"{model_id} сохранена в {path}")


def main():
    parser = argparse.ArgumentParser(description="Локальный ассистент по документам")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db", help="создать SQLite и таблицы")
    sub.add_parser("download-e5", help="скачать закреплённую версию E5")
    sub.add_parser("serve", help="запустить API и локальную страницу")
    args = parser.parse_args()
    if args.command == "init-db":
        initialize_database()
        print(f"База данных готова: {DATABASE_PATH}")
    elif args.command == "download-e5":
        _download_e5(parser)
    else:
        if not DATABASE_PATH.is_file():
            parser.error("База ещё не создана: выполните python -m app init-db.")
        import uvicorn

        settings = load_config()["server"]
        host = settings["host"]
        if host not in ("127.0.0.1", "::1", "localhost") or settings["workers"] != 1:
            parser.error("Для локального запуска требуется loopback и workers = 1 в [server].")
        display_host = f"[{host}]" if ":" in host else host
        print(f"Откройте http://{display_host}:{settings['port']}/", flush=True)
        uvicorn.run("app.main:app", host=host, port=settings["port"], workers=1)


if __name__ == "__main__":
    main()
