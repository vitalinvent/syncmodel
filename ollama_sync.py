# -*- coding: utf-8 -*-
"""Кроссплатформенный экспорт/импорт локальных моделей Ollama.

Требования: Python 3.10+, только стандартная библиотека.
Хранилище переноса имеет стандартную структуру Ollama: blobs/ и manifests/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

APP_NAME = "Ollama Model Transfer"
DEFAULT_REGISTRY = "registry.ollama.ai"
DIGEST_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
COPY_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass
class ModelInfo:
    name: str
    manifest_rel: Path
    digests: list[str] = field(default_factory=list)
    size_bytes: int = 0
    complete: bool = True
    problems: list[str] = field(default_factory=list)


def format_size(size_bytes: int) -> str:
    value = float(max(0, size_bytes))
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ", "ПБ"):
        if value < 1024 or unit == "ПБ":
            return f"{value:.1f} {unit}" if unit != "Б" else f"{int(value)} {unit}"
        value /= 1024
    return f"{value:.1f} ПБ"


def _normal_path(path: Path) -> str:
    """Нормализованное имя пути для сравнений на Windows/Linux."""
    try:
        value = str(path.expanduser().resolve(strict=False))
    except OSError:
        value = str(path.expanduser().absolute())
    return os.path.normcase(os.path.normpath(value))


def paths_overlap(source: Path, destination: Path) -> bool:
    """True, если один из путей совпадает с другим или вложен в него."""
    src = Path(_normal_path(source))
    dst = Path(_normal_path(destination))
    return src == dst or src in dst.parents or dst in src.parents


def is_models_dir(path: Path) -> bool:
    """Проверяет корень хранилища Ollama (в нём должна быть папка manifests)."""
    try:
        return path.expanduser().is_dir() and (path.expanduser() / "manifests").is_dir()
    except (OSError, PermissionError):
        return False


def _systemd_environment_models() -> Path | None:
    """Пытается получить OLLAMA_MODELS из активного systemd unit Ollama."""
    try:
        result = subprocess.run(
            ["systemctl", "show", "ollama.service", "--property=Environment", "--value"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0:
            for item in shlex.split(result.stdout.strip()):
                if item.startswith("OLLAMA_MODELS="):
                    value = item.split("=", 1)[1]
                    if value:
                        return Path(value).expanduser()
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    # Резервный вариант: читаем unit и drop-in файлы, если systemctl недоступен.
    config_files = [
        Path("/etc/systemd/system/ollama.service"),
        Path("/usr/lib/systemd/system/ollama.service"),
        Path("/lib/systemd/system/ollama.service"),
    ]
    config_dirs = [Path("/etc/systemd/system/ollama.service.d")]
    for directory in config_dirs:
        try:
            config_files.extend(sorted(directory.glob("*.conf")))
        except OSError:
            pass

    for config in config_files:
        try:
            for line in config.read_text(encoding="utf-8", errors="ignore").splitlines():
                stripped = line.strip()
                if not stripped.startswith("Environment="):
                    continue
                for item in shlex.split(stripped.split("=", 1)[1]):
                    if item.startswith("OLLAMA_MODELS="):
                        value = item.split("=", 1)[1]
                        if value:
                            return Path(value).expanduser()
        except (OSError, ValueError):
            continue
    return None


def default_models_dir() -> Path:
    """Наиболее вероятное хранилище Ollama для текущей системы."""
    if value := os.environ.get("OLLAMA_MODELS"):
        return Path(value).expanduser()

    if platform.system() == "Linux":
        service_value = _systemd_environment_models()
        if service_value:
            return service_value
        service_default = Path("/usr/share/ollama/.ollama/models")
        if service_default.exists() or Path("/usr/share/ollama").exists():
            return service_default

    return Path.home() / ".ollama" / "models"


def _add_if_path(candidates: list[Path], value: Path | None) -> None:
    if value is not None:
        candidates.append(value.expanduser())


def _mounted_volume_candidates(roots: Iterable[Path]) -> list[Path]:
    """Ищет хранилища на внешних дисках без глубокого сканирования файловой системы."""
    result: list[Path] = []
    child_names = ("models", "ollamamodels", "ollama", "transfer", "export_ollama", "ollama-export")
    for root in roots:
        if not root.is_dir():
            continue
        try:
            first_level = [item for item in root.iterdir() if item.is_dir()]
        except OSError:
            continue

        for first in first_level:
            result.append(first)
            for name in child_names:
                result.append(first / name)
            result.append(first / "ollama" / "models")

            # Для /media и /run/media типичная схема: <root>/<user>/<volume>/models.
            if root in (Path("/media"), Path("/run/media")):
                try:
                    second_level = [item for item in first.iterdir() if item.is_dir()]
                except OSError:
                    continue
                for second in second_level:
                    result.append(second)
                    for name in child_names:
                        result.append(second / name)
                    result.append(second / "ollama" / "models")
    return result


def find_all_models_dirs() -> list[Path]:
    """Ищет известные и пользовательские хранилища Ollama на Windows/Linux."""
    candidates: list[Path] = []
    system = platform.system()

    _add_if_path(candidates, Path(os.environ["OLLAMA_MODELS"]) if os.environ.get("OLLAMA_MODELS") else None)

    if system == "Windows":
        candidates.extend(
            [
                Path.home() / ".ollama" / "models",
                Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".ollama" / "models",
                Path(r"E:\ollamamodels"),
            ]
        )
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            try:
                if drive.exists():
                    candidates.extend(
                        [
                            drive / "ollamamodels",
                            drive / "ollama" / "models",
                            drive / "transfer",
                            drive / "export_ollama",
                            drive / "models",
                        ]
                    )
            except OSError:
                continue
    elif system == "Linux":
        _add_if_path(candidates, _systemd_environment_models())
        candidates.extend(
            [
                Path("/usr/share/ollama/.ollama/models"),
                Path("/var/lib/ollama/.ollama/models"),
                Path.home() / ".ollama" / "models",
                Path("/mnt/llm/models"),
            ]
        )
        candidates.extend(
            _mounted_volume_candidates((Path("/media"), Path("/mnt"), Path("/run/media")))
        )
    else:
        candidates.append(Path.home() / ".ollama" / "models")

    found: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _normal_path(candidate)
        if key in seen:
            continue
        seen.add(key)
        if is_models_dir(candidate):
            try:
                found.append(candidate.resolve())
            except OSError:
                found.append(candidate)
    return found


def _model_name_from_manifest(registry: str, relative: Path) -> str | None:
    parts = relative.parts
    if len(parts) < 2:
        return None
    tag = parts[-1]
    name_parts = parts[1:-1] if parts[0] == "library" else parts[:-1]
    if not name_parts:
        return None
    model_path = "/".join(name_parts)
    if registry != DEFAULT_REGISTRY:
        model_path = f"{registry}/{model_path}"
    return f"{model_path}:{tag}"


def get_models_info(models_dir: Path) -> list[ModelInfo]:
    """Читает manifests всех registry, находит blob-файлы и отмечает неполные модели."""
    models: list[ModelInfo] = []
    manifests_root = models_dir / "manifests"
    if not manifests_root.is_dir():
        return models

    try:
        registries = sorted(p for p in manifests_root.iterdir() if p.is_dir())
    except OSError:
        return models

    for registry_dir in registries:
        try:
            manifest_paths = sorted(p for p in registry_dir.rglob("*") if p.is_file())
        except OSError:
            continue

        for manifest_path in manifest_paths:
            relative = manifest_path.relative_to(registry_dir)
            name = _model_name_from_manifest(registry_dir.name, relative)
            if not name:
                continue

            info = ModelInfo(
                name=name,
                manifest_rel=Path("manifests") / registry_dir.name / relative,
            )
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                raw_digests: list[str] = []
                config = data.get("config")
                if isinstance(config, dict) and isinstance(config.get("digest"), str):
                    raw_digests.append(config["digest"])
                else:
                    info.problems.append("в манифесте отсутствует config.digest")

                layers = data.get("layers", [])
                if not isinstance(layers, list) or not layers:
                    info.problems.append("в манифесте нет layers")
                else:
                    for layer in layers:
                        if isinstance(layer, dict) and isinstance(layer.get("digest"), str):
                            raw_digests.append(layer["digest"])
                        else:
                            info.problems.append("некорректная запись слоя в манифесте")

                # Один и тот же blob может встречаться в нескольких слоях.
                info.digests = list(dict.fromkeys(raw_digests))
                for digest in info.digests:
                    if not DIGEST_RE.fullmatch(digest):
                        info.problems.append(f"неподдерживаемый digest: {digest}")
                        continue
                    blob_path = models_dir / "blobs" / digest.replace(":", "-")
                    try:
                        if blob_path.is_file():
                            info.size_bytes += blob_path.stat().st_size
                        else:
                            info.problems.append(f"отсутствует blob {blob_path.name}")
                    except OSError as exc:
                        info.problems.append(f"не удалось проверить {blob_path.name}: {exc}")
            except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
                info.problems.append(f"ошибка чтения манифеста: {exc}")

            info.complete = not info.problems
            models.append(info)

    return sorted(models, key=lambda item: item.name.lower())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(COPY_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _copy_blob_verified(source: Path, destination: Path, digest: str) -> None:
    """Копирует blob во временный файл, проверяет SHA-256 и атомарно заменяет цель."""
    match = DIGEST_RE.fullmatch(digest)
    if not match:
        raise ValueError(f"Некорректный digest: {digest}")
    expected_hash = match.group(1)

    if not source.is_file():
        raise FileNotFoundError(f"Исходный blob не найден: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            if _sha256_file(destination) == expected_hash:
                print(f"  [ПРОПУСК] Проверенный blob уже есть: {destination.name}")
                return
            print(f"  [ИСПРАВЛЕНИЕ] Существующий blob повреждён: {destination.name}")
        except OSError:
            pass

    temp_path: Path | None = None
    calculated = hashlib.sha256()
    copied = 0
    total = source.stat().st_size
    last_reported = -1
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as target, source.open("rb") as src:
            temp_path = Path(target.name)
            while True:
                chunk = src.read(COPY_CHUNK_SIZE)
                if not chunk:
                    break
                target.write(chunk)
                calculated.update(chunk)
                copied += len(chunk)
                if total >= 1024 * 1024 * 1024:
                    percent = int(copied * 100 / max(total, 1))
                    if percent >= last_reported + 10:
                        print(f"  Копирование {destination.name}: {percent}%")
                        last_reported = percent
            target.flush()
            os.fsync(target.fileno())

        actual_hash = calculated.hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 не совпал для {source.name}: ожидался {expected_hash}, получен {actual_hash}"
            )

        try:
            shutil.copystat(source, temp_path)
        except OSError:
            pass
        os.replace(temp_path, destination)
        temp_path = None
        print(f"  [OK] {destination.name} ({format_size(copied)}) — SHA-256 проверен")
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _copy_manifest_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as target, source.open("rb") as src:
            temp_path = Path(target.name)
            shutil.copyfileobj(src, target, length=COPY_CHUNK_SIZE)
            target.flush()
            os.fsync(target.fileno())
        try:
            shutil.copystat(source, temp_path)
        except OSError:
            pass
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def transfer_single_model(
    model: ModelInfo,
    source_dir: Path,
    destination_dir: Path,
    verified_destination_blobs: set[str] | None = None,
) -> None:
    """Переносит одну модель; манифест публикуется только после всех blob-файлов."""
    if not model.complete:
        details = "; ".join(model.problems)
        raise RuntimeError(f"Модель {model.name} неполная и не будет перенесена: {details}")

    manifest_source = source_dir / model.manifest_rel
    if not manifest_source.is_file():
        raise FileNotFoundError(f"Манифест не найден: {manifest_source}")

    print(f"\n---> {model.name} ({format_size(model.size_bytes)})")
    if verified_destination_blobs is None:
        verified_destination_blobs = set()
    for digest in model.digests:
        if not DIGEST_RE.fullmatch(digest):
            raise ValueError(f"Некорректный digest в модели {model.name}: {digest}")
        blob_name = digest.replace(":", "-")
        blob_destination = destination_dir / "blobs" / blob_name
        blob_key = _normal_path(blob_destination)
        if blob_key in verified_destination_blobs:
            print(f"  [ПРОПУСК] Blob уже проверен в этой операции: {blob_name}")
            continue
        _copy_blob_verified(
            source_dir / "blobs" / blob_name,
            blob_destination,
            digest,
        )
        verified_destination_blobs.add(blob_key)

    # Manifest не попадёт в Ollama, пока все связанные blob-файлы не скопированы и проверены.
    _copy_manifest_atomic(
        manifest_source,
        destination_dir / model.manifest_rel,
    )
    print(f"  [OK] Манифест установлен: {model.manifest_rel}")


def fix_linux_permissions(destination_dir: Path) -> bool:
    """Меняет владельца на ollama; запускается только по явному согласию пользователя."""
    if platform.system() != "Linux":
        print("Изменение владельца доступно только в Linux.")
        return False
    if shutil.which("chown") is None:
        print("Команда chown не найдена.")
        return False
    try:
        targets = [path for path in (destination_dir / "blobs", destination_dir / "manifests") if path.exists()]
        if not targets:
            print("В папке назначения нет blobs/ и manifests/.")
            return False
        result = subprocess.run(
            ["chown", "-R", "ollama:ollama", *(str(path) for path in targets)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"[OK] Владелец файлов изменён на ollama:ollama: {destination_dir}")
            print("При необходимости перезапустите службу: sudo systemctl restart ollama")
            return True
        message = result.stderr.strip() or f"код завершения {result.returncode}"
        print(f"[ВНИМАНИЕ] Не удалось изменить владельца: {message}")
        print("Для стандартной systemd-службы может потребоваться запуск с sudo.")
    except OSError as exc:
        print(f"[ВНИМАНИЕ] Ошибка chown: {exc}")
    return False


def ask_text(prompt: str, allow_empty: bool = False) -> str:
    while True:
        value = input(prompt).strip()
        if value.lower() in {"q", "quit", "exit", "выход"}:
            raise SystemExit(0)
        if value or allow_empty:
            return value
        print("Введите значение или q для выхода.")


def ask_choice(prompt: str, max_value: int, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    while True:
        raw = ask_text(prompt)
        if raw.isdigit():
            value = int(raw)
            if minimum <= value <= max_value:
                return value
        print(f"Введите число от {minimum} до {max_value} или q для выхода.")


def ask_multi_choice(prompt: str, max_value: int) -> list[int]:
    while True:
        raw = ask_text(prompt).lower()
        if raw in {"0", "all", "все", "*"}:
            return list(range(1, max_value + 1))
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        selected: list[int] = []
        valid = bool(parts)
        for part in parts:
            if "-" in part:
                bounds = part.split("-", 1)
                if len(bounds) != 2 or not all(x.isdigit() for x in bounds):
                    valid = False
                    break
                start, end = map(int, bounds)
                if start < 1 or end < start or end > max_value:
                    valid = False
                    break
                selected.extend(range(start, end + 1))
            elif part.isdigit() and 1 <= int(part) <= max_value:
                selected.append(int(part))
            else:
                valid = False
                break
        if valid and selected:
            return list(dict.fromkeys(selected))
        print("Введите номера через запятую (1,3), диапазон (1-3) или 0 для всех.")


def choose_source(found_dirs: list[Path]) -> tuple[Path, list[ModelInfo]]:
    print("\nШАГ 1. Выберите папку-источник:")
    for index, directory in enumerate(found_dirs, 1):
        models = get_models_info(directory)
        total = sum(item.size_bytes for item in models)
        print(f"  [{index}] {directory} — моделей: {len(models)}, размер: {format_size(total)}")
    custom_index = len(found_dirs) + 1
    print(f"  [{custom_index}] Указать путь вручную (корень с папками blobs и manifests)")

    choice = ask_choice(f"Выбор [1-{custom_index}]: ", custom_index)
    if choice == custom_index:
        while True:
            directory = Path(ask_text("Путь к хранилищу-источнику: ").strip('"')).expanduser()
            if is_models_dir(directory):
                models = get_models_info(directory)
                return directory, models
            print(f"Не найден каталог manifests в {directory}. Проверьте путь.")

    directory = found_dirs[choice - 1]
    return directory, get_models_info(directory)


def choose_destination(source_dir: Path, is_export: bool) -> Path:
    default_dir = default_models_dir()
    options: list[Path] = []
    if not is_export:
        if not paths_overlap(source_dir, default_dir):
            options.append(default_dir)
    else:
        export_default = Path.cwd() / "ollama-export"
        if not paths_overlap(source_dir, export_default):
            options.append(export_default)

    for directory in find_all_models_dirs():
        if not paths_overlap(source_dir, directory) and all(
            _normal_path(directory) != _normal_path(existing) for existing in options
        ):
            options.append(directory)

    print("\nШАГ 3. Выберите папку назначения:")
    for index, directory in enumerate(options, 1):
        note = " (путь Ollama по умолчанию)" if _normal_path(directory) == _normal_path(default_dir) else ""
        print(f"  [{index}] {directory}{note}")
    custom_index = len(options) + 1
    print(f"  [{custom_index}] Указать путь вручную / внешний диск")

    choice = ask_choice(f"Выбор [1-{custom_index}]: ", custom_index)
    if choice == custom_index:
        while True:
            destination = Path(ask_text("Путь назначения: ").strip('"')).expanduser()
            if paths_overlap(source_dir, destination):
                print("Источник и назначение совпадают или вложены друг в друга. Выберите другой путь.")
                continue
            return destination

    return options[choice - 1]


def perform_transfer(is_export: bool) -> None:
    found_dirs = find_all_models_dirs()
    if not found_dirs:
        print("Автоматически хранилища не найдены. Можно указать путь вручную.")

    source_dir, models = choose_source(found_dirs)
    if not models:
        print(f"В {source_dir} не найдено моделей.")
        return

    print(f"\nШАГ 2. Модели в {source_dir}:")
    for index, model in enumerate(models, 1):
        state = "ГОТОВА" if model.complete else "НЕПОЛНАЯ"
        print(f"  [{index}] {model.name:<42} | {format_size(model.size_bytes):>10} | {state}")
        if not model.complete:
            for problem in model.problems[:2]:
                print(f"       └─ {problem}")
    print("  [0] Выбрать все модели")

    selected_indices = ask_multi_choice("Какие модели переносить? ", len(models))
    selected_models = [models[index - 1] for index in selected_indices]
    ready_models = [model for model in selected_models if model.complete]
    broken_models = [model for model in selected_models if not model.complete]
    total_size = sum(model.size_bytes for model in ready_models)

    if broken_models:
        print(f"\n[ПРЕДУПРЕЖДЕНИЕ] Неполных моделей пропущено: {len(broken_models)}.")
    if not ready_models:
        print("Нет полных моделей для переноса.")
        return

    destination_dir = choose_destination(source_dir, is_export)
    if paths_overlap(source_dir, destination_dir):
        print("Ошибка: источник и папка назначения совпадают или вложены друг в друга.")
        return

    print("\n" + "=" * 76)
    print(f"Режим:       {'ЭКСПОРТ' if is_export else 'ИМПОРТ'}")
    print(f"Источник:    {source_dir}")
    print(f"Назначение:  {destination_dir}")
    print(f"Моделей:     {len(ready_models)} (неполные пропущены: {len(broken_models)})")
    print(f"Объём:       {format_size(total_size)}")
    print("Проверка:    SHA-256 для каждого blob; манифест записывается последним")
    print("=" * 76)

    confirm = ask_text("Начать копирование? [Y/n]: ", allow_empty=True).lower()
    if confirm in {"n", "no", "н", "нет"}:
        print("Операция отменена.")
        return

    destination_dir.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    failures: list[tuple[str, str]] = []
    verified_destination_blobs: set[str] = set()
    for model in ready_models:
        try:
            transfer_single_model(model, source_dir, destination_dir, verified_destination_blobs)
            succeeded += 1
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append((model.name, str(exc)))
            print(f"[ОШИБКА] {model.name}: {exc}")

    print("\n" + "=" * 76)
    print(f"Перенесено успешно: {succeeded} из {len(ready_models)}")
    if failures:
        print(f"Ошибок: {len(failures)}")
        for name, error in failures:
            print(f"  - {name}: {error}")
    print(f"Папка назначения: {destination_dir}")
    if succeeded and not is_export and platform.system() == "Linux":
        answer = ask_text(
            "Назначить владельцем файлов пользователя ollama? Требуются права root. [y/N]: "
        ).lower()
        if answer in {"y", "yes", "д", "да"}:
            fix_linux_permissions(destination_dir)
    elif succeeded and not is_export:
        print("Импорт завершён. Перезапуск Ollama обычно не требуется; проверьте модели командой: ollama list")
    print("=" * 76)


def show_storages() -> None:
    found_dirs = find_all_models_dirs()
    print(f"\nНайдено хранилищ: {len(found_dirs)}")
    if not found_dirs:
        print("Хранилища автоматически не найдены.")
    for index, directory in enumerate(found_dirs, 1):
        models = get_models_info(directory)
        total = sum(model.size_bytes for model in models)
        complete = sum(model.complete for model in models)
        print(f"\n[{index}] {directory}")
        print(f"    Моделей: {len(models)}, полных: {complete}, объём: {format_size(total)}")
        for model in models:
            state = "OK" if model.complete else "НЕПОЛНАЯ"
            print(f"    - {model.name} ({format_size(model.size_bytes)}) [{state}]")


def interactive_menu() -> None:
    while True:
        print("\n" + "=" * 76)
        print(f"  {APP_NAME} — {platform.system()} {platform.release()}")
        print("=" * 76)
        print(f"Python: {platform.python_version()}")
        print(f"OLLAMA_MODELS: {os.environ.get('OLLAMA_MODELS', 'не задана в этой оболочке')}")
        print(f"Путь назначения по умолчанию: {default_models_dir()}")
        print("\nВыберите действие:")
        print("  [1] ЭКСПОРТ — с этого компьютера на флешку / другой диск")
        print("  [2] ИМПОРТ  — с флешки / другого компьютера в Ollama")
        print("  [3] ПОИСК   — показать найденные хранилища и модели")
        print("  [0] Выход")
        choice = ask_choice("\nВаш выбор [0-3]: ", 3, allow_zero=True)
        if choice == 0:
            print("Выход.")
            return
        if choice == 1:
            perform_transfer(is_export=True)
        elif choice == 2:
            perform_transfer(is_export=False)
        elif choice == 3:
            show_storages()
        input("\nНажмите Enter для возврата в главное меню...")


def cli_mode() -> None:
    parser = argparse.ArgumentParser(
        description="Экспорт и импорт локальных моделей Ollama между Windows и Linux."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Показать модели в хранилище")
    list_parser.add_argument("--dir", type=Path, default=None, help="Корень хранилища Ollama")

    export_parser = subparsers.add_parser("export", help="Экспортировать одну модель")
    export_parser.add_argument("model", help="Имя модели, например qwen3:8b")
    export_parser.add_argument("--from-dir", type=Path, default=None, help="Источник")
    export_parser.add_argument("--to-dir", type=Path, required=True, help="Папка экспорта")

    import_parser = subparsers.add_parser("import", help="Импортировать одну модель")
    import_parser.add_argument("model", help="Имя модели, например qwen3:8b")
    import_parser.add_argument("--from-dir", type=Path, required=True, help="Папка экспорта")
    import_parser.add_argument("--to-dir", type=Path, default=None, help="Хранилище Ollama")
    import_parser.add_argument(
        "--fix-linux-permissions", action="store_true",
        help="После импорта выполнить chown -R ollama:ollama (Linux)",
    )

    args = parser.parse_args()
    if args.command == "list":
        directory = args.dir or default_models_dir()
        if not is_models_dir(directory):
            raise SystemExit(f"Не найдено хранилище Ollama: {directory}")
        models = get_models_info(directory)
        print(f"Хранилище: {directory}\nМоделей: {len(models)}")
        for model in models:
            state = "OK" if model.complete else "НЕПОЛНАЯ: " + "; ".join(model.problems)
            print(f"- {model.name} ({format_size(model.size_bytes)}) [{state}]")
        return

    if args.command == "export":
        source_dir = args.from_dir or default_models_dir()
        destination_dir = args.to_dir
        if not is_models_dir(source_dir):
            raise SystemExit(f"Не найдено хранилище-источник: {source_dir}")
    else:
        source_dir = args.from_dir
        destination_dir = args.to_dir or default_models_dir()
        if not is_models_dir(source_dir):
            raise SystemExit(f"Не найдена папка экспорта: {source_dir}")

    if paths_overlap(source_dir, destination_dir):
        raise SystemExit("Источник и назначение совпадают или вложены друг в друга.")

    models = get_models_info(source_dir)
    model = next((item for item in models if item.name == args.model), None)
    if model is None:
        raise SystemExit(f"Модель '{args.model}' не найдена в {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    transfer_single_model(model, source_dir, destination_dir)

    if args.command == "import" and args.fix_linux_permissions:
        fix_linux_permissions(destination_dir)


def main() -> int:
    try:
        if len(sys.argv) == 1:
            interactive_menu()
        else:
            cli_mode()
        return 0
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"\n[ОШИБКА] {exc}", file=sys.stderr)
        return 1
    except SystemExit as exc:
        # Сохраняем корректные коды argparse и штатного выхода из меню.
        if isinstance(exc.code, int):
            return exc.code
        if exc.code:
            print(exc.code, file=sys.stderr)
            return 2
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
