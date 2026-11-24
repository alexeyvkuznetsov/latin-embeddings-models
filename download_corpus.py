"""Получение проверенной версии LatinISE без повторной публикации текстов.

По умолчанию скачиваются только отсутствующие лемматизированные файлы.
Нелемматизированные тексты доступны с --with-originals.
Наличие и идентичность файлов проверяются по Git blob SHA исходного дерева.
"""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
from pathlib import Path
import argparse
import csv
import json
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
UPSTREAM = 'BarbaraMcG/latinise'
REVISION = '39fe6e72bbec494dfc37693033be8e51622c73ba'
PREFIX = 'christianity_semantic_change/data/'
TREE_URL = f'https://api.github.com/repos/{UPSTREAM}/git/trees/{REVISION}?recursive=1'
RAW_URL = f'https://raw.githubusercontent.com/{UPSTREAM}/{REVISION}/'


def fetch(url):
    """Ограниченное число повторов при временных сетевых ошибках."""
    for attempt in range(3):
        try:
            req = Request(url, headers={'User-Agent': 'latin-embeddings-models/1.0'})
            with urlopen(req, timeout=90) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def git_blob_sha(data):
    return sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def matches_file(path, expected_sha):
    if not path.is_file():
        return False
    data = path.read_bytes()
    # Git checkout on Windows may replace LF with CRLF. The source bytes
    # are never rewritten; only an exact equivalent line-ending check is allowed.
    return (git_blob_sha(data) == expected_sha or
            git_blob_sha(data.replace(b'\r\n', b'\n')) == expected_sha)


def source_files(tree, with_originals=False):
    if tree.get('truncated'):
        raise ValueError('GitHub вернул неполное дерево исходного репозитория.')
    folders = ['new_lemmatized_texts']
    if with_originals:
        folders.append('non_lemmatized_texts')
    files = {}
    for folder in folders:
        prefix = PREFIX + folder + '/'
        items = {x['path'][len(PREFIX):]: x['sha'] for x in tree['tree']
                 if x['type'] == 'blob' and x['path'].startswith(prefix)
                 and x['path'].endswith('.txt')}
        if len(items) != 1153:
            raise ValueError(f'{folder}: ожидается 1153 файла, найдено {len(items)}.')
        files.update(items)
    return files


def check_metadata(data_dir, files):
    with (data_dir / 'metadata.csv').open(encoding='utf-8-sig', newline='') as handle:
        names = {row['file'] for row in csv.DictReader(handle)}
    missing = sorted(name for name in names if 'new_lemmatized_texts/' + name not in files)
    if missing:
        raise ValueError(f'В источнике нет {len(missing)} файлов из metadata.csv: {missing[:5]}')


def download_one(data_dir, relative_path, expected_sha):
    path = data_dir / relative_path
    if path.exists():
        if not matches_file(path, expected_sha):
            raise ValueError(f'Локальный файл отличается от оригинала: {path}')
        return False
    data = fetch(RAW_URL + quote(PREFIX + relative_path, safe='/'))
    if git_blob_sha(data) != expected_sha:
        raise ValueError(f'Полученный файл не совпадает с исходным Git blob: {relative_path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    try:
        tmp.write_bytes(data)
        # Do not replace a file created in the meantime.
        if path.exists():
            raise FileExistsError(path)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def prepare_corpus(data_dir=None, with_originals=False, check_only=False, workers=6):
    data_dir = Path(data_dir) if data_dir else ROOT / 'data'
    tree = json.loads(fetch(TREE_URL))
    files = source_files(tree, with_originals)
    check_metadata(data_dir, files)
    missing, changed = [], []
    for rel, sha in sorted(files.items()):
        path = data_dir / rel
        if not path.exists():
            missing.append((rel, sha))
        elif not matches_file(path, sha):
            changed.append(rel)
    if changed:
        raise ValueError(f'{len(changed)} локальных файлов отличаются от источника: {changed[:5]}. '
                         'Сохраните их отдельно; автоматической замены нет.')
    if check_only and missing:
        raise FileNotFoundError(f'Нет {len(missing)} файлов: {[r for r, _ in missing[:5]]}')
    print(f'Версия источника: {REVISION}. Уже имеется {len(files)-len(missing)} из {len(files)} файлов.')
    if missing:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, _ in enumerate(pool.map(lambda item: download_one(data_dir, *item), missing), 1):
                if i % 100 == 0 or i == len(missing):
                    print(f'Получено {i}/{len(missing)} отсутствующих файлов')
    print(f'Проверено {len(files)} файлов; исходные локальные тексты не изменены.')
    return {'expected': len(files), 'downloaded': len(missing), 'revision': REVISION}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-originals', action='store_true', help='Также получить нелемматизированные тексты')
    parser.add_argument('--check-only', action='store_true', help='Только проверка, без скачивания текстов')
    args = parser.parse_args()
    prepare_corpus(with_originals=args.with_originals, check_only=args.check_only)
