"""Контрольные срезы LatinISE: исключение Вульгаты и двух книг Юстиниана.

Обучение и отбор используют функции src.core без изменения базовых моделей.
"""
from collections import Counter
from pathlib import Path
import shutil
import hashlib

import pandas as pd
from . import core

VULGATE = ('IT-LAT0001',)
JUSTINIAN = ('IT-LAT0867', 'IT-LAT0202')
EXCLUSIONS = {
    'post180_no_vulgate': VULGATE,
    'post180_no_justinian': JUSTINIAN,
    'post180_no_vulgate_no_justinian': VULGATE + JUSTINIAN,
}
# Диагностика для пилота «Свобода и рабство», не используется для выбора медоида.
TARGETS = '''libertas libero seruitus seruitium seruus ancilla famulus famula seruio
libertus libertinus ingenuus mancipium captiuus dominus domina dominium dominatio potestas dominor
liber arbitrium uoluntas licentia obsequium oboedio pareo subicio seruilis manumitto redimo redemptio
captiuitas iugum uinculum liberatio manumissio oboedientia imperium gratia peccatum'''.split()


def read_slice(period):
    path = core.DATA / 'slices' / f'{period}.csv'
    df = pd.read_csv(path)
    if df.file.duplicated().any() or df.id.duplicated().any():
        raise ValueError(f'{path}: повторяющиеся id или file')
    if 'slice' not in df or not df['slice'].eq(period).all():
        raise ValueError(f'{path}: некорректный исходный срез')
    expected = core.metadata().query('slice == @period')
    if set(zip(df.id, df.file)) != set(zip(expected.id, expected.file)):
        raise ValueError(f'{path}: состав не совпадает с metadata.csv; проверьте исходную версию')
    return df.sort_values('file').reset_index(drop=True)


def count_texts(df, period):
    """Та же нормализация и единица контекста, что в core.Sentences."""
    total = Counter()
    removed = {}
    audit = []
    for row in df.itertuples(index=False):
        path = core.CORPUS / row.file
        counts = Counter()
        sentences = 0
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                words = core.tokens(line)
                if words:
                    sentences += 1
                    counts.update(words)
        tokens = sum(counts.values())
        total.update(counts)
        block = ('vulgate' if row.id in VULGATE else
                 'justinian' if row.id in JUSTINIAN else 'other')
        if period == 'post180' and block != 'other':
            removed[row.id] = (counts, sentences)
        audit.append({'period': period, 'id': row.id, 'file': row.file,
                      'creator': row.creator, 'title': row.title, 'block': block,
                      'tokens': tokens, 'sentences': sentences,
                      **{w: counts[w] for w in TARGETS}})
    # Сверяем с опубликованной статистикой, не пересоздаём базовый срез.
    old = pd.read_csv(core.RESULTS / f'{period}_frequencies.csv', keep_default_na=False)
    expected = Counter(dict(zip(old.lemma, old.frequency)))
    if total != expected:
        raise ValueError(f'{period}: частоты локальных текстов не совпадают с исходной таблицей. '
                         'Остановлено; проверьте версию корпуса.')
    summary = pd.read_csv(core.RESULTS / 'corpus_summary.csv').set_index('period').loc[period]
    if (sum(total.values()), sum(r['sentences'] for r in audit), len(df)) != (
            int(summary.tokens), int(summary.sentences), int(summary.texts)):
        raise ValueError(f'{period}: статистика не совпадает с опубликованной')
    return total, removed, audit


def prepare(include_early_audit=True):
    """Сохранить только новые срезы и частотные таблицы; исходные файлы не меняются."""
    late = read_slice('post180')
    required = set(VULGATE + JUSTINIAN)
    if not required <= set(late.id):
        raise ValueError(f'В позднем срезе отсутствуют {required - set(late.id)}')
    total, removed, audit = count_texts(late, 'post180')
    late_sentences = sum(r['sentences'] for r in audit)
    if include_early_audit:
        _, _, early_audit = count_texts(read_slice('pre180'), 'pre180')
        audit = early_audit + audit
    out = core.RESULTS / 'ablation'
    out.mkdir(parents=True, exist_ok=True)
    work = pd.DataFrame(audit)
    work.to_csv(out / 'target_counts_by_work.csv', index=False)
    groups = work.groupby(['period', 'block'], as_index=False)[['tokens', *TARGETS]].sum()
    groups.to_csv(out / 'target_counts_by_block.csv', index=False)
    stats = []
    exclusions = []
    for period, ids in EXCLUSIONS.items():
        part = late[~late.id.isin(ids)].copy()
        part['slice'] = period
        part.to_csv(core.DATA / 'slices' / f'{period}.csv', index=False)
        counts = total.copy()
        sentences = late_sentences
        for work_id in ids:
            cnt, n = removed[work_id]
            counts.subtract(cnt)
            sentences -= n
        counts = +counts
        pd.DataFrame(sorted(counts.items()), columns=['lemma','frequency']).to_csv(
            core.RESULTS / f'{period}_frequencies.csv', index=False)
        stats.append({'period': period, 'texts': len(part), 'sentences': sentences,
                      'tokens': sum(counts.values()), 'lemmas': len(counts),
                      'min_count_5': sum(n >= 5 for n in counts.values()),
                      'min_count_50': sum(n >= 50 for n in counts.values())})
        exclusions.extend({'period': period, 'id': r.id, 'file': r.file,
                           'title': r.title, 'tokens': sum(removed[r.id][0].values())}
                          for r in late[late.id.isin(ids)].itertuples(index=False))
    stats = pd.DataFrame(stats)
    stats.to_csv(out / 'corpus_summary.csv', index=False)
    pd.DataFrame(exclusions).to_csv(out / 'excluded_works.csv', index=False)
    return stats


def train(min_counts=(5,), periods=None):
    """По десять инициализаций на конфигурацию, как в основной библиотеке."""
    periods = list(EXCLUSIONS) if periods is None else list(periods)
    config = core.config()
    if not set(periods) <= EXCLUSIONS.keys() or not set(min_counts) <= set(config['min_counts']):
        raise ValueError('Неизвестный контрольный срез или min_count')
    records = []
    for period in periods:
        if not (core.DATA / 'slices' / f'{period}.csv').is_file():
            raise FileNotFoundError(f'Сначала выполните prepare(): {period}')
        for mc in min_counts:
            for seed in config['seeds']:
                path, status = core.train_one(period, mc, seed)
                records.append({'period': period, 'min_count': mc, 'seed': seed,
                                'status': status, 'path': path.relative_to(core.ROOT).as_posix()})
                print(period, mc, seed, status)
    df = pd.DataFrame(records)
    out = core.RESULTS / 'ablation'
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'training_log.csv', index=False)
    return df


def select(min_counts=(5,), periods=None):
    """Повторяем core.selection для новых срезов; не меняем selection.json базовой серии."""
    periods = list(EXCLUSIONS) if periods is None else list(periods)
    config = core.config()
    if not set(periods) <= EXCLUSIONS.keys() or not set(min_counts) <= set(config['min_counts']):
        raise ValueError('Неизвестный контрольный срез или min_count')
    destination = core.RESULTS / 'selected'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for period in periods:
        anchors = core.choose_anchors(period)
        freq = pd.read_csv(core.RESULTS / f'{period}_frequencies.csv', keep_default_na=False)
        for mc in min_counts:
            models = core.load_runs(period, mc)
            expected = dict(zip(freq.loc[freq.frequency >= mc, 'lemma'], freq.loc[freq.frequency >= mc, 'frequency']))
            for model in models:
                if set(model.wv.key_to_index) != set(expected) or any(
                        model.wv.get_vecattr(w, 'count') != f for w, f in expected.items()):
                    raise ValueError(f'{period} mc{mc}: сохранённая модель не соответствует частотам среза. '
                                     'Удалите устаревшие модели после резервного копирования.')
            scores, per_word = core.agreement(models, anchors.lemma, topn=15)
            ranking = pd.DataFrame({'seed': config['seeds'], 'score_top15': scores}).sort_values(
                ['score_top15', 'seed'], ascending=[False, True])
            stem = f'{period}_mc{mc}'
            ranking.to_csv(destination / f'{stem}_ranking.csv', index=False)
            anchors.merge(per_word, on='lemma').to_csv(destination / f'{stem}_anchors.csv', index=False)
            winner = int(ranking.iloc[0].seed)
            target = destination / f'{stem}.model'
            shutil.copyfile(core.run_path(period, mc, winner), target)
            blob = target.read_bytes()
            records.append({'slice': period, 'min_count': mc, 'selected_seed': winner,
                            'neighbor_medoid_score': float(ranking.iloc[0].score_top15),
                            'margin_to_second': float(ranking.iloc[0].score_top15-ranking.iloc[1].score_top15),
                            'model': target.relative_to(core.ROOT).as_posix(),
                            'git_blob_sha': hashlib.sha1(f'blob {len(blob)}\0'.encode('ascii')+blob).hexdigest(),
                            'sha256': hashlib.sha256(blob).hexdigest()})
            print(stem, 'seed', winner)
    # При запуске дополнительной серии mc50 сохранить результаты ранее выбранных mc5.
    out = core.RESULTS / 'ablation'
    out.mkdir(parents=True, exist_ok=True)
    previous = out / 'selection.csv'
    df = pd.DataFrame(records)
    if previous.exists():
        df = pd.concat([pd.read_csv(previous), df], ignore_index=True).drop_duplicates(
            ['slice', 'min_count'], keep='last')
    df = df.sort_values(['slice', 'min_count']).reset_index(drop=True)
    df.to_csv(previous, index=False)
    core.write_json(out / 'selection.json', df.to_dict(orient='records'))
    return df
