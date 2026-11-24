"""Обучение и выбор статических моделей LatinISE.

Входные тексты считаются уже подготовленными; происхождение проверяет download_corpus.py.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
import json
import shutil
import zlib

import numpy as np
import pandas as pd
from gensim.models import FastText

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
CORPUS = DATA / 'new_lemmatized_texts'
SLICES = ('pre180', 'post180')

# Только для нейтральной выборки, по которой выбирается представитель серии.
FUNCTION_WORDS = set("""et in sum non qui quis is ad que ut cum hic ille sed si a ab de ex e se sui suus per
nec neque aut uel atque ac tu ego nos uos meus tuus noster uester quidem autem etiam tam tamen iam nunc tunc sic
ita quam quod quia quoniam ne an nisi ubi unde post ante inter sub super pro sine apud contra propter omnis ipse
idem iste alius alter nullus nihil quisque aliquis uterque ergo igitur itaque nam namque uero modo solum tantum
adhuc semper numquam saepe magis minus plus ue o quoque item dum donec postquam priusquam licet quamuis etsi
huc illuc hinc inde ibi quo qua sicut uelut tamquam quasi""".split())
FOCAL_WORDS = set('populus plebs plebes gens natio ciuitas ciuis uulgus turba multitudo ecclesia lex ius imperium potestas auctoritas'.split())


def config():
    return json.loads((ROOT / 'config/settings.json').read_text(encoding='utf-8'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def metadata():
    """Произведения по датировкам Лунарди: до 180 и 180–605 гг."""
    df = pd.read_csv(DATA / 'metadata.csv')
    end = pd.to_numeric(df['date_range_end'], errors='coerce')
    df['slice'] = np.where(end < 180, 'pre180', np.where(end <= 605, 'post180', ''))
    df = df[(df['slice'] != '') & df['data_issue'].isna()].copy()
    return df


def files_by_slice(period):
    path = DATA / 'slices' / f'{period}.csv'
    if not path.exists():
        raise FileNotFoundError(f'Сначала выполните ноутбук 00: нет {path}')
    return sorted(pd.read_csv(path)['file'].tolist())


def tokens(line):
    """Приведение регистра, v → u и удаление краевой пунктуации."""
    import unicodedata
    result = []
    for raw in line.split():
        word = raw.strip().lower().replace('v', 'u')
        while word and not unicodedata.category(word[0]).startswith(('L', 'N')):
            word = word[1:]
        while word and not unicodedata.category(word[-1]).startswith(('L', 'N')):
            word = word[:-1]
        if word and any(unicodedata.category(ch).startswith('L') for ch in word):
            result.append(word)
    return result


class Sentences:
    """Список лемм каждой строки: одна строка = одно предложение."""
    def __init__(self, period):
        self.files = files_by_slice(period)

    def __iter__(self):
        for name in self.files:
            with (CORPUS / name).open(encoding='utf-8') as fh:
                for line in fh:
                    sentence = tokens(line)
                    if sentence:
                        yield sentence


def prepare():
    """Только список файлов и статистика готового корпуса; никаких правок и хешей."""
    if not CORPUS.is_dir():
        raise FileNotFoundError(f'Поместите готовые тексты в {CORPUS} или выполните python download_corpus.py')
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = metadata()
    stats = []
    for period in SLICES:
        part = df[df['slice'] == period].sort_values('file')
        missing = [name for name in part.file if not (CORPUS / name).is_file()]
        if missing:
            raise FileNotFoundError(f'{period}: нет {len(missing)} файлов, например {missing[:5]}')
        if part.empty:
            raise ValueError(f'В срезе {period} нет текстов')
        path = DATA / 'slices' / f'{period}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(path, index=False)
        counts = Counter()
        sentences = 0
        for words in Sentences(period):
            sentences += 1
            counts.update(words)
        if not sentences:
            raise ValueError(f'Срез {period} не содержит предложений')
        pd.DataFrame(sorted(counts.items()), columns=['lemma', 'frequency']).to_csv(
            RESULTS / f'{period}_frequencies.csv', index=False)
        stats.append({'period': period, 'texts': len(part), 'sentences': sentences,
                      'tokens': sum(counts.values()), 'lemmas': len(counts),
                      'min_count_5': sum(n >= 5 for n in counts.values()),
                      'min_count_50': sum(n >= 50 for n in counts.values())})
    result = pd.DataFrame(stats)
    RESULTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS / 'corpus_summary.csv', index=False)
    return result


def run_path(period, min_count, seed):
    return RESULTS / 'runs' / period / f'mc{min_count}' / f'seed{seed}.model'


def train_one(period, min_count, seed):
    """Пропускает существующую модель; если корпус менялся, удалите старые модели."""
    path = run_path(period, min_count, seed)
    if path.exists():
        return path, 'уже существует'
    c = config()
    model = FastText(vector_size=c['vector_size'], window=c['window'],
                     min_count=min_count, sg=0, min_n=0, max_n=0,
                     bucket=0, negative=c['negative'], sample=c['sample'],
                     workers=1, seed=seed,
                     hashfxn=lambda w: zlib.crc32(w.encode('utf-8')))
    model.build_vocab(corpus_iterable=Sentences(period))
    model.train(corpus_iterable=Sentences(period),
                total_examples=model.corpus_count, epochs=c['epochs'])
    # Отключённые символьные n-граммы не сохраняются как лишняя большая таблица.
    model.hashfxn = hash
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path), sep_limit=2**63 - 1)
    return path, 'обучена'


def load_runs(period, min_count):
    return [FastText.load(str(run_path(period, min_count, seed)))
            for seed in config()['seeds']]


def choose_anchors(period):
    """300 нейтральных лемм, одинаковых для обоих порогов данного периода."""
    table = pd.read_csv(RESULTS / f'{period}_frequencies.csv')
    table = table[(table.frequency >= 50) & table.lemma.str.fullmatch('[a-z]+', na=False)]
    table = table[~table.lemma.isin(FUNCTION_WORDS | FOCAL_WORDS)]
    table = table.sort_values(['frequency', 'lemma'], ascending=[False, True])
    strata = np.array_split(table, 3)
    rng = np.random.default_rng(2026)
    selected = []
    for label, group in zip(('high', 'middle', 'low'), strata):
        if len(group) < 100:
            raise ValueError(f'Недостаточно контрольных лемм в группе {label}')
        sample = group.iloc[sorted(rng.choice(len(group), 100, replace=False))]
        for row in sample.itertuples(index=False):
            selected.append({'lemma': row.lemma, 'frequency': row.frequency, 'stratum': label})
    return pd.DataFrame(selected)


def agreement(models, words, topn=15):
    """Средний Жаккар списков ближайших соседей между независимыми запусками."""
    words = list(words)
    neighborhoods = [
        {w: set(term for term, _ in model.wv.most_similar(w, topn=topn)) for w in words}
        for model in models
    ]
    scores = np.zeros(len(models))
    word_scores = []
    for word in words:
        values = []
        for a, b in combinations(range(len(models)), 2):
            x, y = neighborhoods[a][word], neighborhoods[b][word]
            similarity = len(x & y) / len(x | y)
            scores[a] += similarity
            scores[b] += similarity
            values.append(similarity)
        word_scores.append({'lemma': word, 'jaccard_mean': np.mean(values),
                            'jaccard_sd': np.std(values)})
    scores /= len(words) * (len(models) - 1)
    return scores, pd.DataFrame(word_scores)


def selection():
    c = config()
    all_results = []
    destination = RESULTS / 'selected'
    destination.mkdir(parents=True, exist_ok=True)
    for period in SLICES:
        anchors = choose_anchors(period)
        for min_count in c['min_counts']:
            models = load_runs(period, min_count)
            scores, per_word = agreement(models, anchors.lemma, topn=15)
            ranking = pd.DataFrame({'seed': c['seeds'], 'score_top15': scores})
            ranking = ranking.sort_values(['score_top15', 'seed'], ascending=[False, True])
            ranking.to_csv(destination / f'{period}_mc{min_count}_ranking.csv', index=False)
            anchors.merge(per_word, on='lemma').to_csv(
                destination / f'{period}_mc{min_count}_anchors.csv', index=False)
            winner = int(ranking.iloc[0].seed)
            target = destination / f'{period}_mc{min_count}.model'
            shutil.copyfile(run_path(period, min_count, winner), target)
            all_results.append({'slice': period, 'min_count': min_count,
                                'selected_seed': winner,
                                'neighbor_medoid_score': float(ranking.iloc[0].score_top15),
                                'margin_to_second': float(ranking.iloc[0].score_top15 - ranking.iloc[1].score_top15),
                                'model': target.relative_to(ROOT).as_posix()})
            print(f'{period}, min_count={min_count}: выбран запуск {winner}')
    df = pd.DataFrame(all_results)
    df.to_csv(destination / 'selection.csv', index=False)
    write_json(destination / 'selection.json', all_results)
    return df


def load_selected():
    path = RESULTS / 'selected' / 'selection.json'
    records = json.loads(path.read_text(encoding='utf-8'))
    models = {(record['slice'], record['min_count']):
              FastText.load(str(ROOT / record['model'].replace('\\', '/'))) for record in records}
    return records, models
