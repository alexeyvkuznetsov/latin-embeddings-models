"""Небольшие проверки без скачивания корпуса LatinISE."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from src import core, validation


def test_synonyms():
    v = KeyedVectors(vector_size=3)
    v.add_vectors(['bonus', 'optimus', 'malus', 'luna', 'sol'], np.array([
        [1,0,0], [.9,.1,0], [-1,0,0], [0,1,0], [0,0,1]], dtype=np.float32))
    questions = pd.DataFrame([
        {'task_id':1,'target':'bonus','gold':'optimus','decoy1':'malus','decoy2':'luna','decoy3':'sol'},
        {'task_id':2,'target':'bonus','gold':'optimus','decoy1':'malus','decoy2':'luna','decoy3':'missing'},
    ])
    answers = validation.evaluate('example', v, questions)
    assert answers.covered.tolist() == [True, False]
    assert bool(answers.iloc[0].correct)


def test_jaccard():
    v = KeyedVectors(vector_size=3)
    v.add_vectors(['alpha', 'beta', 'gamma', 'delta'],
                  np.eye(4, 3, dtype=np.float32) + .01)
    class Model:
        def __init__(self, vec):
            self.wv = vec
    scores, _ = core.agreement([Model(v), Model(v), Model(v)],
                               ['alpha', 'beta'], topn=1)
    assert np.allclose(scores, 1)


def test_training_and_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(core, 'ROOT', tmp_path)
    monkeypatch.setattr(core, 'DATA', tmp_path/'data')
    monkeypatch.setattr(core, 'RESULTS', tmp_path/'results')
    monkeypatch.setattr(core, 'CORPUS', tmp_path/'data/new_lemmatized_texts')
    (tmp_path/'config').mkdir()
    cfg = {'vector_size':16, 'window':5, 'epochs':2, 'min_counts':[5, 50],
           'negative':5,'sample':.001,'seeds':[1,2]}
    (tmp_path/'config/settings.json').write_text(json.dumps(cfg))
    core.CORPUS.mkdir(parents=True)
    # 420 fairly frequent content types provide enough neutral anchors.
    vocab = [f'term{chr(97+i//26)}{chr(97+i%26)}' for i in range(420)]
    sentence = ' '.join(vocab)
    for f in ('early.txt','late.txt'):
        (core.CORPUS/f).write_text((sentence+'\n')*52)
    pd.DataFrame([
        {'file':'early.txt','date_range_end':170,'data_issue':''},
        {'file':'late.txt','date_range_end':400,'data_issue':''},
    ]).to_csv(core.DATA/'metadata.csv',index=False)
    summary = core.prepare()
    assert summary.tokens.min()>20000
    assert len(pd.read_csv(core.DATA/'slices/pre180.csv')) == 1
    for period in core.SLICES:
        for mc in cfg['min_counts']:
            for seed in cfg['seeds']:
                path,status = core.train_one(period,mc,seed)
                assert path.exists() and status == 'обучена'
                assert core.train_one(period,mc,seed)[1] == 'уже существует'
    # synthetic corpus ties can still produce a valid representative model
    result = core.selection()
    assert len(result) == 4
    _, models = core.load_selected()
    assert len(models) == 4


def test_load_selected_legacy_windows_path(tmp_path, monkeypatch):
    monkeypatch.setattr(core, 'ROOT', tmp_path)
    monkeypatch.setattr(core, 'RESULTS', tmp_path / 'results')
    folder = core.RESULTS / 'selected'
    folder.mkdir(parents=True)
    records = [{'slice': 'pre180', 'min_count': 5,
                'model': r'results\selected\pre180_mc5.model'}]
    (folder / 'selection.json').write_text(json.dumps(records))
    expected = folder / 'pre180_mc5.model'
    seen = []
    monkeypatch.setattr(core.FastText, 'load', lambda path: seen.append(Path(path)) or 'loaded')
    _, models = core.load_selected()
    assert seen == [expected]
    assert models[('pre180', 5)] == 'loaded'
