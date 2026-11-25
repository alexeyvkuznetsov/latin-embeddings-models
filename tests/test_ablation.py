"""Проверка дополнительного обучения без скачивания корпуса."""
import json
import pandas as pd
from src import core, ablation


def test_ablation_pipeline(tmp_path, monkeypatch):
    for attr, path in {'ROOT': tmp_path, 'DATA': tmp_path/'data',
                       'RESULTS': tmp_path/'results', 'CORPUS': tmp_path/'data/new_lemmatized_texts'}.items():
        monkeypatch.setattr(core, attr, path)
    (tmp_path/'config').mkdir()
    cfg = {'min_counts':[5,50], 'vector_size':12, 'window':5, 'epochs':1,
           'negative':5,'sample':.001,'seeds':[1,2]}
    (tmp_path/'config/settings.json').write_text(json.dumps(cfg))
    core.CORPUS.mkdir(parents=True)
    words = ['term'+chr(97+i//26)+chr(97+i%26) for i in range(420)]
    rows = []
    for i, ident in enumerate(['early', *ablation.VULGATE, *ablation.JUSTINIAN, 'other']):
        name = f'{i}.txt'
        text = (' '.join(words) + ' libertas\n')*52
        (core.CORPUS/name).write_text(text, encoding='utf-8')
        rows.append({'id':ident,'file':name,'title':ident,'creator':'Test',
                     'date_range_end':100 if ident=='early' else 400, 'data_issue':''})
    pd.DataFrame(rows).to_csv(core.DATA/'metadata.csv', index=False)
    core.prepare()
    baseline = (core.RESULTS/'corpus_summary.csv').read_bytes()
    old_selection = core.RESULTS/'selected/selection.json'
    old_selection.parent.mkdir(parents=True)
    old_selection.write_text('[]')
    stats = ablation.prepare()
    assert stats.texts.tolist() == [3,2,1]
    assert stats.tokens.tolist() == [3*52*421,2*52*421,52*421]
    assert (core.RESULTS/'corpus_summary.csv').read_bytes() == baseline
    assert old_selection.read_text() == '[]'
    audit = pd.read_csv(core.RESULTS/'ablation/target_counts_by_work.csv')
    assert len(audit)==5 and audit.libertas.sum()==260
    period = 'post180_no_vulgate_no_justinian'
    logs = ablation.train(periods=[period])
    assert len(logs)==2
    selected = ablation.select(periods=[period])
    assert len(selected)==1
    assert (core.ROOT / selected.iloc[0].model).is_file()
    assert old_selection.read_text() == '[]'
    assert len(selected.iloc[0].sha256)==64
    assert ablation.train(periods=[period]).status.eq('уже существует').all()
