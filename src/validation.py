"""Independent intrinsic, diagnostic and literature-reference checks.

No verification test is used to select random seeds or retune training hyperparameters.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import csv
import unicodedata
import urllib.request

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from . import core

BENCHMARK_FILE='syn-selection-benchmark-Latin.tsv'
BENCHMARK_SOURCES=(
 'https://raw.githubusercontent.com/CIRCSE/Lemma-Embeddings-for-Latin/master/syn-selection-benchmark-Latin.tsv',
 'https://zenodo.org/records/4022818/files/syn-selection-benchmark-Latin.tsv?download=1',
)
LASLA_FILE='allLASLAlemmi-vector-100-nocase-w5-CBOW.vec'
LASLA_URL='https://embeddings.lila-erc.eu/samples/download/word2vec/'+LASLA_FILE


def download_once(path,urls):
    path=Path(path)
    if path.is_file():return path
    path.parent.mkdir(parents=True,exist_ok=True)
    errors=[]
    for url in urls:
        tmp=path.with_suffix(path.suffix+'.part')
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'latinise-final-reproducibility/1.0'})
            with urllib.request.urlopen(request,timeout=120) as inp,tmp.open('wb') as dest:
                while True:
                    block=inp.read(1<<20)
                    if not block:break
                    dest.write(block)
            if tmp.stat().st_size<100 or b'<html' in tmp.read_bytes()[:200].lower():
                raise ValueError('Not a valid data download.')
            tmp.replace(path)
            return path
        except Exception as e:
            errors.append(f'{url}: {e}')
            tmp.unlink(missing_ok=True)
    raise RuntimeError('Download failed. Put the file manually at '+str(path)+'\n'+'\n'.join(errors))


def ensure_benchmark(download=True):
    path=core.DATA/'benchmark'/BENCHMARK_FILE
    if not path.exists():
        if not download:raise FileNotFoundError(path)
        download_once(path,BENCHMARK_SOURCES)
    return path


def ensure_opera(download=True):
    path=core.DATA/'reference_models'/LASLA_FILE
    if not path.exists():
        if not download:raise FileNotFoundError(path)
        download_once(path,[LASLA_URL])
    return path


def load_opera(path=None):
    path=Path(path) if path else ensure_opera()
    return KeyedVectors.load_word2vec_format(str(path),binary=False,unicode_errors='replace')


def norm(word):
    w=unicodedata.normalize('NFKD',str(word).strip().lower())
    return ''.join(ch for ch in w if not unicodedata.combining(ch)).replace('v','u').replace('j','i')


def parse_benchmark(path):
    with Path(path).open('r',encoding='utf-8-sig',newline='') as inp:
        rows=[r for r in csv.reader(inp,delimiter='\t') if r and any(x.strip() for x in r)]
    if not rows:raise ValueError('Empty benchmark.')
    if any(x.lower() in ('target','synonym','gold','lemma') for x in rows[0]):rows=rows[1:]
    parsed=[]
    for i,row in enumerate(rows,1):
        if len(row)!=5 or any(not x.strip() for x in row):
            raise ValueError(f'Bad official 5-column benchmark task {i}: {row!r}')
        parsed.append({'task_id':i,'target':row[0].strip(),'gold':row[1].strip(),
            'decoy1':row[2].strip(),'decoy2':row[3].strip(),'decoy3':row[4].strip()})
    return pd.DataFrame(parsed)


def lookup(wv):
    result={}
    ambiguous=set()
    for key in wv.key_to_index:
        n=norm(key)
        if n in result and result[n]!=key:ambiguous.add(n)
        else:result[n]=key
    for n in ambiguous:result.pop(n,None)
    return result


def evaluate(name,wv,tasks):
    index=lookup(wv)
    output=[]
    for row in tasks.itertuples(index=False):
        words=[row.target,row.gold,row.decoy1,row.decoy2,row.decoy3]
        keys=[index.get(norm(word)) for word in words]
        missing=[word for word,key in zip(words,keys) if key is None]
        result={'model':name,'task_id':row.task_id,'covered':not missing,
                'missing':', '.join(missing),'correct':pd.NA,'winner':'','gold_cosine':np.nan}
        if not missing:
            sims=[float(wv.similarity(keys[0],key)) for key in keys[1:]]
            winner=int(np.argmax(sims))
            result.update({'correct':winner==0,'winner':words[winner+1],'gold_cosine':sims[0]})
        output.append(result)
    return pd.DataFrame(output)


def _accuracy(df,ids=None):
    sampled=df if ids is None else df[df.task_id.isin(ids)]
    covered=sampled[sampled.covered]
    return {'n_eligible':len(sampled),'n_covered':len(covered),
        'coverage':len(covered)/len(sampled) if len(sampled) else np.nan,
        'accuracy':float(covered.correct.astype(bool).mean()) if len(covered) else np.nan}


def synonym_reports(selected,opera,tasks,folder):
    evaluated={}
    for (period,mc),m in selected.items():
        name=f'{period}_mc{mc}'
        evaluated[name]=evaluate(name,m.wv,tasks)
    evaluated['opera_latina']=evaluate('opera_latina',opera,tasks)
    long=pd.concat(evaluated.values(),ignore_index=True)
    long.to_csv(folder/'synonym_items.csv',index=False)
    summary=[]
    for name,df in evaluated.items():
        summary.append({'model':name,'evaluation':'own_available',**_accuracy(df)})
    # Paired controls: same five words per question in both compared models.
    comparisons=[('pre180_mc5','opera_latina'),('pre180_mc50','opera_latina'),
      ('pre180_mc5','pre180_mc50'),('post180_mc5','post180_mc50')]
    paired=[]
    for a,b in comparisons:
        aa=evaluated[a].set_index('task_id')
        bb=evaluated[b].set_index('task_id')
        shared=sorted(set(aa.index[aa.covered])&set(bb.index[bb.covered]))
        for model in (a,b):
            summary.append({'model':model,'evaluation':f'shared:{a}|{b}',**_accuracy(evaluated[model],shared)})
        for task in shared:
            paired.append({'comparison':f'{a}|{b}','task_id':task,'model_a':a,'model_b':b,
               'a_correct':bool(aa.loc[task,'correct']),'b_correct':bool(bb.loc[task,'correct'])})
    pd.DataFrame(summary).to_csv(folder/'synonym_summary.csv',index=False)
    pd.DataFrame(paired,columns=['comparison','task_id','model_a','model_b','a_correct','b_correct']).to_csv(folder/'synonym_paired.csv',index=False)
    return pd.DataFrame(summary)


def diagnostic_pairs(selected,folder):
    source=core.DATA/'diagnostic_pairs.csv'
    pairs=pd.read_csv(source)
    out=[]
    for (s,mc),model in selected.items():
        runs=core.load_runs(s,mc)
        for record in pairs.itertuples(index=False):
            a,b=record.word1,record.word2
            # The diagnostic list is NOT a gold standard. Antonyms may also be close!
            sims=[]
            for run in runs:
                if a in run.wv.key_to_index and b in run.wv.key_to_index:
                    sims.append(float(run.wv.similarity(a,b)))
            out.append({'slice':s,'min_count':mc,'word1':a,'word2':b,'relation_type':record.relation_type,
                'n_runs':len(sims),'mean_cosine':float(np.mean(sims)) if sims else np.nan,
                'sd_cosine':float(np.std(sims,ddof=1)) if len(sims)>1 else np.nan})
    result=pd.DataFrame(out)
    result.to_csv(folder/'diagnostic_pairs.csv',index=False)
    return result


def lunardi_reference(selected,folder):
    reference=pd.read_csv(core.DATA/'lunardi_reference_neighbours.csv')
    rows=[]
    for (s,mc),model in selected.items():
        for lemma in sorted(reference.lemma.unique()):
            ref=reference[(reference.slice==s)&(reference.lemma==lemma)]
            if ref.empty or lemma not in model.wv.key_to_index:continue
            published=list(ref.sort_values('rank').neighbour)
            ours=[word for word,_ in model.wv.most_similar(lemma,topn=10)]
            exact=(int(ref.window.iloc[0])==core.config()['window'] and int(ref.min_count.iloc[0])==mc)
            rows.append({'slice':s,'min_count':mc,'lemma':lemma,
              'reference_window':int(ref.window.iloc[0]),'reference_min_count':int(ref.min_count.iloc[0]),
              'matching_hyperparameters':exact,'overlap_top10':len(set(ours)&set(published)),
              'reference_neighbors':', '.join(published),'our_neighbors':', '.join(ours),
              'note':'qualitative reference only; corpus/case/tokenization may differ'})
    result=pd.DataFrame(rows)
    result.to_csv(folder/'lunardi_neighbors.csv',index=False)
    return result


def neighbor_reproducibility(folder):
    """Measures 10-seed stability on only the prior 300 neutral anchors, not subject terms."""
    records=[]
    for mc in core.config()['min_counts']:
        for s in core.SLICES:
            anchors=pd.read_csv(core.RESULTS/'selected'/f'{s}_mc{mc}_anchors.csv')
            records.append({'slice':s,'min_count':mc,'mean_anchor_jaccard':float(anchors.jaccard_mean.mean()),
                'median_anchor_jaccard':float(anchors.jaccard_mean.median()),'n_anchors':len(anchors)})
    report=pd.DataFrame(records)
    report.to_csv(folder/'stability_summary.csv',index=False)
    return report


def verification(download=True):
    core.verify_audit()
    _,selected=core.load_selected()
    folder=core.RESULTS/'validation'
    folder.mkdir(parents=True,exist_ok=True)
    pairs=diagnostic_pairs(selected,folder)
    refs=lunardi_reference(selected,folder)
    stability=neighbor_reproducibility(folder)
    print('Diagnostics and Lunardi examples saved.')
    # Network-dependent part can be resumed independently if downloads fail.
    path=ensure_benchmark(download=download)
    opera_path=ensure_opera(download=download)
    tasks=parse_benchmark(path)
    opera=load_opera(opera_path)
    synonyms=synonym_reports(selected,opera,tasks,folder)
    core.write_json(folder/'external_sources.json',{'benchmark':str(path.relative_to(core.ROOT)),
      'opera_file':str(opera_path.relative_to(core.ROOT)),
      'benchmark_tasks':len(tasks),
      'notes':'synonym accuracy must be reported with coverage; only paired subsets compare models'})
    return {'synonyms':synonyms,'stability':stability,'lunardi':refs,'diagnostic_pairs':pairs}
