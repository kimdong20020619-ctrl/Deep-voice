"""Collect public-domain Citizen DJ excerpts with frozen source provenance."""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import unicodedata
import urllib.request
import urllib.parse
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/raw/citizen-dj-20260923'
REPO = 'https://api.github.com/repos/LibraryOfCongress/citizen-dj'
ASSET = 'https://s3.amazonaws.com/citizen-dj-assets.labs.loc.gov/audio/samplepacks/loc-fma/'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def fetch(url, path, limit=4*1024**2):
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(url, headers={'User-Agent':'DeepVoice-source-audit/1.0'})
    with urllib.request.urlopen(request, timeout=25) as response:
        blob = response.read(limit+1)
    assert 0 < len(blob) <= limit, 'Transfer bound exceeded'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return blob


def normalize(value):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', value).casefold())


def fields(blob):
    return dict(re.findall(r'^([A-Za-z_]+): "(.*)"\s*$', blob.decode('utf-8'), re.M))


def main(collect=False):
    OUT.mkdir(parents=True, exist_ok=True)
    commit = json.loads(fetch(REPO+'/commits/master', OUT/'commit.json'))['sha']
    raw = 'https://raw.githubusercontent.com/LibraryOfCongress/citizen-dj/'+commit+'/'
    catalog = json.loads(fetch(raw+'_data/loc-fma.json', OUT/'catalog.json'))
    fetch(raw+'_collections/loc-fma.md', OUT/'rights.md')
    template = fetch(raw+'_layouts/use.html', OUT/'download-template.html').decode()
    assert '/audio/samplepacks/{{ page.uid }}/{{ clip.filename }}.wav' in template
    listing = json.loads(fetch(REPO+'/contents/_items/loc-fma?ref='+commit, OUT/'listing.json'))
    names = [x['name'] for x in listing if x['name'].endswith('.md')]
    def get_item(name):
        return fields(fetch(raw+'_items/loc-fma/'+name, OUT/'metadata'/name))
    with ThreadPoolExecutor(max_workers=4) as pool:
        items = list(pool.map(get_item, names))
    print('METADATA',len(items),'artists',sorted({x.get('contributors','') for x in items}),flush=True)
    inventory=list(csv.DictReader((ROOT/'data/raw/musan-metadata-20260912/music_inventory.csv').open(encoding='utf-8')))
    # Conservatively exclude every MUSAN artist, including unused tracks.
    excluded_artists={normalize(r['artist_raw']) for r in inventory}
    index={x['id']:x for x in catalog['items']}
    # Group related project names conservatively; not a verified identity assertion.
    related_names = {'Monplaisir', 'Komiku', 'Soft and Furious', 'Loyalty Freak Music',
                     'Frederic Lardon', 'Frederic Lardon feat Laura Palmée', 'Cuicuitte',
                     'cuicuitte (feat renoncule) / comme ça', 'Comme Jospin', 'Alpha Hydrae'}
    def artist_group(item):
        return 'related_projects_conservative_group' if item['contributors'] in related_names else normalize(item['contributors'])
    eligible=[x for x in items if x['uid'] in index
              and normalize(x.get('contributors','')) not in excluded_artists
              and 'dedicated to the public domain' in x.get('rights','')]
    groups={}
    for x in sorted(eligible,key=lambda x:sha(('citizen-20260923:'+x['uid']).encode())):
        groups.setdefault(artist_group(x), x)
    selected=list(groups.values())[:12]
    plan=dict(commit=commit, selected=selected, eligible_tracks=len(eligible), artist_groups=len(groups),
              selection='All music genres, public-domain item statement, exclude all normalized MUSAN artist strings, one hash-ordered item per conservative artist group, up to 12 groups; no model scores used',
              related_project_names=sorted(related_names), related_group_identity_verified=False,
              revision_reason='Metadata Instrumental tag omitted many music genres; related project names grouped before any inference or listening labels',
              artist_aliases_verified=False, independent_holdout=False)
    (OUT/'selection.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SELECTED',[(x['uid'],x['contributors']) for x in selected],flush=True)
    if not collect:
        return
    known_hashes=set()
    for path in (ROOT/'data/raw').rglob('*manifest.json'):
        if path.parent==OUT:
            continue
        obj=json.loads(path.read_text(encoding='utf-8'))
        for row in obj.get('rows',[])+obj.get('samples',[]):
            for key in ('source_sha256','clip_sha256','pcm_sha256','sha256'):
                if row.get(key):known_hashes.add(row[key])
    rows=[]
    rejected=[]
    for item in selected:
        # Use the first published excerpt; never choose based on model scores.
        clip=index[item['uid']]['clips'][0]
        name=clip['filename']+'.wav'
        assert Path(name).name==name
        url=ASSET+urllib.parse.quote(name)
        original=OUT/'originals'/name
        blob=fetch(url,original,32*1024**2)
        assert sha(blob) not in known_hashes, 'Previously used exact bytes'
        sr,audio=wavfile.read(io.BytesIO(blob))
        assert audio.ndim in (1,2) and len(audio)>0
        original_dtype=str(audio.dtype)
        if np.issubdtype(audio.dtype,np.integer):
            if audio.dtype==np.uint8:audio=(audio.astype(np.float64)-128)/128
            else:audio=audio.astype(np.float64)/float(2**(np.iinfo(audio.dtype).bits-1))
        else:audio=audio.astype(np.float64)
        if audio.ndim==2:audio=audio.mean(axis=1)
        assert np.isfinite(audio).all()
        if len(audio)<8*sr:
            rejected.append(dict(ID=item['uid'],reason='published_excerpt_shorter_than_8s',seconds=len(audio)/sr))
            continue
        start=(len(audio)-8*sr)//2
        signal=audio[start:start+8*sr]
        factor=math.gcd(sr,16000)
        signal=resample_poly(signal,16000//factor,sr//factor)
        assert len(signal)==128000 and np.max(np.abs(signal))>1e-5
        gain=min(1.0,0.999/max(np.max(np.abs(signal)),1e-12))
        pcm=np.rint(signal*gain*32767).astype('<i2')
        pcm_hash=sha(pcm.tobytes())
        assert pcm_hash not in known_hashes and pcm_hash not in {r['pcm_sha256'] for r in rows}
        identity='N'+str(len(rows)+1).zfill(2)
        path=OUT/'clips'/(identity+'.wav');path.parent.mkdir(exist_ok=True)
        wavfile.write(path,16000,pcm)
        row=dict(ID=identity,source_id=item['uid'],title=item['title'],artist=item['contributors'],artist_group=artist_group(item),
                 source_page='https://citizen-dj.labs.loc.gov'+item['permalink'],download_url=url,
                 source_sha256=sha(blob),clip_sha256=sha(path.read_bytes()),pcm_sha256=pcm_hash,
                 original_path=original.relative_to(ROOT).as_posix(),clip_path=path.relative_to(ROOT).as_posix(),
                 source_sample_rate=sr,source_dtype=original_dtype,source_excerpt_seconds=len(audio)/sr,
                 published_excerpt_start_ms=clip['start'],crop_start_in_excerpt_seconds=start/sr,gain=gain,
                 clip_seconds=8,license='Public domain dedication per Library of Congress item statement',
                 metadata_sha256=sha((OUT/'metadata'/(item['uid']+'.md')).read_bytes()),
                 file_fake=0,authenticity_basis='Historical official music archive provenance; not a forensic guarantee of production method',
                 voice_presence=None,music_presence=None,voice_fake=None,music_fake=None,component_labels_reviewed=False,
                 source_representation='Citizen DJ published WAV excerpt, not full original recording',independent_holdout=False)
        rows.append(row)
        print('SAVED',identity,item['contributors'],flush=True)
    assert rows, 'No usable excerpts'
    manifest=dict(rows=rows,rejected=rejected,selection=plan,excluded_known_hashes_count=len(known_hashes),
                  limitations=['Exact file hashes and stored PCM hashes checked; full-recording/near-duplicate/alias/pretraining overlap unverified.',
                               'FMA shares source ecosystem with MUSAN; this is not established cross-dataset independence.',
                               'Music genre metadata is not crop-level component verification; listening required.',
                               'Single-class real-music collection cannot produce Music EER or competition Score.'],
                  ai_usage=dict(date='2026-09-23',tool='Codex/Python/web',work='Source collection and review preparation',limitations='No training or inference'))
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    instructions='새 실제 음악 성분 확인\n\n각 clips/Nxx.wav를 듣고 음악만 / 목소리만 / 둘 다 / 잘 모르겠음 중 하나로 답해주세요.\n말·노래·허밍은 목소리입니다. REAL/FAKE를 귀로 판단하는 작업이 아닙니다.\n음원은 의회도서관 Citizen DJ의 공개 구간이며 중앙 8초, 16kHz 모노 PCM16으로 변환했습니다.\n원본과 변환·출처 기록은 manifest.json, 곡별 권리 안내는 provenance에 있습니다.\n'
    bundle=Path.home()/'Downloads/fresh_real_music_review.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('README.txt',instructions)
        z.write(OUT/'manifest.json','manifest.json')
        z.write(OUT/'rights.md','provenance/rights.md')
        for row in rows:
            z.write(ROOT/row['clip_path'],'clips/'+row['ID']+'.wav')
            z.write(OUT/'metadata'/(row['source_id']+'.md'),'provenance/'+row['source_id']+'.md')
    with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
    print('COMPLETE',len(rows),bundle,sha(bundle.read_bytes()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--collect',action='store_true')
    main(parser.parse_args().collect)
