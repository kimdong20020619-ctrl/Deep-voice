"""Collect 24 new metadata-matched pairs without evaluating or training any model."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import zlib

import numpy as np
from scipy.io import wavfile
from prepare_external_music import ROOT,OUT,URL,SIZE
from prepare_paired_speech import ChunkedRemoteZip


def describe(meta):
    assert meta['duration']==60 and meta['sample_rate']==22050
    music_seconds=sum(float(s['duration']) for s in meta['segments'] if any(x['type']=='music' for x in s['sources']))
    signature=[(s['event_type'],float(s['duration']),
        [(Path(x['file']).name,float(x['start']),float(x['duration'])) for x in s['sources'] if x['type']=='speech']) for s in meta['segments']]
    music=sorted({Path(s['file']).name for seg in meta['segments'] for s in seg['sources'] if s['type']=='music'})
    speech=sorted({Path(s['file']).name for seg in meta['segments'] for s in seg['sources'] if s['type']=='speech'})
    return music_seconds,signature,music,speech


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('metadata','audio','receipts'):
        (OUT/name).mkdir(exist_ok=True)
    assert json.loads((OUT/'record.json').read_text())['metadata']['license']['id']=='cc-by-4.0'
    remote=ChunkedRemoteZip(URL,SIZE,budget=256*1024**2)
    with zipfile.ZipFile(remote) as archive:
        def fetch(name,path):
            entry=archive.getinfo(name)
            assert 0<entry.file_size<8*1024**2
            if path.exists():
                blob=path.read_bytes()
                assert len(blob)==entry.file_size and zlib.crc32(blob)==entry.CRC
                return blob
            blob=archive.read(name)
            path.write_bytes(blob)
            return blob

        plan_path=OUT/'pair_plan.json'
        if plan_path.exists():
            pairs=json.loads(plan_path.read_text(encoding='utf-8'))
        else:
            candidates=sorted({Path(n).stem.removesuffix('_ai') for n in archive.namelist()
                if n.startswith('AI-OpenBMAT/ai/gen_') and n.endswith('_ai.json')},
                key=lambda s:hashlib.sha256(('external-20260917:'+s).encode()).hexdigest())
            pairs=[]
            used_music,used_speech=set(),set()
            rejected=[]
            for pair in candidates[:160]:
                ai_path=f'AI-OpenBMAT/ai/{pair}_ai.json'
                ai=json.loads(fetch(ai_path,OUT/'metadata'/(pair+'_ai.json')))
                a_seconds,a_signature,a_music,a_speech=describe(ai)
                if a_seconds<8:
                    rejected.append(dict(pair=pair,reason='less_than_8s_music'))
                    continue
                human_path=f'AI-OpenBMAT/human/{pair}.json'
                human=json.loads(fetch(human_path,OUT/'metadata'/(pair+'_human.json')))
                h_seconds,h_signature,h_music,h_speech=describe(human)
                if (a_signature!=h_signature or a_seconds!=h_seconds or
                    ai['reference_structure']['file']!=human['reference_structure']['file']):
                    rejected.append(dict(pair=pair,reason='unmatched_structure_or_speech'))
                    continue
                tokens={'ai:'+s for s in a_music}|{'human:'+s for s in h_music}
                if tokens & used_music or set(a_speech) & used_speech:
                    rejected.append(dict(pair=pair,reason='repeated_source'))
                    continue
                assert ai['artisan']=='ai' and human['artisan']=='human'
                record=dict(pair_id=pair,music_seconds=a_seconds,music_source_tokens=sorted(tokens),
                    music_source_group=hashlib.sha256('|'.join(sorted(tokens)).encode()).hexdigest(),
                    speech_source_ids=a_speech,reference_id=ai['reference_structure']['file'])
                pairs.append(record)
                used_music.update(tokens)
                used_speech.update(a_speech)
                print('selected',len(pairs),pair,'music_seconds',a_seconds,flush=True)
                if len(pairs)==24:
                    break
            (OUT/'selection_rejections.json').write_text(json.dumps(rejected,indent=2),encoding='utf-8')
            assert len(pairs)==24, 'Insufficient eligible pairs; no automatic weakening of filters'
            plan_path.write_text(json.dumps(pairs,indent=2),encoding='utf-8')
        rows=[]
        prior_hashes=set()
        for parent in ('file-probe-20260915','music-pilot-20260912'):
            data=json.loads((ROOT/'data/raw'/parent/'manifest.json').read_text(encoding='utf-8'))
            for row in data['rows']:
                prior_hashes.update(row[k] for k in ('source_sha256','clip_sha256') if k in row)
        for pair in pairs:
            for label,kind in ((0,'human'),(1,'ai')):
                suffix='_ai' if label else ''
                name=f'AI-OpenBMAT/{kind}/{pair["pair_id"]}{suffix}.wav'
                identity=pair['pair_id']+'_'+kind
                path=OUT/'audio'/(identity+'.wav')
                blob=fetch(name,path)
                sha=hashlib.sha256(blob).hexdigest()
                assert sha not in prior_hashes
                sr,audio=wavfile.read(io.BytesIO(blob))
                assert sr==22050 and audio.ndim==1 and audio.dtype==np.int16
                assert 59*sr<=len(audio)<=61*sr and np.isfinite(audio).all() and np.any(audio)
                meta_path=OUT/'metadata'/(pair['pair_id']+'_'+kind+'.json')
                row=dict(pair,ID=identity,file_fake=label,split='external_check',source=name,
                    source_url='https://zenodo.org/records/20031232',generator='Suno_v3.5' if label else 'human',
                    source_sha256=sha,clip_sha256=sha,metadata_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                    sample_rate=sr,samples=len(audio),license='CC BY 4.0')
                rows.append(row)
                (OUT/'receipts'/(identity+'.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
                print('audio',len(rows),'/',2*len(pairs),identity,flush=True)
        manifest=dict(rows=rows,purpose='frozen_external_source_check',
            selection='First 24 SHA256-ordered pairs with >=8s music, matched speech/timing and nonoverlapping source tokens; no predictions consulted',
            full_archive_checksum_verified=False,previous_source_exact_hash_overlap=0,
            near_duplicate_or_pretraining_overlap_verified=False,transferred_bytes_this_run=remote.transferred)
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print('READY',len(rows),'external files',flush=True)


if __name__=='__main__':
    main()
