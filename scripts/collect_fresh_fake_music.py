"""Collect frozen fresh FakeMusicCaps candidates without model predictions."""
import hashlib
import io
import json
import re
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from prepare_paired_speech import ChunkedRemoteZip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/raw/fresh-fake-music-20260923'
PLAN = ROOT / 'data/research/fresh-music-20260922/selection.json'
SOURCE = 'https://zenodo.org/records/15063698'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    selected=plan['fake_candidates']
    assert len(selected)==20 and len({r['caption_group'] for r in selected})==20
    prior_hashes,prior_captions=set(),set()
    generators={r['generator'] for r in selected}
    for p in (ROOT/'data/raw').rglob('*manifest.json'):
        if p.parent==OUT:
            continue
        obj=json.loads(p.read_text(encoding='utf-8'))
        rows=obj.get('rows',[])+obj.get('samples',[])
        text=json.dumps(rows)
        for generator in generators:
            prior_captions.update(re.findall(re.escape(generator)+r'/([^/"\\]+)\.wav',text))
        prior_captions.update(obj.get('excluded_caption_ids',[]))
        for row in rows:
            prior_hashes.update(row[k] for k in ('source_sha256','clip_sha256','pcm_sha256','sha256') if row.get(k))
    assert not ({r['caption_group'] for r in selected}&prior_captions)
    for folder in ('originals','clips','receipts'):
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    record_path=OUT/'record.json'
    if not record_path.exists():
        with urllib.request.urlopen('https://zenodo.org/api/records/15063698',timeout=30) as f:
            blob=f.read(512*1024+1)
        assert len(blob)<=512*1024
        record_path.write_bytes(blob)
    record=json.loads(record_path.read_text(encoding='utf-8'))
    assert record['metadata']['license']['id'].lower()=='cc-by-nc-4.0'
    entry=next(x for x in record['files'] if x['key']=='FakeMusicCaps.zip')
    assert entry['size']==12889873014
    assert entry['checksum']=='md5:db418dc95ab7dc378a55f29d6021fd66'
    remote=ChunkedRemoteZip(SOURCE+'/files/FakeMusicCaps.zip?download=1',entry['size'],budget=64*1024**2)
    rows=[]
    with zipfile.ZipFile(remote) as archive:
        for i,selected_row in enumerate(selected,1):
            identity=f'F{i:02d}'
            name=selected_row['path']
            info=archive.getinfo(name)
            assert 0<info.file_size<2*1024**2
            assert info.file_size==selected_row['bytes']
            assert f'{info.CRC:08x}'==selected_row['crc32']
            original=OUT/'originals'/(identity+'.wav')
            if original.exists():blob=original.read_bytes()
            else:
                blob=archive.read(name)
                assert len(blob)==info.file_size and zlib.crc32(blob)==info.CRC
                original.write_bytes(blob)
            assert len(blob)==info.file_size and zlib.crc32(blob)==info.CRC
            assert sha(blob) not in prior_hashes
            sr,audio=wavfile.read(io.BytesIO(blob))
            assert sr==16000 and audio.ndim==1 and len(audio)>=128000
            if audio.dtype==np.int16:
                audio=audio.astype(np.float64)/32768
            else:
                assert audio.dtype in (np.dtype('float32'),np.dtype('float64'))
                audio=audio.astype(np.float64)
            assert np.isfinite(audio).all()
            start_sample=(len(audio)-128000)//2
            signal=audio[start_sample:start_sample+128000]
            peak=float(np.max(np.abs(signal)))
            assert peak>1e-5
            gain=min(1.0,0.999/max(peak,1e-12))
            pcm=np.rint(signal*gain*32767).astype('<i2')
            start=start_sample/sr
            clip=OUT/'clips'/(identity+'.wav');wavfile.write(clip,16000,pcm)
            assert len(pcm)==128000 and np.any(pcm)
            pcm_hash=sha(pcm.astype('<i2',copy=False).tobytes())
            assert pcm_hash not in prior_hashes and pcm_hash not in {r['pcm_sha256'] for r in rows}
            assert sha(clip.read_bytes()) not in prior_hashes
            row=dict(ID=identity,source=name,generator=selected_row['generator'],caption_group=selected_row['caption_group'],
                     source_url=SOURCE,source_sha256=sha(blob),clip_sha256=sha(clip.read_bytes()),pcm_sha256=pcm_hash,
                     original_path=original.relative_to(ROOT).as_posix(),clip_path=clip.relative_to(ROOT).as_posix(),
                     archive_member_crc32=f'{info.CRC:08x}',crop_start_seconds=start,clip_seconds=8,
                     transform='Central 8 seconds; source mono 16kHz; same peak-reduction and PCM16 rule as collected real music; no repetition',
                     source_crop_peak=peak,gain=gain,
                     license='CC BY-NC 4.0',file_fake=1,voice_presence=None,music_presence=None,
                     voice_fake=None,music_fake=None,component_labels_reviewed=False,independent_holdout=False)
            receipt=OUT/'receipts'/(identity+'.json')
            if receipt.exists():assert json.loads(receipt.read_text(encoding='utf-8'))==row
            receipt.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
            rows.append(row)
            print('SAVED',identity,selected_row['generator'],flush=True)
    result=dict(rows=rows,selection_sha256=sha(PLAN.read_bytes()),record_sha256=sha(record_path.read_bytes()),
                transferred_bytes_this_run=remote.transferred,full_archive_checksum_verified=False,
                exclusions=dict(caption_groups=sorted(prior_captions),stored_hash_count=len(prior_hashes)),
                limitations=['New caption IDs within the same dataset and generators; not unseen-generator validation.',
                             'Exact/stored PCM hash comparison does not establish near-duplicate or pretraining independence.',
                             'Component authenticity remains null until actual crop content is reviewed.'],
                ai_usage=dict(date='2026-09-23',tool='Codex/Python/web',action='Collect frozen generated-music candidates and prepare listening ZIP',limit='No inference or training'))
    (OUT/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    credit=('FakeMusicCaps, Luca Comanducci, Paolo Bestagini, Stefano Tubaro. Dataset v2 (2025).\n'
            'Source: https://doi.org/10.5281/zenodo.15063698\n'
            'License: Creative Commons Attribution-NonCommercial 4.0 International\n'
            'https://creativecommons.org/licenses/by-nc/4.0/\n'
            'Changes: central 8-second crop and PCM16 conversion. Original paths and hashes in manifest.json.\n'
            'Citation: FakeMusicCaps: a Dataset for Detection and Attribution of Synthetic Music Generated via Text-to-Music Models (2024). https://doi.org/10.48550/arXiv.2409.10684\n')
    (OUT/'ATTRIBUTION.txt').write_text(credit,encoding='utf-8')
    instructions=('새 생성 음악 성분 확인\nF01~F20을 듣고 음악만 / 목소리만 / 둘 다 / 잘 모르겠음으로 답해주세요.\n'
                  '말·노래·허밍은 목소리입니다. 진짜/가짜를 귀로 판단하는 작업이 아닙니다.\n'
                  '각 8초이며 과거 C01~C10과 다른 파일입니다. 앞서 N01~N09를 들었다면 다시 들을 필요 없습니다.\n')
    bundle=Path.home()/'Downloads/fresh_fake_music_review.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('README.txt',instructions)
        for name in ('manifest.json','ATTRIBUTION.txt','record.json'):
            z.write(OUT/name,name)
        for row in rows:z.write(ROOT/row['clip_path'],'clips/'+row['ID']+'.wav')
    with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
    print('COMPLETE',len(rows),bundle,sha(bundle.read_bytes()),flush=True)


if __name__=='__main__':
    main()
