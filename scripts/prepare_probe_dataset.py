"""Build source-disjoint FILE-probe feasibility data, excluding the previous pilot."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
import urllib.request
import zipfile

from scipy.io import wavfile
import numpy as np
from prepare_music_pilot import crop, LimitedReader, MUSAN_URL, REPO
from inspect_fakemusiccaps import RemoteZip, URL, SIZE, is_audio
from plan_fakemusiccaps_split import caption_split, GENERATOR_SPLITS


def normalized_crop(blob):
    sr, audio = wavfile.read(io.BytesIO(blob))
    if sr != 16000 or audio.ndim != 1 or len(audio) < 128000:
        raise ValueError('Invalid source format or length')
    if audio.dtype == np.int16:
        audio = audio.astype(np.float64) / 32768
    elif audio.dtype.kind != 'f':
        raise ValueError('Unsupported source encoding')
    if not np.isfinite(audio).all():
        raise ValueError('Nonfinite source')
    start = (len(audio)-128000)//2
    clip = audio[start:start+128000].astype(np.float64)
    peak = float(np.max(np.abs(clip)))
    if float(np.sqrt(np.mean(clip**2))) < 1e-6:
        raise ValueError('Silent crop')
    # Apply exactly the same peak normalization to both real and generated audio.
    pcm = np.rint(clip * (0.95 / peak) * 32768).astype(np.int16)
    return pcm, start / sr


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    metadata = REPO / 'data/raw/musan-metadata-20260912'
    previous = json.loads((REPO / 'data/raw/music-pilot-20260912/manifest.json').read_text(encoding='utf-8'))
    excluded_artists = {r['source_group'] for r in previous['rows'] if r['file_fake'] == 0}
    excluded_captions = {Path(r['source']).stem for r in previous['rows'] if r['file_fake'] == 1}
    excluded_captions.add('-0Gj8-vB1q4')
    artists = defaultdict(list)
    license_bytes = (metadata / '04_LICENSE.txt').read_bytes()
    licensing = license_bytes.decode('utf-8')
    for line in (metadata / '03_ANNOTATIONS.txt').read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if fields and fields[3] not in excluded_artists and fields[0] in licensing:
            artists[fields[3]].append(fields[0])
    groups = sorted(artists)
    random.Random(20260915).shuffle(groups)
    assert len(groups) >= 16
    plan = []
    for index, artist in enumerate(groups):
        split = 'holdout' if index < 4 else 'development' if index < 8 else 'train'
        for track in sorted(artists[artist])[:2]:
            plan.append(dict(source=f'musan/music/fma/{track}.wav', file_fake=0,
                             group=artist, split=split, origin=MUSAN_URL))
    fake_manifest = json.loads((REPO / 'data/raw/fakemusiccaps-audit-20260911-v2/manifest.json').read_text(encoding='utf-8'))
    candidates = sorted(e['path'] for e in fake_manifest['files'] if e['path'].endswith('.wav') and not e['path'].startswith('__MACOSX/') and not Path(e['path']).name.startswith('._'))
    random.Random(20260915).shuffle(candidates)
    used = set(excluded_captions)
    for split in ('train', 'development', 'holdout'):
        needed = sum(r['split'] == split for r in plan if r['file_fake'] == 0)
        selected = 0
        for path in candidates:
            generator, filename = path.split('/')
            group = Path(filename).stem
            if group in used or caption_split(group) != split or GENERATOR_SPLITS[generator] != split:
                continue
            plan.append(dict(source=path, file_fake=1, group=group, generator=generator,
                             split=split, origin='https://zenodo.org/records/15063698'))
            used.add(group)
            selected += 1
            if selected == needed:
                break
        assert selected == needed
    for index, row in enumerate(plan):
        row['ID'] = f'R{index:04d}'
    plan_path = out / 'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text(encoding='utf-8')) == plan
    else:
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'test').mkdir(exist_ok=True)
    (out / 'originals').mkdir(exist_ok=True)
    (out / 'receipts').mkdir(exist_ok=True)
    (out / 'MUSAN_LICENSE.txt').write_bytes(license_bytes)
    (out / 'MUSAN_ANNOTATIONS.txt').write_bytes((metadata / '03_ANNOTATIONS.txt').read_bytes())

    def done(row):
        receipt = out / 'receipts' / (row['ID'] + '.json')
        if not receipt.exists():
            return False
        record = json.loads(receipt.read_text(encoding='utf-8'))
        assert all(record[k] == v for k, v in row.items())
        assert hashlib.sha256((out / 'test' / (row['ID'] + '.wav')).read_bytes()).hexdigest() == record['clip_sha256']
        return True

    def save(row, blob):
        pcm, start = normalized_crop(blob)
        target = out / 'test' / (row['ID'] + '.wav')
        (out / 'originals' / (row['ID'] + '.wav')).write_bytes(blob)
        wavfile.write(target, 16000, pcm)
        record = dict(row, source_sha256=hashlib.sha256(blob).hexdigest(),
                      clip_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), crop_start_seconds=start)
        (out / 'receipts' / (row['ID'] + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print('saved', row['ID'], row['split'], row['source'], flush=True)
    pending = {r['source']: r for r in plan if not r['file_fake'] and not done(r)}
    if pending:
        with urllib.request.urlopen(MUSAN_URL, timeout=60) as response:
            with tarfile.open(fileobj=LimitedReader(response, 1536 * 1024**2), mode='r|gz') as archive:
                for entry in archive:
                    if entry.name in pending:
                        if not 0 < entry.size < 32 * 1024**2:
                            raise ValueError('Unexpected source size')
                        save(pending.pop(entry.name), archive.extractfile(entry).read())
                    if not pending:
                        break
        assert not pending, list(pending)
    fake_pending = [r for r in plan if r['file_fake'] and not done(r)]
    if fake_pending:
        with zipfile.ZipFile(RemoteZip(URL, SIZE, budget=96 * 1024**2)) as archive:
            for row in fake_pending:
                entry = archive.getinfo(row['source'])
                assert is_audio(entry) and entry.file_size < 2 * 1024**2
                save(row, archive.read(entry))
    assert all(done(r) for r in plan)
    records = [json.loads((out / 'receipts' / (r['ID'] + '.json')).read_text(encoding='utf-8')) for r in plan]
    (out / 'manifest.json').write_text(json.dumps({'purpose': 'file_probe_feasibility_not_final_competition_validation',
        'excluded_pilot_artists': sorted(excluded_artists), 'excluded_caption_ids': sorted(excluded_captions),
        'rows': records}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    build(parser.parse_args().out_dir)
