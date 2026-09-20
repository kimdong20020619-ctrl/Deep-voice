"""Collect frozen MUSAN candidates with explicit mirror and representation provenance."""
import hashlib
import io
import json
import shutil
import tarfile
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from prepare_music_pilot import LimitedReader, crop

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/raw/component-music-20260920'
DOWNLOAD_URL = ('https://huggingface.co/datasets/huseinzol05/musan-mirror/resolve/'
                'f52353cc256ad55582fd64674c2bb568e65d5f7e/musan.tar.gz')


class ResumableReader:
    """Retry a failed bounded range without losing the tar decompressor state."""
    def __init__(self):
        self.position = 0
        self.buffer = b''
        self.total = None

    def read(self, size):
        if size < 0:
            raise ValueError('Unbounded read')
        if not self.buffer:
            end = self.position + 4*1024**2 - 1
            if end >= 5*1024**3:
                raise ValueError('Transfer limit reached')
            for attempt in range(3):
                try:
                    request = urllib.request.Request(DOWNLOAD_URL, headers={
                        'Range': f'bytes={self.position}-{end}', 'Accept-Encoding': 'identity'})
                    with urllib.request.urlopen(request, timeout=30) as response:
                        assert response.status == 206, 'Server must support byte ranges'
                        interval, total = response.headers['Content-Range'].split('/')
                        assert interval == f'bytes {self.position}-{end}'
                        if self.total is not None:
                            assert self.total == total
                        self.total = total
                        blob = response.read(4*1024**2)
                        assert len(blob) == 4*1024**2
                    self.buffer = blob
                    break
                except (OSError, AssertionError):
                    if attempt == 2:
                        raise
            self.position += len(self.buffer)
        result, self.buffer = self.buffer[:size], self.buffer[size:]
        return result


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def collect_individual(missing, save):
    """Read viewer-exported WAVs, explicitly preserving the mirror provenance."""
    def page(offset):
        query = urllib.parse.urlencode(dict(dataset='shadwl/musan', config='default',
            split='train', length=100, offset=offset))
        for attempt in range(3):
            try:
                with urllib.request.urlopen('https://datasets-server.huggingface.co/rows?'+query, timeout=30) as response:
                    return json.load(response)
            except OSError:
                if attempt == 2:
                    raise
    total = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, 2400, 400):
            for result in pool.map(page, range(start, start+400, 100)):
                for item in result['rows']:
                    source = item['row']['path']
                    if source not in missing:
                        continue
                    url = item['row']['audio'][0]['src']
                    revision = url.split('/--/')[1]
                    assert revision == '4659edba2da70ebb8b9eccf6c997e7199a685119'
                    row = dict(missing[source], download_representation='dataset_viewer_exported_wav',
                        download_url=url.split('?')[0], mirror_revision=revision,
                        mirror_repository='shadwl/musan')
                    with urllib.request.urlopen(url, timeout=30) as response:
                        blob = response.read(64*1024**2 + 1)
                    assert 0 < len(blob) <= 64*1024**2
                    total += len(blob)
                    save(row, blob)
                    missing.pop(source)
            print('inspected rows', start+400, 'remaining', len(missing), flush=True)
            if not missing:
                return total
            if start+400 >= result['num_rows_total']:
                break
    raise ValueError(f'Missing individual sources: {list(missing)}')


def main():
    audit_path = ROOT / 'data/research/component-labels-20260920/audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    # The source text contradicts itself on 3.0 versus 4.0; do not resolve by regex.
    rejected = {'music-jamendo-0070': 'License prose says 4.0 but parenthesis says 3.0.',
                'music-jamendo-0108': 'JPMOUNIER may alias selected mounier; pending identity review.',
                'music-hd-0048': 'Performer MIT but source Bernd Krueger conflicts with artist independence.'}
    selected = [r for r in audit['fresh_instrumental_candidates'] if r['track_id'] not in rejected]
    for row in selected:
        if row['track_id'] == 'music-hd-0002':
            row['license'] = 'CC BY-SA 3.0 DE'
    for folder in ('originals', 'clips', 'receipts', 'licenses'):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    metadata = ROOT / 'data/raw/musan-metadata-20260912'
    meta = json.loads((metadata / 'manifest.json').read_text(encoding='utf-8'))
    for item in meta['metadata']:
        if any(item['path'].rsplit('/', 1)[0] == r['source'].removeprefix('musan/').rsplit('/', 1)[0] for r in selected):
            blob = (metadata / item['local_file']).read_bytes()
            assert sha(blob) == item['sha256']
            shutil.copy2(metadata / item['local_file'], OUT / 'licenses' / item['local_file'])
    excluded = set(audit['exclusions']['exact_hashes'])
    rows = []

    def save(row, blob):
        identity = row['track_id']
        digest = sha(blob)
        assert digest not in excluded
        assert digest not in {x['source_sha256'] for x in rows}
        sr, audio = wavfile.read(io.BytesIO(blob))
        assert sr == 16000 and audio.ndim == 1 and audio.dtype == np.int16
        pcm, start = crop(blob)
        original = OUT / 'originals' / (identity + '.wav')
        original.write_bytes(blob)
        target = OUT / 'clips' / (identity + '.wav')
        wavfile.write(target, sr, pcm)
        record = dict(row, audio_downloaded=True, source_sha256=digest,
            clip_sha256=sha(target.read_bytes()), crop_start_seconds=start,
            original_seconds=len(audio)/sr, clip_seconds=len(pcm)/sr,
            clip_rms=float(np.sqrt(np.mean((pcm.astype(np.float64)/32768)**2))),
            original_distribution_url='https://www.openslr.org/17/',
            mirror_bytes_match_official_archive_verified=False, full_archive_checksum_verified=False,
            file_fake=0, voice_presence=None, music_presence=None,
            voice_fake=None, music_fake=None, component_labels_reviewed=False)
        (OUT / 'receipts' / (identity + '.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
        rows.append(record)
        print('saved', len(rows), '/', len(selected), identity, flush=True)

    missing = {}
    for row in selected:
        original = OUT / 'originals' / (row['track_id'] + '.wav')
        receipt = OUT / 'receipts' / (row['track_id'] + '.json')
        if original.exists() and receipt.exists():
            blob = original.read_bytes()
            previous = json.loads(receipt.read_text())
            assert sha(blob) == previous['source_sha256']
            save(dict(row, **{k: previous[k] for k in ('download_representation', 'download_url',
                       'mirror_revision', 'mirror_repository') if k in previous}), blob)
        else:
            missing[row['source']] = row
    transferred = 0
    if missing and '--archive' not in __import__('sys').argv:
        transferred = collect_individual(missing, save)
    elif missing:
        budget = 5 * 1024**3
        reader = ResumableReader()
        bounded = LimitedReader(reader, budget)
        last = time.monotonic()
        with tarfile.open(fileobj=bounded, mode='r|gz') as archive:
            for entry in archive:
                if time.monotonic() - last > 25:
                    print('scanning', entry.name, 'MiB', round((budget-bounded.remaining)/1024**2), flush=True)
                    last = time.monotonic()
                if entry.name not in missing:
                    continue
                assert entry.isfile() and 0 < entry.size < 64*1024**2
                blob = archive.extractfile(entry).read()
                assert len(blob) == entry.size
                save(dict(missing.pop(entry.name), archive_url=DOWNLOAD_URL,
                          download_representation='archive_member'), blob)
                if not missing:
                    break
        transferred = budget - bounded.remaining
    assert not missing, list(missing)
    result = dict(rows=rows, rejected_candidates=rejected, selection_audit_sha256=sha(audit_path.read_bytes()),
                  transferred_bytes_this_run=transferred, full_archive_checksum_verified=False,
                  purpose='real_music_component_review_not_full_score_validation',
                  full_competition_score_available=False)
    (OUT / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('COMPLETE', len(rows), 'originals; component labels remain unreviewed', flush=True)


if __name__ == '__main__':
    import sys
    if '--inspect-viewer' in sys.argv:
        query = urllib.parse.urlencode(dict(dataset='shadwl/musan', config='default', split='train', length=100, offset=0))
        with urllib.request.urlopen('https://datasets-server.huggingface.co/rows?'+query, timeout=30) as response:
            info = json.load(response)
        (OUT / 'viewer_probe.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        print('viewer rows', len(info.get('rows', [])))
    elif '--inspect-mirror' in sys.argv:
        api = 'https://huggingface.co/api/datasets/burak-ozenc/dsp-game/tree/ac2ed607c95e7489a2cd087870b71e81e746ca54/data/musan/musan/music?recursive=true&limit=1000'
        with urllib.request.urlopen(api, timeout=30) as response:
            info = json.load(response)
        paths = [x for x in info if x['path'].endswith('.wav')]
        (OUT / 'individual_mirror_index.json').write_text(json.dumps(paths, indent=2), encoding='utf-8')
        print('individual files', len(paths), flush=True)
    else:
        main()
