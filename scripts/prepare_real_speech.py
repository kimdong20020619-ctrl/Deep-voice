"""Fetch eight distinct-speaker LibriSpeech sources, bounded and without full extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
from prepare_music_pilot import LimitedReader

URL = 'https://openslr.trmal.net/resources/12/dev-clean.tar.gz'


def flac_info(blob):
    if blob[:4] != b'fLaC' or blob[4] & 127 != 0 or int.from_bytes(blob[5:8], 'big') != 34:
        raise ValueError('Expected FLAC STREAMINFO')
    bits = int.from_bytes(blob[18:26], 'big')
    return {'sample_rate': bits >> 44, 'channels': ((bits >> 41) & 7)+1,
            'samples': bits & ((1 << 36)-1)}


def collect(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Collection already complete')
    rows, speakers = [], set()
    for receipt in sorted(out.glob('speech_*.json')):
        row = json.loads(receipt.read_text(encoding='utf-8'))
        assert hashlib.sha256((out / row['local_path']).read_bytes()).hexdigest() == row['sha256']
        rows.append(row)
        speakers.add(row['speaker'])
    with urllib.request.urlopen(URL, timeout=60) as response:
        with tarfile.open(fileobj=LimitedReader(response, 384*1024**2), mode='r|gz') as archive:
            for entry in archive:
                if entry.isfile() and Path(entry.name).name in ('LICENSE.TXT', 'README.TXT', 'SPEAKERS.TXT'):
                    if entry.size < 1024**2:
                        (out / Path(entry.name).name).write_bytes(archive.extractfile(entry).read())
                if not entry.isfile() or not entry.name.endswith('.flac'):
                    continue
                speaker = Path(entry.name).name.split('-')[0]
                if speaker in speakers:
                    continue
                if not 0 < entry.size < 4*1024**2:
                    continue
                blob = archive.extractfile(entry).read()
                info = flac_info(blob)
                if info['sample_rate'] != 16000 or info['channels'] != 1 or info['samples'] < 128000:
                    continue
                name = f'speech_{len(rows):02d}.flac'
                row = dict(source=entry.name, url=URL, speaker=speaker, local_path=name,
                           sha256=hashlib.sha256(blob).hexdigest(), **info)
                (out / name).write_bytes(blob)
                (out / (Path(name).stem+'.json')).write_text(json.dumps(row, indent=2), encoding='utf-8')
                rows.append(row)
                speakers.add(speaker)
                print('saved', name, speaker, info['samples']/16000, flush=True)
                if len(rows) == 8:
                    break
    assert len(rows) == len(speakers) == 8
    (out / 'manifest.json').write_text(json.dumps({'source': 'https://www.openslr.org/12/',
        'license': 'CC BY 4.0', 'full_archive_checksum_verified': False,
        'local_validation': 'FLAC STREAMINFO only; full decode checked in Colab', 'rows': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    collect(parser.parse_args().out_dir)
