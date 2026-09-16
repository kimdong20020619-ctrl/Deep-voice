"""Bounded official LJSpeech/WaveFake collection with matched utterance groups."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

from inspect_fakemusiccaps import RemoteZip, wav_info
from prepare_music_pilot import LimitedReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/raw/paired-speech-20260916'
LJ_URL = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
WF_URL = 'https://zenodo.org/records/5642694/files/generated_audio.zip?download=1'


class ChunkedRemoteZip(RemoteZip):
    def read(self, count=-1):
        count = self.size-self.position if count < 0 else min(count, self.size-self.position)
        parts = []
        while count > 0:
            size = min(count, 4*1024**2)
            start = self.position
            if self.transferred + size > self.budget:
                raise ValueError('Transfer budget exceeded')
            request = urllib.request.Request(self.url, headers={
                'Range': f'bytes={start}-{start+size-1}', 'Accept-Encoding':'identity',
                'User-Agent':'DeepVoice-dataset-audit/1.0'})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=30) as response:
                        assert response.status == 206
                        assert response.headers.get('Content-Range') == f'bytes {start}-{start+size-1}/{self.size}'
                        chunk = response.read(size)
                    if not chunk:
                        raise OSError('Empty range response')
                    break
                except (urllib.error.URLError, TimeoutError, OSError):
                    if attempt == 2:
                        raise
                    time.sleep(1)
            # An interrupted short response resumes from the exact next byte.
            # zipfile independently verifies each fully assembled member's CRC.
            self.transferred += len(chunk)
            self.position += len(chunk)
            parts.append(chunk)
            count -= len(chunk)
        return b''.join(parts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'originals').mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():
        print('Already collected; use manifest.json')
        return
    receipt = OUT/'real_sources.json'
    if receipt.exists():
        real = json.loads(receipt.read_text())
    else:
        real = []
        with urllib.request.urlopen(LJ_URL, timeout=60) as response:
            with tarfile.open(fileobj=LimitedReader(response, 256*1024**2), mode='r|bz2') as archive:
                for entry in archive:
                    if not entry.isfile() or not entry.name.endswith('.wav') or entry.size > 2*1024**2:
                        continue
                    blob = archive.extractfile(entry).read()
                    info = wav_info(blob)
                    if info['seconds'] < 8.1 or info['channels'] != 1:
                        continue
                    identity = Path(entry.name).stem
                    path = 'originals/'+identity+'_real.wav'
                    (OUT/path).write_bytes(blob)
                    real.append(dict(utterance=identity, source=entry.name, local_path=path,
                                     source_sha256=hashlib.sha256(blob).hexdigest(), **info))
                    print('real', len(real), identity, flush=True)
                    if len(real) == 24:
                        break
        assert len(real) == 24
        receipt.write_text(json.dumps(real, indent=2), encoding='utf-8')
    # Fixed order based on identity, independent of model outputs.
    real.sort(key=lambda r: hashlib.sha256(('speech-v1:'+r['utterance']).encode()).hexdigest())
    license_path = OUT/'WaveFake_LICENSE.txt'
    if not license_path.exists():
        with urllib.request.urlopen('https://zenodo.org/records/5642694/files/LICENSE?download=1', timeout=60) as response:
            blob = response.read(65536)
        assert hashlib.md5(blob).hexdigest() == '9cc9e1ad97513505bfb75fc148a70005'
        license_path.write_bytes(blob)
    remote = ChunkedRemoteZip(WF_URL, 28918626084, budget=192*1024**2)
    rows = []
    with zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        (OUT/'archive_index.json').write_text(json.dumps(names), encoding='utf-8')
        folders = sorted({name.rpartition('/')[0] for name in names if name.endswith('.wav')})
        print('folders', folders, flush=True)
        for i, original in enumerate(real):
            split = 'train' if i < 16 else 'development'
            row = dict(original, ID='S_'+original['utterance']+'_real', split=split,
                       file_fake=0, generator='human', speaker='Linda_Johnson',
                       license='Public Domain', source_url='https://keithito.com/LJ-Speech-Dataset/')
            assert hashlib.sha256((OUT/row['local_path']).read_bytes()).hexdigest() == row['source_sha256']
            rows.append(row)
            generators = ('ljspeech_hifiGAN', 'ljspeech_melgan') if split == 'train' else ('ljspeech_waveglow',)
            for generator in generators:
                suffix = {'ljspeech_hifiGAN':'_generated.wav', 'ljspeech_melgan':'_gen.wav', 'ljspeech_waveglow':'.wav'}[generator]
                matches = [name for name in names if name.endswith('/'+original['utterance']+suffix) and generator in name.split('/')]
                assert len(matches) == 1, (generator, original['utterance'], matches)
                name = matches[0]
                path = 'originals/'+original['utterance']+'_'+generator+'.wav'
                target = OUT/path
                if target.exists():
                    blob = target.read_bytes()
                    assert len(blob) == archive.getinfo(name).file_size
                    import zlib
                    assert zlib.crc32(blob) == archive.getinfo(name).CRC
                else:
                    assert archive.getinfo(name).file_size < 2*1024**2
                    blob = archive.read(name)
                    target.write_bytes(blob)
                info = wav_info(blob)
                assert info['seconds'] >= 8 and info['channels'] == 1
                rows.append(dict(ID='S_'+original['utterance']+'_'+generator, utterance=original['utterance'],
                    split=split, file_fake=1, generator=generator, speaker='Linda_Johnson',
                    license='CC BY-SA 4.0', source_url='https://zenodo.org/records/5642694',
                    source=name, local_path=path, source_sha256=hashlib.sha256(blob).hexdigest(), **info))
                print('fake', len(rows), name, flush=True)
    manifest = dict(rows=rows, full_archive_checksums_verified=False,
                    limitation='Single reference speaker shared across splits; utterances and fake generators disjoint. Development only, not a speaker-disjoint holdout.',
                    transferred_wavefake_bytes=remote.transferred)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('READY', len(rows), 'speech sources', flush=True)


if __name__ == '__main__':
    main()
