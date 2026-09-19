"""Inspect official AI-OpenBMAT archive with bounded range access."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

from prepare_paired_speech import ChunkedRemoteZip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/raw/ai-openbmat-20260917'
API = 'https://zenodo.org/api/records/20031232'
URL = 'https://zenodo.org/records/20031232/files/AI-OpenBMAT.zip?download=1'
SIZE = 7312205691


def inspect():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'record.json'
    if not path.exists():
        with urllib.request.urlopen(API,timeout=30) as response:
            blob=response.read(256*1024)
        record=json.loads(blob)
        assert record['metadata']['license']['id']=='cc-by-4.0'
        assert record['files'][0]['size']==SIZE
        path.write_bytes(blob)
    remote=ChunkedRemoteZip(URL,SIZE,budget=32*1024**2)
    with zipfile.ZipFile(remote) as z:
        files=[dict(path=e.filename,size=e.file_size,crc=e.CRC) for e in z.infolist() if not e.is_dir()]
        (OUT/'index.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
        print('FILES',len(files),flush=True)
        print('EXAMPLES',files[:12],flush=True)
        names=[e['path'] for e in files if e['path'].lower().endswith('.json') and '__MACOSX' not in e['path']]
        for i,name in enumerate(names[:4]):
            assert z.getinfo(name).file_size<1024**2
            blob=z.read(name)
            (OUT/f'example_{i}.json').write_bytes(blob)
            print(name,blob.decode()[:9000],flush=True)
        for name in z.namelist():
            if Path(name).name.lower() in ('license','license.txt','readme','readme.md','readme.txt'):
                assert z.getinfo(name).file_size<1024**2
                blob=z.read(name)
                (OUT/('archive_'+Path(name).name)).write_bytes(blob)
                print('NOTICE',name,blob.decode(errors='replace')[:4000],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pair')
    args=parser.parse_args()
    if args.pair:
        with zipfile.ZipFile(ChunkedRemoteZip(URL,SIZE,budget=32*1024**2)) as z:
            for kind in ('ai','human'):
                suffix='_ai' if kind=='ai' else ''
                name=f'AI-OpenBMAT/{kind}/{args.pair}{suffix}.json'
                cached=OUT/f'{args.pair}_{kind}.json'
                blob=cached.read_bytes() if cached.exists() else z.read(name)
                (OUT/f'{args.pair}_{kind}.json').write_bytes(blob)
                value=json.loads(blob)
                print(kind,'reference',value['reference_structure']['file'],
                      'sources',sorted({(s['type'],Path(s['file']).name) for seg in value['segments'] for s in seg['sources']}),flush=True)
    else:
        inspect()
