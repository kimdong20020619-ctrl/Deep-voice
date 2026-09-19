"""Archive official model metadata and pinned source without installing packages."""
import hashlib
import json
from pathlib import Path
import urllib.request
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/research/sonics-20260919'
REV = '9156ffad151f797c71556923c4a02fa01fa8fc91'


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = []
    def save(url, path):
        target = OUT/path
        target.parent.mkdir(parents=True, exist_ok=True)
        blob = target.read_bytes() if target.exists() else fetch(url)
        target.write_bytes(blob)
        receipts.append(dict(url=url, path=path, sha256=hashlib.sha256(blob).hexdigest()))
        return blob
    tree = json.loads(save(f'https://api.github.com/repos/awsaf49/sonics/git/trees/{REV}?recursive=1', 'tree.json'))
    for entry in tree['tree']:
        name = entry['path']
        if (name.startswith('sonics/') and name.endswith('.py')) or name in ('LICENSE', 'README.md', 'test.py', 'train.py', 'setup.py', 'requirements.txt'):
            save(f'https://raw.githubusercontent.com/awsaf49/sonics/{REV}/{name}', 'upstream/'+name)
    for variant in ('alpha-5s', 'alpha-120s'):
        repo = 'awsaf49/sonics-spectttra-'+variant
        meta = json.loads(save(f'https://huggingface.co/api/models/{repo}?blobs=true', variant+'/metadata.json'))
        revision = meta['sha']
        for name in ('config.json', 'README.md'):
            save(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', variant+'/'+name)
        print(variant, revision, meta.get('cardData',{}).get('license'), flush=True)
        print(json.loads((OUT/variant/'config.json').read_text()), flush=True)
    (OUT/'receipts.json').write_text(json.dumps(receipts, indent=2), encoding='utf-8')
    print('READY', len(receipts), flush=True)


if __name__ == '__main__':
    main()
