"""Audit label evidence and freeze source exclusions without using predictions."""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/research/component-labels-20260920'


def read_manifest(name):
    return json.loads((ROOT / 'data/raw' / name / 'manifest.json').read_text(encoding='utf-8'))['rows']


def main():
    metadata = ROOT / 'data/raw/musan-metadata-20260912'
    with (metadata / 'music_inventory.csv').open(encoding='utf-8', newline='') as stream:
        inventory = list(csv.DictReader(stream))
    by_id = {r['track_id']: r for r in inventory}
    assert len(by_id) == len(inventory)
    licenses = {}
    for license_path in sorted(metadata.glob('*_LICENSE.txt')):
        for block in re.split(r'={5,}', license_path.read_text(encoding='utf-8')):
            ids = re.findall(r'(?m)^music-[a-z-]+\d+\s*$', block)
            terms = set(re.findall(r'CC BY(?:-[A-Z]+)* \d+\.\d+', block))
            if len(terms) == 1:
                for identity in ids:
                    licenses[identity.strip()] = next(iter(terms))
    audited = []
    used_artists, used_ids, used_hashes = set(), set(), set()
    counts = {}
    for dataset in ('music-pilot-20260912', 'file-probe-20260915'):
        rows = read_manifest(dataset)
        local_counts = Counter()
        for row in rows:
            used_hashes.update(row[k] for k in ('source_sha256', 'clip_sha256') if k in row)
            identity = Path(row['source']).stem
            annotation = by_id.get(identity)
            if row['file_fake'] == 0 and row['source'].startswith('musan/'):
                assert annotation is not None, identity
                used_ids.add(identity)
                used_artists.add(annotation['artist_raw'])
                local_counts[annotation['annotated_vocals']] += 1
            # A track-level Y does not establish presence in an arbitrary crop.
            audited.append(dict(dataset=dataset, ID=row['ID'], source=row['source'],
                file_fake=row['file_fake'],
                track_vocals_annotation=annotation['annotated_vocals'] if annotation else None,
                voice_presence=None, music_presence=None, voice_fake=None, music_fake=None,
                component_label_status='unreviewed_crop_not_eligible_for_full_competition_score'))
        counts[dataset] = dict(local_counts)
    candidates = []
    for row in inventory:
        identity = row['track_id']
        if (row['annotated_vocals'] != 'N'
                or row['artist_raw'] in used_artists or identity in used_ids):
            continue
        terms = licenses.get(identity)
        if terms not in ('CC BY 3.0', 'CC BY-SA 3.0', 'CC BY 4.0', 'CC BY-SA 4.0'):
            continue
        candidates.append(dict(track_id=identity, artist=row['artist_raw'], license=terms,
            source=f"musan/{row['source']}/{identity}.wav",
            annotation_url=row['annotation_url'], license_url=row['license_url'],
            audio_downloaded=False, crop_labels_verified=False))
    candidates.sort(key=lambda r: hashlib.sha256(('component-20260920:'+r['track_id']).encode()).hexdigest())
    selected, artists = [], set()
    for row in candidates:
        if row['artist'] not in artists:
            selected.append(row)
            artists.add(row['artist'])
        if len(selected) == 12:
            break
    external = read_manifest('ai-openbmat-20260917')
    speech = read_manifest('paired-speech-20260916')
    exclusion = dict(
        musan_track_ids=sorted(used_ids), musan_artist_strings=sorted(used_artists),
        exact_hashes=sorted(used_hashes | {r['source_sha256'] for r in external + speech}),
        ai_openbmat_pair_ids=sorted({r['pair_id'] for r in external}),
        ai_openbmat_music_source_tokens=sorted({x for r in external for x in r['music_source_tokens']}),
        ai_openbmat_nonmusic_source_ids=sorted({x for r in external for x in r['speech_source_ids']}),
        ai_openbmat_reference_ids=sorted({r['reference_id'] for r in external}),
        paired_speech_utterances=sorted({r['utterance'] for r in speech}),
        paired_speech_speakers=sorted({r['speaker'] for r in speech}),
        scope='Listed manifests only; not a complete inventory of all past or pretrained sources')
    assert not ({r['track_id'] for r in selected} & used_ids)
    assert not ({r['artist'] for r in selected} & used_artists)
    assert len({r['artist'] for r in selected}) == len(selected)
    result = dict(track_annotation_counts=counts, audited_rows=audited,
        fresh_instrumental_candidates=selected, eligible_candidate_tracks=len(candidates),
        exclusions=exclusion, full_competition_score_available=False,
        caveats=[
            'AI-OpenBMAT speech source tags also cover clapping/environmental sounds (paper section 2.2).',
            'AI-OpenBMAT music authenticity does not by itself prove DACON instrumental-component labels.',
            'MUSAN vocal annotations describe full tracks, not the extracted crops.',
            'Fresh candidate means track/artist-string disjoint from the two listed music manifests only.',
            'Artist aliases, recording duplicates, other datasets and pretraining overlap are unverified.',
            'No candidate audio has been collected or heard by this audit; no new GPU bundle is ready.'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(counts=counts, selected_candidates=len(selected),
                         eligible_tracks=len(candidates), full_score_available=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
