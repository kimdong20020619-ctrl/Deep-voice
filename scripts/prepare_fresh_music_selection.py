"""Freeze metadata-only music candidates, excluding recorded source groups."""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/research/fresh-music-20260922/selection.json'
GENERATORS = ('MusicGen_medium', 'audioldm2', 'musicldm', 'mustango', 'stable_audio_open')


def main():
    metadata = ROOT / 'data/raw/musan-metadata-20260912'
    inventory = list(csv.DictReader((metadata / 'music_inventory.csv').open(encoding='utf-8')))
    used_tracks, used_captions, inputs = set(), set(), {}
    catalog = None
    for path in sorted((ROOT / 'data/raw').rglob('*manifest.json')):
        blob = path.read_bytes()
        obj = json.loads(blob)
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(blob).hexdigest()
        # Archive inventories are candidates, not evidence of prior use.
        prior = obj.get('rows', []) + obj.get('samples', [])
        text = json.dumps(prior)
        used_tracks.update(re.findall(r'music-(?:hd|jamendo|fma)-\d+', text))
        for generator in GENERATORS:
            used_captions.update(re.findall(re.escape(generator) + r'/([^/"\\]+)\.wav', text))
        used_captions.update(obj.get('excluded_caption_ids', []))
        if path.parent.name == 'fakemusiccaps-audit-20260911-v2':
            catalog = obj
    assert catalog is not None
    used_artists = {r['artist_raw'] for r in inventory if r['track_id'] in used_tracks}
    terms_by_id = {}
    for path in sorted(metadata.glob('*_LICENSE.txt')):
        blob = path.read_bytes()
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(blob).hexdigest()
        for block in re.split(r'={5,}', blob.decode('utf-8')):
            ids = re.findall(r'(?m)^music-[a-z-]+\d+\s*$', block)
            terms = set(re.findall(r'CC BY(?:-[A-Z]+)* \d+\.\d+', block))
            versions = set(re.findall(r'\b[34]\.0\b', block))
            if len(versions) != 1:
                continue
            if len(terms) == 1 and next(iter(terms)) in ('CC BY 3.0', 'CC BY-SA 3.0', 'CC BY 4.0', 'CC BY-SA 4.0'):
                for identity in ids:
                    terms_by_id[identity.strip()] = dict(license_hint=next(iter(terms)), license_block=block.strip(), license_evidence=path.relative_to(ROOT).as_posix())
    # Previously unresolved authorship/license cases remain excluded.
    blocked_artists = {'JPMOUNIER', 'MIT', 'MOUNIER'}
    eligible = [r for r in inventory if r['annotated_vocals']=='N' and r['track_id'] not in used_tracks and r['artist_raw'] not in used_artists | blocked_artists and r['track_id'] in terms_by_id and r['track_id']!='music-jamendo-0070']
    order = lambda key: hashlib.sha256(('fresh-music-20260922:'+key).encode()).hexdigest()
    real, artists = [], set()
    for row in sorted(eligible, key=lambda r: order(r['track_id'])):
        if row['artist_raw'] in artists:
            continue
        artists.add(row['artist_raw'])
        real.append(dict(row, **terms_by_id[row['track_id']], file_fake=0, voice_presence=None, music_presence=None, music_fake=None, status='metadata_candidate_pending_license_and_audio_review'))
        if len(real)==10:
            break
    fake, chosen_captions = [], set()
    available = {}
    for generator in GENERATORS:
        candidates = [r for r in catalog['files'] if r['path'].startswith(generator+'/') and r['path'].endswith('.wav') and Path(r['path']).stem not in used_captions]
        available[generator]=len(candidates)
        selected=[]
        for row in sorted(candidates,key=lambda r:order(r['path'])):
            caption=Path(row['path']).stem
            if caption in chosen_captions:
                continue
            chosen_captions.add(caption)
            selected.append(dict(row, generator=generator, caption_group=caption, file_fake=1, voice_presence=None, music_presence=None, music_fake=None, status='metadata_candidate_pending_license_and_audio_review'))
            if len(selected)==4:
                break
        assert len(selected)==4, generator
        fake.extend(selected)
    assert real, 'No eligible real artist remains; do not weaken exclusions'
    assert not ({r['track_id'] for r in real} & used_tracks)
    assert not ({r['artist_raw'] for r in real} & used_artists)
    assert not (chosen_captions & used_captions)
    assert len(chosen_captions)==len(fake)
    inventory_path=metadata/'music_inventory.csv'
    inputs[inventory_path.relative_to(ROOT).as_posix()]=hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    result=dict(date='2026-09-22', status='metadata_selection_only_insufficient_real_sources', inputs_sha256=inputs,
        selection_rule='SHA256 order, up to 10 distinct unused MUSAN artist strings and 4 unused caption groups per generator; predictions not read',
        excluded_track_ids=sorted(used_tracks), excluded_artist_strings=sorted(used_artists), excluded_caption_groups=sorted(used_captions),
        blocked_artist_strings=sorted(blocked_artists), eligible_real_tracks=len(eligible), available_fake_by_generator=available,
        real_candidates=real, fake_candidates=fake,
        evaluation_plan=dict(models=['current DF-Arena baseline', 'SONICS alpha-5s fixed mean'],
            conditions=['central 8s original', 'same crop MP3'],
            selection_uses_predictions=False, tuning_allowed=False, full_competition_score=False,
            acceptance='Both EER and AUC must improve against baseline in each codec and generator comparison; failure prevents adoption. Small-sample improvement is only a screening result, requiring speech/mixed and server validation.',
            component_policy='Keep all selected file-authenticity candidates. Music metrics require reviewed music presence and supported component authenticity. Do not replace poor predictions or unhelpful generators.'),
        remaining=['Verify current source/license and exact per-track terms before download.', 'Download and hash originals; check exact and decoded-audio duplicates against existing material.', 'Review actual 8s components before model inference; uncertain components remain null.', 'Expand real-source artist coverage before GPU evaluation; current pool is insufficient.', 'Freeze final audio manifest and model/config hashes before GPU execution.'],
        limitations=['Same source datasets and generators as previous work; not unseen-generator or cross-dataset validation.', 'Artist-string and caption-ID exclusions do not establish recording, alias, or pretraining independence.', 'No audio downloaded, no labels inferred, no GPU execution, no model change.'],
        ai_usage=dict(date='2026-09-22',tool='Codex/Python',action='Freeze source-excluded metadata candidates',limitation='Internal research artifact, not competition submission prose'))
    serialized=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(serialized,encoding='utf-8')
    print(json.dumps(dict(real=len(real),fake=len(fake),eligible_real_tracks=len(eligible),excluded_tracks=len(used_tracks),excluded_captions=len(used_captions),output=str(OUT)),ensure_ascii=False))


if __name__ == '__main__':
    main()
