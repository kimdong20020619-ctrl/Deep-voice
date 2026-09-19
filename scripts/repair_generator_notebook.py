"""Update only the notebook download cell; preserve the existing bundle hash."""
import ast
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
tree=ast.parse((ROOT/'scripts/make_generator_check.py').read_text(encoding='utf-8'))
addition=next(n.value.right.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='download' for t in n.targets))
path=ROOT/'notebooks/colab_generator_check.ipynb'
nb=json.loads(path.read_text(encoding='utf-8'))
original=''.join(nb['cells'][3]['source'])
prefix=original.split('entry=plan["df_model"]')[0]
nb['cells'][3]['source']=(prefix+addition.lstrip()).splitlines(True)
ast.parse(''.join(nb['cells'][3]['source']))
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
shutil.copy2(path,Path.home()/'Downloads'/path.name)
repair='from pathlib import Path\nimport hashlib, shutil\nfrom huggingface_hub import hf_hub_download\n'+addition.lstrip()
ast.parse(repair)
(Path.home()/'Downloads/generator_weight_repair.txt').write_text(repair,encoding='utf-8')
print('Updated notebook and repair cell; bundle unchanged')
