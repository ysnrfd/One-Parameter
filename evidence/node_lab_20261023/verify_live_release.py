from pathlib import Path
import hashlib
import json
import urllib.request

root = Path(__file__).resolve().parents[2]
base = 'http://127.0.0.1:8765'
def read(path):
    with urllib.request.urlopen(base + path, timeout=10) as response:
        return response.read(), response.headers.get_content_type()
def get(path):
    return json.loads(read(path)[0])
health = get('/api/health')
assert health == {'ok': True, 'service': 'single-parameter-studio', 'version': '1.3.0'}, health
baseline = json.loads((root / 'evidence/node_lab_20261023/workspace_before.json').read_bytes())
project = get('/api/project')
assert project['runs'] == baseline['runs'], 'Original experiments, configuration, dataset or engine state changed'
workspace_keys = ('id', 'name', 'notes', 'active_run_id', 'view')
assert all(project['workspace'][key] == baseline['workspace'][key] for key in workspace_keys)
catalog = get('/api/graph/catalog')
graphs = get('/api/graph/state')
assert not graphs['jobs'] and not graphs['artifacts'] and not graphs['plugins']
assert graphs['queue_length'] == 0
assert len(catalog['types']) == 21
assets = {}
for name in ('index.html', 'studio.css', 'studio.js', 'graph.css', 'graph.js', 'graph_core.js'):
    content, mime = read('/' + name)
    source = root / 'single_parameter_observatory/studio_web' / name
    assert content == source.read_bytes(), name
    assets[name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(), 'mime': mime}
report = {'status': 'passed', 'url': base + '/#nodes', 'health': health,
          'original_runs_unchanged': True, 'original_workspace_fields_unchanged': list(workspace_keys),
          'original_run_ids': [run['id'] for run in project['runs']],
          'original_runs_sha256': hashlib.sha256(json.dumps(project['runs'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
          'idle_graph_count': len(graphs['graphs']), 'queued_jobs': graphs['queue_length'], 'jobs': len(graphs['jobs']), 'plugins': len(graphs['plugins']), 'core_node_types': len(catalog['types']),
          'graph_state_keys': sorted(graphs), 'graph_catalog_keys': sorted(catalog), 'assets': assets}
(root / 'evidence/node_lab_20261023/live_release.json').write_text(json.dumps(report, indent=2) + chr(10), encoding='utf-8')
print(json.dumps(report, indent=2))
