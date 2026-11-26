import json, pytest
from benchmark.bench import compute_metrics, save_results

FAKE = [
    {'status': 200, 'latency': 1.0, 'output_tokens': 100, 'generate_elapsed': 0.8},
    {'status': 200, 'latency': 2.0, 'output_tokens': 100, 'generate_elapsed': 1.6},
    {'status': 200, 'latency': 3.0, 'output_tokens': 50,  'generate_elapsed': 2.5},
    {'status': 500, 'latency': 9.9, 'error': 'boom'},
]

def test_percentiles():
    m = compute_metrics(FAKE, wall=10.0)
    assert m['p99_latency'] == pytest.approx(2.98)
    assert m['p50_latency'] == pytest.approx(2.0)

def test_failed_requests_are_excluded():
    m = compute_metrics(FAKE, wall=10.0)
    assert m['max_latency'] == 3.0        
    assert m['request_per_second'] == 0.3

ALL_FAILED = [
    {'status': 500, 'latency': 9.9, 'error': 'boom'},
    {'status': 'HTTPError', 'latency': -1, 'error': 'timeout'},
]

def test_all_failed():
    m = compute_metrics(ALL_FAILED, wall=10.0)
    assert m['max_latency'] is None
    assert m['p50_latency'] is None
    assert m['p99_latency'] is None
    assert m['request_per_second'] == 0

ONE_SUCCESSFUL = [
    {'status': 200, 'latency': 1.5, 'output_tokens': 100, 'generate_elapsed': 0.8},
    {'status': 500, 'latency': 9.9, 'error': 'boom'},
]

def test_one_successful():
    m = compute_metrics(ONE_SUCCESSFUL, wall=10.0)
    assert m['max_latency'] == 1.5
    assert m['p50_latency'] == 1.5
    assert m['p99_latency'] == 1.5
    assert m['request_per_second'] == 0.1

def test_save_results(tmp_path):
    path = tmp_path / "results" / "x.json"
    save_results(str(path), {"tag": "t"}, {"p50_latency": 2.0}, FAKE)

    data = json.loads(path.read_text())
    assert path.parent.exists()
    assert path.exists()
    assert len(data["results"]) == 4