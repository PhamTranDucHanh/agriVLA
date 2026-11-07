from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

from PIL import Image
import pytest
import yaml

from agri_vla.browser import BrowserDataset, handler_for, page_of, search_text, serve_dataset
from agri_vla.dataset import DatasetError, load_compose_config
from agri_vla.scenes import compose
from http.server import ThreadingHTTPServer
from test_generation import fixture_project


def test_replace_refuses_a_non_browser_process(monkeypatch):
    import os
    from types import SimpleNamespace
    from agri_vla.browser import replace_browser
    from agri_vla.preflight import project_root
    monkeypatch.setattr("agri_vla.browser.subprocess.run",lambda *a,**kw:SimpleNamespace(stdout=f"State Listener\nLISTEN users:((python,pid={os.getpid()},fd=3))\n"))
    stopped=[]
    monkeypatch.setattr("agri_vla.browser.os.kill",lambda *a:stopped.append(a))
    with pytest.raises(DatasetError,match="Từ chối"):
        replace_browser(8765,project_root())
    assert not stopped


@pytest.fixture(scope="module")
def web_project(fixture_project):
    path = fixture_project / "dataset.yaml"
    config = yaml.safe_load(path.read_text())
    config["paths"]["images_dir"] = "images"
    path.write_text(yaml.safe_dump(config, allow_unicode=True))
    loaded = load_compose_config(path)
    compose(loaded)
    return loaded, BrowserDataset(loaded, fixture_project / "generated")


@pytest.fixture(scope="module")
def web_server(web_project):
    _, app = web_project
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", app
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def get(base, path, *, headers=None):
    with urlopen(Request(base + path, headers=headers or {}), timeout=10) as response:
        return response.read(), response.headers


def test_http_can_visit_every_record_and_source_without_gaps(web_server):
    base, app = web_server
    for endpoint, identifiers, key in (("samples", set(app.by_id), "sample_id"), ("sources", set(app.sources), "source_image_id")):
        seen = []
        first = json.loads(get(base, f"/api/{endpoint}?page_size=48")[0])
        for page in range(1, first["pages"] + 1):
            payload = json.loads(get(base, f"/api/{endpoint}?page_size=48&page={page}")[0])
            seen.extend(item[key] for item in payload["items"])
        assert len(seen) == len(identifiers) == len(set(seen))
        assert set(seen) == identifiers


def test_filters_search_and_quarantine_are_truthful(web_server):
    base, app = web_server
    query = urlencode({"split": "val", "language": "en", "wording": "unseen", "intent": "STOP_ALL", "action": "zero"})
    data = json.loads(get(base, "/api/samples?" + query)[0])
    assert data["total"] == 20
    assert all(r["target_action"] == [0, 0, 0] and r["language"] == "en" for r in data["items"])
    text = json.loads(get(base, "/api/samples?" + urlencode({"q": "dung tuoi toan bo"}))[0])
    assert text["total"] > 0 and all(r["intent"] == "STOP_ALL" for r in text["items"])
    assert search_text("Đất / DỪNG") == "dat / dung"
    quarantined = json.loads(get(base, "/api/sources?eligibility=quarantine&wilt=5")[0])
    assert quarantined["total"] == 1
    assert all(not r["eligible"] and r["record_uses"] == 0 and r["split"] is None for r in quarantined["items"])
    empty = json.loads(get(base, "/api/samples?q=does-not-exist-agrivla")[0])
    assert empty["total"] == 0 and empty["page"] == 1 and empty["items"] == []


def test_detail_links_translation_quartet_and_exact_original_image(web_server):
    base, app = web_server
    row = next(r for r in app.records if r["split"] == "test")
    detail = json.loads(get(base, "/api/sample?" + urlencode({"id": row["sample_id"]}))[0])
    assert len(detail["related"]) == 4
    assert {(r["language"], r["wording_partition"]) for r in detail["related"]} == {("vi", "seen"), ("en", "seen"), ("vi", "unseen"), ("en", "unseen")}
    for source in detail["sources"]:
        original, headers = get(base, source["original_url"])
        assert original == Path(source["path"]).read_bytes()
        assert headers["Content-Type"].startswith("image/")
        thumb = get(base, source["thumbnail_url"])[0]
        with Image.open(BytesIO(thumb)) as image:
            assert image.mode == "RGB" and image.width <= 320 and image.height <= 240
        listing = get(base, source["folder_url"])[0].decode()
        assert source["source_image_id"] in listing
        assert str(app.folders["images"]) in listing
    downloaded, headers = get(base, "/api/sample?" + urlencode({"id": row["sample_id"], "download": 1}))
    assert json.loads(downloaded) == detail["record"]
    assert "attachment" in headers["Content-Disposition"]
    last = json.loads(get(base, f"/api/sample?index={len(app.records)}")[0])
    assert last["ordinal"] == len(app.records) and last["record"]["sample_id"] == app.records[-1]["sample_id"]
    related = json.loads(get(base, "/api/samples?" + urlencode({"scene_id": row["scene_id"]}))[0])
    assert related["total"] > 4


@pytest.mark.parametrize("path,status", [
    ("/api/samples?split=other", 400), ("/api/samples?page_size=999", 400),
    ("/api/samples?page=-1", 400), ("/api/sample?index=0", 400),
    ("/api/sample?id=missing", 404), ("/original/missing.jpg", 404),
    ("/files/%2e%2e/.git/config", 400), ("/files/raw/.venv/secrets", 404),
    ("/.git/config", 404), ("/assets/../browser.py", 400),
])
def test_bad_queries_and_unlisted_files_are_rejected(web_server, path, status):
    base, _ = web_server
    with pytest.raises(HTTPError) as info:
        get(base, path)
    assert info.value.code == status
    assert "error" in json.loads(info.value.read())


def test_host_and_write_methods_cannot_access_data(web_server):
    base, _ = web_server
    with pytest.raises(HTTPError) as info:
        get(base, "/api/summary", headers={"Host": "external.example"})
    assert info.value.code == 400
    with pytest.raises(HTTPError) as info:
        urlopen(Request(base + "/api/summary", method="POST", data=b"{}"), timeout=10)
    assert info.value.code == 405


def test_all_folder_links_and_assets_are_local_readable_and_head_works(web_server):
    base, app = web_server
    summary = json.loads(get(base, "/api/summary")[0])
    assert summary["records"] == len(app.records)
    assert summary["quarantined"] == 1
    for folder in summary["folders"]:
        data = get(base, folder["url"])[0].decode()
        assert folder["path"] in data
    for path in ("/", "/assets/browser.css", "/assets/browser.js"):
        body, headers = get(base, path)
        assert body and "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    name = "/files/manifests/train.jsonl"
    assert get(base, name)[0] == (app.directory / "train.jsonl").read_bytes()
    with urlopen(Request(base + name, method="HEAD"), timeout=10) as response:
        assert response.read() == b"" and int(response.headers["Content-Length"]) > 0


def test_dry_run_never_opens_socket_or_writes(web_project, monkeypatch):
    config, app = web_project
    before = sorted(app.base.rglob("*"))
    monkeypatch.setattr("agri_vla.browser.ThreadingHTTPServer", lambda *a, **kw: pytest.fail("Socket opened during dry-run"))
    result = serve_dataset(config, app.directory, dry_run=True)
    assert result["status"] == "pass" and result["records"] == len(app.records)
    assert sorted(app.base.rglob("*")) == before


def test_browser_dry_run_does_not_import_ml(web_project):
    config, app = web_project
    code = r'''
import importlib.abc, sys
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','torchvision','lerobot','transformers','peft'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0,BlockML())
from agri_vla.cli import main
assert main(['browse','--config',sys.argv[1],'--manifests',sys.argv[2],'--dry-run'])==0
'''
    result = subprocess.run([sys.executable, "-c", code, str(config["_config_path"]), str(app.directory)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout[-1000:]
