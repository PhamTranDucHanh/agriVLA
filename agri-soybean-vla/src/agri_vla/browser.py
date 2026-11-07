"""Read-only, loopback dataset browser. Runtime uses only existing data dependencies."""

from collections import Counter, defaultdict
import errno
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
from importlib.resources import files
from io import BytesIO
import json
import math
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import time
import unicodedata
from urllib.parse import parse_qs, quote, unquote, urlsplit

from PIL import Image, ImageOps

from .dataset import DatasetError, read_jsonl, require, SPLITS
from .split import digest, load_split_config, relative_file
from .validate import validate_dataset


def search_text(value) -> str:
    text = unicodedata.normalize("NFD", str(value).casefold().replace("đ", "d"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def page_of(items, query):
    try:
        page = int(query.get("page", "1"))
        size = int(query.get("page_size", "12"))
    except ValueError as error:
        raise DatasetError("page và page_size phải là số nguyên") from error
    require(page >= 1 and size in (12, 24, 48), "page ≥1; page_size là 12, 24 hoặc 48")
    pages = max(1, math.ceil(len(items) / size))
    page = min(page, pages)
    return {"items": items[(page - 1) * size:page * size], "total": len(items),
            "page": page, "pages": pages, "page_size": size}


class BrowserDataset:
    def __init__(self, config: dict, directory: Path, *, validated=None):
        self.report, inputs, records = validate_dataset(config, directory) if validated is None else validated
        self.base, self.directory = inputs.base, directory.resolve()
        self.records, self.by_id, self.by_case = [], {}, defaultdict(list)
        self.usage = Counter()
        eligible = {r["source_image_id"]: r for rows in inputs.rows.values() for r in rows}
        sc = load_split_config(relative_file(self.base, config["generation"]["source_split_config"]))
        source_path = relative_file(self.base, sc["source"]["manifest"])
        # validate_dataset has already verified this pinned source manifest.
        self.sources = {r["source_image_id"]: r for r in read_jsonl(source_path)}
        self.source_split = {source: r["split"] for source, r in eligible.items()}
        for split in SPLITS:
            for line, row in enumerate(records[split], 1):
                item = {**row, "ordinal": len(self.records) + 1, "manifest_line": line}
                self.records.append(item)
                self.by_id[item["sample_id"]] = item
                self.by_case[item["case_id"]].append(item)
                self.usage.update(item["source_image_ids"])
        self.search_records = {r["sample_id"]: search_text(" ".join(map(str, (r["sample_id"], r["scene_id"], r["case_id"],
                                r["instruction"], r["template_family_id"], *r["source_image_ids"], *r["source_groups"])))) for r in self.records}
        self.search_sources = {s: search_text(f"{s} {r['split_group']} {r['original_path']}") for s, r in self.sources.items()}
        self.source_items = [self.source_item(source) for source in sorted(self.sources)]
        self.file_entries = {}
        self.add_files("manifests", self.directory, ["train.jsonl", "val.jsonl", "test.jsonl", "dataset_lock.json", "qa_report.json", "DATASET_CARD.md"])
        split_dir = relative_file(self.base, config["generation"]["source_split_dir"])
        self.add_files("source-splits", split_dir, ["train.sources.jsonl", "val.sources.jsonl", "test.sources.jsonl", "excluded.sources.jsonl", "split_lock.json", "split_report.json", "group_stats.csv"])
        raw = relative_file(self.base, config["paths"]["raw_dir"])
        self.add_files("raw", raw, [config["files"][k]["name"] for k in ("csv", "zip")])
        self.add_files("source-report", source_path.parent, [source_path.name, "source_report.json", "quarantine.csv", "group_stats.csv"])
        for source, row in self.sources.items():
            image = relative_file(self.base, row["original_path"])
            self.file_entries["images/" + source] = image
            self.file_entries["raw/images/" + source] = image
        self.folders = {"manifests": self.directory, "images": relative_file(self.base, config["paths"]["images_dir"]),
                        "raw": raw, "source-splits": split_dir, "source-report": source_path.parent}
        self.summary = {"status": "pass", "dataset_revision":self.directory.name, "records": len(self.records), "cases": len(self.by_case),
                        "scenes": len({r["scene_id"] for r in self.records}), "used_source_images": len(self.usage),
                        "raw_source_images": len(self.sources), "quarantined": sum(not r["eligible_v0"] for r in self.sources.values()),
                        "splits": {s: len(records[s]) for s in SPLITS}, "dataset_lock_sha256": self.report["dataset_lock_sha256"],
                        "options": {"intent": sorted({r["intent"] for r in self.records}),
                                    "kind": sorted({r["generation_metadata"]["kind"] for r in self.records}),
                                    "group": sorted({r["split_group"] for r in self.sources.values()})},
                        "folders": [{"key": key, "path": str(path), "url": "/files/" + key + "/"} for key, path in self.folders.items()]}

    def add_files(self, alias, directory, names):
        for name in names:
            path = directory / name
            if path.is_file():
                self.file_entries[alias + "/" + name] = path

    def source_item(self, source):
        r = self.sources[source]
        return {"source_image_id": source, "split": self.source_split.get(source), "group": r["split_group"],
                "annotation": r["annotation"], "eligible": r["eligible_v0"], "record_uses": self.usage[source],
                "timestamp": r.get("timestamp"), "width": r["width"], "height": r["height"],
                "path": str(relative_file(self.base, r["original_path"])), "sha256": r["sha256"],
                "original_url": "/original/" + quote(source, safe=""), "thumbnail_url": "/thumbnail/" + quote(source, safe=""),
                "folder_url": "/files/images/?q=" + quote(source, safe="")}

    def samples(self, query):
        allowed = {"split": ("", *SPLITS), "language": ("", "vi", "en"), "wording": ("", "seen", "unseen"),
                   "intent": ("", *self.summary["options"]["intent"]), "kind": ("", *self.summary["options"]["kind"]),
                   "group": ("", *self.summary["options"]["group"]), "wilt": ("", "0", "1", "2", "3", "4"),
                   "action": ("", "zero", "on", "high")}
        for key, values in allowed.items():
            require(query.get(key, "") in values, f"Bộ lọc {key} không hợp lệ")
        needle = search_text(query.get("q", ""))
        selected = []
        for row in self.records:
            if any(query.get(key) and row[key] != query[key] for key in ("split", "language", "intent", "scene_id", "case_id")):
                continue
            if query.get("wording") and row["wording_partition"] != query["wording"]:
                continue
            if query.get("kind") and row["generation_metadata"]["kind"] != query["kind"]:
                continue
            if query.get("group") and query["group"] not in row["source_groups"]:
                continue
            if query.get("wilt") and int(query["wilt"]) not in row["teacher_annotations"]["wilt_scores"]:
                continue
            if query.get("source_id") and query["source_id"] not in row["source_image_ids"]:
                continue
            action = row["target_action"]
            if query.get("action") and not {"zero": not any(action), "on": any(action), "high": action == [3, 3, 3]}[query["action"]]:
                continue
            if needle and needle not in self.search_records[row["sample_id"]]:
                continue
            selected.append(self.sample_card(row))
        return page_of(selected, query)

    @staticmethod
    def sample_card(row):
        keys = ("sample_id", "scene_id", "case_id", "split", "ordinal", "manifest_line", "instruction", "language", "wording_partition",
                "intent", "intent_params", "target_action", "source_image_ids", "source_groups", "template_family_id", "teacher_annotations", "generation_metadata")
        return {key: row[key] for key in keys}

    def source_list(self, query):
        require(query.get("split", "") in ("", *SPLITS), "Split không hợp lệ")
        require(query.get("wilt", "") in ("", "0", "1", "2", "3", "4", "5"), "Nhãn nguồn không hợp lệ")
        require(query.get("eligibility", "") in ("", "eligible", "quarantine", "unused"), "Trạng thái nguồn không hợp lệ")
        require(query.get("group", "") in ("", *self.summary["options"]["group"]), "Nhóm nguồn không hợp lệ")
        needle = search_text(query.get("q", ""))
        selected = []
        for row in self.source_items:
            if query.get("split") and row["split"] != query["split"]:
                continue
            if query.get("group") and row["group"] != query["group"]:
                continue
            if query.get("wilt") and str(row["annotation"]) != query["wilt"]:
                continue
            status = query.get("eligibility")
            if status == "eligible" and not row["eligible"] or status == "quarantine" and row["eligible"] or status == "unused" and (row["record_uses"] or not row["eligible"]):
                continue
            if needle and needle not in self.search_sources[row["source_image_id"]]:
                continue
            selected.append(row)
        return page_of(selected, query)

    def detail(self, query):
        if "index" in query:
            try:
                index = int(query["index"])
            except ValueError as error:
                raise DatasetError("Số thứ tự mẫu phải là số nguyên") from error
            require(1 <= index <= len(self.records), "Số thứ tự mẫu ngoài dataset")
            row = self.records[index - 1]
        else:
            row = self.by_id.get(query.get("id"))
            if row is None:
                raise FileNotFoundError("Không có sample ID này")
        record = {k: v for k, v in row.items() if k not in {"ordinal", "manifest_line"}}
        return {"record": record, "ordinal": row["ordinal"], "manifest_line": row["manifest_line"],
                "sources": [self.source_item(source) | {"metadata": self.sources[source]} for source in row["source_image_ids"]],
                "related": [self.sample_card(r) for r in self.by_case[row["case_id"]]]}

    def checked_image(self, source):
        row = self.sources.get(source)
        if row is None:
            raise FileNotFoundError("Không có ảnh nguồn này")
        path = relative_file(self.base, row["original_path"])
        data = path.read_bytes()
        require(digest(data) == row["sha256"], "Ảnh gốc đã đổi checksum; dừng xem ảnh và kiểm tra lại dataset")
        return data

    @lru_cache(maxsize=256)
    def thumbnail(self, source, mtime_ns, file_size):
        with Image.open(BytesIO(self.checked_image(source))) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (320, 240), Image.Resampling.LANCZOS)
            output = BytesIO()
            thumb.save(output, format="JPEG", quality=85)
            return output.getvalue()

    def folder_page(self, alias, query):
        require(alias in self.folders, "Thư mục không thuộc dataset")
        needle = search_text(query.get("q", ""))
        prefix = alias + "/"
        entries = [(name[len(prefix):], path) for name, path in sorted(self.file_entries.items()) if name.startswith(prefix) and
                   "/" not in name[len(prefix):] and (not needle or needle in search_text(name))]
        escape = html.escape
        rows = []
        if alias == "raw":
            rows.append('<tr><td><a href="/files/images/">images/ — thư mục ảnh gốc</a></td><td></td></tr>')
        for name, path in entries:
            link = "/files/" + quote(prefix + name, safe="/")
            rows.append(f'<tr><td><a href="{escape(link, quote=True)}">{escape(name)}</a></td><td>{path.stat().st_size:,} bytes</td></tr>')
        return ('<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Thư mục dataset local</title><link rel="stylesheet" href="/assets/browser.css"><main class="file-page">'
                '<a href="/?view=files">← Trình duyệt dataset</a>'
                f'<h1>Thư mục {escape(alias)}</h1><p>Đọc trực tiếp từ thư mục thật trên máy chạy server:</p><code class="path">{escape(str(self.folders[alias]))}</code>'
                '<p>Nhấp file để mở hoặc tải bản gốc. Có thể chọn và copy đường dẫn trên để mở bằng Files của Linux.</p>'
                f'<form><label>Tìm tên file <input name="q" value="{escape(query.get("q", ""), quote=True)}"></label><button>Tìm</button><a href="/files/{alias}/">Tất cả</a></form>'
                f'<p>{len(entries):,} file phù hợp</p><table><thead><tr><th>Tệp gốc</th><th>Kích thước</th></tr></thead><tbody>{"".join(rows)}</tbody></table></main></html>').encode()


def handler_for(app: BrowserDataset):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AgriVLALocal/1.0"

        def log_message(self, format, *args):
            # Keep terminal usable while loading image grids; log only failures.
            if len(args) > 1 and str(args[1]).startswith(("4", "5")):
                super().log_message(format, *args)

        def headers_for(self, status, content_type, length, *, disposition=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            if disposition:
                self.send_header("Content-Disposition", disposition)
            self.end_headers()

        def send_bytes(self, data, content_type, status=200, *, disposition=None):
            self.headers_for(status, content_type, len(data), disposition=disposition)
            if self.command != "HEAD":
                self.wfile.write(data)

        def send_json(self, value, status=200, *, disposition=None):
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8", status, disposition=disposition)

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            self.send_json({"error": "Trình duyệt chỉ đọc dữ liệu"}, 405)

        def do_GET(self):
            try:
                port = self.server.server_address[1]
                allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
                require(self.headers.get("Host") in allowed, "Host phải là localhost hoặc 127.0.0.1 của server")
                origin = self.headers.get("Origin")
                require(not origin or origin in {"http://" + host for host in allowed}, "Origin không hợp lệ")
                request = urlsplit(self.path)
                path = unquote(request.path)
                require(".." not in PurePosixPath(path).parts and "\\" not in path and "\x00" not in path, "Đường dẫn không hợp lệ")
                query = {key: values[-1] for key, values in parse_qs(request.query, keep_blank_values=True).items()}
                if path in ("/", "/index.html"):
                    self.send_bytes(files("agri_vla").joinpath("web/browser.html").read_bytes(), "text/html; charset=utf-8")
                elif path.startswith("/assets/"):
                    name = path.removeprefix("/assets/")
                    if name not in {"browser.css", "browser.js"}:
                        raise FileNotFoundError("Không có asset này")
                    self.send_bytes(files("agri_vla").joinpath("web/" + name).read_bytes(), "text/css; charset=utf-8" if name.endswith(".css") else "text/javascript; charset=utf-8")
                elif path == "/api/health":
                    self.send_json({"status": "ready", "records": len(app.records), "dataset_lock_sha256": app.summary["dataset_lock_sha256"]})
                elif path == "/api/summary":
                    self.send_json(app.summary)
                elif path == "/api/samples":
                    self.send_json(app.samples(query))
                elif path == "/api/sources":
                    self.send_json(app.source_list(query))
                elif path == "/api/sample":
                    detail = app.detail(query)
                    if query.get("download") == "1":
                        self.send_json(detail["record"], disposition='attachment; filename="' + detail["record"]["sample_id"] + '.json"')
                    else:
                        self.send_json(detail)
                elif path.startswith(("/original/", "/thumbnail/")):
                    kind, source = path.lstrip("/").split("/", 1)
                    if source not in app.sources:
                        raise FileNotFoundError("Không có ảnh nguồn này")
                    if kind == "thumbnail":
                        stat = relative_file(app.base, app.sources[source]["original_path"]).stat()
                        data = app.thumbnail(source, stat.st_mtime_ns, stat.st_size)
                        mime = "image/jpeg"
                    else:
                        data = app.checked_image(source)
                        with Image.open(BytesIO(data)) as image:
                            mime = Image.MIME.get(image.format, "application/octet-stream")
                    self.send_bytes(data, mime)
                elif path.startswith("/files/"):
                    key = path.removeprefix("/files/")
                    if key.endswith("/"):
                        self.send_bytes(app.folder_page(key.rstrip("/"), query), "text/html; charset=utf-8")
                    else:
                        target = app.file_entries.get(key)
                        if target is None:
                            raise FileNotFoundError("File không nằm trong danh sách dataset")
                        require((target.resolve().is_relative_to(app.base) or target.resolve().is_relative_to(app.directory)) and not target.is_symlink(), "File không thuộc dataset local")
                        if key.startswith(("images/", "raw/images/")):
                            data = app.checked_image(key.rsplit("/", 1)[-1])
                            self.send_bytes(data, mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                        else:
                            with target.open("rb") as stream:
                                self.headers_for(200, mimetypes.guess_type(target.name)[0] or "text/plain; charset=utf-8", target.stat().st_size)
                                if self.command != "HEAD":
                                    while chunk := stream.read(1024 * 1024):
                                        self.wfile.write(chunk)
                else:
                    raise FileNotFoundError("Không có trang này")
            except FileNotFoundError as error:
                self.send_json({"error": str(error)}, 404)
            except (DatasetError, ValueError, KeyError) as error:
                self.send_json({"error": str(error)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError:
                self.send_json({"error": "Không đọc được file local; kiểm tra terminal và chạy lại validator"}, 500)

    return Handler


def replace_browser(port, module):
    """Explicit restart: stop only this user's same-repo data-browser process."""
    result = subprocess.run(["ss","-ltnp",f"( sport = :{port} )"],capture_output=True,text=True,check=True)
    pids = set(map(int,re.findall(r"pid=(\d+)",result.stdout)))
    listeners = result.stdout.splitlines()[1:]
    require(not listeners or pids, "Không xác minh được PID cổng; không dừng dịch vụ khác")
    for pid in pids:
        proc = Path("/proc")/str(pid)
        cmd = (proc/"cmdline").read_bytes().decode().strip("\0").split("\0")
        uid = int((proc/"status").read_text().split("Uid:")[1].split()[0])
        cwd = (proc/"cwd").resolve()
        trusted = (uid == os.getuid() and cwd in (module,module.parent) and len(cmd)>=4
                   and Path(cmd[0]).absolute().parent == module/".venv/bin"
                   and cmd[1:3] == ["-m","agri_vla.cli"] and cmd[3] in ("browse","browse-revision"))
        require(trusted, f"Từ chối dừng PID {pid}: không phải browser cùng repo/user")
    for pid in pids:
        os.kill(pid,signal.SIGTERM)
    if pids:
        time.sleep(.2)


def serve_dataset(config, directory, *, port=8765, dry_run=False, validated=None, replace=False):
    require(type(port) is int and 0 <= port <= 65535, "Port phải nằm trong 0–65535")
    if not dry_run:
        print("Đang kiểm tra dataset lock, manifests và ảnh gốc…", flush=True)
    app = BrowserDataset(config, directory, validated=validated)
    if dry_run:
        return {"status": "pass", "dry_run": True, "listen": f"127.0.0.1:{port}", **app.summary}
    if replace:
        from .preflight import project_root
        replace_browser(port,project_root())
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(app))
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EPERM):
            raise DatasetError("Môi trường đang chặn socket local. Chạy scripts/browse.sh trong terminal local có quyền mở localhost.") from error
        raise DatasetError(f"Không mở được port {port}: {error}. Thử --port 8766.") from error
    address = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"\nDataset QA PASS · {len(app.records):,} mẫu · {len(app.sources):,} ảnh nguồn", flush=True)
    print(f"Mở trình duyệt: {address}", flush=True)
    print("Giữ terminal này mở. Nhấn Ctrl+C để dừng server.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng trình duyệt dataset.", flush=True)
    finally:
        server.server_close()
    return {"status": "stopped", "url": address}
