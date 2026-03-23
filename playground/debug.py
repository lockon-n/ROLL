import urllib.request

tests = [
    ("http://127.0.0.1:45101/get_or_create_runtime_env", b"abc"),
    ("http://172.18.0.5:45101/get_or_create_runtime_env", b"abc"),
]

for url, body in tests:
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            print(url, r.status, r.read(200))
    except Exception as e:
        print(url, repr(e))

