## Python

- os.sep

- os.path.getsize

- rsplit

- 

  

```python
//img_path——>label_path
def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]
```

```python
//hash value
def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash
```