import nltk

_REQUIRED_PACKAGES = [
    "punkt",
    "stopwords",
]

def ensure_nltk():
    """Ensure essential NLTK datasets are available."""
    for pkg in _REQUIRED_PACKAGES:
        try:
            if pkg == "stopwords":
                nltk.data.find(f"corpora/{pkg}")
            else:
                nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
