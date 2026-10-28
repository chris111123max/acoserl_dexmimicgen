"""Run the unmodified official downloader without loading simulation dependencies.
Only substitutes the package path needed for the default download directory.
Explicit --path and --tasks arguments retain the official Hub download behavior.
"""
import runpy, sys, types
from pathlib import Path
root = Path(__file__).resolve().parents[1]
pkg = types.ModuleType('dexmimicgen')
pkg.__path__ = [str(root / 'third_party/dexmimicgen/dexmimicgen')]
sys.modules['dexmimicgen'] = pkg
runpy.run_path(str(root / 'third_party/dexmimicgen/scripts/download_hf_dataset.py'), run_name='__main__')
