"""Integrity and path isolation tests for the reproduction tool."""
import argparse,importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('repro',Path(__file__).with_name('reproduce.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ReproductionContracts(unittest.TestCase):
 def test_path_traversal_is_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaises(ValueError):m.safe(Path(t),'../outside')
 def test_corrupt_or_missing_asset_fails(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);f=root/'data.bin';f.write_bytes(b'fixed');entry={'relative':'data.bin','bytes':5,'sha256':m.sha(f)}
   m.REPO=root;(root/'bundle.json').write_text(json.dumps({'profile':'validation','files':[entry],'code_sha256':{},'file_count':1}))
   m.verify(argparse.Namespace(bundle=t))
   f.write_bytes(b'wrong')
   with self.assertRaises(ValueError):m.verify(argparse.Namespace(bundle=t))
   f.unlink()
   with self.assertRaises(ValueError):m.verify(argparse.Namespace(bundle=t))
 def test_nonempty_workspace_is_not_overwritten(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);bundle=root/'bundle';bundle.mkdir();(bundle/'bundle.json').write_text(json.dumps({'runtime_original':'/old'}));work=root/'run';work.mkdir();(work/'owner.txt').write_text('keep')
   with self.assertRaises(ValueError):m.hydrate(argparse.Namespace(bundle=str(bundle),workspace=str(work),copy=True))
   self.assertEqual((work/'owner.txt').read_text(),'keep')
class FreshWorkspaceTests(unittest.TestCase):
 def test_old_checkpoints_are_not_imported(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);bundle=root/'bundle';repo=root/'repo';(repo/'experiments').mkdir(parents=True)
   entries=[]
   for relative,content in [('roots/banelp-refine-v1/data/x.wav',b'audio'),('roots/banelp-refine-v1/initial/step.pt',b'oldstate')]:
    f=bundle/relative;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(content);entries.append({'relative':relative})
   (bundle/'bundle.json').write_text(json.dumps({'runtime_original':'/old','files':entries}))
   m.REPO=repo
   with patch.object(m.subprocess,'run'):
    m.hydrate(argparse.Namespace(bundle=str(bundle),workspace=str(root/'work'),copy=True))
   self.assertFalse((root/'work/runtimes/banelp-refine-v1/initial/step.pt').exists())
   self.assertEqual((root/'work/runtimes/banelp-refine-v1/data/x.wav').read_bytes(),b'audio')

class ArchiveTests(unittest.TestCase):
 def test_stream_checksum_and_archive_prefix(self):
  import tarfile
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);bundle=root/'bundle';bundle.mkdir();(bundle/'data.bin').write_bytes(b'payload')
   (bundle/'bundle.json').write_text(json.dumps({'bytes':7,'files':[{'relative':'data.bin'}]}))
   archive=root/'assets.tar';m.pack(argparse.Namespace(bundle=str(bundle),archive=str(archive)))
   self.assertEqual(archive.with_suffix('.tar.sha256').read_text().split()[0],m.sha(archive))
   with tarfile.open(archive) as tar:self.assertEqual(tar.getnames(),['mss-bundle/bundle.json','mss-bundle/data.bin'])

if __name__=='__main__':unittest.main()
