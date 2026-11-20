#!/usr/bin/env python3
# Copyright (c) 2026 Hedy AI
"""Check release replacement, deterministic contents and failure preservation."""
import pathlib
import tarfile
import tempfile
import unittest
from unittest import mock

from package import write_archive


class PackageTest(unittest.TestCase):
    def test_replace_with_deterministic_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / 'release.tar.bz2'
            output.write_bytes(b'previous release')
            files = {'b.fst': b'second rule', 'a.fst': b'first rule'}
            write_archive(files, output)
            first = output.read_bytes()
            write_archive(dict(reversed(list(files.items()))), output)
            self.assertEqual(output.read_bytes(), first)
            with tarfile.open(output) as archive:
                self.assertEqual(archive.getnames(), ['english-tn/a.fst', 'english-tn/b.fst'])
                for member in archive.getmembers():
                    self.assertEqual(archive.extractfile(member).read(), files[pathlib.Path(member.name).name])
                    self.assertEqual(member.mtime, 0)
                    self.assertEqual(member.mode, 0o644)
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_write_failure_preserves_previous_release(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / 'release.tar.bz2'
            output.write_bytes(b'previous release')
            with mock.patch.object(tarfile.TarFile, 'addfile', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    write_archive({'a.fst': b'new rule'}, output)
            self.assertEqual(output.read_bytes(), b'previous release')
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_replace_failure_preserves_previous_release(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / 'release.tar.bz2'
            output.write_bytes(b'previous release')
            with mock.patch.object(pathlib.Path, 'replace', side_effect=OSError('replace failed')):
                with self.assertRaises(OSError):
                    write_archive({'a.fst': b'new rule'}, output)
            self.assertEqual(output.read_bytes(), b'previous release')
            self.assertEqual(list(output.parent.iterdir()), [output])


if __name__ == '__main__':
    unittest.main()
