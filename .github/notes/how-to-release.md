# Steps to release

1. Change ../../new-release.sh, and run it, git add changed files
2. Change ../../CHANGELOG.md
3. git commit and push
4. Trigger ../workflows/build-xcframework-shared-sherpa-with-static-onnxruntime.yaml
5. Trigger ../workflows/build-xcframework.yaml
6. Change ../../Package.swift
7. Create a PR
8. Merge it
9. Create a tag and push it
10. Trigger ../workflows/build-wheels*
11. Trigger ../workflows/npm*
12. Trigger ../workflows/release-go.yaml
13. Trigger ../workflows/release-rust.yaml
14. Trigger ../workflows/dot-net.yaml
15. Trigger ../workflows/release-dart-package.yaml

