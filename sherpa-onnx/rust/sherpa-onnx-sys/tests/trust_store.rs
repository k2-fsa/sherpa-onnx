//! Regression tests for the build script's trust-anchor selection (issue #3832).
//!
//! `cargo test` does not run `#[test]` functions declared inside `build.rs`, so
//! the pure decision logic lives in `build_support/trust_store.rs` and is
//! included by both the build script and this test.
//!
//! These tests are deliberately offline: they exercise the anchor-selection
//! decision with in-memory inputs and never open a socket.

use rustls::pki_types::{CertificateDer, TrustAnchor};

include!("../build_support/trust_store.rs");

/// A DER blob that is not a certificate, so rustls cannot turn it into an anchor.
fn malformed_anchor() -> CertificateDer<'static> {
    CertificateDer::from(vec![0x00, 0x01, 0x02, 0x03])
}

/// Real, parseable certificates standing in for OS-provided ones.
fn native_anchors() -> Vec<CertificateDer<'static>> {
    let pem = include_str!("data/test_root_ca.pem");
    rustls_pemfile::certs(&mut pem.as_bytes())
        .collect::<Result<Vec<_>, _>>()
        .expect("test fixture must contain a parseable certificate")
}

/// Stands in for ureq's compiled-in Mozilla root list.
fn bundled_anchors() -> Vec<TrustAnchor<'static>> {
    webpki_roots::TLS_SERVER_ROOTS.to_vec()
}

fn no_errors() -> [std::io::Error; 0] {
    []
}

#[test]
fn native_anchors_are_added_on_top_of_the_bundled_ones() {
    // The corporate-proxy case from issue #3832: the internal root CA must be
    // trusted *in addition to* the public roots, never instead of them.
    let bundled = bundled_anchors();
    let expected_bundled = bundled.len();
    let native = native_anchors();
    let expected_native = native.len();
    assert!(expected_bundled > 0, "fixture must provide bundled roots");
    assert!(expected_native > 0, "fixture must provide native anchors");

    let store = build_root_store(bundled, native, &no_errors())
        .expect("valid anchors must produce a root store");

    assert_eq!(store.len(), expected_bundled + expected_native);
}

#[test]
fn bundled_roots_are_kept_when_no_native_anchor_exists() {
    // A stripped container exposes no OS trust store. The download must keep
    // working exactly as before, i.e. through the bundled roots.
    let expected = bundled_anchors().len();

    let store = build_root_store(bundled_anchors(), Vec::new(), &no_errors())
        .expect("bundled roots alone must still produce a usable store");

    assert_eq!(store.len(), expected);
}

#[test]
fn bundled_roots_are_kept_when_every_native_anchor_is_unusable() {
    let expected = bundled_anchors().len();

    let store = build_root_store(bundled_anchors(), vec![malformed_anchor()], &no_errors())
        .expect("an unusable native anchor must not discard the bundled roots");

    assert_eq!(store.len(), expected);
}

#[test]
fn no_usable_anchor_at_all_falls_back_to_the_default_client() {
    // Verifying against an empty store would reject every connection, so the
    // caller must be told to leave ureq on its own defaults instead.
    assert!(build_root_store(Vec::new(), Vec::new(), &no_errors()).is_none());
    assert!(build_root_store(Vec::new(), vec![malformed_anchor()], &no_errors()).is_none());
}

#[test]
fn load_errors_alone_do_not_discard_usable_anchors() {
    // Partial failures are normal (an unreadable file in SSL_CERT_DIR, ...).
    // As long as one anchor loaded it must still be used.
    let errors = [std::io::Error::other("unreadable trust anchor")];
    let expected = bundled_anchors().len() + native_anchors().len();

    let store = build_root_store(bundled_anchors(), native_anchors(), &errors)
        .expect("usable anchors must survive unrelated load errors");

    assert_eq!(store.len(), expected);
}

#[test]
fn malformed_native_anchors_are_skipped_without_dropping_valid_ones() {
    let expected = bundled_anchors().len() + native_anchors().len();
    let mut mixed = vec![malformed_anchor()];
    mixed.extend(native_anchors());

    let store = build_root_store(bundled_anchors(), mixed, &no_errors())
        .expect("a malformed anchor must not discard the valid ones");

    assert_eq!(store.len(), expected);
}
