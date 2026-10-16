/// Build the set of trust anchors used to download the prebuilt archives.
///
/// The archives are fetched over HTTPS with `ureq`, whose default root set is
/// the bundled Mozilla list (`webpki-roots`). On a machine behind a
/// TLS-inspecting corporate proxy the traffic is re-signed by an internal root
/// CA which the OS already trusts but that list never contains, so the download
/// fails with `invalid peer certificate: UnknownIssuer`. See issue #3832.
///
/// The anchors are therefore *additive*: the bundled roots are kept and the
/// platform ones (which `rustls-native-certs` reads from the OS trust store,
/// honouring `SSL_CERT_FILE` / `SSL_CERT_DIR`) are added on top. Keeping both
/// means a machine that worked before keeps working, and a machine behind such
/// a proxy starts working.
///
/// Returns `None` when not a single anchor is usable, so the caller can leave
/// `ureq` on its own defaults rather than failing the build: verifying against
/// an empty store would reject every connection, which would turn a working
/// build into a broken one on stripped containers with no certificate store.
fn build_root_store<B, I, E>(bundled: B, native: I, errors: &[E]) -> Option<rustls::RootCertStore>
where
    B: IntoIterator<Item = rustls::pki_types::TrustAnchor<'static>>,
    I: IntoIterator<Item = rustls::pki_types::CertificateDer<'static>>,
{
    let mut roots = rustls::RootCertStore::empty();
    roots
        .roots
        .extend(bundled);
    let bundled_count = roots.len();

    // `add_parsable_certificates` skips unusable entries instead of giving up on
    // the whole set: an OS trust store legitimately contains certificates that
    // rustls cannot use, and those must not discard the valid ones.
    let (added, rejected) = roots.add_parsable_certificates(native);

    if roots.is_empty() {
        eprintln!(
            "sherpa-onnx-sys: no usable root certificates found \
             ({rejected} rejected, {} load error(s)); \
             leaving the HTTP client on its default trust anchors",
            errors.len()
        );
        return None;
    }

    if added == 0 {
        eprintln!(
            "sherpa-onnx-sys: no native root certificates were added \
             ({rejected} rejected, {} load error(s)); \
             continuing with the {bundled_count} bundled root certificate(s)",
            errors.len()
        );
    } else if rejected > 0 || !errors.is_empty() {
        eprintln!(
            "sherpa-onnx-sys: added {added} native root certificate(s) to the \
             {bundled_count} bundled one(s) ({rejected} rejected, \
             {} load error(s))",
            errors.len()
        );
    }

    Some(roots)
}
