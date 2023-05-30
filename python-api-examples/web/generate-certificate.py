#!/usr/bin/env python3

"""
pip install pyopenssl
"""

from OpenSSL import crypto

# The code in this file is modified from
# https://stackoverflow.com/questions/27164354/create-a-self-signed-x509-certificate-in-python

"""
This script generates 3 files:
    - private.key
    - selfsigned.crt
    - cert.pem

You need cert.pem when you start a https server
or a secure websocket server.

Note: You need to change serialNumber if you want to generate
a new certificate as two different certificates cannot share
the same serial number if they are issued by the same organization.

Otherwise, you may get the following error from within you browser:

  An error occurred during a connection to 127.0.0.1:6007. You have received an
  invalid certificate. Please contact the server administrator or email
  correspondent and give them the following information: Your certificate
  contains the same serial number as another certificate issued by the
  certificate authority. Please get a new certificate containing a unique
  serial number. Error code: SEC_ERROR_REUSED_ISSUER_AND_SERIAL

"""


def cert_gen(
    emailAddress="https://github.com/k2-fsa/k2",
    commonName="sherpa",
    countryName="CN",
    localityName="k2-fsa",
    stateOrProvinceName="k2-fsa",
    organizationName="k2-fsa",
    organizationUnitName="k2-fsa",
    serialNumber=3,
    validityStartInSeconds=0,
    validityEndInSeconds=10 * 365 * 24 * 60 * 60,
    KEY_FILE="private.key",
    CERT_FILE="selfsigned.crt",
    ALL_IN_ONE_FILE="cert.pem",
):
    # can look at generated file using openssl:
    # openssl x509 -inform pem -in selfsigned.crt -noout -text
    # create a key pair
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 4096)
    # create a self-signed cert
    cert = crypto.X509()
    cert.get_subject().C = countryName
    cert.get_subject().ST = stateOrProvinceName
    cert.get_subject().L = localityName
    cert.get_subject().O = organizationName  # noqa
    cert.get_subject().OU = organizationUnitName
    cert.get_subject().CN = commonName
    cert.get_subject().emailAddress = emailAddress
    cert.set_serial_number(serialNumber)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(validityEndInSeconds)
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, "sha512")
    with open(CERT_FILE, "wt") as f:
        f.write(
            crypto.dump_certificate(crypto.FILETYPE_PEM, cert).decode("utf-8")
        )
    with open(KEY_FILE, "wt") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k).decode("utf-8"))

    with open(ALL_IN_ONE_FILE, "wt") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k).decode("utf-8"))
        f.write(
            crypto.dump_certificate(crypto.FILETYPE_PEM, cert).decode("utf-8")
        )
    print(f"Generated {CERT_FILE}")
    print(f"Generated {KEY_FILE}")
    print(f"Generated {ALL_IN_ONE_FILE}")


cert_gen()
