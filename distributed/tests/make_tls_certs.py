"""Make the custom certificate and private key files used by TLS tests.

Code heavily borrowed from Lib/test/certdata/make_ssl_certs.py in CPython.
Changes: filenames, dropped the SAN/ECC stuff, create a smaller set of
certs which is all we need for our tests. Is excluded from pre-commit so
we can keep the diff against upstream tight for ease of updating.
"""

from __future__ import annotations

import os
import pprint
import shutil
import tempfile
from subprocess import *

startdate = "20180829142316Z"
enddate = "20371028142316Z"

req_template = """
    [ default ]
    base_url               = http://testca.pythontest.net/testca

    [req]
    distinguished_name     = req_distinguished_name
    prompt                 = no

    [req_distinguished_name]
    C                      = XY
    L                      = Dask-distributed
    O                      = Dask
    CN                     = {hostname}

    [req_x509_extensions_nosan]

    [req_x509_extensions_simple]
    subjectAltName         = @san

    [req_x509_extensions_full]
    subjectAltName         = @san
    keyUsage               = critical,keyEncipherment,digitalSignature
    extendedKeyUsage       = serverAuth,clientAuth
    basicConstraints       = critical,CA:false
    subjectKeyIdentifier   = hash
    authorityKeyIdentifier = keyid:always,issuer:always
    authorityInfoAccess    = @issuer_ocsp_info
    crlDistributionPoints  = @crl_info

    [ issuer_ocsp_info ]
    caIssuers;URI.0        = $base_url/pycacert.cer
    OCSP;URI.0             = $base_url/ocsp/

    [ crl_info ]
    URI.0                  = $base_url/revocation.crl

    [san]
    DNS.1 = {hostname}
    {extra_san}

    [dir_sect]
    C                      = XY
    L                      = Castle Anthrax
    O                      = Python Software Foundation
    CN                     = dirname example

    [princ_name]
    realm = EXP:0, GeneralString:KERBEROS.REALM
    principal_name = EXP:1, SEQUENCE:principal_seq

    [principal_seq]
    name_type = EXP:0, INTEGER:1
    name_string = EXP:1, SEQUENCE:principals

    [principals]
    princ1 = GeneralString:username

    [ ca ]
    default_ca      = CA_default

    [ CA_default ]
    dir = cadir
    database  = $dir/index.txt
    crlnumber = $dir/crl.txt
    default_md = sha256
    startdate = {startdate}
    default_startdate = {startdate}
    enddate = {enddate}
    default_enddate = {enddate}
    default_days = 7000
    default_crl_days = 7000
    certificate = tls-ca-cert.pem
    private_key = tls-ca-key.pem
    serial    = $dir/serial
    RANDFILE  = $dir/.rand
    policy          = policy_match

    [ policy_match ]
    countryName             = match
    stateOrProvinceName     = optional
    organizationName        = match
    organizationalUnitName  = optional
    commonName              = supplied
    emailAddress            = optional

    [ policy_anything ]
    countryName   = optional
    stateOrProvinceName = optional
    localityName    = optional
    organizationName  = optional
    organizationalUnitName  = optional
    commonName    = supplied
    emailAddress    = optional


    [ v3_ca ]

    subjectKeyIdentifier=hash
    authorityKeyIdentifier=keyid:always,issuer
    basicConstraints = critical, CA:true
    keyUsage = critical, digitalSignature, keyCertSign, cRLSign

    """

here = os.path.abspath(os.path.dirname(__file__))


def make_cert_key(hostname, sign=False, extra_san='',
                  ext='req_x509_extensions_full', key='rsa:3072'):
    print("creating cert for " + hostname)
    tempnames = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tempnames.append(f.name)
    req_file, cert_file, key_file = tempnames
    if sign:
        reqext = 'req_x509_extensions_simple'
    else:
        reqext = ext
    try:
        req = req_template.format(
            hostname=hostname,
            extra_san=extra_san,
            startdate=startdate,
            enddate=enddate
        )
        with open(req_file, 'w') as f:
            f.write(req)
        args = ['req', '-new', '-nodes', '-days', '7000',
                '-newkey', key, '-keyout', key_file,
                '-extensions', reqext,
                '-config', req_file]
        if sign:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                tempnames.append(f.name)
                reqfile = f.name
            args += ['-out', reqfile ]

        else:
            args += ['-x509', '-out', cert_file ]
        check_call(['openssl'] + args)

        if sign:
            args = [
                'ca',
                '-config', req_file,
                '-extensions', ext,
                '-out', cert_file,
                '-outdir', 'cadir',
                '-policy', 'policy_anything',
                '-batch', '-infiles', reqfile
            ]
            check_call(['openssl'] + args)


        with open(cert_file, 'r') as f:
            cert = f.read()
        with open(key_file, 'r') as f:
            key = f.read()
        return cert, key
    finally:
        for name in tempnames:
            os.remove(name)

TMP_CADIR = 'cadir'

def unmake_ca():
    shutil.rmtree(TMP_CADIR)

def make_ca():
    os.mkdir(TMP_CADIR)
    with open(os.path.join('cadir','index.txt'),'a+') as f:
        pass # empty file
    with open(os.path.join('cadir','crl.txt'),'a+') as f:
        f.write("00")
    with open(os.path.join('cadir','index.txt.attr'),'w+') as f:
        f.write('unique_subject = no')
    # random start value for serial numbers
    with open(os.path.join('cadir','serial'), 'w') as f:
        f.write('CB2D80995A69525B\n')

    with tempfile.NamedTemporaryFile("w") as t:
        req = req_template.format(
            hostname='our-ca-server',
            extra_san='',
            startdate=startdate,
            enddate=enddate
        )
        t.write(req)
        t.flush()
        with tempfile.NamedTemporaryFile() as f:
            args = ['req', '-config', t.name, '-new',
                    '-nodes',
                    '-newkey', 'rsa:3072',
                    '-keyout', 'tls-ca-key.pem',
                    '-out', f.name,
                    '-subj', '/C=XY/L=Dask-distributed/O=Dask CA/CN=our-ca-server']
            check_call(['openssl'] + args)
            args = ['ca', '-config', t.name,
                    '-out', 'tls-ca-cert.pem', '-batch', '-outdir', TMP_CADIR,
                    '-keyfile', 'tls-ca-key.pem',
                    '-selfsign', '-extensions', 'v3_ca', '-infiles', f.name ]
            check_call(['openssl'] + args)
            args = ['ca', '-config', t.name, '-gencrl', '-out', 'revocation.crl']
            check_call(['openssl'] + args)

def print_cert(path):
    import _ssl
    pprint.pprint(_ssl._test_decode_cert(path))


if __name__ == '__main__':
    os.chdir(here)
    cert, key = make_cert_key('localhost', ext='req_x509_extensions_simple')
    with open('tls-self-signed-cert.pem', 'w') as f:
        f.write(cert)
    with open('tls-self-signed-key.pem', 'w') as f:
        f.write(key)

    # For certificate matching tests
    make_ca()
    with open('tls-ca-cert.pem') as f:
        ca_cert = f.read()

    cert, key = make_cert_key('localhost', sign=True)
    with open('tls-cert.pem', 'w') as f:
        f.write(cert)
    with open('tls-cert-chain.pem', 'w') as f:
        f.write(cert)
        f.write(ca_cert)
    with open('tls-key.pem', 'w') as f:
        f.write(key)
    with open('tls-key-cert.pem', 'w') as f:
        f.write(key)
        f.write(cert)

    unmake_ca()

    print_cert('tls-self-signed-cert.pem')
    print_cert('tls-key-cert.pem')
    print_cert('tls-cert-chain.pem')
