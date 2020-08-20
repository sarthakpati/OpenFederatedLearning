#!/bin/bash

#1. Create Root CA

# 1.1 Create Directories
mkdir -p ca/root-ca/private ca/root-ca/db
chmod 700 ca/root-ca/private

# 1.2 Create database
cp /dev/null ca/root-ca/db/root-ca.db
cp /dev/null ca/root-ca/db/root-ca.db.attr
echo 01 > ca/root-ca/db/root-ca.crt.srl
echo 01 > ca/root-ca/db/root-ca.crl.srl

#1.3 Create CA request
openssl req -new \
    -config config/root-ca.conf \
    -out ca/root-ca.csr \
    -keyout ca/root-ca/private/root-ca.key

#1.4 Create CA certificate
openssl ca -selfsign \
    -config config/root-ca.conf \
    -in ca/root-ca.csr \
    -out ca/root-ca.crt \
    -extensions root_ca_ext

# 2. Create Signing Cert
# 2.1 Create directories
mkdir -p ca/signing-ca/private ca/signing-ca/db
chmod 700 ca/signing-ca/private

#2.2 Create database
cp /dev/null ca/signing-ca/db/signing-ca.db
cp /dev/null ca/signing-ca/db/signing-ca.db.attr
echo 01 > ca/signing-ca/db/signing-ca.crt.srl
echo 01 > ca/signing-ca/db/signing-ca.crl.srl

#2.3 Create Signing Cert CSR
openssl req -new \
    -config config/signing-ca.conf \
    -out ca/signing-ca.csr \
    -keyout ca/signing-ca/private/signing-ca.key

#2.4 Sign Signing Cert CSR
openssl ca \
    -config config/root-ca.conf \
    -in ca/signing-ca.csr \
    -out ca/signing-ca.crt \
    -extensions signing_ca_ext

#3 Create Certificate Chain
cat ca/root-ca.crt >> cert_chain.crt
cat ca/signing-ca.crt >> cert_chain.crt
