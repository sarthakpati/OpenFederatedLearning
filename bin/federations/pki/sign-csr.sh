#!/bin/bash
if [ "$#" -ne 2 ];
then
    echo "Usage: sign-csr.sh <csr filepath> <challenge string>"
    exit 1;
fi

fname=$1
fname_no_ext="${fname%.*}"
echo $fname_no_ext
challenge=$2

csr_hash=$(openssl sha256 $fname | sed 's/.*\(......\)/\1/')

echo "CSR $fname has the the csr hash $csr_hash"

if [ "$challenge" = "$csr_hash" ]; then
    echo "Challenge strings match, signing cert."
    openssl ca -config config/signing-ca.conf -batch -in $fname -out $fname_no_ext.crt
    echo "Send $fname_no_ext.crt back to requesting party."
else
    echo "Challenge string $challenge does not match the CSR hash $csr_hash. Aborting"
fi
