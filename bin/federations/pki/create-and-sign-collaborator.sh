#!/bin/bash
function valid_fqdn()
{
    local fqdn=$1
    local stat=0
    result=`echo $fqdn | grep -P '(?=^.{1,254}$)(^(?>(?!\d+\.)[a-zA-Z0-9_\-]{1,63}\.?)+(?:[a-zA-Z]{2,})$)'`
    if [[ -z "$result" ]]
    then
        stat=1
    fi
    return $stat
}

if [ "$#" -ne 1 ];
then
    echo "Usage: create-collaborator FQDN/CN"
    exit 1;
fi

FQDN=$1
subject_alt_name="DNS:$FQDN"

if valid_fqdn $FQDN; 
then 
    echo "Valid FQDN";
    extensions="client_reqext_san"
else 
    echo "Note: collaborator CN is not a valid FQDN and will not be added to the DNS entry of the subject alternative names";
    extensions="client_reqext"
fi

FQDN=$1

fname="$FQDN"

echo "Creating collaborator key pair with following settings: CN=$FQDN SAN=$subject_alt_name"

SAN=$subject_alt_name openssl req -new -config config/client.conf -out $fname.csr -keyout $fname.key -subj "/CN=$FQDN" -reqexts $extensions
openssl ca -config config/signing-ca.conf -batch -in $fname.csr -out $fname.crt

mkdir -p client
mv $fname.crt $fname.key client
rm $fname.csr
