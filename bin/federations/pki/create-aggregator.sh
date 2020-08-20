#!/bin/bash

# valid_id function from this article: https://www.linuxjournal.com/content/validating-ip-address-bash-script
function valid_ip()
{
    local  ip=$1
    local  stat=1

    if [[ $ip =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        OIFS=$IFS
        IFS='.'
        ip=($ip)
        IFS=$OIFS
        [[ ${ip[0]} -le 255 && ${ip[1]} -le 255 \
            && ${ip[2]} -le 255 && ${ip[3]} -le 255 ]]
        stat=$?
    fi;
    return $stat
}

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

if [ -z $1 ];
then
    echo "Usage: create-aggregator FQDN [IP_ADDRESS]"
    echo "IP address is optional."
    exit 1;
fi

if valid_fqdn $1; 
then 
    echo "Valid FQDN";
else 
    echo "Invalid FQDN";
    exit 1;
fi

FQDN=$1
subject_alt_name="DNS:$FQDN"

if [ -z $2 ];
then
    echo "No IP specified. IP address will not be included in subject alt name."
else
    if valid_ip $2; 
    then 
        echo "Valid IP address";
    else 
        echo "Invalid IP address.";
        exit 1;
    fi
    subject_alt_name="$subject_alt_name,IP:$2"
fi

echo "Creating debug client key pair with following settings: CN=$FQDN SAN=$subject_alt_name"

fname="$FQDN"

SAN=$subject_alt_name openssl req -new -config config/server.conf -subj "/CN=$FQDN" -out $fname.csr -keyout $fname.key

mkdir -p server
mv $fname.csr $fname.key server

echo "Send the following 6 hex values to the signing party to confirm your CSR:"
openssl sha256 server/$fname.csr | sed 's/.*\(......\)/\1/'
