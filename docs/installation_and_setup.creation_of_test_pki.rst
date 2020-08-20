.. # Copyright (C) 2020 Intel Corporation
.. # Licensed under the Apache License, Version 2.0 (the "License");
.. # you may not use this file except in compliance with the License.
.. # You may obtain a copy of the License at
.. #
.. #     http://www.apache.org/licenses/LICENSE-2.0
.. #
.. # Unless required by applicable law or agreed to in writing, software
.. # distributed under the License is distributed on an "AS IS" BASIS,
.. # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. # See the License for the specific language governing permissions and
.. # limitations under the License.

.. _Creating Certs:

**************************
Creating the Test PKI (Optional)
**************************

See :ref:`PKI Requirements` for an overview of how |prod| uses TLS with digital certificates.

This section explains how to use some convenience open-ssl scripts to generate a test PKI. 

**NOTE: Proper consideration needs to be taken when deploying Open Federated Learning.  The scripts that create the custom PKI for the project do not have a mechanism to sign any of the certificates by a proper Certificate Authority.  The CA certificate that is created will only be self-signed.  Trust in this certificate therefore will be based on your abilty to propertly secure the private key for that certificate, as well as securely distribute the CA Certificate to your endpoints.**
   
**Use these scripts at your own risk.** 


.. note::
  You do not need to create a PKI to use the single process simulation mode, as there is no network involved in simulation, and thus no TLS.


.. note::
  This tutorial relies on :ref:`SCN Mode`, which is a convenience feature for testing and devlopment and should never be used in production.

Create the Certificate Authority and Signing Key
######################

1.	Change the directory to bin/federations/pki:

.. code-block:: console

  $ cd bin/federations/pki

2.	Run the Certificate Authority script. This will create a `Certificate Authority <https://en.wikipedia.org/wiki/Certificate_authority>`_
for the Federation on this node. All certificates will be
signed by this signing key. Follow the command-line instructions and enter
in the information as prompted. The script will create a simple database
file to keep track of all issued certificates.

.. code-block:: console

  $ bash setup_ca.sh

.. note::
  This is not a proper way to manage a real CA. We are not giving guidance on how to protect keys!

This will create a root key, signing key, and public cert_chain. You should find the 'cert_chain.crt' under bin/federations/pki:

.. code-block:: console

  $ ls -l .
  drwxr-xr-x 4 msheller intelall 4096 Jul  1 12:52 ca
  -rw-r--r-- 1 msheller intelall 9079 Aug 10 08:58 cert_chain.crt
  drwxr-xr-x 2 msheller intelall 4096 Aug 19 16:23 client
  drwxr-xr-x 2 msheller intelall 4096 Jun 10 15:13 config
  -rw-r--r-- 1 msheller intelall 1684 Aug 18 12:02 create-aggregator.sh
  -rw-r--r-- 1 msheller intelall 1660 Aug 18 12:02 create-and-sign-aggregator.sh
  -rw-r--r-- 1 msheller intelall 1035 Aug 18 12:02 create-and-sign-collaborator.sh
  -rw-r--r-- 1 msheller intelall 1083 Aug 18 12:02 create-collaborator.sh
  -rw-r--r-- 1 msheller intelall   99 Jun 10 15:13 README.md
  drwxr-xr-x 2 msheller intelall 4096 Aug 19 16:10 server
  -rw-r--r-- 1 msheller intelall 1394 Jun 10 15:13 setup_ca.sh
  -rw-r--r-- 1 msheller intelall  629 Aug 13 13:56 sign-csr.sh

Every aggregator or collaborator machine will need this cert_chain.crt file in this same location. This is how that node can verify signatures made by your signing key. Without this file on each machine, the TLS connections will fail.

3.	Run the aggregator cert script, replacing AGG.FQDN
with the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_
for the aggregator machine. You may optionally include the
IP address for the aggregator, replacing [IP_ADDRESS].

.. code-block:: console

  $ bash create-and-sign-aggregator.sh AGG.FQDN

.. note::
   You can discover the FQDN with the Linux command:

   .. code-block:: console

     $ hostname --all-fqdns | awk '{print $1}'

After creating this certificate, you should see the following files under bin/federations/pki/server:

.. code-block:: console

  $ ls -l ./server
  -rw-r--r-- 1 msheller intelall 4704 Aug 19 16:10 AGG.FQDN.crt
  -rw------- 1 msheller intelall 1708 Aug 19 16:10 AGG.FQDN.key

You will need to move these files to the same location on aggregator node.

4.	Next we create collaborator certificates. Normally, you want to create a certificate for each collaborator.
However, in testing environments, this is overly-burdensome to manage, and not necessary if you are only testing that TLS is working and all the machines are under your control and trusted.
Instead, we will create a single collaborator certificate for all our test collaborator processes.
**This is not appropriate for actual TLS use cases.**
Pick a name for your test_collaborator certificate. You will be passing this name as an argument to every collaborator/aggregator process.

.. code-block:: console

  $ bash create-and-sign-collaborator.sh MICAH.TEST.COLLABORATOR.CERT


.. note::
  I don't advise using my name :) Pick something more meaningful.


Now you should have the following files under bin/federations/pki/client:

.. code-block:: console

  $ ls -l ./client
  -rw-r--r-- 1 msheller intelall 4655 Aug 19 16:23 MICAH.TEST.COLLABORATOR.CERT.crt
  -rw------- 1 msheller intelall 1704 Aug 19 16:23 MICAH.TEST.COLLABORATOR.CERT.key

Each collaborator machine will need a copy of these files in this same location.

.. note::
  Beating a dead horse here: a production-worthy PKI involves some real form of identity verification. Generating keys, signing them, then giving them out is NOT proper key management. This is for testing/development purposes only!

Summary of Test PKI files
#########################

After creating and transfering files around you should have:

.. list-table:: Collaborator PKI Files (on each collaborator machine)
   :widths: 50 50
   :header-rows: 1

   * - File Type
     - Filename
   * - Certificate chain
     - bin/federations/pki/cert_chain.crt
   * - Collaborator certificate
     - bin/federations/pki/client/MICAH.TEST.COLLABORATOR.CERT.crt
   * - Collaborator key
     - bin/federations/pki/client/MICAH.TEST.COLLABORATOR.CERT.key


.. list-table:: Aggregator PKI Files
   :widths: 50 50
   :header-rows: 1

   * - File Type
     - Filename
   * - Certificate chain
     - bin/federations/pki/cert_chain.crt
   * - Aggregator certificate
     - bin/federations/pki/server/AGG.FQDN.crt
   * - Aggregator key
     - bin/federations/pki/server/AGG.FQDN.key

