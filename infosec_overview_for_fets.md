### Purpose of this document
This document is intended to help InfoSec analysis processes of the collaborators involved in the initial stages of the FeTS initiative by summarizing key points of the code functionality. Please be aware that this document may not be up to date with the latest code, particularly as this repository is intended to be EOL'd soon in favor of the long-term openfl project (see main README.md for link).

In this context, the "FeTS software" refers to the FeTS front-end and the specific configuration of openfl used by the FeTS intiative. Thus we will say that the "FeTS software uses mutually authenticated TLS" because the FeTS initiative uses that configuration of openfl. Openfl supports other configurations, but they are not relevant to this document. Some of the FeTS software comes as submodules from repositories maintained at this time by the University of Pennsylvania (UPenn)

#### Network Connectivity Overview
The FeTS software uses a hub-and-spoke topology between _collaborator_ clients that generate model parameter updates from their data and the _aggregator_ server that combines their training updates into new models. Key details about this functionality are:
* Connections are made using request/response gRPC connections.
* The _aggregator_ listens for connections on port 50051, so all _collaborators_ must be able to send outgoing traffic on this port.
* All connections are initiated by the _collaborator_.
* The _collaborator_ does not open any listening sockets.
* Connections are secured using mutually-authenticated TLS.
* Each request response pair is done on a new TLS connection.
* The PKI for FeTS is currently created specifically for FeTS using openssl tools. The team at UPenn acts as the certificate authority to verify each identity before signing.
* Currently, the _collaborator_ polls the _aggregator_ at a fixed interval. We have had a request to enable client-side configuration of this interval and hope to support that feature soon.
* Connection timeouts are set to gRPC defaults.
* If the _aggregator_ is not available, the _collaborator_ will retry connections indefinitely. This is currently useful so that we can take the aggregator down for bugfixes without _collaborator_ processes crashing.
* Note that there is currently an issue with the library OS the _aggregator_ runs on that can cause TLS decryption failures on the _aggregator_. Currently, the _collaborator_ simply retries the message. We believe that the failure is in the network stack on the _aggregator_ and have alerted the developers of the library OS we use.

#### Overview of Contents of Network Messages
Network messages are well defined protobufs which can be found in https://github.com/IntelLabs/OpenFederatedLearning/blob/fets/openfl/proto/collaborator_aggregator_interface.proto
Key points about the network messages/protocol:
* No executable code is ever sent to the _collaborator_. All algorithms come with the installer.
* The _collaborator_ typically sends a "job" request, to which the aggregator responds with one of the following:
  - JOB_DOWNLOAD_MODEL: this prompts the _collaborator_ to download the current model weights. Note that this is not code. It is a list of tensors (matrices). The model algorithm code is static. The _collaborator_ will overwrite its current model weights with these values. The current FeTS model requires 112 MB of tensors on each download/upload.
  - JOB_UPLOAD_RESULTS: this prompts the _collaborator_ to upload some result. It includes a name from a list of values defined as the layer names of the network, or the metrics "shared model validation", "local model validation", "loss" (training loss). The metrics are each a single floating point value. The model layers are 112 MB of tensors (just as with the download).
  - JOB_SLEEP: this prompts the _collaborator_ to do nothing until its next polling interval (generally sent when the _collaborator_ is done for the round and other _collaborators_ are still training/validating).
  - JOB_QUIT: this prompts the _collaborator_ process to exit because the training of the model is completed.
* The "job" request is a unary->unary gRPC. Based on the job received, the following gRPCs will be of the following types:
  - JOB_DOWNLOAD_MODEL: this results in a sequence of unary->datastream requests where the _collaborator_ requests each model tensor in a loop.
  - JOB_UPLOAD_RESULTS: this results in a sequence of datastream->unary requests where the _collaborator_ uploads each model tensor and the three metrics (2 validation, 1 training loss).

#### Testing a Collaborator
In order to test a _collaborator_, the _aggregator_ can be set not aggregate updates from specific _collaborators_ so that they can test functionality without impacting the running model training/validation. We hope this enables InfoSec runtime analysis against the live _aggregator_ without the need to access any private data on the _collaborator_ machine running the test. To do this, please coordinate with the FeTS aggregator admins.
