# OpenFHE Integration Guide

## Overview

OpenFHE is an open-source library for Fully Homomorphic Encryption (FHE) that provides efficient implementations of major FHE schemes:

- **BGV**: Brakerski-Gentry-Vaikuntanathan scheme
- **BFV**: Brakerski-Fan-Vercauteren scheme
- **CKKS**: Cheon-Kim-Kim-Song scheme (for approximate arithmetic)
- **FHEW**: Fast Homomorphic Encryption for Boolean circuits

## Installation

### Option 1: Build from Source

```bash
# Clone OpenFHE repository
git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe-development

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build (use -j for parallel compilation)
make -j$(nproc)

# Install
sudo make install
```

### Option 2: Docker

```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgmp-dev \
    libntl-dev

# Clone and build OpenFHE
RUN git clone https://github.com/openfheorg/openfhe-development.git && \
    cd openfhe-development && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install
```

### Option 3: Python Bindings

```bash
# Install openfhe-python wrapper
pip install openfhe
```

## Using OpenFHE for SwarmBrain

### Use Case: Encrypted Model Update Validation

We use OpenFHE CKKS scheme to validate federated learning updates without decryption.

```cpp
#include "openfhe.h"

using namespace lbcrypto;

// Initialize CKKS cryptosystem
void initializeCKKS() {
    // Set crypto context parameters
    CCParams<CryptoContextCKKSRNS> parameters;

    parameters.SetMultiplicativeDepth(3);
    parameters.SetScalingModSize(50);
    parameters.SetBatchSize(8192);

    // Create crypto context
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // Enable features
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    // Generate keys
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalRotateKeyGen(keyPair.secretKey);

    return cc;
}

// Encrypt model updates
Ciphertext<DCRTPoly> encryptModelUpdate(
    CryptoContext<DCRTPoly> cc,
    PublicKey<DCRTPoly> publicKey,
    std::vector<double>& modelUpdate
) {
    // Create plaintext from model update
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(modelUpdate);

    // Encrypt
    auto ctxt = cc->Encrypt(publicKey, ptxt);

    return ctxt;
}

// Aggregate encrypted updates (homomorphic addition)
Ciphertext<DCRTPoly> aggregateEncryptedUpdates(
    CryptoContext<DCRTPoly> cc,
    std::vector<Ciphertext<DCRTPoly>>& encryptedUpdates
) {
    // Sum all encrypted updates
    auto aggregated = encryptedUpdates[0];

    for (size_t i = 1; i < encryptedUpdates.size(); i++) {
        aggregated = cc->EvalAdd(aggregated, encryptedUpdates[i]);
    }

    // Average
    int numUpdates = encryptedUpdates.size();
    aggregated = cc->EvalMult(aggregated, 1.0 / numUpdates);

    return aggregated;
}
```

### Python Example

```python
import openfhe as fhe

# Create crypto context
params = fhe.CCParamsCKKSRNS()
params.SetMultiplicativeDepth(3)
params.SetScalingModSize(50)
params.SetBatchSize(8192)

cc = fhe.GenCryptoContext(params)

# Enable operations
cc.Enable(fhe.PKESchemeFeature.PKE)
cc.Enable(fhe.PKESchemeFeature.KEYSWITCH)
cc.Enable(fhe.PKESchemeFeature.LEVELEDSHE)

# Generate keys
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)

# Encrypt model updates
model_update = [0.1, 0.2, 0.3, 0.4]  # Example gradient
plaintext = cc.MakeCKKSPackedPlaintext(model_update)
ciphertext = cc.Encrypt(keys.publicKey, plaintext)

# Homomorphic operations
aggregated = cc.EvalAdd(ciphertext, ciphertext)  # Add updates
scaled = cc.EvalMult(aggregated, 0.5)  # Average

# Decrypt (only on trusted server)
result = cc.Decrypt(keys.secretKey, scaled)
print(result)
```

## Integration with SwarmBrain

### Architecture

```
┌────────────────────┐
│   Robot Edge       │
│                    │
│  1. Train locally  │
│  2. Compute update │
│  3. Encrypt update │──────────┐
│     (OpenFHE CKKS) │          │
└────────────────────┘          │
                                 │ Encrypted
┌────────────────────┐          │ updates
│   Robot Edge       │          │
│                    │          │
│  1. Train locally  │          ▼
│  2. Compute update │    ┌──────────────┐
│  3. Encrypt update │───▶│  Aggregator  │
│     (OpenFHE CKKS) │    │              │
└────────────────────┘    │  Homomorphic │
                          │  aggregation │
                          │  (no decrypt)│
                          └──────┬───────┘
                                 │
                                 │ Aggregated
                                 │ encrypted
                                 │ update
                                 ▼
                          ┌──────────────┐
                          │ Cloud Server │
                          │              │
                          │  1. Decrypt  │
                          │  2. Validate │
                          │  3. Update   │
                          │     global   │
                          │     model    │
                          └──────────────┘
```

### Implementation Steps

1. **Edge Device** (`robot_control/policies/encrypted_policy.py`):
   - Train policy locally
   - Compute model update (gradient or weight diff)
   - Encrypt using OpenFHE CKKS
   - Send encrypted update to aggregator

2. **Aggregator** (`learning/secure_aggregation/fhe_aggregator.py`):
   - Collect encrypted updates from robots
   - Perform homomorphic aggregation (sum + average)
   - Forward aggregated encrypted update to server
   - **Never** decrypt at aggregator level

3. **Server** (`learning/federated_client/fl_server.py`):
   - Receive aggregated encrypted update
   - Decrypt using private key
   - Validate update (check for anomalies)
   - Apply to global model
   - Distribute updated model

### Performance Considerations

- **CKKS** is best for floating-point operations (model weights)
- **BGV/BFV** is better for integer operations
- **Batch encoding** improves throughput (pack multiple values)
- **Bootstrapping** is expensive - minimize multiplicative depth
- **Approximate arithmetic** in CKKS allows some noise

### Security Properties

- **Semantic Security**: Ciphertext reveals no information about plaintext
- **Circuit Privacy**: Evaluator learns nothing from homomorphic operations
- **Post-Quantum**: OpenFHE schemes are resistant to quantum attacks

## References

- [OpenFHE Documentation](https://openfhe-development.readthedocs.io/)
- [OpenFHE GitHub](https://github.com/openfheorg/openfhe-development)
- [CKKS Paper](https://eprint.iacr.org/2016/421)
- [OpenFHE Python Bindings](https://github.com/openfheorg/openfhe-python)
