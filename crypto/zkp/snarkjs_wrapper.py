"""
Snarkjs Wrapper for Python

Provides Python interface to snarkjs CLI for generating and verifying zk-SNARKs.

Requirements:
    - Node.js and npm installed
    - snarkjs: npm install -g snarkjs
    - circom: npm install -g circom

Usage:
    wrapper = SnarkjsWrapper(
        circuit_path="circuits/reputation_tier.circom",
        protocol="groth16"
    )

    # Setup (one-time)
    wrapper.compile_circuit()
    wrapper.generate_keys()

    # Generate proof
    witness = {"reputation_score": 75, "robot_id": 12345, "salt": 67890, "claimed_tier": 3}
    proof = wrapper.generate_proof(witness)

    # Verify proof
    is_valid = wrapper.verify_proof(proof, public_signals=[3])
"""

import json
import subprocess
import tempfile
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class SnarkProof:
    """Structured zk-SNARK proof"""
    pi_a: List[str]  # G1 point (3 elements)
    pi_b: List[List[str]]  # G2 point (3x2 elements)
    pi_c: List[str]  # G1 point (3 elements)
    protocol: str
    curve: str


class SnarkjsWrapper:
    """
    Python wrapper for snarkjs CLI.

    Manages circuit compilation, trusted setup, proof generation, and verification.
    """

    def __init__(
        self,
        circuit_path: str,
        protocol: str = "groth16",
        build_dir: str = "build/circuits",
        ptau_file: Optional[str] = None
    ):
        """
        Initialize snarkjs wrapper.

        Args:
            circuit_path: Path to .circom circuit file
            protocol: ZK protocol (groth16, plonk, fflonk)
            build_dir: Directory for compiled artifacts
            ptau_file: Path to powers of tau file (optional, will download if needed)
        """
        self.circuit_path = Path(circuit_path)
        self.protocol = protocol
        self.build_dir = Path(build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)

        self.circuit_name = self.circuit_path.stem
        self.logger = logging.getLogger(__name__)

        # Artifact paths
        self.r1cs_path = self.build_dir / f"{self.circuit_name}.r1cs"
        self.wasm_path = self.build_dir / f"{self.circuit_name}_js"
        self.zkey_path = self.build_dir / f"{self.circuit_name}.zkey"
        self.vkey_path = self.build_dir / f"{self.circuit_name}_verification_key.json"

        # Powers of tau file
        if ptau_file:
            self.ptau_path = Path(ptau_file)
        else:
            # Use default ptau for small circuits (2^12 constraints)
            self.ptau_path = self.build_dir / "powersOfTau28_hez_final_12.ptau"

        # Check if snarkjs is installed
        self._check_snarkjs_installed()

    def _check_snarkjs_installed(self):
        """Check if snarkjs and circom are installed."""
        try:
            subprocess.run(["snarkjs", "version"], capture_output=True, check=True)
            subprocess.run(["circom", "--version"], capture_output=True, check=True)
            self.logger.info("snarkjs and circom are installed")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error(
                "snarkjs or circom not found. Please install:\n"
                "  npm install -g snarkjs circom"
            )
            raise RuntimeError("snarkjs/circom not installed") from e

    def compile_circuit(self) -> bool:
        """
        Compile Circom circuit to R1CS and WASM.

        Returns:
            True if compilation succeeded
        """
        if not self.circuit_path.exists():
            raise FileNotFoundError(f"Circuit not found: {self.circuit_path}")

        self.logger.info(f"Compiling circuit: {self.circuit_path}")

        try:
            # Compile circuit
            cmd = [
                "circom",
                str(self.circuit_path),
                "--r1cs",
                "--wasm",
                "--sym",
                "-o",
                str(self.build_dir)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            self.logger.info(f"Circuit compiled successfully")
            self.logger.debug(f"Compile output: {result.stdout}")

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Circuit compilation failed: {e.stderr}")
            raise

    def download_ptau(self):
        """Download powers of tau ceremony file if not present."""
        if self.ptau_path.exists():
            self.logger.info(f"Powers of tau file already exists: {self.ptau_path}")
            return

        self.logger.info("Downloading powers of tau file...")

        # Download from Hermez trusted setup ceremony
        # This is a universally trusted setup for circuits up to 2^12 constraints
        ptau_url = "https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_12.ptau"

        try:
            import requests
            response = requests.get(ptau_url, stream=True)
            response.raise_for_status()

            with open(self.ptau_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            self.logger.info(f"Downloaded ptau file to {self.ptau_path}")

        except Exception as e:
            self.logger.error(f"Failed to download ptau file: {e}")
            raise

    def generate_keys(self, entropy: Optional[str] = None) -> bool:
        """
        Generate proving and verification keys.

        Args:
            entropy: Optional entropy for key generation

        Returns:
            True if key generation succeeded
        """
        # Ensure ptau file exists
        self.download_ptau()

        # Ensure circuit is compiled
        if not self.r1cs_path.exists():
            self.logger.warning("R1CS not found, compiling circuit first")
            self.compile_circuit()

        self.logger.info(f"Generating {self.protocol} keys...")

        try:
            if self.protocol == "groth16":
                # Groth16 setup
                # 1. Start new zkey
                cmd_setup = [
                    "snarkjs",
                    "groth16",
                    "setup",
                    str(self.r1cs_path),
                    str(self.ptau_path),
                    str(self.zkey_path)
                ]
                subprocess.run(cmd_setup, capture_output=True, check=True)

                # 2. Export verification key
                cmd_export = [
                    "snarkjs",
                    "zkey",
                    "export",
                    "verificationkey",
                    str(self.zkey_path),
                    str(self.vkey_path)
                ]
                subprocess.run(cmd_export, capture_output=True, check=True)

                self.logger.info("Groth16 keys generated successfully")
                return True

            elif self.protocol == "plonk":
                # PLONK setup
                cmd_setup = [
                    "snarkjs",
                    "plonk",
                    "setup",
                    str(self.r1cs_path),
                    str(self.ptau_path),
                    str(self.zkey_path)
                ]
                subprocess.run(cmd_setup, capture_output=True, check=True)

                # Export verification key
                cmd_export = [
                    "snarkjs",
                    "zkey",
                    "export",
                    "verificationkey",
                    str(self.zkey_path),
                    str(self.vkey_path)
                ]
                subprocess.run(cmd_export, capture_output=True, check=True)

                self.logger.info("PLONK keys generated successfully")
                return True

            else:
                raise ValueError(f"Unsupported protocol: {self.protocol}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Key generation failed: {e.stderr}")
            raise

    def generate_proof(
        self,
        witness: Dict[str, Any],
        public_signals: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a zk-SNARK proof.

        Args:
            witness: Dictionary of circuit inputs (private + public)
            public_signals: Optional list of public signals (extracted from witness if not provided)

        Returns:
            Proof dictionary with 'proof' and 'publicSignals' keys
        """
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(witness, f)
            input_path = f.name

        try:
            # Calculate witness
            witness_path = self.build_dir / "witness.wtns"

            cmd_witness = [
                "snarkjs",
                "wtns",
                "calculate",
                str(self.wasm_path / f"{self.circuit_name}.wasm"),
                input_path,
                str(witness_path)
            ]
            subprocess.run(cmd_witness, capture_output=True, check=True)

            # Generate proof
            proof_path = self.build_dir / "proof.json"
            public_path = self.build_dir / "public.json"

            if self.protocol == "groth16":
                cmd_prove = [
                    "snarkjs",
                    "groth16",
                    "prove",
                    str(self.zkey_path),
                    str(witness_path),
                    str(proof_path),
                    str(public_path)
                ]
            elif self.protocol == "plonk":
                cmd_prove = [
                    "snarkjs",
                    "plonk",
                    "prove",
                    str(self.zkey_path),
                    str(witness_path),
                    str(proof_path),
                    str(public_path)
                ]
            else:
                raise ValueError(f"Unsupported protocol: {self.protocol}")

            subprocess.run(cmd_prove, capture_output=True, check=True)

            # Load proof and public signals
            with open(proof_path) as f:
                proof = json.load(f)

            with open(public_path) as f:
                public_signals_output = json.load(f)

            result = {
                'proof': proof,
                'publicSignals': public_signals_output,
                'protocol': self.protocol,
                'curve': 'bn128'
            }

            self.logger.info("Proof generated successfully")
            return result

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Proof generation failed: {e.stderr}")
            raise

        finally:
            # Cleanup temporary files
            os.unlink(input_path)

    def verify_proof(
        self,
        proof: Dict[str, Any],
        public_signals: Optional[List[Any]] = None
    ) -> bool:
        """
        Verify a zk-SNARK proof.

        Args:
            proof: Proof dictionary (from generate_proof)
            public_signals: Optional public signals (extracted from proof if not provided)

        Returns:
            True if proof is valid
        """
        # Extract public signals if not provided
        if public_signals is None:
            public_signals = proof.get('publicSignals', [])

        # Create temporary files for verification
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(proof.get('proof', proof), f)
            proof_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(public_signals, f)
            public_path = f.name

        try:
            if self.protocol == "groth16":
                cmd_verify = [
                    "snarkjs",
                    "groth16",
                    "verify",
                    str(self.vkey_path),
                    str(public_path),
                    str(proof_path)
                ]
            elif self.protocol == "plonk":
                cmd_verify = [
                    "snarkjs",
                    "plonk",
                    "verify",
                    str(self.vkey_path),
                    str(public_path),
                    str(proof_path)
                ]
            else:
                raise ValueError(f"Unsupported protocol: {self.protocol}")

            result = subprocess.run(
                cmd_verify,
                capture_output=True,
                text=True,
                check=False  # Don't raise on failure, check return code
            )

            is_valid = result.returncode == 0 and "OK" in result.stdout

            if is_valid:
                self.logger.info("Proof verification succeeded")
            else:
                self.logger.warning(f"Proof verification failed: {result.stdout}")

            return is_valid

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Proof verification error: {e.stderr}")
            return False

        finally:
            # Cleanup
            os.unlink(proof_path)
            os.unlink(public_path)

    def is_setup_complete(self) -> bool:
        """Check if circuit setup is complete (compiled + keys generated)."""
        return (
            self.r1cs_path.exists() and
            self.wasm_path.exists() and
            self.zkey_path.exists() and
            self.vkey_path.exists()
        )
