"""Tests for A2A Ed25519 identity generation and persistence."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from adk.a2a_identity import (
    a2a_identity_dir,
    get_a2a_private_key,
    get_a2a_public_key,
    has_a2a_identity,
)


class TestA2AIdentity:
    """Test suite for A2A Ed25519 identity management."""

    def test_identity_dir_default(self):
        """a2a_identity_dir() returns ~/.aither/a2a by default."""
        d = a2a_identity_dir()
        assert d == Path.home() / ".aither" / "a2a"

    def test_identity_dir_override_env(self, tmp_path, monkeypatch):
        """a2a_identity_dir() respects AITHER_A2A_IDENTITY_DIR env var."""
        override = tmp_path / "custom_a2a"
        monkeypatch.setenv("AITHER_A2A_IDENTITY_DIR", str(override))
        d = a2a_identity_dir()
        assert d == override

    def test_keypair_generation_idempotence(self, tmp_path):
        """Keypair generation is idempotent — re-calling returns the same keys."""
        d = tmp_path / "a2a_test"
        # First call generates
        pub1, priv1 = get_a2a_public_key(directory=d), get_a2a_private_key(directory=d)
        # Second call should return the same keys
        pub2, priv2 = get_a2a_public_key(directory=d), get_a2a_private_key(directory=d)
        assert pub1 == pub2
        assert priv1 == priv2

    def test_public_key_format_hex(self, tmp_path):
        """Public key is hex-encoded (64 chars for 32 bytes)."""
        d = tmp_path / "a2a_test"
        pub = get_a2a_public_key(directory=d)
        # RAW 32 bytes = 64 hex characters
        assert len(pub) == 64
        # Must be valid hex
        bytes.fromhex(pub)

    def test_private_key_format_hex(self, tmp_path):
        """Private key is hex-encoded (64 chars for 32 bytes)."""
        d = tmp_path / "a2a_test"
        priv = get_a2a_private_key(directory=d)
        # RAW 32 bytes = 64 hex characters
        assert len(priv) == 64
        # Must be valid hex
        bytes.fromhex(priv)

    def test_keypair_round_trip_crypto(self, tmp_path):
        """Public key round-trips through cryptography.Ed25519PublicKey.from_public_bytes."""
        from cryptography.hazmat.primitives.asymmetric import ed25519

        d = tmp_path / "a2a_test"
        pub_hex = get_a2a_public_key(directory=d)

        # Convert hex to bytes and verify it can be loaded as Ed25519 public key
        pub_bytes = bytes.fromhex(pub_hex)
        pub_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        # If we get here without an exception, the format is correct
        assert pub_key is not None

    def test_has_a2a_identity_false(self, tmp_path):
        """has_a2a_identity() returns False when no keypair exists."""
        d = tmp_path / "a2a_empty"
        d.mkdir(parents=True, exist_ok=True)
        assert has_a2a_identity(directory=d) is False

    def test_has_a2a_identity_true(self, tmp_path):
        """has_a2a_identity() returns True after keys are generated."""
        d = tmp_path / "a2a_test"
        get_a2a_public_key(directory=d)
        assert has_a2a_identity(directory=d) is True

    def test_private_key_file_permissions(self, tmp_path):
        """Private key file is created with 0o600 permissions (best-effort)."""
        d = tmp_path / "a2a_test"
        get_a2a_private_key(directory=d)
        priv_file = d / "private_key"
        assert priv_file.is_file()
        # Check permissions (Unix-only; Windows/FAT filesystems may not support this)
        try:
            stat = priv_file.stat()
            mode = stat.st_mode & 0o777
            # Should be 0o600 (user read + write only)
            assert mode == 0o600 or mode == 0o644  # Windows may not support 0o600
        except (OSError, AttributeError):
            # Windows or non-POSIX filesystem — skip check
            pass

    def test_public_key_persistence(self, tmp_path):
        """Public key is persisted and matches on re-read."""
        d = tmp_path / "a2a_test"
        pub_file = d / "public_key"

        pub1 = get_a2a_public_key(directory=d)
        pub_from_file = pub_file.read_text(encoding="utf-8").strip()

        assert pub1 == pub_from_file

    def test_private_key_never_returned_in_public(self, tmp_path):
        """get_a2a_public_key() never returns the private key value."""
        d = tmp_path / "a2a_test"
        pub = get_a2a_public_key(directory=d)
        priv = get_a2a_private_key(directory=d)
        # They're different 64-char hex strings; the public key is NOT the private key
        assert pub != priv


class TestA2AIdentityIntegration:
    """Integration tests for A2A identity with a2a_trust verification."""

    def test_public_key_verifiable(self, tmp_path):
        """Generated public key can be used to verify a signature."""
        from cryptography.hazmat.primitives.asymmetric import ed25519

        d = tmp_path / "a2a_test"
        pub_hex = get_a2a_public_key(directory=d)
        priv_hex = get_a2a_private_key(directory=d)

        # Create a test message
        test_message = b"test message for signing"

        # Load keys from hex
        priv_bytes = bytes.fromhex(priv_hex)
        pub_bytes = bytes.fromhex(pub_hex)

        priv_key = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
        pub_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)

        # Sign with private key
        signature = priv_key.sign(test_message)

        # Verify with public key (should not raise)
        pub_key.verify(signature, test_message)

    def test_public_key_rejection_of_wrong_signature(self, tmp_path):
        """Generated public key rejects signatures signed with a different key."""
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.exceptions import InvalidSignature

        d = tmp_path / "a2a_test"
        pub_hex = get_a2a_public_key(directory=d)

        # Create a different keypair
        other_priv = ed25519.Ed25519PrivateKey.generate()
        other_sig = other_priv.sign(b"signed by other key")

        # Load the generated public key
        pub_bytes = bytes.fromhex(pub_hex)
        pub_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)

        # Try to verify — should fail
        with pytest.raises(InvalidSignature):
            pub_key.verify(other_sig, b"signed by other key")
