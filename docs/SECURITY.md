# Security Notes

SYNTH Miner is a miner client. It is not a wallet and not a transaction signer.

Expected inputs:

- payout address
- optional pool URL
- optional performance parameters

Never enter a private key, seed phrase, keystore, or exchange credential into the miner.

The official pool verifies all submitted work server-side. Invalid nonce slots, expired leases, duplicate submissions, and sequence conflicts are rejected by the pool.

Report security issues privately to the project maintainer before public disclosure.
