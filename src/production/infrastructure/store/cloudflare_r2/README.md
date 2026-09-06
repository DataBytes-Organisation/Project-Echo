# Cloudflare R2 engine and simulator prototype

This prototype reproduces the Project Echo storage behaviours used by the
production engine and simulator without changing either production component.

- Engine: list sorted species names from the first dataset directory.
- Simulator: group objects by species, randomly select an audio object, and
  download its bytes.
- Shared storage: connect, paginate listings, upload, and download.

The existing GCP code, dependencies, containers, and configuration remain the
production path.

## Configuration

Create a private Standard R2 bucket, leave public access disabled, and create a
token with Object Read & Write access restricted to the prototype bucket. Set
the variables shown in `.env.example` in the local environment. Never commit
real credentials.

Objects must use this layout:

```text
prototype/<species name>/<audio file>
```

Install `requirements.txt`, then run from the parent `store` directory:

```powershell
python -m cloudflare_r2.demo
```

Expected final output:

```text
R2 connection: PASS
Engine species loading: PASS (... species)
Engine/simulator species agreement: PASS
Simulator random audio download: PASS (...)
Prototype integration: PASS
```

This is a prototype acceptance check, not a production migration. Do not remove
GCP until Krish validates this output and the later end-to-end integration.
